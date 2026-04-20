"""
etl_pipeline.py
===============
Reusable ETL utilities for hospital patient datasets.

Exported functions
------------------
basic_clean(df)          -> DataFrame   # generic, dataset-agnostic cleaning
hospital_clean(df)       -> DataFrame   # hospital-specific cleaning on top of basic_clean
run_pipeline(raw_path,
             processed_path,
             hospital_label) -> DataFrame   # end-to-end: read → clean → save → report
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # removed in pandas 3.x


# ──────────────────────────────────────────────────────────────────────────────
# 1.  GENERIC / BASIC CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset-agnostic cleaning steps applied to every dataframe.

    Steps
    -----
    1. Strip leading/trailing whitespace from column names.
    2. Strip leading/trailing whitespace from all string cells.
    3. Replace empty-string cells with NaN.
    4. Drop fully-duplicate rows.
    5. Reset the index.

    Returns a *copy* — never mutates the original.
    """
    df = df.copy()

    # 1. Clean column names
    df.columns = df.columns.str.strip()

    # 2. Strip string values
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # 3. Empty strings → NaN
    df[str_cols] = df[str_cols].replace("", np.nan)

    # 4. Drop fully-duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"  [basic_clean] Dropped {before - after} fully-duplicate rows.")

    # 5. Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  HOSPITAL-SPECIFIC CLEANING
# ──────────────────────────────────────────────────────────────────────────────

# ── 2a. Categorical standardisation maps ─────────────────────────────────────

_GENDER_MAP: dict[str, str] = {
    # all case variants → canonical title-case
    "male":   "Male",
    "female": "Female",
}

_REGION_MAP: dict[str, str] = {
    "north": "North",
    "south": "South",
    "east":  "East",
    "west":  "West",
}

# Valid domain sets (used for out-of-domain detection)
_VALID_GENDER          = {"Male", "Female"}
_VALID_REGION          = {"North", "South", "East", "West"}
_VALID_ECONOMIC_STATUS = {"Low", "Middle", "High"}
_VALID_OUTCOME         = {"Recovered", "Under Treatment", "Deceased"}
_VALID_INSURANCE       = {"Yes", "No"}
_VALID_DIAGNOSIS       = {"Asthma", "Covid", "Hypertension", "Diabetes", "Flu"}
_VALID_HOSPITAL_TYPE   = {"Government", "Private"}


def _standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise case/spacing in every categorical column."""

    # Gender: case-insensitive map
    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .str.strip()
            .str.lower()
            .map(_GENDER_MAP)           # unknown values become NaN — handled later
        )

    # Region: title-case
    if "Region" in df.columns:
        df["Region"] = (
            df["Region"]
            .str.strip()
            .str.lower()
            .map(_REGION_MAP)
        )

    # Remaining categoricals: strip & title-case (already title-cased in source,
    # but this future-proofs against new data)
    for col in ("Economic_Status", "Outcome", "Insurance",
                "Diagnosis", "Hospital_Type", "Hospital"):
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Re-map "Under Treatment" which title() breaks to "Under Treatment" ✓
    # (title() is fine here because each word starts uppercase already)

    return df


def _drop_out_of_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows whose categorical values fall outside the known valid sets.
    Nulls are kept here (imputed in a later step).
    """
    domain_checks = {
        "Gender":          _VALID_GENDER,
        "Region":          _VALID_REGION,
        "Economic_Status": _VALID_ECONOMIC_STATUS,
        "Outcome":         _VALID_OUTCOME,
        "Insurance":       _VALID_INSURANCE,
        "Diagnosis":       _VALID_DIAGNOSIS,
        "Hospital_Type":   _VALID_HOSPITAL_TYPE,
    }
    before = len(df)
    for col, valid_set in domain_checks.items():
        if col not in df.columns:
            continue
        mask_invalid = df[col].notna() & ~df[col].isin(valid_set)
        if mask_invalid.any():
            print(f"  [domain] Dropping {mask_invalid.sum()} rows with invalid '{col}': "
                  f"{df.loc[mask_invalid, col].unique().tolist()}")
            df = df[~mask_invalid]

    after = len(df)
    if before != after:
        print(f"  [domain] Total rows removed: {before - after}")
    return df.reset_index(drop=True)


def _fix_numeric_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business-rule–based outlier handling for numeric columns.

    Age                   : must be 0–120 → out-of-range → NaN
    Treatment_Cost        : must be >= 0  → negative values → NaN
    Doctor_Experience_Years: must be 0–60 → out-of-range → NaN
    Cleanliness_Score     : must be 1–5   → out-of-range → NaN
    Doctor_Availability   : must be 1–14  → out-of-range → NaN
    """
    rules: list[tuple[str, str, float, float]] = [
        # (column, description, min_valid, max_valid)
        ("Age",                    "Age (0-120)",              0,   120),
        ("Treatment_Cost",         "Treatment_Cost (>=0)",     0,   1e9),
        ("Doctor_Experience_Years","Doctor_Experience_Years (0-60)", 0, 60),
        ("Cleanliness_Score",      "Cleanliness_Score (1-5)",  1,   5),
        ("Doctor_Availability",    "Doctor_Availability (1-14)", 1, 14),
    ]

    for col, desc, lo, hi in rules:
        if col not in df.columns:
            continue
        mask_bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if mask_bad.any():
            print(f"  [outlier] {mask_bad.sum()} invalid '{col}' values ({desc}) → NaN")
            df.loc[mask_bad, col] = np.nan

    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute remaining missing values with domain-appropriate strategies.

    Numeric
    -------
    Age                     : median (robust to skew)
    Treatment_Cost          : median per Diagnosis group (cost varies by condition)
    Doctor_Experience_Years : median per Hospital group
    Cleanliness_Score       : median per Hospital (scores are hospital-level)
    Doctor_Availability     : mode per Hospital

    Categorical
    -----------
    Gender                  : mode
    Region                  : mode per Hospital
    Insurance               : 'No'  (conservative assumption)
    Economic_Status         : mode
    Outcome                 : not imputed — too clinically sensitive; rows dropped
    Diagnosis               : not imputed — clinical field; rows dropped
    """

    # ── Numeric ──────────────────────────────────────────────────────────────

    if "Age" in df.columns and df["Age"].isna().any():
        median_age = df["Age"].median()
        n = df["Age"].isna().sum()
        df["Age"] = df["Age"].fillna(median_age)
        print(f"  [impute] Age: filled {n} nulls with global median ({median_age:.1f})")

    if "Treatment_Cost" in df.columns and df["Treatment_Cost"].isna().any():
        if "Diagnosis" in df.columns:
            group_median = df.groupby("Diagnosis")["Treatment_Cost"].transform("median")
            global_median = df["Treatment_Cost"].median()
            filled = group_median.fillna(global_median)
            n = df["Treatment_Cost"].isna().sum()
            df["Treatment_Cost"] = df["Treatment_Cost"].fillna(filled)
            print(f"  [impute] Treatment_Cost: filled {n} nulls with per-Diagnosis median")
        else:
            n = df["Treatment_Cost"].isna().sum()
            df["Treatment_Cost"] = df["Treatment_Cost"].fillna(df["Treatment_Cost"].median())
            print(f"  [impute] Treatment_Cost: filled {n} nulls with global median")

    if "Doctor_Experience_Years" in df.columns and df["Doctor_Experience_Years"].isna().any():
        if "Hospital" in df.columns:
            group_median = df.groupby("Hospital")["Doctor_Experience_Years"].transform("median")
            global_median = df["Doctor_Experience_Years"].median()
            filled = group_median.fillna(global_median)
            n = df["Doctor_Experience_Years"].isna().sum()
            df["Doctor_Experience_Years"] = df["Doctor_Experience_Years"].fillna(filled)
            print(f"  [impute] Doctor_Experience_Years: filled {n} nulls with per-Hospital median")
        else:
            n = df["Doctor_Experience_Years"].isna().sum()
            df["Doctor_Experience_Years"] = df["Doctor_Experience_Years"].fillna(
                df["Doctor_Experience_Years"].median()
            )
            print(f"  [impute] Doctor_Experience_Years: filled {n} nulls with global median")

    if "Cleanliness_Score" in df.columns and df["Cleanliness_Score"].isna().any():
        if "Hospital" in df.columns:
            group_median = df.groupby("Hospital")["Cleanliness_Score"].transform("median")
            global_median = df["Cleanliness_Score"].median()
            filled = group_median.fillna(global_median)
            n = df["Cleanliness_Score"].isna().sum()
            df["Cleanliness_Score"] = df["Cleanliness_Score"].fillna(filled)
            print(f"  [impute] Cleanliness_Score: filled {n} nulls with per-Hospital median")

    if "Doctor_Availability" in df.columns and df["Doctor_Availability"].isna().any():
        if "Hospital" in df.columns:
            group_mode = df.groupby("Hospital")["Doctor_Availability"].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else x.median()
            )
            n = df["Doctor_Availability"].isna().sum()
            df["Doctor_Availability"] = df["Doctor_Availability"].fillna(group_mode)
            print(f"  [impute] Doctor_Availability: filled {n} nulls with per-Hospital mode")

    # ── Categorical ───────────────────────────────────────────────────────────

    if "Gender" in df.columns and df["Gender"].isna().any():
        mode_val = df["Gender"].mode().iloc[0]
        n = df["Gender"].isna().sum()
        df["Gender"] = df["Gender"].fillna(mode_val)
        print(f"  [impute] Gender: filled {n} nulls with mode ('{mode_val}')")

    if "Region" in df.columns and df["Region"].isna().any():
        if "Hospital" in df.columns:
            group_mode = df.groupby("Hospital")["Region"].transform(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
            n = df["Region"].isna().sum()
            df["Region"] = df["Region"].fillna(group_mode)
            print(f"  [impute] Region: filled {n} nulls with per-Hospital mode")

    if "Insurance" in df.columns and df["Insurance"].isna().any():
        n = df["Insurance"].isna().sum()
        df["Insurance"] = df["Insurance"].fillna("No")
        print(f"  [impute] Insurance: filled {n} nulls with 'No' (conservative default)")

    if "Economic_Status" in df.columns and df["Economic_Status"].isna().any():
        mode_val = df["Economic_Status"].mode().iloc[0]
        n = df["Economic_Status"].isna().sum()
        df["Economic_Status"] = df["Economic_Status"].fillna(mode_val)
        print(f"  [impute] Economic_Status: filled {n} nulls with mode ('{mode_val}')")

    # Drop rows where clinically critical fields are still null
    critical_cols = [c for c in ("Outcome", "Diagnosis") if c in df.columns]
    if critical_cols:
        before = len(df)
        df.dropna(subset=critical_cols, inplace=True)
        after = len(df)
        if before != after:
            print(f"  [impute] Dropped {before - after} rows with null critical fields "
                  f"({critical_cols})")

    return df.reset_index(drop=True)


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their correct final dtypes."""

    int_cols   = ["Cleanliness_Score", "Doctor_Availability"]
    float_cols = ["Age", "Treatment_Cost", "Doctor_Experience_Years"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # Categorical columns → pandas Categorical (saves memory, enables ordering)
    cat_map: dict[str, list[str] | None] = {
        "Gender":          ["Male", "Female"],
        "Region":          ["North", "South", "East", "West"],
        "Economic_Status": ["Low", "Middle", "High"],
        "Outcome":         ["Recovered", "Under Treatment", "Deceased"],
        "Insurance":       ["Yes", "No"],
        "Diagnosis":       None,   # unordered
        "Hospital_Type":   None,
    }
    for col, cats in cat_map.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=cats, ordered=bool(cats))

    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight feature engineering that supports downstream analysis
    without leaking any target information.
    """
    # Age group
    if "Age" in df.columns:
        bins   = [0, 17, 35, 60, 120]
        labels = ["Child", "Young Adult", "Middle-Aged", "Senior"]
        df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    # Cost tier (percentile-based, per-dataset)
    if "Treatment_Cost" in df.columns:
        df["Cost_Tier"] = pd.qcut(
            df["Treatment_Cost"],
            q=4,
            labels=["Low", "Medium", "High", "Very High"],
            duplicates="drop",
        )

    return df


# ── 2b. Main hospital_clean entry point ──────────────────────────────────────

def hospital_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full hospital-specific cleaning pipeline.

    Order of operations
    -------------------
    1. basic_clean          – whitespace, empty strings, dedup
    2. standardise cats     – case normalisation
    3. out-of-domain check  – drop rows with unrecognisable categories
    4. numeric outliers     – business-rule range checks → NaN
    5. impute missing       – smart group-aware imputation
    6. enforce dtypes       – correct final types
    7. derived features     – Age_Group, Cost_Tier
    """
    print("  Step 1/7 : basic_clean …")
    df = basic_clean(df)

    print("  Step 2/7 : standardise categoricals …")
    df = _standardise_categoricals(df)

    print("  Step 3/7 : drop out-of-domain values …")
    df = _drop_out_of_domain(df)

    print("  Step 4/7 : fix numeric outliers …")
    df = _fix_numeric_outliers(df)

    print("  Step 5/7 : impute missing values …")
    df = _impute_missing(df)

    print("  Step 6/7 : enforce dtypes …")
    df = _enforce_dtypes(df)

    print("  Step 7/7 : add derived features …")
    df = _add_derived_features(df)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  END-TO-END PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    raw_path: str | Path,
    processed_path: str | Path,
    hospital_label: str = "Hospital",
) -> pd.DataFrame:
    """
    Read → clean → save → print quality report.

    Parameters
    ----------
    raw_path        : path to the raw CSV
    processed_path  : destination path for the cleaned CSV
    hospital_label  : display name used in the quality report

    Returns
    -------
    Cleaned DataFrame (also saved to processed_path).
    """
    raw_path       = Path(raw_path)
    processed_path = Path(processed_path)

    print(f"\n{'='*60}")
    print(f"  Pipeline: {hospital_label}")
    print(f"  Source  : {raw_path}")
    print(f"{'='*60}")

    # ── Read ──────────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(raw_path)
    print(f"\n[read]  {len(df_raw):,} rows × {df_raw.shape[1]} columns loaded.\n")

    # ── Clean ────────────────────────────────────────────────────────────────
    df_clean = hospital_clean(df_raw)

    # ── Save ─────────────────────────────────────────────────────────────────
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    print(f"\n[save]  Cleaned dataset saved → {processed_path}")

    # ── Quality report ───────────────────────────────────────────────────────
    _print_quality_report(df_raw, df_clean, hospital_label)

    return df_clean


# ──────────────────────────────────────────────────────────────────────────────
# 4.  QUALITY REPORT HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _print_quality_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n{'─'*60}")
    print(f"  Quality Report : {label}")
    print(f"{'─'*60}")
    print(f"  Rows  : {len(df_raw):>7,}  →  {len(df_clean):>7,}  "
          f"(removed {len(df_raw) - len(df_clean):,})")
    print(f"  Cols  : {df_raw.shape[1]:>7}  →  {df_clean.shape[1]:>7}  "
          f"(added {df_clean.shape[1] - df_raw.shape[1]} derived cols)")

    null_before = df_raw.isnull().sum().sum()
    null_after  = df_clean.isnull().sum().sum()
    print(f"  Nulls : {null_before:>7,}  →  {null_after:>7,}")

    dupes_before = df_raw.duplicated().sum()
    dupes_after  = df_clean.duplicated().sum()
    print(f"  Dupes : {dupes_before:>7,}  →  {dupes_after:>7,}")

    print(f"\n  Remaining nulls per column:")
    remaining = df_clean.isnull().sum()
    remaining = remaining[remaining > 0]
    if remaining.empty:
        print("    (none)")
    else:
        for col, cnt in remaining.items():
            print(f"    {col:<30} {cnt:>6,}")
    print(f"{'─'*60}\n")
