# 🏥 HealthOps Analytics: Hospital Operations & Patient Outcome Optimization

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tableau](https://img.shields.io/badge/Visualization-Tableau-orange.svg)](https://www.tableau.com/)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)](https://github.com/Pinfinity07/SectionB_G11_HealthOps_Analytics)

## 📋 Project Overview
This project provides a comprehensive data-driven analysis of hospital operations, patient outcomes, and treatment costs for **Section B — Group 11**. By leveraging a dataset of 24,000 records across multiple hospital facilities, we aim to identify operational inefficiencies, cost drivers, and key factors affecting patient recovery.

The goal is to move beyond descriptive statistics into **decision-ready intelligence**, providing the Strategy Lead and Hospital Administrators with evidence-backed recommendations to optimize resource allocation and care quality.

---

## 📂 Repository Structure
- **`notebooks/`**: Modularized pipeline for ETL, EDA, Statistical Analysis, and Tableau Load Prep.
- **`data/processed/`**: Cleaned and enriched analytical datasets.
- **`docs/`**: Project documentation, data dictionary, and handoff notes.
- **`tableau/`**: Tableau workbook templates and dashboard exports.

---

## 📊 Dataset Profile
The analysis is based on three cleaned hospital datasets, synthesized into a unified analytical fact table.

- **Sample Size**: 24,000 Patient Records
- **Facilities**: 3 Hospitals (Medilife, Sunrise, Carepoint)
- **Clinical Scope**: 5 Primary Diagnoses (Asthma, Covid, Diabetes, Flu, Hypertension)
- **Quality Metrics**: Doctor experience, facility cleanliness, and provider availability scores.

For a full breakdown of variables, see the [Data Dictionary](docs/data_dictionary.md).

---

## 👥 The Team (Section B - Group 11)

| Role | Primary Responsibility | Owner Name |
|:---|:---|:---|
| **Project Lead** | Repo setup, timeline management, submission compliance, and Gate 1. | Manan Bansal |
| **Data Lead** | Dataset sourcing, raw data commit, and owning `docs/data_dictionary.md`. | Arjun Singh |
| **ETL Lead** | Notebooks 01 and 02 — Python extraction and cleaning pipeline. | Shrihari K N |
| **Analysis Lead** | Notebooks 03 and 04 — EDA and statistical analysis in Python. | Udit Jain |
| **Visualization Lead** | Tableau dashboard design, publishing to Tableau Public, and screenshots. | Manan Bansal |
| **Strategy Lead** | Problem statement, KPI framework, and business recommendations. | Harshita Joshi |
| **PPT & Quality Lead** | Final report PDF, presentation deck, and contribution quality. | Pragya Kashyap |

---

## 🚀 Key Insights Summary
1. **Diagnosis as a Cost Driver**: Diagnosis accounts for the largest variance in treatment cost (p < 0.001), with Covid-19 identified as the most resource-intensive.
2. **Hospital Efficiency**: Significant cost variance detected between facilities (e.g., Sunrise Hosp. vs Carepoint), warranting a review of operational overhead.
3. **Outcome Predictors**: Patient age and diagnosis are statistically associated with recovery rates, while insurance status showed no significant correlation with treatment outcomes in this population.
4. **Operations Index**: We developed a weighted **Operational Access Index** to rank hospitals based on actual service capacity and facility quality.

---

## 🛠️ Usage
1. Follow the `notebooks/` in numerical order (01-05) to reproduce the analytical pipeline.
2. The final Tableau-optimized dataset is located at `data/processed/hospital_tableau_ready.csv`.
3. View the [Analysis Handoff Notes](docs/analysis_lead_handoff.md) for detailed statistical evidence.
