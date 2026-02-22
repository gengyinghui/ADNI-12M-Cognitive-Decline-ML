# ADNI-12M-Cognitive-Decline-ML

Reproducible machine learning pipeline for predicting 12-month cognitive
decline (≥3-point MMSE decline) in Alzheimer's disease using ADNI
clinical and MRI data.

------------------------------------------------------------------------

## Overview

This repository provides a fully reproducible analysis pipeline for
developing and validating machine learning models to predict 12-month
cognitive decline in Alzheimer's disease patients.

The pipeline includes:

-   Data preprocessing and cohort construction\
-   Feature engineering\
-   Model development (Logistic Regression, Random Forest)\
-   5-fold stratified cross-validation\
-   Isotonic calibration\
-   Performance evaluation (ROC, AUC, Brier score, calibration, decision
    curve analysis)\
-   SHAP-based model interpretability\
-   Automatic generation of all manuscript figures and tables

------------------------------------------------------------------------

## Repository Structure

-   `00_build_analysis_ready_from_ADNIMERGE.py` -- data preprocessing
    and cohort construction\
-   `01_archive_legacy_model.py` -- legacy model reference\
-   `02_plot_outputs_from_csv.py` -- plotting outputs\
-   `03_dca_plot_R_template.R` -- decision curve analysis\
-   `04_make_table1_descriptives.py` -- descriptive statistics\
-   `05_export_submission_tables.py` -- main tables export\
-   `06_make_supp_tables_S2_S4.py` -- supplementary tables S2--S4\
-   `07_make_supp_figures_S1_S4.py` -- supplementary figures S1--S4\
-   `08_sensitivity_ventricles_icv.py` -- sensitivity analysis\
-   `09_make_supp_tables_S6_bootstrap_CI.py` -- bootstrap CI analysis\
-   `10_make_supp_tables_S7_S8_S9.py` -- additional supplementary
    tables\
-   `11_make_table1_and_suppS2_from_raw_ADNIMERGE.py` -- table
    generation from raw data\
-   `config.py` -- configuration\
-   `run_all.py` -- full pipeline execution\
-   `requirements.txt` -- dependencies

------------------------------------------------------------------------

## Reproducibility

All analyses are fully reproducible using the provided scripts.

To run the complete pipeline:

``` bash
pip install -r requirements.txt
python run_all.py
```

All preprocessing steps are performed within cross-validation folds to
avoid data leakage.

------------------------------------------------------------------------

## Data Availability

The data used in this study are obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI).

Data are publicly available upon registration at:\
https://adni.loni.usc.edu/

Due to data use agreements, raw data are not included in this
repository.

------------------------------------------------------------------------

## Code Availability

All analysis scripts required to reproduce the results, figures, and
tables are provided in this repository.

Repository:\
https://github.com/gengyinghui/ADNI-12M-Cognitive-Decline-ML

------------------------------------------------------------------------

## Reproducibility Statement

All results reported in the manuscript can be reproduced using the
provided scripts.

The full pipeline, including data preprocessing, model training,
validation, calibration, and figure/table generation, can be executed
via:

``` bash
python run_all.py
```

------------------------------------------------------------------------

## License

This project is licensed under the MIT License.

------------------------------------------------------------------------

## Citation

Geng Y, Zhang H, et al.\
Machine-learning prediction and risk stratification of 12-month
cognitive decline in Alzheimer's disease using routine clinical and MRI
data.\
(Currently under review)
