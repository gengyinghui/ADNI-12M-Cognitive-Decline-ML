#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for the Scientific Reports reproducibility package.

All "rules" are taken from the submitted manuscript:
- Outcome: 12-month MMSE decline >=3 points
- Analytic sample: baseline AD with complete 12-month MMSE (n=306 in the paper)
- Candidate predictors: demographics/genetics + cognition/severity + MRI volumes
- Internal validation: 5-fold stratified CV, within-fold preprocessing, median/mode single imputation,
  standardization, one-hot encoding, isotonic calibration within fold.
- RF tuned hyperparameters: n_estimators=200, max_features='sqrt', max_depth=None
- Threshold metrics: Table 3 (0.25 / 0.40 / 0.50) + Supp Table S3 (0.20/0.30/0.40/0.50/0.60)
- Risk strata: low <0.25, intermediate 0.25–0.50, high >=0.50
- DCA thresholds: 0.20–0.80
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

RANDOM_SEED: int = 20260221

# Fixed thresholds and risk strata (manuscript)
TABLE3_THRESHOLDS = [0.25, 0.40, 0.50]
SUPP_S3_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]
DCA_THRESHOLD_GRID = [round(x, 2) for x in [i/100 for i in range(20, 81)]]
RISK_STRATA = [
    ("low", 0.0, 0.25),
    ("intermediate", 0.25, 0.50),
    ("high", 0.50, 1.0000001),
]

# Columns expected in the analysis-ready CSV created by 00_build_analysis_ready_from_ADNIMERGE.py.
# The script lowercases all column names; we keep the lowercase names here.
RID_COL = "rid"
Y_COL = "decline_3pt"
MMSE_CHANGE_COL = "mmse_change"

# Candidate predictor columns (lowercase), per manuscript.
# Note: ADNIMERGE naming can vary across releases; we therefore allow synonyms via COLUMN_SYNONYMS.
CANDIDATE_PREDICTORS: List[str] = [
    # Demographics/genetics
    "age",
    "ptgender",     # sex
    "pteducat",     # years education
    "apoe4",        # allele count -> derive carrier
    # Baseline cognition / severity
    "mmse_bl",
    "adas13_bl",
    "adas11_bl",
    "adasq4_bl",
    "cdrsb_bl",
    "faq_bl",
    # MRI volumes (baseline)
    "hippocampus_bl",
    "ventricles_bl",
    "wholebrain_bl",
    "entorhinal_bl",
    "fusiform_bl",
    "midtemp_bl",
    "icv_bl",
    # Derived ratio used in Supplementary Table S4; computed in preprocessing
    # "hippo_icv",
]

# Synonyms mapping: when a preferred column is missing, use the first existing alias.
COLUMN_SYNONYMS: Dict[str, Sequence[str]] = {
    "ptgender": ("ptgender", "gender", "sex"),
    "pteducat": ("pteducat", "educ", "education", "edu", "yrs_edu", "yearseducation"),
    "apoe4": ("apoe4", "apoe4allele", "apoe4_count", "apoe_e4"),
    "mmse_bl": ("mmse_bl", "mmse", "mmscore", "mmse_total"),
    "adas13_bl": ("adas13_bl", "adas13", "adas_cog13", "adascog13"),
    "adas11_bl": ("adas11_bl", "adas11", "adas_cog11", "adascog11"),
    "adasq4_bl": ("adasq4_bl", "adasq4", "adas_q4"),
    "cdrsb_bl": ("cdrsb_bl", "cdrsb", "cdrsb_total"),
    "faq_bl": ("faq_bl", "faq"),
    # MRI: ADNI sometimes uses volumes with suffix _ucsf, _ba, etc; keep a few common aliases.
    "hippocampus_bl": ("hippocampus_bl", "hippocampus", "hippocampus_ucsf", "hippocampus_u"),
    "ventricles_bl": ("ventricles_bl", "ventricles", "ventricles_ucsf", "ventricles_u"),
    "wholebrain_bl": ("wholebrain_bl", "wholebrain", "wholebrain_ucsf", "wholebrain_u"),
    "entorhinal_bl": ("entorhinal_bl", "entorhinal", "entorhinal_ucsf", "entorhinal_u"),
    "fusiform_bl": ("fusiform_bl", "fusiform", "fusiform_ucsf", "fusiform_u"),
    "midtemp_bl": ("midtemp_bl", "midtemp", "middletemporal", "middle_temporal", "midtemp_ucsf", "midtemp_u"),
    "icv_bl": ("icv_bl", "icv", "intracranialvolume", "intracranial_vol", "icv_ucsf", "icv_u"),
}

# Model hyperparameters (as reported)
RF_PARAMS = dict(
    n_estimators=200,
    max_features="sqrt",
    max_depth=None,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

LOGREG_PARAMS = dict(
    solver="liblinear",
    max_iter=1000,
    random_state=RANDOM_SEED,
)

def resolve_column(df_columns: Sequence[str], preferred: str) -> str | None:
    """Return the actual column name present in df_columns matching preferred or its synonyms."""
    cols = set(df_columns)
    for c in COLUMN_SYNONYMS.get(preferred, (preferred,)):
        if c in cols:
            return c
    return None

def apoe4_carrier_from_count(apoe4_series):
    """Return 1 if >=1 allele, 0 if 0, NaN if missing."""
    import numpy as np
    x = apoe4_series.astype("float")
    return np.where(np.isnan(x), np.nan, (x >= 1).astype(int))
