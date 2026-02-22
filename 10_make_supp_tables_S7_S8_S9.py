#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_make_supp_tables_S7_S8_S9.py

Creates Supplementary Tables S7–S9 from existing outputs.

Expected upstream outputs (from 01_rebuild_models_and_outputs.py):
- output/threshold_metrics.csv            (RF threshold performance; Table3 + SuppTableS3)
- output/dca_net_benefit.csv              (long format: model, threshold, net_benefit)
- output/table4_performance_summary.csv   (used elsewhere; not required here)

Outputs:
- output/SuppTable_S7_clinician_cutoffs.csv
- output/SuppTable_S8_net_benefit_key_thresholds.csv
- output/SuppTable_S9_RF_hyperparameters.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config


def _net_reduction(nb_model: float, nb_all: float, pt: float) -> float:
    """
    Net reduction in unnecessary interventions per 100 patients compared with treating all
    (Vickers):
        (NB_model - NB_all) * (1-pt)/pt * 100
    """
    if pt <= 0 or pt >= 1:
        return np.nan
    return (nb_model - nb_all) * (1 - pt) / pt * 100.0


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    thr_path = out_dir / "threshold_metrics.csv"
    dca_path = out_dir / "dca_net_benefit.csv"

    if not thr_path.exists():
        raise FileNotFoundError("Missing output/threshold_metrics.csv. Run 01_rebuild_models_and_outputs.py first.")
    if not dca_path.exists():
        raise FileNotFoundError("Missing output/dca_net_benefit.csv. Run 01_rebuild_models_and_outputs.py first.")

    thr = pd.read_csv(thr_path)

    # --- S7: clinician-oriented interpretation of key cutoffs (RF)
    thr_t3 = thr[(thr["table"].astype(str).str.lower() == "table3") & (thr["threshold"].isin(config.TABLE3_THRESHOLDS))].copy()
    thr_t3 = thr_t3.sort_values("threshold")

    notes = []
    for t in thr_t3["threshold"].tolist():
        if abs(t - 0.25) < 1e-9:
            notes.append("Lower threshold (rule-out / early recall): higher sensitivity, lower specificity; captures most decliners but increases false positives.")
        elif abs(t - 0.40) < 1e-9:
            notes.append("Primary cut-off (optimal balance): selected by Youden’s J; used for main confusion matrix and classification results.")
        elif abs(t - 0.50) < 1e-9:
            notes.append("Higher threshold (rule-in / intensified follow-up): higher specificity, lower sensitivity; suitable when resources are limited and false positives should be minimized.")
        else:
            notes.append("")

    s7 = thr_t3[["threshold", "sensitivity", "specificity", "ppv", "npv", "youden_j", "tp", "fp", "tn", "fn"]].copy()
    s7.rename(columns={
        "threshold": "probability_threshold",
        "ppv": "PPV",
        "npv": "NPV",
        "youden_j": "Youden_J",
    }, inplace=True)
    s7["clinical_interpretation"] = notes
    s7.to_csv(out_dir / "SuppTable_S7_clinician_cutoffs.csv", index=False)

    # --- S8: net benefit at key thresholds (DCA)
    dca = pd.read_csv(dca_path)
    # pivot to wide: one row per threshold
    dca_w = dca.pivot_table(index="threshold", columns="model", values="net_benefit", aggfunc="first").reset_index()
    key_pts = [0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    dca_k = dca_w[dca_w["threshold"].isin(key_pts)].copy().sort_values("threshold")

    required = {"rf_calibrated", "baseline_logistic", "treat_all", "treat_none"}
    missing = required - set(dca_k.columns)
    if missing:
        raise ValueError(f"dca_net_benefit.csv is missing expected models: {missing}. Found: {list(dca_w.columns)}")

    rows = []
    for _, r in dca_k.iterrows():
        pt = float(r["threshold"])
        nb_rf = float(r["rf_calibrated"])
        nb_lr = float(r["baseline_logistic"])
        nb_all = float(r["treat_all"])
        nb_none = float(r["treat_none"])
        rows.append({
            "threshold_probability": pt,
            "net_benefit_RF": nb_rf,
            "net_benefit_logistic": nb_lr,
            "net_benefit_treat_all": nb_all,
            "net_benefit_treat_none": nb_none,
            "net_reduction_unnecessary_interventions_per_100_vs_treat_all_RF": _net_reduction(nb_rf, nb_all, pt),
            "net_reduction_unnecessary_interventions_per_100_vs_treat_all_logistic": _net_reduction(nb_lr, nb_all, pt),
        })
    pd.DataFrame(rows).to_csv(out_dir / "SuppTable_S8_net_benefit_key_thresholds.csv", index=False)

    # --- S9: RF hyperparameters (fixed per manuscript)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        max_depth=None,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    params = rf.get_params(deep=True)
    s9 = pd.DataFrame([{"parameter": k, "value": str(v)} for k, v in sorted(params.items(), key=lambda x: x[0])])
    note = pd.DataFrame([{
        "parameter": "_NOTE_",
        "value": "Key tuned hyperparameters per manuscript: n_estimators=200, max_features='sqrt', max_depth=None. Other parameters are sklearn defaults unless specified."
    }])
    pd.concat([note, s9], ignore_index=True).to_csv(out_dir / "SuppTable_S9_RF_hyperparameters.csv", index=False)

    print("Done. Wrote Supplementary Tables S7–S9 to:", out_dir)


if __name__ == "__main__":
    main()
