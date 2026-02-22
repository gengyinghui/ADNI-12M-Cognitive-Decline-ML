#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_make_supp_tables_S6_bootstrap_CI.py

Supplementary Table S6: Bootstrap 95% confidence intervals for key metrics
(AUC, Brier score, calibration slope/intercept) for both models using
out-of-fold (OOF) predictions from 5-fold CV.

Inputs (from ./output created by 01_rebuild_models_and_outputs.py):
- oof_predictions.csv

Output:
- ./output/SuppTable_S6_bootstrap_CI.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

import config


def _calibration_slope_intercept(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(logit, y_true)
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])
    return slope, intercept


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = {"AUC": [], "Brier": [], "calibration_slope": [], "calibration_intercept": []}
    idx = np.arange(n)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        yb = y[samp]
        pb = p[samp]
        # guard: if resample has one class only, skip
        if len(np.unique(yb)) < 2:
            continue
        stats["AUC"].append(roc_auc_score(yb, pb))
        stats["Brier"].append(brier_score_loss(yb, pb))
        slope, intercept = _calibration_slope_intercept(yb, pb)
        stats["calibration_slope"].append(slope)
        stats["calibration_intercept"].append(intercept)

    out = {}
    for k, v in stats.items():
        arr = np.asarray(v, dtype=float)
        out[k+"_mean"] = float(np.mean(arr))
        out[k+"_ci_low"] = float(np.quantile(arr, 0.025))
        out[k+"_ci_high"] = float(np.quantile(arr, 0.975))
        out[k+"_n_boot_used"] = int(len(arr))
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    oof_path = out_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError("Missing output/oof_predictions.csv. Run 01_rebuild_models_and_outputs.py first.")
    oof = pd.read_csv(oof_path)
    y = oof["y_true"].astype(int).to_numpy()

    # Use calibrated OOF probabilities to match manuscript evaluation
    p_log = oof["lr_prob"].to_numpy()
    p_rf = oof["rf_prob_cal"].to_numpy()

    n_boot = 2000
    seed = config.RANDOM_SEED

    res_log = _bootstrap_ci(y, p_log, n_boot=n_boot, seed=seed + 1)
    res_rf = _bootstrap_ci(y, p_rf, n_boot=n_boot, seed=seed + 2)

    rows = []
    for name, res in [("Model 1 (baseline clinical logistic)", res_log), ("Model 2 (RF, main model)", res_rf)]:
        rows.append({
            "model": name,
            "AUC_mean": res["AUC_mean"], "AUC_95CI_low": res["AUC_ci_low"], "AUC_95CI_high": res["AUC_ci_high"],
            "Brier_mean": res["Brier_mean"], "Brier_95CI_low": res["Brier_ci_low"], "Brier_95CI_high": res["Brier_ci_high"],
            "calibration_slope_mean": res["calibration_slope_mean"], "calibration_slope_95CI_low": res["calibration_slope_ci_low"], "calibration_slope_95CI_high": res["calibration_slope_ci_high"],
            "calibration_intercept_mean": res["calibration_intercept_mean"], "calibration_intercept_95CI_low": res["calibration_intercept_ci_low"], "calibration_intercept_95CI_high": res["calibration_intercept_ci_high"],
            "n_boot_used": min(res["AUC_n_boot_used"], res["Brier_n_boot_used"], res["calibration_slope_n_boot_used"], res["calibration_intercept_n_boot_used"]),
            "n_boot_target": n_boot,
        })

    s6 = pd.DataFrame(rows)
    s6.to_csv(out_dir / "SuppTable_S6_bootstrap_CI.csv", index=False)
    print("Done. Wrote:", out_dir / "SuppTable_S6_bootstrap_CI.csv")


if __name__ == "__main__":
    main()
