#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_plot_outputs_from_csv.py

Generates manuscript main figures from CSV outputs created by 01_rebuild_models_and_outputs.py.

Manuscript mapping:
- Figure 1: ROC curves (Logistic baseline vs RF) using out-of-fold calibrated probabilities
- Figure 2: Calibration plot (RF) using deciles of out-of-fold calibrated probabilities
- Figure 3: Decision-curve analysis (Logistic baseline vs RF; treat-all / treat-none)

Outputs are written to output/figures with submission-friendly names:
  Figure1_ROC.png/.pdf
  Figure2_Calibration.png/.pdf
  Figure3_DCA.png/.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------- Figure 1: ROC --------
    roc_lr = out_dir / "roc_points_logistic.csv"
    roc_rf = out_dir / "roc_points_rf.csv"
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if roc_lr.exists():
        d = pd.read_csv(_need(roc_lr))
        ax.plot(d["fpr"], d["tpr"], label="Baseline clinical (Logistic)")
    if roc_rf.exists():
        d = pd.read_csv(_need(roc_rf))
        ax.plot(d["fpr"], d["tpr"], label="Random forest (RF)")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 1. ROC curves (out-of-fold calibrated probabilities)")
    ax.legend(loc="lower right")

    _save(fig, fig_dir / "Figure1_ROC.png")
        # Re-create PDF properly (avoid blank)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if roc_lr.exists():
        d = pd.read_csv(roc_lr)
        ax.plot(d["fpr"], d["tpr"], label="Baseline clinical (Logistic)")
    if roc_rf.exists():
        d = pd.read_csv(roc_rf)
        ax.plot(d["fpr"], d["tpr"], label="Random forest (RF)")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Figure 1. ROC curves (out-of-fold calibrated probabilities)")
    ax.legend(loc="lower right")
    _save(fig, fig_dir / "Figure1_ROC.pdf")

    # -------- Figure 2: Calibration (RF) --------
    cal = out_dir / "calibration_deciles_rf.csv"
    cal_lr = out_dir / "calibration_deciles_logistic.csv"
    if cal.exists():
        d = pd.read_csv(_need(cal))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(d["pred_mean"], d["obs_rate"], marker="o", label="RF (deciles)")
        if cal_lr.exists():
            dlr = pd.read_csv(cal_lr)
            ax.plot(dlr["mean_pred"], dlr["mean_obs"], marker="s", label="Baseline logistic (deciles)")
        if cal_lr.exists():
            dlr = pd.read_csv(cal_lr)
            ax.plot(dlr["mean_pred"], dlr["mean_obs"], marker="s", label="Baseline logistic (deciles)")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_title("Figure 2. Calibration plot (RF, out-of-fold)")
        ax.legend(loc="upper left")
        _save(fig, fig_dir / "Figure2_Calibration.png")
        # pdf
        d = pd.read_csv(cal)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(d["pred_mean"], d["obs_rate"], marker="o", label="RF (deciles)")
        if cal_lr.exists():
            dlr = pd.read_csv(cal_lr)
            ax.plot(dlr["mean_pred"], dlr["mean_obs"], marker="s", label="Baseline logistic (deciles)")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_title("Figure 2. Calibration plot (RF, out-of-fold)")
        ax.legend(loc="upper left")
        _save(fig, fig_dir / "Figure2_Calibration.pdf")

    # -------- Figure 3: DCA --------
    dca = out_dir / "dca_net_benefit.csv"
    if dca.exists():
        d = pd.read_csv(_need(dca))
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for model in ["baseline_logistic", "rf_calibrated", "treat_all", "treat_none"]:
            sub = d[d["model"] == model]
            if len(sub):
                ax.plot(sub["threshold"], sub["net_benefit"], label=model.replace("_", " "))

        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title("Figure 3. Decision-curve analysis (out-of-fold calibrated probabilities)")
        ax.legend(loc="best")
        _save(fig, fig_dir / "Figure3_DCA.png")

        # pdf
        d = pd.read_csv(dca)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for model in ["baseline_logistic", "rf_calibrated", "treat_all", "treat_none"]:
            sub = d[d["model"] == model]
            if len(sub):
                ax.plot(sub["threshold"], sub["net_benefit"], label=model.replace("_", " "))
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title("Figure 3. Decision-curve analysis (out-of-fold calibrated probabilities)")
        ax.legend(loc="best")
        _save(fig, fig_dir / "Figure3_DCA.pdf")

    print(f"[OK] Main figures written to: {fig_dir}")

if __name__ == "__main__":
    main()
