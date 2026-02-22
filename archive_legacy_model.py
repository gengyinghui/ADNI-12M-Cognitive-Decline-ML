#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_rebuild_models_and_outputs.py (manuscript-aligned)

This script implements the manuscript-specified modelling pipeline:
- Analytic sample: baseline AD with complete 12-month MMSE (from 00_build...)
- Outcome: decline_3pt (MMSE drop >= 3)
- 5-fold stratified CV
- Within-fold preprocessing (median/mode single imputation, scaling, one-hot encoding)
- RF hyperparameters fixed as reported (n_estimators=200, max_features=sqrt, max_depth=None)
- Within-fold isotonic regression calibration -> out-of-fold calibrated probabilities
- Outputs: OOF predictions, performance metrics, Table 2/3/4 inputs, and key CSVs used for plots.

Run:
  python 01_rebuild_models_and_outputs.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit as sp_logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, confusion_matrix

import config

def _find_input_csv(root: Path) -> Path:
    candidates = [
        root / "data" / "derived" / "ADNIMERGE_ready_m12_AD_withlabel.csv",
        root / "data" / "analysis" / "ADNIMERGE_ready_m12_AD_withlabel.csv",
        root / "ADNIMERGE_ready_m12_AD_withlabel.csv",
    ]
    for p in candidates:
        # DEBUG: print candidate paths during standalone execution
        # print('DEBUG candidates:', [str(x) for x in candidates])
        if p.exists():
            # print('DEBUG exists', p, p.exists())
            return p
    raise FileNotFoundError("Analysis-ready CSV not found. Run 00_build_analysis_ready_from_ADNIMERGE.py first.")

def _make_preprocessor(X: pd.DataFrame):
    # Identify categorical vs continuous using pandas dtype + known fields
    cat_cols = []
    cont_cols = []
    for c in X.columns:
        if c in ("ptgender", "apoe4_carrier"):
            cat_cols.append(c)
        else:
            # treat numeric as continuous, else categorical
            if pd.api.types.is_numeric_dtype(X[c]):
                cont_cols.append(c)
            else:
                cat_cols.append(c)

    cont_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cont", cont_pipe, cont_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, cont_cols, cat_cols

def _get_feature_names(pre: ColumnTransformer) -> list[str]:
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []

def _calibration_slope_intercept(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """
    Logistic calibration model: logit(y) ~ a + b*logit(p).
    Intercept=a, slope=b.
    """
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    x = sp_logit(p)
    # Fit via sklearn LogisticRegression with no penalty (approx by huge C)
    lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(x.reshape(-1, 1), y_true)
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_[0])
    return slope, intercept

def _threshold_metrics(y_true: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y_pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    youden = sens + spec - 1 if (sens is not np.nan and spec is not np.nan) else np.nan
    return dict(
        threshold=thr,
        sensitivity=sens,
        specificity=spec,
        ppv=ppv,
        npv=npv,
        youden_j=youden,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )

def _risk_stratification(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    ridcol = config.resolve_column(df.columns, config.RID_COL) or config.RID_COL
    mmse_chg = config.resolve_column(df.columns, config.MMSE_CHANGE_COL) or config.MMSE_CHANGE_COL
    out = df[[ridcol, "y_true", prob_col, mmse_chg]].copy()
    strata = []
    for name, lo, hi in config.RISK_STRATA:
        m = (out[prob_col] >= lo) & (out[prob_col] < hi)
        tmp = out.loc[m].copy()
        strata.append(dict(
            risk_stratum=name,
            n=int(len(tmp)),
            events=int(tmp["y_true"].sum()),
            event_rate=float(tmp["y_true"].mean()) if len(tmp) else np.nan,
            mean_mmse_change=float(tmp[config.MMSE_CHANGE_COL].mean()) if len(tmp) else np.nan,
        ))
    return pd.DataFrame(strata)

def _dca_net_benefit(y_true: np.ndarray, p: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """
    Decision curve net benefit:
      NB(t) = TP/N - FP/N * t/(1-t)
    """
    n = len(y_true)
    out = []
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        out.append(dict(threshold=t, net_benefit=nb, tp=int(tp), fp=int(fp)))
    return pd.DataFrame(out)

def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = _find_input_csv(root)
    df = pd.read_csv(in_csv, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    # Resolve required columns
    rid = config.resolve_column(df.columns, config.RID_COL) or config.RID_COL
    ycol = config.resolve_column(df.columns, config.Y_COL) or config.Y_COL
    mmse_change = config.resolve_column(df.columns, config.MMSE_CHANGE_COL) or config.MMSE_CHANGE_COL
    if rid not in df.columns or ycol not in df.columns:
        raise ValueError(f"Required columns not found: {rid}, {ycol}")

    # Build explicit predictor matrix with synonym handling
    X = pd.DataFrame({rid: df[rid].values})
    resolved = {}
    for pref in config.CANDIDATE_PREDICTORS:
        actual = config.resolve_column(df.columns, pref)
        if actual is None:
            raise ValueError(f"Missing candidate predictor column '{pref}'. Provide it in analysis-ready CSV or add synonym in config.py.")
        resolved[pref] = actual
        X[pref] = df[actual].values

    # Derived predictors used in Supplementary Table S4
    X["apoe4_carrier"] = config.apoe4_carrier_from_count(pd.Series(X["apoe4"]))
    # hippo/icv ratio (for missingness reporting and optional sensitivity)
    X["hippo_icv"] = X["hippocampus_bl"].astype(float) / X["icv_bl"].astype(float)

    y = df[ycol].astype(int).values
    rid_vals = df[rid].values
    mmse_change_vals = df[mmse_change].values if mmse_change in df.columns else np.full(len(df), np.nan)

    # Model 1: baseline clinical logistic regression predictors (as manuscript)
    baseline_cols = ["age", "ptgender", "mmse_bl", "adas13_bl"]

    # Model 2: RF all candidate predictors (plus derived apoe4_carrier, hippo_icv)
    rf_cols = [c for c in X.columns if c != rid]

    # Prepare containers for OOF
    oof = pd.DataFrame({ config.RID_COL: rid_vals, "y_true": y, config.MMSE_CHANGE_COL: mmse_change_vals })

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    # ---- Baseline logistic regression OOF (no calibration in manuscript; we keep raw probs) ----
    p_lr = np.zeros(len(df), dtype=float)

    # ---- RF OOF: raw + calibrated within fold (isotonic) ----
    p_rf_raw = np.zeros(len(df), dtype=float)
    p_rf_cal = np.zeros(len(df), dtype=float)

    fold_rows = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y[tr], y[te]

        # Baseline LR pipeline
        pre_lr, _, _ = _make_preprocessor(Xtr[baseline_cols])
        lr = LogisticRegression(**config.LOGREG_PARAMS)
        pipe_lr = Pipeline([("pre", pre_lr), ("model", lr)])
        pipe_lr.fit(Xtr[baseline_cols], ytr)
        p_lr[te] = pipe_lr.predict_proba(Xte[baseline_cols])[:, 1]

        # RF pipeline (reported tuned params)
        pre_rf, _, _ = _make_preprocessor(Xtr[rf_cols])
        rf = RandomForestClassifier(**config.RF_PARAMS)
        pipe_rf = Pipeline([("pre", pre_rf), ("model", rf)])
        pipe_rf.fit(Xtr[rf_cols], ytr)
        pr_tr = pipe_rf.predict_proba(Xtr[rf_cols])[:, 1]
        pr_te = pipe_rf.predict_proba(Xte[rf_cols])[:, 1]
        p_rf_raw[te] = pr_te

        # Isotonic calibration within fold: fit on training predictions, apply to validation
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pr_tr, ytr)
        p_rf_cal[te] = iso.transform(pr_te)

        fold_rows.append({"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)), "events_train": int(ytr.sum()), "events_test": int(yte.sum())})

    oof["lr_prob"] = p_lr
    oof["rf_prob_raw"] = p_rf_raw
    oof["rf_prob_cal"] = p_rf_cal

    oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    # ---- Performance summary (Table 4 inputs) ----
    def perf_row(name: str, p: np.ndarray) -> dict:
        auc = roc_auc_score(y, p)
        brier = brier_score_loss(y, p)
        slope, intercept = _calibration_slope_intercept(y, p)
        return dict(model=name, auc=auc, brier=brier, calib_slope=slope, calib_intercept=intercept)

    perf = pd.DataFrame([
        perf_row("baseline_logistic", oof["lr_prob"].values),
        perf_row("rf_calibrated", oof["rf_prob_cal"].values),
    ])
    perf.to_csv(out_dir / "table4_performance_summary.csv", index=False)

    # ---- Threshold metrics: Table 3 + Supp Table S3 ----
    thr_rows = []
    for thr in config.TABLE3_THRESHOLDS:
        thr_rows.append(dict(table="Table3", **_threshold_metrics(y, oof["rf_prob_cal"].values, thr)))
    for thr in config.SUPP_S3_THRESHOLDS:
        thr_rows.append(dict(table="SuppTableS3", **_threshold_metrics(y, oof["rf_prob_cal"].values, thr)))
    pd.DataFrame(thr_rows).to_csv(out_dir / "threshold_metrics.csv", index=False)

    # ---- Confusion matrix at 0.40 for Supp Table S1 ----
    cm = _threshold_metrics(y, oof["rf_prob_cal"].values, 0.40)
    pd.DataFrame([{
        "threshold": 0.40,
        "TP": cm["tp"], "FP": cm["fp"], "TN": cm["tn"], "FN": cm["fn"],
        "N": int(len(y)),
    }]).to_csv(out_dir / "SuppTable_S1_confusion_matrix_thr0.40.csv", index=False)

    # ---- Risk stratification (Table 2 inputs + Supp Fig S2/S3 inputs) ----
    strata = _risk_stratification(oof, "rf_prob_cal")
    strata.to_csv(out_dir / "table2_risk_strata.csv", index=False)

    # ---- ROC curve points (Figure 1) ----
    fpr, tpr, thr = roc_curve(y, oof["rf_prob_cal"].values)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(out_dir / "roc_points_rf.csv", index=False)
    # Logistic ROC points (Figure 1)
    fpr, tpr, thr = roc_curve(y, oof["lr_prob"].values)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(out_dir / "roc_points_logistic.csv", index=False)


    # ---- Calibration deciles (Figure 2) ----
    cal_df = oof[["y_true", "rf_prob_cal"]].copy()
    cal_df["decile"] = pd.qcut(cal_df["rf_prob_cal"], 10, duplicates="drop")
    cal_sum = cal_df.groupby("decile", observed=True).agg(
        pred_mean=("rf_prob_cal", "mean"),
        obs_rate=("y_true", "mean"),
        n=("y_true", "size"),
    ).reset_index()
    cal_sum.to_csv(out_dir / "calibration_deciles_rf.csv", index=False)
    # Calibration deciles for baseline logistic (optional overlay in Figure 2)
    cal_df_lr = oof[["y_true", "lr_prob"]].copy()
    cal_df_lr["decile"] = pd.qcut(cal_df_lr["lr_prob"], q=10, duplicates="drop")
    cal_lr = cal_df_lr.groupby("decile").agg(
        mean_pred=("lr_prob", "mean"),
        mean_obs=("y_true", "mean"),
        n=("y_true", "size"),
    ).reset_index()
    cal_lr.to_csv(out_dir / "calibration_deciles_logistic.csv", index=False)


    # ---- DCA curves (Figure 3) ----
    dca_lr = _dca_net_benefit(y, oof["lr_prob"].values, config.DCA_THRESHOLD_GRID)
    dca_lr["model"] = "baseline_logistic"
    dca_rf = _dca_net_benefit(y, oof["rf_prob_cal"].values, config.DCA_THRESHOLD_GRID)
    dca_rf["model"] = "rf_calibrated"
    dca = pd.concat([dca_lr, dca_rf], ignore_index=True)
    # Add treat-all and treat-none reference strategies
    prev = float(y.mean())
    ref = []
    for t in config.DCA_THRESHOLD_GRID:
        # treat-none net benefit is 0 by definition
        ref.append({"threshold": t, "net_benefit": 0.0, "tp": 0, "fp": 0, "model": "treat_none"})
        # treat-all net benefit: prevalence - (1-prevalence)*t/(1-t)
        nb_all = prev - (1.0 - prev) * (t / (1.0 - t))
        ref.append({"threshold": t, "net_benefit": nb_all, "tp": int((y==1).sum()), "fp": int((y==0).sum()), "model": "treat_all"})
    dca = pd.concat([dca, pd.DataFrame(ref)], ignore_index=True)

    dca.to_csv(out_dir / "dca_net_benefit.csv", index=False)

    # ---- Fold summary ----
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_folds_summary.csv", index=False)

    # ---- Save feature name map for RF pipeline (needed for importance/SHAP) ----
    # Fit RF on full analytic sample for interpretability plots (as manuscript).
    pre_full, _, _ = _make_preprocessor(X[rf_cols])
    rf_full = RandomForestClassifier(**config.RF_PARAMS)
    pipe_full = Pipeline([("pre", pre_full), ("model", rf_full)])
    pipe_full.fit(X[rf_cols], y)
    feature_names = _get_feature_names(pre_full)
    # Importances align to transformed feature space.
    importances = pipe_full.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(out_dir / "rf_feature_importance.csv", index=False)

    meta = {
        "input_csv": str(in_csv),
        "n": int(len(df)),
        "event_rate": float(np.mean(y)),
        "rf_params": config.RF_PARAMS,
        "logreg_params": config.LOGREG_PARAMS,
        "predictors_rf": rf_cols,
        "predictors_baseline_lr": baseline_cols,
        "column_resolution": resolved,
        "random_seed": config.RANDOM_SEED,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done. Outputs written to:", out_dir)

if __name__ == "__main__":
    main()