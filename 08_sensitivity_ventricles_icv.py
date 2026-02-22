#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_sensitivity_ventricles_icv.py

Manuscript-aligned sensitivity analysis (Supplementary Table S5):
Replace absolute ventricular volume (ventricles_bl) and intracranial volume (icv_bl)
with the ventricles/ICV ratio, then rerun the RF pipeline under the same validation
scheme (5-fold stratified CV, within-fold preprocessing + isotonic calibration).

Outputs (written to ./output):
- oof_predictions_sensitivity_ventricles_icv.csv
- table4_performance_summary_sensitivity_ventricles_icv.csv
- dca_net_benefit_sensitivity_ventricles_icv.csv
- SuppTable_S5_sensitivity_ventricles_icv.csv   (side-by-side main vs sensitivity)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import config


def _find_input_csv(root: Path) -> Path:
    """Find analysis-ready input CSV in common locations."""
    candidates = [
        root / "data" / "analysis" / "analysis_ready.csv",
        root / "data" / "analysis" / "ADNIMERGE_ready_m12_AD_withlabel.csv",
        root / "ADNIMERGE_ready_m12_AD_withlabel.csv",
        root / "output" / "analysis_ready.csv",
        root / "output" / "analysis_ready_from_ADNIMERGE.csv",
        root / "output" / "analysis_ready_from_ADNIMERGE_lowercase.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any csv with "withlabel" or "analysis_ready"
    for p in list(root.glob("*.csv")) + list((root / "data").rglob("*.csv")):
        if "withlabel" in p.name.lower() or "analysis_ready" in p.name.lower():
            return p
    raise FileNotFoundError("Could not locate analysis-ready CSV. Place it in data/analysis/ or project root.")


def _make_preprocessor(df: pd.DataFrame, categorical: list[str], continuous: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    cont_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical),
            ("cont", cont_pipe, continuous),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _calibration_slope_intercept(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic calibration model: y ~ intercept + slope * logit(p)."""
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(logit, y_true)
    slope = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])
    return slope, intercept


def _dca_net_benefit(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """Decision curve net benefit for model, treat-all, treat-none."""
    y = np.asarray(y).astype(int)
    n = len(y)
    out = []
    prevalence = y.mean()
    for t in thresholds:
        preds = (p >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        nb_model = (tp / n) - (fp / n) * (t / (1 - t))
        nb_none = 0.0
        # Treat-all: everyone positive -> TP = events, FP = non-events
        tp_all = int(y.sum())
        fp_all = int((1 - y).sum())
        nb_all = (tp_all / n) - (fp_all / n) * (t / (1 - t))
        out.append({
            "threshold": float(t),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
            "prevalence": float(prevalence),
            "n": int(n),
        })
    return pd.DataFrame(out)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv = _find_input_csv(root)
    df = pd.read_csv(in_csv, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    rid = config.resolve_column(df.columns, config.RID_COL) or config.RID_COL
    ycol = config.resolve_column(df.columns, config.Y_COL) or config.Y_COL

    if rid not in df.columns or ycol not in df.columns:
        raise ValueError(f"Required columns not found: {rid}, {ycol}")

    # Resolve predictors and compute ventricles/icv ratio
    # Start from manuscript predictors but replace ventricles_bl + icv_bl with ventricles_icv
    predictors = []
    resolved = {}
    for pref in config.CANDIDATE_PREDICTORS:
        actual = config.resolve_column(df.columns, pref)
        if actual is None:
            raise ValueError(f"Missing required predictor column for sensitivity analysis: {pref}")
        resolved[pref] = actual

    vent = resolved["ventricles_bl"]
    icv = resolved["icv_bl"]
    df["ventricles_icv"] = df[vent] / df[icv]
    # build X
    for pref in config.CANDIDATE_PREDICTORS:
        if pref in ("ventricles_bl", "icv_bl"):
            continue
        predictors.append(resolved[pref])
    predictors.append("ventricles_icv")

    X = df[[rid] + predictors].copy()
    y = df[ycol].astype(int).to_numpy()

    # Define categorical vs continuous (same as main: ptgender categorical, apoe4 categorical-ish)
    # We keep apoe4 as categorical (0/1/2) and ptgender as categorical; everything else continuous.
    cat_cols = []
    cont_cols = []
    for c in predictors:
        if c in (resolved["ptgender"], resolved["apoe4"]):
            cat_cols.append(c)
        else:
            cont_cols.append(c)

    pre = _make_preprocessor(df, categorical=cat_cols, continuous=cont_cols)

    # RF fixed hyperparameters per manuscript
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        max_depth=None,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    oof_rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        Xtr = X.iloc[tr].drop(columns=[rid])
        Xte = X.iloc[te].drop(columns=[rid])
        ytr = y[tr]
        yte = y[te]

        # Fit preprocessing on train only
        Xt_tr = pre.fit_transform(Xtr)
        Xt_te = pre.transform(Xte)

        rf.fit(Xt_tr, ytr)
        p_tr = rf.predict_proba(Xt_tr)[:, 1]
        p_te = rf.predict_proba(Xt_te)[:, 1]

        # within-fold isotonic calibration
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, ytr)
        p_te_cal = iso.transform(p_te)

        for i, idx in enumerate(te):
            oof_rows.append({
                rid: X.iloc[idx][rid],
                "fold": fold,
                "y_true": int(yte[i]),
                "proba_rf_raw": float(p_te[i]),
                "proba_rf_calibrated": float(p_te_cal[i]),
            })

    oof = pd.DataFrame(oof_rows).sort_values(by=[rid, "fold"])
    oof.to_csv(out_dir / "oof_predictions_sensitivity_ventricles_icv.csv", index=False)

    # performance
    y_true = oof["y_true"].to_numpy()
    p_cal = oof["proba_rf_calibrated"].to_numpy()
    auc = roc_auc_score(y_true, p_cal)
    brier = brier_score_loss(y_true, p_cal)
    slope, intercept = _calibration_slope_intercept(y_true, p_cal)
    perf = pd.DataFrame([{
        "model": "RF_sensitivity_ventricles_icv",
        "auc": float(auc),
        "brier": float(brier),
        "calib_slope": float(slope),
        "calib_intercept": float(intercept),
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
    }])
    perf.to_csv(out_dir / "table4_performance_summary_sensitivity_ventricles_icv.csv", index=False)

    # DCA grid
    dca = _dca_net_benefit(y_true, p_cal, thresholds=config.DCA_THRESHOLD_GRID)
    dca.to_csv(out_dir / "dca_net_benefit_sensitivity_ventricles_icv.csv", index=False)

    # Build Supplementary Table S5 by comparing main RF vs sensitivity RF
    main_perf_path = out_dir / "table4_performance_summary.csv"
    if not main_perf_path.exists():
        raise FileNotFoundError("Run 01_rebuild_models_and_outputs.py first to create table4_performance_summary.csv")

    main_perf = pd.read_csv(main_perf_path)
    # pick RF main model row
    rf_main = main_perf[main_perf["model"].str.contains("RF", case=False, na=False)].head(1)
    if rf_main.empty:
        raise ValueError("Could not find RF row in table4_performance_summary.csv")

    s5 = pd.DataFrame([
        {
            "analysis": "Main (absolute ventricles + ICV)",
            "auc": float(rf_main["auc"].values[0]),
            "brier": float(rf_main["brier"].values[0]),
            "calib_slope": float(rf_main["calib_slope"].values[0]),
            "calib_intercept": float(rf_main["calib_intercept"].values[0]),
        },
        {
            "analysis": "Sensitivity (ventricles/ICV ratio)",
            "auc": float(auc),
            "brier": float(brier),
            "calib_slope": float(slope),
            "calib_intercept": float(intercept),
        },
    ])
    s5.to_csv(out_dir / "SuppTable_S5_sensitivity_ventricles_icv.csv", index=False)

    print("Done. Created Supplementary Table S5 and sensitivity outputs in:", out_dir)


if __name__ == "__main__":
    main()
