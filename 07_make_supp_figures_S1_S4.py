# -*- coding: utf-8 -*-
"""
07_make_supp_figures_S1_S4.py
Generate Supplementary Figure S4 (SHAP summary / beeswarm + bar) for RF model.

✅ Fixes:
- Robust numeric/categorical detection (prevents "Male" hitting median imputer)
- Command-line args: --data, --outdir
- Default paths still supported if you don't pass args
- Works with sparse output from OneHotEncoder (densifies for SHAP)
- SHAP plot normalization: prevents accidental "SHAP interaction value" plots
- Better layout: normal-looking beeswarm, tight save

Run (recommended):
  python .\07_make_supp_figures_S1_S4.py --data .\data\derived\ADNIMERGE_ready_m12_AD_withlabel.csv --outdir .\output

Outputs:
  output/figures/SuppFigure_S4_SHAP_summary_top20.png
  output/figures/SuppFigure_S4_SHAP_bar_top20.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype, is_bool_dtype

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import shap


# ----------------------------
# Helpers
# ----------------------------
def _need(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to input CSV. If not provided, use default ./data/derived/ADNIMERGE_ready_m12_AD_withlabel.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. If not provided, use default ./output",
    )
    return parser.parse_args()


def _get_target_col(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    if "decline_3pt" in df.columns:
        return "decline_3pt"
    raise KeyError("Target column not found. Expected 'label' or 'decline_3pt'.")


def _detect_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    drop_cols = {
        target_col,
        "rid", "ptid", "viscode", "examdate", "examdate_bl",
        "imageuid", "imageuid_bl", "update_stamp",
        "colprot", "origprot", "site", "fsversion", "fsversion_bl",
        "dx", "dx_bl",
    }
    cols = [c for c in df.columns if c not in drop_cols]
    cols = [c for c in cols if df[c].notna().any()]
    return cols


def _split_num_cat(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Robustly split features:
    - bool => categorical
    - numeric dtype => numeric
    - everything else => categorical
    This prevents strings like 'Male' being treated as numeric.
    """
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for c in X.columns:
        if is_bool_dtype(X[c]):
            cat_cols.append(c)
        elif is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def _normalize_shap_values(shap_values) -> np.ndarray:
    """
    Normalize shap_values into a 2D array (n_samples, n_features).

    Why:
    - TreeExplainer for classifiers can return:
      * list([class0, class1]) each (n, p)
      * OR ndarray (n, p, n_classes) in some versions
    - If we accidentally pass 3D to shap.summary_plot, it may render as
      "SHAP interaction value" and create weird tall plots.

    Strategy:
    - If list with 2 classes => use class 1
    - If ndarray 3D => take last axis class 1 if possible, else squeeze
    """
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            sv = shap_values[1]
        else:
            sv = shap_values[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            # choose class axis if present
            sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
        return sv

    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    return sv


def _save_shap_beeswarm(
    out_path: Path,
    shap_values_2d: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list[str],
    title: str,
    max_display: int = 20,
    figsize: tuple[float, float] = (7.5, 5.5),
    dpi: int = 300,
) -> None:
    """
    Save a normal-looking SHAP beeswarm.
    Key: don't pre-create huge figure; let shap create, then resize and tight-save.
    """
    # Clear any existing figures
    plt.close("all")

    shap.summary_plot(
        shap_values_2d,
        X_proc,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)

    ax = plt.gca()
    ax.set_xlabel("SHAP value")  # enforce correct label
    # title: use fig.suptitle to avoid being clipped
    fig.suptitle(title, y=0.98, fontsize=12)

    # Tight save to avoid huge blank canvas
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_shap_bar(
    out_path: Path,
    shap_values_2d: np.ndarray,
    X_proc: np.ndarray,
    feature_names: list[str],
    title: str,
    max_display: int = 20,
    figsize: tuple[float, float] = (7.5, 5.5),
    dpi: int = 300,
) -> None:
    plt.close("all")

    shap.summary_plot(
        shap_values_2d,
        X_proc,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)
    fig.suptitle(title, y=0.98, fontsize=12)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = _parse_args()

    root = Path(__file__).resolve().parent
    default_data = root / "data" / "derived" / "ADNIMERGE_ready_m12_AD_withlabel.csv"
    default_outdir = root / "output"

    data_path = Path(args.data) if args.data else default_data
    out_dir = Path(args.outdir) if args.outdir else default_outdir

    _need(data_path)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading: {data_path}")
    df = pd.read_csv(data_path)

    target_col = _get_target_col(df)
    print(f"[INFO] Target column: {target_col}")

    # Ensure target is 0/1 numeric
    y = df[target_col]
    if y.dtype == "O":
        y = y.map({"0": 0, "1": 1, "false": 0, "true": 1, "False": 0, "True": 1}).fillna(y)
    y = pd.to_numeric(y, errors="raise").astype(int)

    feature_cols = _detect_feature_cols(df, target_col)
    X = df[feature_cols].copy()

    num_cols, cat_cols = _split_num_cat(X)
    print(f"[INFO] Features: {len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", rf),
    ])

    print("[INFO] Fitting pipeline (pre + RF)...")
    pipe.fit(X, y)

    # Transform X for SHAP
    X_proc = pipe.named_steps["pre"].transform(X)

    # SHAP needs dense arrays
    try:
        from scipy import sparse
        if sparse.issparse(X_proc):
            X_proc = X_proc.toarray()
    except Exception:
        pass

    X_proc = np.asarray(X_proc)
    if X_proc.ndim != 2:
        X_proc = X_proc.reshape(X_proc.shape[0], -1)

    # Feature names after preprocessing
    try:
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
        feature_names = [str(x) for x in feature_names]
    except Exception:
        feature_names = [f"x{i}" for i in range(X_proc.shape[1])]

    print("[INFO] Computing SHAP values...")
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_proc)
    shap_values_2d = _normalize_shap_values(shap_values)

    # Sanity check to avoid interaction plots
    if shap_values_2d.ndim != 2:
        raise RuntimeError(f"Unexpected shap_values shape: {np.asarray(shap_values_2d).shape}. Expected 2D.")

    # ---- S4: beeswarm (top 20) ----
    out1 = fig_dir / "SuppFigure_S4_SHAP_summary_top20.png"
    print(f"[INFO] Saving: {out1.name}")
    _save_shap_beeswarm(
        out_path=out1,
        shap_values_2d=shap_values_2d,
        X_proc=X_proc,
        feature_names=feature_names,
        title="Supplementary Figure S4. SHAP summary plot (RF, top 20)",
        max_display=20,
        figsize=(7.5, 5.5),
        dpi=300,
    )

    # ---- S4: bar (top 20) ----
    out2 = fig_dir / "SuppFigure_S4_SHAP_bar_top20.png"
    print(f"[INFO] Saving: {out2.name}")
    _save_shap_bar(
        out_path=out2,
        shap_values_2d=shap_values_2d,
        X_proc=X_proc,
        feature_names=feature_names,
        title="Supplementary Figure S4. SHAP importance (RF, top 20)",
        max_display=20,
        figsize=(7.5, 5.5),
        dpi=300,
    )

    print(f"[OK] Done. Figures written to: {fig_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()