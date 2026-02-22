#!/usr/bin/env python3
"""Generate Table 1 (baseline characteristics, n=411) and Supplementary Table S2 (baseline categorical)
from raw ADNIMERGE.csv.

This matches the manuscript definition:
- Baseline AD cohort: VISCODE == 'bl' and DX_bl == 'AD' (n=411).
- 12-month MMSE: VISCODE == 'm12' matched by RID.
Outputs are written to output/tables and output/supplementary_tables.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def _fmt_num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def describe_continuous(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce')
    n = int(s.notna().sum())
    miss = int(s.isna().sum())
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "missing": miss}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "missing": miss
    }

def main() -> None:
    raw_path = os.environ.get("ADNIMERGE_RAW_PATH", os.path.join(ROOT, "data", "raw", "ADNIMERGE.csv"))
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw ADNIMERGE.csv not found at: {raw_path}. Set ADNIMERGE_RAW_PATH or place file in data/raw/ADNIMERGE.csv")

    df = _read_csv(raw_path)

    # Baseline AD cohort
    bl = df[(df["VISCODE"].astype(str).str.lower() == "bl") & (df["DX_bl"] == "AD")].copy()
    # ensure unique RID
    bl = bl.sort_values(["RID", "EXAMDATE"]).drop_duplicates("RID", keep="first")
    assert bl["RID"].nunique() == 411, f"Expected 411 baseline AD participants, got {bl['RID'].nunique()}"

    # Month 12 MMSE
    m12 = df[df["VISCODE"].astype(str).str.lower() == "m12"][["RID", "MMSE"]].copy()
    m12 = m12.rename(columns={"MMSE": "MMSE_m12"})
    m12 = m12.sort_values(["RID"]).drop_duplicates("RID", keep="first")

    bl = bl.merge(m12, on="RID", how="left")
    bl["MMSE_change_12m"] = pd.to_numeric(bl["MMSE_m12"], errors="coerce") - pd.to_numeric(bl["MMSE"], errors="coerce")

    # Variables in manuscript Table 1
    cont_vars = [
        ("AGE", "Age, years"),
        ("MMSE", "MMSE at baseline"),
        ("ADAS13", "ADAS-Cog 13 at baseline"),
        ("CDRSB", "CDR-SB at baseline"),
        ("Hippocampus", "Hippocampal volume, mm³"),
        ("ICV", "Intracranial volume, mm³"),
        ("MMSE_m12", "MMSE at 12 months"),
        ("MMSE_change_12m", "12-month MMSE change"),
    ]

    rows = []
    for col, label in cont_vars:
        stats = describe_continuous(bl[col] if col in bl.columns else pd.Series([np.nan]*len(bl)))
        rows.append({
            "Variable": label,
            "n": int(len(bl)),
            "Mean": stats["mean"],
            "SD": stats["sd"],
            "Median": stats["median"],
            "Min": stats["min"],
            "Max": stats["max"],
            "Missing": stats["missing"],
        })

    table1 = pd.DataFrame(rows)

    # Supplementary Table S2: Sex and APOE4 categories
    sex_map = {"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"}
    sex = bl["PTGENDER"].map(sex_map).fillna(bl["PTGENDER"].astype(str))
    sex_counts = sex.value_counts(dropna=False)

    apoe = pd.to_numeric(bl["APOE4"], errors="coerce")
    apoe_cat = pd.cut(apoe, bins=[-0.5,0.5,1.5,2.5], labels=["0 allele","1 allele","2 allele"])
    apoe_cat = apoe_cat.astype("object")
    apoe_cat[pd.isna(apoe)] = "Missing"
    apoe_counts = apoe_cat.value_counts(dropna=False).reindex(["0 allele","1 allele","2 allele","Missing"])

    # Percentages: use available data denominator for each variable as per note
    sex_total = sex.notna().sum()
    apoe_total = apoe.notna().sum()  # denominator for non-missing
    # But note says % calculated using number with available data, and also list Missing as count; we'll compute % over (available) for categories and Missing over total.
    # We'll follow the table shown in your supplementary_Tables.docx: percentages appear over total n=411 (including missing) for APOE categories + missing.
    denom_total = len(bl)

    s2_rows=[]
    for cat in ["Male","Female"]:
        n=int(sex_counts.get(cat,0))
        s2_rows.append({"Variable":"Sex","Category":cat,"n":n,"%": round(100*n/denom_total,2)})
    for cat in ["0 allele","1 allele","2 allele","Missing"]:
        n=int(apoe_counts.get(cat,0))
        s2_rows.append({"Variable":"APOE-ε4 status","Category":cat,"n":n,"%": round(100*n/denom_total,2)})
    supp_s2=pd.DataFrame(s2_rows)

    out_tables=os.path.join(ROOT,"output","tables")
    out_supp=os.path.join(ROOT,"output","supplementary_tables")
    _ensure_dir(out_tables); _ensure_dir(out_supp)

    table1_path=os.path.join(out_tables,"Table1_baseline_n411.csv")
    table1_legacy=os.path.join(out_tables,"table1_AD_descriptive_reconstructed.csv")
    s2_path=os.path.join(out_supp,"SuppTable_S2_baseline_categorical_n411.csv")
    s2_legacy=os.path.join(ROOT,"output","SuppTable_S2_baseline_categorical_n411.csv")

    table1.to_csv(table1_path, index=False)
    table1.to_csv(table1_legacy, index=False)
    supp_s2.to_csv(s2_path, index=False)
    supp_s2.to_csv(s2_legacy, index=False)

    print(f"Wrote: {table1_path}")
    print(f"Wrote: {table1_legacy}")
    print(f"Wrote: {s2_path}")
    print(f"Wrote: {s2_legacy}")

if __name__ == "__main__":
    main()
