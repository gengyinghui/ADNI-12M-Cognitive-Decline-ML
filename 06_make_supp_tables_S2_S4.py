#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Supplementary Tables required by the manuscript:
- Supp Table S2: baseline categorical characteristics in baseline AD cohort (n=411 in paper)
- Supp Table S4: missingness of candidate predictors in analytic sample (n=306 in paper)

Run:
  python 06_make_supp_tables_S2_S4.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import config

def _find_adnimerge(root: Path) -> Path:
    candidates = [
        root / "ADNIMERGE.csv",
        root / "data" / "raw" / "ADNIMERGE.csv",
        root / "data" / "ADNIMERGE.csv",
        Path("/mnt/data/ADNIMERGE.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("ADNIMERGE.csv not found. Place it next to the package folder or in data/raw/.")

def _find_analysis_ready(root: Path) -> Path:
    candidates = [
        root / "data" / "derived" / "ADNIMERGE_ready_m12_AD_withlabel.csv",
        root / "ADNIMERGE_ready_m12_AD_withlabel.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Analysis-ready CSV not found. Run 00_build_analysis_ready_from_ADNIMERGE.py first.")

def main():
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Supp Table S2 (baseline cohort n=411) ----
    adni_path = _find_adnimerge(root)
    df = pd.read_csv(adni_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    vis = config.resolve_column(df.columns, "viscode") or ("viscode2" if "viscode2" in df.columns else None)
    dx = config.resolve_column(df.columns, "dx_bl") or ("dx" if "dx" in df.columns else None)
    if vis is None or dx is None:
        raise ValueError("Could not find viscode/viscode2 or dx_bl/dx in ADNIMERGE.")

    bl = df.loc[df[vis].astype(str).str.lower().eq("bl")].copy()
    bl_ad = bl.loc[bl[dx].astype(str).str.upper().eq("AD")].copy()

    sex_col = config.resolve_column(bl_ad.columns, "ptgender") or "ptgender"
    apoe_col = config.resolve_column(bl_ad.columns, "apoe4") or "apoe4"

    # Sex counts
    sex = bl_ad[sex_col].astype(str).replace({"nan": np.nan})
    sex_counts = sex.value_counts(dropna=False)

    rows = []
    for cat, n in sex_counts.items():
        if pd.isna(cat):
            label = "Missing"
        else:
            # standardize common labels
            s = str(cat).strip().lower()
            if s in ("m", "male", "1"):
                label = "Male"
            elif s in ("f", "female", "2"):
                label = "Female"
            else:
                label = str(cat)
        denom = int(sex.notna().sum())
        pct = (n / denom * 100) if denom > 0 and label != "Missing" else (n / len(sex) * 100)
        rows.append({"Variable": "Sex", "Category": label, "n": int(n), "%": float(pct)})

    # APOE4 allele distribution (0/1/2 + Missing)
    ap = pd.to_numeric(bl_ad[apoe_col], errors="coerce")
    for allele in [0, 1, 2]:
        n = int((ap == allele).sum())
        denom = int(ap.notna().sum())
        pct = (n / denom * 100) if denom > 0 else np.nan
        rows.append({"Variable": "APOE-ε4 status", "Category": f"{allele} allele" if allele==1 else f"{allele} alleles" if allele==2 else "0 allele", "n": n, "%": float(pct)})

    n_miss = int(ap.isna().sum())
    rows.append({"Variable": "APOE-ε4 status", "Category": "Missing", "n": n_miss, "%": float(n_miss / len(ap) * 100)})

    supp_s2 = pd.DataFrame(rows)
    supp_s2.to_csv(out_dir / "SuppTable_S2_baseline_categorical_n411.csv", index=False)

    # ---- Supp Table S4 (missingness in analytic sample n=306) ----
    ar_path = _find_analysis_ready(root)
    ar = pd.read_csv(ar_path, low_memory=False)
    ar.columns = [c.lower() for c in ar.columns]

    # Resolve to the preferred canonical names
    missing_rows = []
    n = len(ar)
    domain_map = {
        "age": "Demographics/genetics",
        "ptgender": "Demographics/genetics",
        "pteducat": "Demographics/genetics",
        "apoe4": "Demographics/genetics",
        "mmse_bl": "Clinical/cognitive",
        "adas11_bl": "Clinical/cognitive",
        "adas13_bl": "Clinical/cognitive",
        "adasq4_bl": "Clinical/cognitive",
        "cdrsb_bl": "Clinical/cognitive",
        "faq_bl": "Clinical/cognitive",
        "hippocampus_bl": "MRI-derived",
        "ventricles_bl": "MRI-derived",
        "wholebrain_bl": "MRI-derived",
        "entorhinal_bl": "MRI-derived",
        "fusiform_bl": "MRI-derived",
        "midtemp_bl": "MRI-derived",
        "icv_bl": "MRI-derived",
        "hippo_icv": "MRI-derived",
    }

    # Build a temp df with canonical columns
    tmp = pd.DataFrame()
    for pref in config.CANDIDATE_PREDICTORS:
        actual = config.resolve_column(ar.columns, pref)
        if actual is None:
            raise ValueError(f"Missing '{pref}' in analysis-ready CSV (or its synonyms).")
        tmp[pref] = ar[actual]
    # Derived hippo/icv ratio
    tmp["hippo_icv"] = pd.to_numeric(tmp["hippocampus_bl"], errors="coerce") / pd.to_numeric(tmp["icv_bl"], errors="coerce")

    for col in list(tmp.columns):
        miss = int(pd.isna(tmp[col]).sum())
        pct = miss / n * 100
        missing_rows.append({
            "Predictor": col,
            "Domain": domain_map.get(col, "Other"),
            "Missing_n": miss,
            "Missing_pct": round(pct, 1),
        })

    supp_s4 = pd.DataFrame(missing_rows)
    supp_s4.to_csv(out_dir / "SuppTable_S4_missingness_n306.csv", index=False)

    print("Done. Supplementary tables written to:", out_dir)

if __name__ == "__main__":
    main()
