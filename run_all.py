#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scientific Reports reproducibility runner.

Inputs (either option):
A) Raw ADNI file:
   - ADNIMERGE.csv (place beside this folder or in data/raw/)
   -> enables reconstruction of the n=411 baseline cohort tables (Table 1; Supp Table S2)

B) Analysis-ready file:
   - data/analysis/ADNIMERGE_ready_m12_AD_withlabel.csv
   -> enables model rebuilding, evaluation, plots, and Supp Tables S5–S9.

Run:
  python run_all.py
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import sys

BASE_SCRIPTS = [
    "01_rebuild_models_and_outputs.py",
    "02_plot_outputs_from_csv.py",
    "07_make_supp_figures_S1_S4.py",
    "08_sensitivity_ventricles_icv.py",
    "09_make_supp_tables_S6_bootstrap_CI.py",
    "10_make_supp_tables_S7_S8_S9.py",
]

RAW_DEPENDENT_SCRIPTS = [
    "00_build_analysis_ready_from_ADNIMERGE.py",
    "11_make_table1_and_suppS2_from_raw_ADNIMERGE.py",
    "06_make_supp_tables_S2_S4.py",
]

EXPORT_SCRIPT = "05_export_submission_tables.py"

def _run(here: Path, script: str) -> None:
    p = here / script
    if not p.exists():
        raise FileNotFoundError(p)
    print("\n=== Running:", script, "===\n")
    subprocess.check_call([sys.executable, str(p)], cwd=str(here))

def main() -> None:
    here = Path(__file__).resolve().parent
    raw1 = here / "ADNIMERGE.csv"
    raw2 = here / "data" / "raw" / "ADNIMERGE.csv"
    has_raw = raw1.exists() or raw2.exists()

    if has_raw:
        for s in RAW_DEPENDENT_SCRIPTS:
            _run(here, s)
    else:
        print("NOTE: ADNIMERGE.csv not found -> skipping Table 1 / Supp Table S2 reconstruction scripts (they require raw data).")

    for s in BASE_SCRIPTS:
        _run(here, s)

    # Try export step (creates output/final/Submission_Tables.xlsx + OUTPUT_MANIFEST.md)
    try:
        _run(here, EXPORT_SCRIPT)
    except Exception as e:
        print("\nWARNING: Export step could not run (likely because Table 1 / Supp Table S2 CSVs are missing without ADNIMERGE.csv).")
        print("You can still submit the scripts; to regenerate Table 1 and Supp Table S2, provide ADNIMERGE.csv and rerun.")
        print("Details:", repr(e))

    print("\nAll done. See output/ (and output/final/ if export succeeded).\n")

if __name__ == "__main__":
    main()
