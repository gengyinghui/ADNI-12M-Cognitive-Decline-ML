#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build analysis-ready AD-only 12-month MMSE decline dataset from ADNIMERGE.csv
Output: ADNIMERGE_ready_m12_AD_withlabel.csv

This script reconstructs the core preprocessing step that may be missing from the
reproducibility package (which starts from an analysis-ready CSV).
"""
from pathlib import Path
import pandas as pd
import numpy as np

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    root = Path(__file__).resolve().parent
    candidate_paths = [
        root / 'ADNIMERGE.csv',
        root / 'data' / 'raw' / 'ADNIMERGE.csv',
        root / 'data' / 'ADNIMERGE.csv',
        Path('/mnt/data/ADNIMERGE.csv'),
    ]
    adni_path = next((p for p in candidate_paths if p.exists()), None)
    if adni_path is None:
        raise FileNotFoundError('ADNIMERGE.csv not found. Put it beside this scripts folder or in data/raw/.')
    out_dir = root / 'data' / 'derived'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(adni_path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    vis_col = first_existing(df, ['viscode','viscode2'])
    dx_col = first_existing(df, ['dx_bl','dx'])
    rid_col = first_existing(df, ['rid'])
    mmse_col = first_existing(df, ['mmse'])
    if not all([vis_col, dx_col, rid_col, mmse_col]):
        raise ValueError(f'Missing required columns. Found vis={vis_col}, dx={dx_col}, rid={rid_col}, mmse={mmse_col}')

    vis = df[vis_col].astype(str).str.lower()
    bl = df.loc[vis.eq('bl')].copy()
    dx = bl[dx_col].astype(str).str.upper()
    bl_ad = bl.loc[dx.eq('AD')].copy()

    m12 = df.loc[df[vis_col].astype(str).str.lower().eq('m12'), [rid_col, mmse_col]].copy()
    m12 = m12.rename(columns={mmse_col: 'mmse_m12'})

    ad = bl_ad.merge(m12, on=rid_col, how='left')
    if 'mmse_bl' not in ad.columns and 'mmse' in ad.columns:
        ad = ad.rename(columns={'mmse': 'mmse_bl'})
    if 'mmse_bl' not in ad.columns:
        c = first_existing(ad, ['mmscore','mmse_total'])
        if c:
            ad = ad.rename(columns={c: 'mmse_bl'})
        else:
            raise ValueError('Baseline MMSE column not found after merge.')

    ad['mmse_change'] = ad['mmse_m12'] - ad['mmse_bl']
    ad['decline_3pt'] = np.where(ad['mmse_change'] <= -3, 1, 0)
    ad.loc[ad['mmse_m12'].isna(), 'decline_3pt'] = np.nan

    ad_ready = ad.loc[ad['mmse_m12'].notna()].copy()

    out_path = out_dir / 'ADNIMERGE_ready_m12_AD_withlabel.csv'
    ad_ready.to_csv(out_path, index=False)
    print(f'Saved: {out_path}')
    print(f'AD baseline rows: {len(bl_ad)}')
    print(f'Analysis-ready rows with 12m MMSE: {len(ad_ready)}')
    print(f'12m decline rate (>=3-point MMSE drop): {ad_ready["decline_3pt"].mean():.3f}')

if __name__ == '__main__':
    main()
