#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

def cont(s):
    s = pd.to_numeric(s, errors="coerce")
    return f"{s.mean():.2f} ({s.std(ddof=1):.2f})"

def cat(s):
    c = s.value_counts(dropna=False)
    n = len(s)
    return " | ".join([f"{k}: {int(v)} ({v/n*100:.1f}%)" for k,v in c.items()])

def main():
    root = Path(__file__).resolve().parent
    df = pd.read_csv(root/"data"/"derived"/"ADNIMERGE_ready_m12_AD_withlabel.csv")
    y = "decline_3pt"
    exclude = {"rid","ptid","viscode","dx_bl","mmse_m12","mmse_change",y}
    rows=[]
    for c in [x for x in df.columns if x not in exclude]:
        if df[c].dtype == "object":
            rows.append({"variable":c, "type":"categorical", "overall":cat(df[c]), "decline0":cat(df[df[y]==0][c]), "decline1":cat(df[df[y]==1][c])})
        else:
            rows.append({"variable":c, "type":"continuous", "overall":cont(df[c]), "decline0":cont(df[df[y]==0][c]), "decline1":cont(df[df[y]==1][c])})
    pd.DataFrame(rows).to_csv(root/"output"/"tables"/"table1_AD_descriptive_reconstructed.csv", index=False)
    print("Saved table1_AD_descriptive_reconstructed.csv")

if __name__ == "__main__":
    main()
