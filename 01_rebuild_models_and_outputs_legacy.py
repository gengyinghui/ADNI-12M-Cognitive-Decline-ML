#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstructed reproducible pipeline for ADNI AD 12-month cognitive decline prediction.
Starts from analysis-ready cohort file and regenerates main modeling outputs.
Note: If original raw ADNI preprocessing scripts exist, add them to this package before submission.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit as sp_logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

RANDOM_SEED = 2025
THRESHOLDS = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]

def calibration_intercept_slope(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    x = sp_logit(p).reshape(-1, 1)
    mdl = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    mdl.fit(x, y_true)
    return float(mdl.intercept_[0]), float(mdl.coef_[0][0])

def threshold_metrics(y_true, p):
    rows=[]; y_true=np.asarray(y_true).astype(int); n=len(y_true); prev=y_true.mean()
    for t in THRESHOLDS:
        pred=(p>=t).astype(int)
        tp=int(((pred==1)&(y_true==1)).sum()); fp=int(((pred==1)&(y_true==0)).sum())
        tn=int(((pred==0)&(y_true==0)).sum()); fn=int(((pred==0)&(y_true==1)).sum())
        sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
        ppv=tp/(tp+fp) if tp+fp else np.nan; npv=tn/(tn+fn) if tn+fn else np.nan
        acc=(tp+tn)/n; f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan
        rows.append(dict(threshold=t,TP=tp,FP=fp,TN=tn,FN=fn,prevalence=round(float(prev),4),
                         sensitivity=round(float(sens),4),specificity=round(float(spec),4),
                         PPV=round(float(ppv),4),NPV=round(float(npv),4),accuracy=round(float(acc),4),
                         F1=round(float(f1),4),balanced_accuracy=round(float(np.nanmean([sens,spec])),4),
                         youden_J=round(float(sens+spec-1),4)))
    return pd.DataFrame(rows)

def calibration_points(y, p, n_bins=10):
    tmp = pd.DataFrame({"y":y,"p":p}).sort_values("p").reset_index(drop=True)
    tmp["bin"] = pd.qcut(tmp.index, q=n_bins, labels=False, duplicates="drop")
    out = tmp.groupby("bin").agg(predicted=("p","mean"), observed=("y","mean")).reset_index(drop=True)
    return out

def dca_points(y_true, p):
    rows=[]; y=np.asarray(y_true).astype(int); n=len(y); prev=y.mean()
    for t in np.linspace(0.05,0.80,40):
        pred=(p>=t).astype(int)
        tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum()
        w=t/(1-t)
        rows.append((t, tp/n - fp/n*w, prev - (1-prev)*w, 0.0))
    return pd.DataFrame(rows, columns=["threshold","nb_model","nb_all","nb_none"])

def main():
    root = Path(__file__).resolve().parent
    data_path = root/"data"/"derived"/"ADNIMERGE_ready_m12_AD_withlabel.csv"
    out_i = root/"output"/"intermediate"; out_t = root/"output"/"tables"
    out_i.mkdir(parents=True, exist_ok=True); out_t.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "dx_bl" in df.columns:
        df = df[df["dx_bl"].astype(str).str.upper()=="AD"].copy()

    exclude = {"rid","ptid","viscode","dx_bl","mmse_m12","mmse_change","decline_3pt"}
    X = df[[c for c in df.columns if c not in exclude]].copy()
    y = df["decline_3pt"].astype(int).values
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    pre_logit = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], verbose_feature_names_out=False)
    logit_pipe = Pipeline([("prep", pre_logit), ("clf", LogisticRegression(max_iter=5000, solver="liblinear", random_state=RANDOM_SEED))])

    pre_rf = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], verbose_feature_names_out=False)
    rf_pipe = Pipeline([("prep", pre_rf), ("clf", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1))])
    rf_grid = {
        "clf__n_estimators":[300,500],
        "clf__max_depth":[None,4,6],
        "clf__min_samples_split":[2,5],
        "clf__min_samples_leaf":[1,2],
        "clf__max_features":["sqrt",0.5]
    }

    oof = pd.DataFrame(index=df.index, data={"RID":df["rid"].values, "y_true":y,
                                             "rf_prob_raw":np.nan, "rf_prob":np.nan,
                                             "logit_prob_raw":np.nan, "logit_prob":np.nan})
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold,(tr,va) in enumerate(outer.split(X,y), start=1):
        Xtr,Xva = X.iloc[tr],X.iloc[va]
        ytr = y[tr]

        # logistic + isotonic
        logit_pipe.fit(Xtr,ytr)
        p_tr = logit_pipe.predict_proba(Xtr)[:,1]
        p_va = logit_pipe.predict_proba(Xva)[:,1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_tr,ytr)
        oof.loc[va,"logit_prob_raw"] = p_va
        oof.loc[va,"logit_prob"] = iso.transform(p_va)

        # RF tuning + isotonic
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED+fold)
        tuner = GridSearchCV(rf_pipe, rf_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
        tuner.fit(Xtr,ytr)
        p_tr = tuner.predict_proba(Xtr)[:,1]
        p_va = tuner.predict_proba(Xva)[:,1]
        iso_rf = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_tr,ytr)
        oof.loc[va,"rf_prob_raw"] = p_va
        oof.loc[va,"rf_prob"] = iso_rf.transform(p_va)

    oof.to_csv(out_i/"predictions_AD_oof.csv", index=False)

    # metrics
    metrics=[]
    for model_col, name in [("logit_prob","Logistic"), ("rf_prob","RandomForest")]:
        p = oof[model_col].to_numpy(dtype=float)
        ci, cs = calibration_intercept_slope(y, p)
        metrics.append(dict(model=name, AUC=roc_auc_score(y,p), Brier=brier_score_loss(y,p), Calib_Intercept=ci, Calib_Slope=cs))
    pd.DataFrame(metrics).to_csv(out_t/"metrics_AD_12m.csv", index=False)

    # ROC/calibration/DCA
    fpr,tpr,_ = roc_curve(y, oof["rf_prob"])
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(out_i/"ROC_points_RF_AD.csv", index=False)
    calibration_points(y, oof["rf_prob"]).to_csv(out_i/"Calibration_points_RF_AD.csv", index=False)
    dca = dca_points(y, oof["rf_prob"])
    dca[["threshold","nb_model"]].rename(columns={"nb_model":"net_benefit"}).to_csv(out_i/"DCA_points_model_AD.csv", index=False)
    dca[["threshold","nb_all"]].rename(columns={"nb_all":"net_benefit"}).to_csv(out_i/"DCA_points_treatall_AD.csv", index=False)
    dca[["threshold","nb_none"]].rename(columns={"nb_none":"net_benefit"}).to_csv(out_i/"DCA_points_treatnone_AD.csv", index=False)

    threshold_metrics(y, oof["rf_prob"]).to_csv(out_t/"threshold_performance_RF.csv", index=False)
    threshold_metrics(y, oof["logit_prob"]).to_csv(out_t/"threshold_performance_Logistic.csv", index=False)

    # risk stratification
    tmp = oof[["RID","y_true","rf_prob"]].copy().merge(df[["rid","mmse_change"]], left_on="RID", right_on="rid", how="left").drop(columns=["rid"])
    tmp["risk_stratum"] = pd.cut(tmp["rf_prob"], bins=[-np.inf,0.25,0.50,np.inf], labels=["Low (<0.25)","Mid (0.25-0.50)","High (>=0.50)"], right=False)
    rs = tmp.groupby("risk_stratum", observed=True).agg(N=("y_true","size"), events=("y_true","sum"), event_rate=("y_true","mean"), mean_mmse_change=("mmse_change","mean")).reset_index()
    rs["event_rate"] = rs["event_rate"].round(3)
    rs["mean_mmse_change"] = rs["mean_mmse_change"].round(2)
    rs.to_csv(out_t/"risk_stratification_summary.csv", index=False)
    tmp.to_csv(out_i/"RiskStratification_RF_AD.csv", index=False)

    # feature importance (simple reconstructed full-data RF with minimal encoding)
    Xfi = X.copy()
    for c in cat_cols:
        Xfi[c] = Xfi[c].astype("category").cat.codes.replace(-1, np.nan)
    Xfi = Xfi.fillna(Xfi.median(numeric_only=True))
    rffi = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
    rffi.fit(Xfi, y)
    fi = pd.DataFrame({"feature":Xfi.columns, "importance":rffi.feature_importances_}).sort_values("importance", ascending=False).head(12)
    fi.to_csv(out_i/"FeatureImportance_Top12_RF_AD.csv", index=False)

    print("Reconstructed outputs exported.")

if __name__ == "__main__":
    main()
