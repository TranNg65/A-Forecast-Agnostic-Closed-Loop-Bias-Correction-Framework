"""
svr_local/train.py
──────────────────
Trains LinearSVR and Nystroem-RBF+LinearSVR per-customer demand forecasters
on the identical SGSC train/val split used by the XGBoost model.

Outputs (in outputs/svr_unit_local/):
  scaler.pkl              StandardScaler fit on the training subsample
  linear_svr.pkl          fitted LinearSVR pipeline
  nystroem_svr.pkl        fitted Nystroem + LinearSVR pipeline
  metrics_comparison.json MAE / RMSE / sMAPE / CV(RMSE) on the FULL validation
                          set for LinearSVR, Nystroem+LinearSVR, and the
                          previously trained XGBoost booster.

Usage:
    cd /home/tran/trace_ws
    python3 -m svr_local.train
"""

import gc
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from xg_boost_local.data import (
    load_weather,
    load_train_val_unit_data,
    compute_customer_baseline,
    attach_customer_baseline,
)
from xg_boost_local.config import (
    USAGE_TRAIN_FILE, USAGE_VAL_FILE, FEATURE_COLS,
    OUTPUT_DIR as XGB_OUTPUT_DIR,
)

from .config import (
    OUTPUT_DIR,
    TRAIN_SUBSAMPLE_PER_CUSTOMER, RANDOM_STATE,
    LINEAR_SVR_PARAMS, NYSTROEM_PARAMS, NYSTROEM_LSVR_PARAMS,
)


# ── Metric helpers (shared shape with xg_boost_local.features.compute_metrics)─

def _safe_smape(y, yhat):
    eps = 1e-6
    return float(np.mean(np.abs(y - yhat) / (0.5 * (np.abs(y) + np.abs(yhat)) + eps)) * 100.0)


def _metrics(y, yhat, label):
    y    = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mae   = float(np.mean(np.abs(y - yhat)))
    rmse  = float(np.sqrt(np.mean((y - yhat) ** 2)))
    smape = _safe_smape(y, yhat)
    ybar  = float(np.mean(y))
    cvrmse = rmse / ybar * 100.0 if ybar > 0 else float("nan")
    return {"split": label, "MAE": mae, "RMSE": rmse,
            "sMAPE_pct": smape, "CV_RMSE_pct": cvrmse, "n": int(y.size)}


# ── Data preparation ────────────────────────────────────────────────────────

def _subsample_training(train_df, per_customer, random_state):
    """Return a stratified sample of `per_customer` rows from each customer."""
    rng = np.random.default_rng(random_state)
    sampled_idx = []
    grouped = train_df.groupby("customer_ID", sort=False).indices
    for cid, idx in grouped.items():
        if len(idx) <= per_customer:
            sampled_idx.append(idx)
        else:
            sampled_idx.append(rng.choice(idx, size=per_customer, replace=False))
    sampled_idx = np.concatenate(sampled_idx)
    rng.shuffle(sampled_idx)
    return train_df.iloc[sampled_idx].reset_index(drop=True)


def _to_xy(df, feature_cols):
    X = np.empty((len(df), len(feature_cols)), dtype=np.float32)
    for i, c in enumerate(feature_cols):
        X[:, i] = df[c].to_numpy(dtype=np.float32)
    y_kw  = df["unit_kw"].to_numpy(dtype=np.float32)
    y_log = np.log1p(y_kw)
    return X, y_kw, y_log


def _predict_in_batches(predict_fn, X, batch=200_000):
    """Run `predict_fn` over X in fixed-size batches to keep memory bounded."""
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch):
        end = min(start + batch, len(X))
        out[start:end] = predict_fn(X[start:end])
    return out


# ── XGBoost reference predictions on the same validation set ────────────────

def _xgb_val_predictions(X_val):
    import xgboost as xgb
    model_path = os.path.join(XGB_OUTPUT_DIR, "model_unit_demand.json")
    if not os.path.exists(model_path):
        print(f"  [xgb] reference model not found at {model_path}; skipping.")
        return None
    booster = xgb.Booster()
    booster.load_model(model_path)
    dmat = xgb.DMatrix(X_val, feature_names=FEATURE_COLS)
    return np.expm1(booster.predict(dmat))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("SVR Unit-Demand Forecaster — LinearSVR and Nystroem+LinearSVR")
    print("=" * 72)

    print("\n── Step 1: weather ──")
    weather = load_weather()

    print("\n── Step 2: train + val data (same loader as XGBoost) ──")
    train_df, val_df = load_train_val_unit_data(weather, USAGE_TRAIN_FILE, USAGE_VAL_FILE)
    gc.collect()

    print("\n── Step 3: customer baseline (TRAIN only) ──")
    baseline = compute_customer_baseline(train_df)
    train_df = attach_customer_baseline(train_df, baseline)
    val_df   = attach_customer_baseline(val_df,   baseline)
    del baseline; gc.collect()

    # Match XGBoost's feature dropping behaviour
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    print(f"\n── Step 4: stratified subsample for SVR ({TRAIN_SUBSAMPLE_PER_CUSTOMER}/customer) ──")
    train_sub = _subsample_training(train_df, TRAIN_SUBSAMPLE_PER_CUSTOMER, RANDOM_STATE)
    print(f"  Subsample: {len(train_sub):,} rows  "
          f"({train_sub['customer_ID'].nunique()} customers)")
    del train_df; gc.collect()

    X_tr, y_tr_kw, y_tr_log = _to_xy(train_sub, feature_cols)
    del train_sub; gc.collect()

    print("\n── Step 5: validation feature matrix ──")
    X_val, y_val_kw, _ = _to_xy(val_df, feature_cols)
    del val_df; gc.collect()
    print(f"  X_val shape: {X_val.shape}")

    print("\n── Step 6: impute NaN (median) + standardise features ──")
    n_nan_tr  = int(np.isnan(X_tr).sum())
    n_nan_val = int(np.isnan(X_val).sum())
    print(f"  NaN count — train: {n_nan_tr:,}  val: {n_nan_val:,}")
    imputer = SimpleImputer(strategy="median")
    X_tr  = imputer.fit_transform(X_tr).astype(np.float32)
    X_val = imputer.transform(X_val).astype(np.float32)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f, protocol=5)

    results = {}

    # ── 6a: LinearSVR ──
    lsvr_path = os.path.join(OUTPUT_DIR, "linear_svr.pkl")
    print("\n── Step 6a: LinearSVR ──")
    if os.path.exists(lsvr_path):
        print(f"  → reusing {lsvr_path}")
        with open(lsvr_path, "rb") as f:
            lsvr = pickle.load(f)["model"]
        t_fit = 0.0
    else:
        t0 = time.perf_counter()
        lsvr = LinearSVR(**LINEAR_SVR_PARAMS)
        lsvr.fit(X_tr_s, y_tr_log)
        t_fit = time.perf_counter() - t0
        print(f"  fit time: {t_fit:.1f} s", flush=True)
        with open(lsvr_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": lsvr,
                         "feature_cols": feature_cols}, f, protocol=5)

    y_pred_log = _predict_in_batches(lsvr.predict, X_val_s)
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    m_l = _metrics(y_val_kw, y_pred, "Val (LinearSVR)")
    m_l["fit_seconds"] = t_fit
    results["LinearSVR"] = m_l
    print(f"  Val   MAE={m_l['MAE']:.4f}  RMSE={m_l['RMSE']:.4f}  "
          f"sMAPE={m_l['sMAPE_pct']:.2f}%  CV(RMSE)={m_l['CV_RMSE_pct']:.2f}%", flush=True)

    # ── 6b: Nystroem + LinearSVR ──
    # Memory-conscious: only the TRAIN Nystroem features are materialised at
    # once (≈500k × 512 ≈ 1 GB).  The validation Nystroem features are
    # streamed in chunks during evaluation to avoid OOM on the 7.2M × n_comps
    # transformed matrix.
    print(f"\n── Step 6b: Nystroem ({NYSTROEM_PARAMS['n_components']} comps, "
          f"gamma={NYSTROEM_PARAMS['gamma']}) + LinearSVR ──")
    nys_path = os.path.join(OUTPUT_DIR, "nystroem_svr.pkl")
    if os.path.exists(nys_path):
        print(f"  → reusing {nys_path}")
        with open(nys_path, "rb") as f:
            nb = pickle.load(f)
        nys, nlsvr = nb["nystroem"], nb["model"]
        t_nys = t_fit = 0.0
    else:
        t0 = time.perf_counter()
        nys = Nystroem(**NYSTROEM_PARAMS)
        Z_tr = nys.fit_transform(X_tr_s).astype(np.float32)
        t_nys = time.perf_counter() - t0
        print(f"  Nystroem fit+transform (train) time: {t_nys:.1f} s  "
              f"Z_tr shape: {Z_tr.shape}", flush=True)

        t0 = time.perf_counter()
        nlsvr = LinearSVR(**NYSTROEM_LSVR_PARAMS)
        nlsvr.fit(Z_tr, y_tr_log)
        t_fit = time.perf_counter() - t0
        print(f"  LinearSVR on Nystroem features: {t_fit:.1f} s", flush=True)
        del Z_tr; gc.collect()

        with open(nys_path, "wb") as f:
            pickle.dump({"scaler": scaler, "nystroem": nys, "model": nlsvr,
                         "feature_cols": feature_cols}, f, protocol=5)

    # Stream X_val_s → Nystroem.transform → predict, avoiding the full Z_val
    print("  Streaming validation predictions in chunks ...")
    t0 = time.perf_counter()
    y_pred_log = np.empty(len(X_val_s), dtype=np.float32)
    CHUNK = 100_000
    for start in range(0, len(X_val_s), CHUNK):
        end  = min(start + CHUNK, len(X_val_s))
        Zb   = nys.transform(X_val_s[start:end]).astype(np.float32)
        y_pred_log[start:end] = nlsvr.predict(Zb)
        del Zb
    t_eval = time.perf_counter() - t0
    print(f"  Streaming eval time: {t_eval:.1f} s")
    y_pred = np.clip(np.expm1(y_pred_log), 0.0, None)
    m_n = _metrics(y_val_kw, y_pred, "Val (Nystroem+LinearSVR)")
    m_n["fit_seconds"] = t_nys + t_fit
    results["Nystroem_LinearSVR"] = m_n
    print(f"  Val   MAE={m_n['MAE']:.4f}  RMSE={m_n['RMSE']:.4f}  "
          f"sMAPE={m_n['sMAPE_pct']:.2f}%  CV(RMSE)={m_n['CV_RMSE_pct']:.2f}%")

    # ── 6c: XGBoost reference on identical X_val ──
    print("\n── Step 6c: XGBoost reference on identical X_val ──")
    y_pred_xgb = _xgb_val_predictions(X_val)
    if y_pred_xgb is not None:
        m_x = _metrics(y_val_kw, y_pred_xgb, "Val (XGBoost)")
        results["XGBoost"] = m_x
        print(f"  Val   MAE={m_x['MAE']:.4f}  RMSE={m_x['RMSE']:.4f}  "
              f"sMAPE={m_x['sMAPE_pct']:.2f}%  CV(RMSE)={m_x['CV_RMSE_pct']:.2f}%")

    # ── Summary ──
    json_path = os.path.join(OUTPUT_DIR, "metrics_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics → {json_path}")

    print("\n" + "=" * 72)
    print("FORECASTER COMPARISON  (validation set, same train/val split)")
    print("=" * 72)
    print(f"  {'Model':<26} {'MAE':>10} {'RMSE':>10} {'sMAPE%':>10} {'CV(RMSE)%':>11}")
    print(f"  {'-'*68}")
    for key in ("XGBoost", "Nystroem_LinearSVR", "LinearSVR"):
        if key not in results:
            continue
        r = results[key]
        print(f"  {key:<26} {r['MAE']:>10.4f} {r['RMSE']:>10.4f} "
              f"{r['sMAPE_pct']:>10.2f} {r['CV_RMSE_pct']:>11.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
