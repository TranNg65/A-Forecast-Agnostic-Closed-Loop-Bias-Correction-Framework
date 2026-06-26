"""
xgboost_unit/train.py
─────────────────────
Trains XGBoost to predict 30-min unit electricity demand from weather,
time, and unit demographic features.

Run split_dataset.py FIRST to generate pre-split input files:
  {SPLIT_DIR}/unit_only.csv
  {SPLIT_DIR}/unit_usage_train.csv
  {SPLIT_DIR}/unit_usage_val.csv

Lag features are computed on the combined train+val timeseries (so shift(48)
truly means 24 hours ago), then masked to the day-ahead information set before
the split is evaluated.  This keeps training and optimisation inference on the
same causal feature contract.

Usage:
    # from trace_ws/:
    python3 -m xg_boost_local.train

"""
import gc
import os
import pickle
import tempfile
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import xgboost as xgb

from .config import (
    USAGE_TRAIN_FILE, USAGE_VAL_FILE, OUTPUT_DIR,
    XGB_PARAMS, XGB_MONOTONE, FEATURE_COLS,
)
from .data import (
    load_weather,
    load_train_val_unit_data,
    compute_customer_baseline,
    attach_customer_baseline,
)
from .features import compute_metrics
from .plots import (
    plot_actual_vs_predicted,
    plot_customer_traces,
    plot_feature_importance,
    plot_residuals,
    plot_predicted_vs_actual_scatter,
    plot_per_customer_errors,
)


def build_monotone_vector(feature_cols):
    """XGBoost expects a tuple/list matching feature order: 0/1/-1 per feature."""
    return tuple(XGB_MONOTONE.get(c, 0) for c in feature_cols)


def main():
    print("=" * 70)
    print("XGBoost Unit Demand Model (Weather + Demographics)")
    print("=" * 70)

    print("\n── Step 1: Loading BOM 2013 weather (Station 066212) ──")
    weather = load_weather()

    print("\n── Step 2: Loading and preparing TRAIN + VAL data ──")
    train_df, val_df = load_train_val_unit_data(
        weather, USAGE_TRAIN_FILE, USAGE_VAL_FILE,
    )
    gc.collect()

    print("\n── Step 3: Computing customer × hour-of-week baseline (TRAIN only) ──")
    # Free val_df before the groupby to avoid OOM on the memory spike (~2 GB saved)
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as _f:
        _val_tmp = _f.name
        pickle.dump(val_df, _f, protocol=5)
    del val_df; gc.collect()

    baseline = compute_customer_baseline(train_df)
    train_df = attach_customer_baseline(train_df, baseline)

    with open(_val_tmp, 'rb') as _f:
        val_df = pickle.load(_f)
    os.unlink(_val_tmp)
    val_df   = attach_customer_baseline(val_df,   baseline)
    print(f"  Baseline rows: {len(baseline):,}  "
          f"(customers × half-hours-of-week)")
    del baseline; gc.collect()

    missing_cols = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing_cols:
        print(f"\n  WARNING — missing feature columns: {missing_cols}")
    feature_cols_final = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"  Features used: {len(feature_cols_final)}")

    n_train_rows = len(train_df)
    n_val_rows   = len(val_df)
    n_customers  = val_df['customer_ID'].nunique()
    print(f"  Train : {n_train_rows:,} rows  "
          f"({pd.to_datetime(train_df['timestamp']).dt.normalize().nunique()} days)")
    print(f"  Val   : {n_val_rows:,} rows  "
          f"({pd.to_datetime(val_df['timestamp']).dt.normalize().nunique()} days)")

    # Extract column-by-column to avoid a 2× DataFrame copy when selecting
    # mixed-dtype columns (int8/int16/float32) before casting to float32.
    y_train_raw = train_df['unit_kw'].to_numpy(dtype='float32')
    y_train     = np.log1p(y_train_raw)
    X_train = np.empty((len(train_df), len(feature_cols_final)), dtype='float32')
    for i, col in enumerate(feature_cols_final):
        X_train[:, i] = train_df[col].to_numpy(dtype='float32')
    del train_df; gc.collect()

    y_val            = val_df['unit_kw'].to_numpy(dtype='float32')
    y_val_log        = np.log1p(y_val)
    val_timestamps   = pd.to_datetime(val_df['timestamp']).to_numpy()
    val_customer_ids = val_df['customer_ID'].to_numpy()
    X_val = np.empty((len(val_df), len(feature_cols_final)), dtype='float32')
    for i, col in enumerate(feature_cols_final):
        X_val[:, i] = val_df[col].to_numpy(dtype='float32')
    del val_df; gc.collect()

    print("\n── Step 4: Training ──")
    params = dict(XGB_PARAMS)
    params['monotone_constraints'] = build_monotone_vector(feature_cols_final)

    # Build DMatrix first, then delete the raw arrays (frees ~9 GB) before the
    # training loop starts so the GPU + histogram structures have headroom.
    print("  Building DMatrix (train) …")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols_final)
    del X_train, y_train; gc.collect()

    print("  Building DMatrix (val) …")
    dval = xgb.DMatrix(X_val, label=y_val_log, feature_names=feature_cols_final)
    del X_val, y_val_log; gc.collect()

    # xgb.train() params (strip sklearn-only keys)
    _SKLEARN_KEYS = {'n_estimators', 'early_stopping_rounds', 'n_jobs', 'random_state'}
    xgb_params = {k: v for k, v in params.items() if k not in _SKLEARN_KEYS}
    xgb_params['nthread'] = params.get('n_jobs', -1)
    xgb_params['seed']    = params.get('random_state', 42)

    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=params['early_stopping_rounds'],
        verbose_eval=100,
    )

    # Save immediately after training before any other booster operations.
    model_path = os.path.join(OUTPUT_DIR, 'model_unit_demand.json')
    booster.save_model(model_path)
    print(f"\n  Model saved → {model_path}")

    print("\n── Step 5: Evaluating ──")
    # Reload from disk to get a CPU booster — set_param resets weights in XGBoost 3.x
    # and predict(device=) is not available in this build.
    cpu_booster = xgb.Booster()
    cpu_booster.load_model(model_path)
    y_pred_train = np.expm1(cpu_booster.predict(dtrain))
    del dtrain; gc.collect()
    y_pred_val   = np.expm1(cpu_booster.predict(dval))

    print(f"  Best iteration: {cpu_booster.best_iteration}")
    m_train = compute_metrics(y_train_raw, y_pred_train, 'Train')
    m_val   = compute_metrics(y_val,       y_pred_val,   'Val  ')

    print("\n── Step 6: Generating plots ──")
    plot_actual_vs_predicted(val_timestamps, y_val, y_pred_val, m_val)
    plot_customer_traces(val_timestamps, val_customer_ids, y_val, y_pred_val)
    plot_feature_importance(cpu_booster, feature_cols_final)
    plot_residuals(val_timestamps, y_val, y_pred_val)
    plot_predicted_vs_actual_scatter(y_val, y_pred_val)
    plot_per_customer_errors(val_customer_ids, y_val, y_pred_val)

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Target variable   : unit_kw (kWh per 30-minute interval)")
    print(f"  Customers in model: {n_customers}")
    print(f"  Features used     : {len(feature_cols_final)}")
    print(f"  Split strategy    : per-month tail (last 20% of days each month)")
    print(f"  Training rows     : {n_train_rows:,}")
    print(f"  Validation rows   : {n_val_rows:,}")
    print(f"  Objective         : {params.get('objective', 'reg:squarederror')}")
    print(f"  Device            : {params.get('device', 'cpu').upper()}")
    print(f"  Best XGB iteration: {cpu_booster.best_iteration}")
    print()
    print(f"  {'Split':<10} {'MAE':>10} {'RMSE':>10} {'sMAPE%':>10} {'CV(RMSE)%':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Train':<10} {m_train['mae']:>10.4f} {m_train['rmse']:>10.4f} "
          f"{m_train['smape']:>10.2f} {m_train['cv_rmse']:>10.2f}")
    print(f"  {'Val':<10} {m_val['mae']:>10.4f} {m_val['rmse']:>10.4f} "
          f"{m_val['smape']:>10.2f} {m_val['cv_rmse']:>10.2f}")
    print()
    print(f"  Model  → {OUTPUT_DIR}/model_unit_demand.json")
    print(f"  Plots  → {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
