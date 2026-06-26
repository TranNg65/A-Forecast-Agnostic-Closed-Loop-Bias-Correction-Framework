"""SVR forecaster configuration.

Identical data/feature/split pipeline to xg_boost_local so per-customer
half-hourly forecasts are directly comparable between XGBoost and SVR.

Kernel SVR is O(n^2)-O(n^3) and infeasible on the full ~29M training rows,
so two scalable variants are supported:

  * LinearSVR        — linear epsilon-insensitive baseline (primal, liblinear).
  * Nystroem + LinearSVR  — RBF kernel approximation via Nystroem features.

Both use the same StandardScaler-normalised feature matrix as the XGBoost
training, so the validation metrics are directly comparable.
"""

import os

# Reuse all paths and feature definitions from the XGBoost training pipeline.
from xg_boost_local.config import (
    USAGE_TRAIN_FILE, USAGE_VAL_FILE, UNIT_FILE,
    SGSC_WEATHER_FILE, FEATURE_COLS,
)

# ── Output directory (mirrors xg_boost_local/output structure) ──────────────
OUTPUT_DIR = '/home/tran/trace_ws/outputs/svr_unit_local'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Training subsample ──────────────────────────────────────────────────────
# Stratified by customer; same fraction across all 2081 SGSC apartments so the
# customer distribution is preserved.  ~500k rows ≈ 240 per customer.
TRAIN_SUBSAMPLE_PER_CUSTOMER = 240
RANDOM_STATE                  = 42

# ── Hyperparameters ─────────────────────────────────────────────────────────
LINEAR_SVR_PARAMS = {
    'C':            1.0,
    'epsilon':      0.05,    # tolerance band on log1p(unit_kw) target
    'loss':         'epsilon_insensitive',
    'max_iter':     5000,
    'dual':         True,    # primal not supported with epsilon_insensitive
    'random_state': RANDOM_STATE,
}

NYSTROEM_PARAMS = {
    'kernel':       'rbf',
    'gamma':        0.05,
    'n_components': 512,
    'random_state': RANDOM_STATE,
}

NYSTROEM_LSVR_PARAMS = {
    'C':            1.0,
    'epsilon':      0.05,
    'loss':         'epsilon_insensitive',
    'max_iter':     8000,
    'dual':         True,
    'random_state': RANDOM_STATE,
}
