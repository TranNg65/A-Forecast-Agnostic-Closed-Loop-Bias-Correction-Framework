import os
import pandas as pd

try:
    import holidays as holidays_lib
    AU_HOLIDAYS_2013 = {pd.Timestamp(d) for d in
                        holidays_lib.Australia(state='NSW', years=2013).keys()}
except ImportError:
    AU_HOLIDAYS_2013 = set()
    print("NOTE: 'holidays' package not installed — is_public_holiday will be 0")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS — local machine
# ══════════════════════════════════════════════════════════════════════════════
SPLIT_DIR        = '/home/tran/trace_ws/data/mf_data/split'
UNIT_FILE        = os.path.join(SPLIT_DIR, 'unit_only.csv')
USAGE_TRAIN_FILE = os.path.join(SPLIT_DIR, 'unit_usage_train.csv')
USAGE_VAL_FILE   = os.path.join(SPLIT_DIR, 'unit_usage_val.csv')

SGSC_WEATHER_FILE = '/home/tran/trace_ws/data/mf_data/strata/bom_station_066212_2013.csv'

OUTPUT_DIR = '/home/tran/trace_ws/outputs/xgboost_unit_local'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS — tuned for a laptop with:
#   • NVIDIA RTX 3080 Mobile (16 GB VRAM — workstation/studio SKU)
#   • 32 GB system RAM
#   • 32 CPU cores (used for data loading / histogram precompute)
#
# 16 GB VRAM lets us keep max_bin=512 and max_leaves=512 (similar to the
# Gadi A100 config). The mobile GPU is slower than A100 per-iteration, so
# we use a low learning rate (0.003) and let early stopping (300 rounds)
# decide when to halt. If you see CUDA OOM, drop max_bin to 256 then
# max_leaves to 256.
XGB_PARAMS = {
    'objective':             'reg:pseudohubererror',
    'huber_slope':           0.2,
    'n_estimators':          3000,
    'max_depth':             9,
    'max_leaves':            512,
    'grow_policy':           'lossguide',
    'learning_rate':         0.003,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'colsample_bylevel':     0.8,
    'max_bin':               512,
    'min_child_weight':      3,
    'gamma':                 0.05,
    'reg_alpha':             0.05,
    'reg_lambda':            1.0,
    'random_state':          42,
    'n_jobs':                os.cpu_count(),
    'early_stopping_rounds': 300,
    'tree_method':           'hist',
    'device':                'cuda',
}

# ══════════════════════════════════════════════════════════════════════════════
# MONOTONIC CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════
XGB_MONOTONE = {
    'heat_stress':       1,
    'cold_stress':       1,
    'heat_stress_roll3': 1,
    'cold_stress_roll3': 1,
}

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE LIST — identical to the Gadi version
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # ── Time (cyclical) ───────────────────────────────────────────────────
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos',
    # ── Time (calendar) ───────────────────────────────────────────────────
    'tod', 'day_of_week', 'month', 'day_of_year',
    'is_weekend', 'is_summer', 'is_winter',
    'is_public_holiday', 'is_day_before_holiday', 'is_day_after_holiday',
    'is_school_holiday',
    # ── Unit demand lags — short horizon (30 min .. 12 hr) ────────────────
    'unit_lag_1', 'unit_lag_2', 'unit_lag_3',
    'unit_lag_6', 'unit_lag_12', 'unit_lag_24',
    # ── Unit demand lags — day scale ──────────────────────────────────────
    'unit_lag_48',
    'unit_lag_96',
    'unit_lag_336',
    'unit_lag_672',
    'unit_rolling_mean_48',
    'unit_rolling_std_48',
    'unit_rolling_max_48',
    'unit_rolling_mean_336',
    # ── Customer baseline ─────────────────────────────────────────────────
    'customer_how_mean',
    # ── Weather ───────────────────────────────────────────────────────────
    'max_temp_c', 'min_temp_c',
    'solar_exposure_mj_m2',
    'rainfall_mm',
    'heat_stress', 'cold_stress',
    'temp_range',
    'temp_x_tod',
    'max_temp_c_lag1', 'min_temp_c_lag1',
    'max_temp_c_roll3', 'max_temp_c_roll7',
    'heat_stress_roll3', 'cold_stress_roll3',
    # ── Unit demographics ─────────────────────────────────────────────────
    'electricity_usage_enc', 'gas_usage_enc', 'income_enc',
    'has_aircon', 'has_gas', 'has_gas_heating',
    'has_gas_hot_water', 'has_gas_cooking', 'is_renting',
    'eDaily', 'peakTime',
    # ── Occupant/household composition ────────────────────────────────────
    'num_occupants',
    'num_children_0_10',
    'num_children_11_17',
    'num_occupants_70plus',
    'has_children_enc',
]
