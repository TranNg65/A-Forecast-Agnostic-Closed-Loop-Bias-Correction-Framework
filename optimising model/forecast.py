"""
forecast.py
───────────
Wraps the XGBoost unit-demand model (model_unit_demand.json) to produce
per-unit, per-interval point forecasts and proportional uncertainty estimates
for a single day.

Returns (12, 48) arrays in kW — matching D_u units used throughout model.py.
The XGBoost model is cached after the first load.

Public API
----------
get_forecast(date, ...)                     → (D_u_hat, D_u_sigma)
get_corrected_forecast(date, ...)           → (D_u_hat, D_u_sigma, correction)
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

_DIR     = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.normpath(os.path.join(_DIR, '..'))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from xg_boost_local.config import OUTPUT_DIR, FEATURE_COLS as _FEATURE_COLS, AU_HOLIDAYS_2013
from xg_boost_local.features import (
    add_time_features,
    add_weather_daily_lags,
    add_weather_features,
    add_holiday_features,
    NSW_SCHOOL_HOLIDAYS_2013,
)

_MODEL_PATH = os.path.join(OUTPUT_DIR, 'model_unit_demand.json')

_USAGE_FILE   = os.path.join(_DIR, '..', 'data', 'strata', 'unit_usage.csv')
_UNITS_FILE   = os.path.join(_DIR, '..', 'data', 'strata', 'units.csv')
_WEATHER_FILE = os.path.join(_DIR, '..', 'data', 'strata', 'bom_station_066212_2013.csv')

_PERIODS_PER_DAY = 48
_HISTORY_DAYS    = 15   # lag_672 needs 14 days; 15 gives one spare day

_xgb_model  = None   # module-level cache; avoids reloading on each call
_full_usage  = None
_forecast_cache = {}  # date_str -> (D_u_hat, D_u_sigma); avoids re-running the
                      # 7-day rolling lookback that overlaps across days.
_CACHE_MAX = 64


def _load_full_usage(usage_file):
    global _full_usage
    if _full_usage is None:
        _full_usage = pd.read_csv(usage_file, parse_dates=['timestamp'])
    return _full_usage


def _load_xgb():
    global _xgb_model
    if _xgb_model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"XGBoost model not found: {_MODEL_PATH}\n"
                f"Run 'python3 -m xg_boost_local.train' from trace_ws/ to train the model first."
            )
        print(f"  [forecast] Loading XGBoost model …", flush=True)
        m = xgb.Booster()
        m.load_model(_MODEL_PATH)
        _xgb_model = m
        print(f"  [forecast] Loaded ({m.num_features()} features)")
    return _xgb_model


# ── Feature engineering ───────────────────────────────────────────────────────

def _load_demographics(units_file):
    units = pd.read_csv(units_file)
    lmh   = {'LOW': 0, 'MED': 1, 'HI': 2}
    units['electricity_usage_enc'] = units['electricity_usage_group'].map(lmh).astype('int8')
    units['gas_usage_enc']         = units['gas_usage_group'].map(lmh).astype('int8')
    units['income_enc']            = units['income_group'].map(lmh).astype('int8')
    for src, dst in [
        ('HAS_AIRCON', 'has_aircon'), ('HAS_GAS', 'has_gas'),
        ('HAS_GAS_HEATING', 'has_gas_heating'),
        ('HAS_GAS_HOT_WATER', 'has_gas_hot_water'),
        ('HAS_GAS_COOKING', 'has_gas_cooking'),
        ('IS_RENTING', 'is_renting'),
    ]:
        units[dst] = (units[src] == 'Y').astype('int8')
    units['has_children_enc'] = units['has_children'].map(
        {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}
    ).astype('float32')
    for col in ['num_occupants', 'num_children_0_10',
                'num_children_11_17', 'num_occupants_70plus']:
        if col in units.columns:
            units[col] = units[col].astype('int8')
    demo_cols = [
        'electricity_usage_enc', 'gas_usage_enc', 'income_enc',
        'has_aircon', 'has_gas', 'has_gas_heating',
        'has_gas_hot_water', 'has_gas_cooking', 'is_renting',
        'eDaily', 'peakTime',
        'num_occupants', 'num_children_0_10', 'num_children_11_17',
        'num_occupants_70plus', 'has_children_enc',
    ]
    return units.set_index('unit_id')[[c for c in demo_cols if c in units.columns]]


def _build_feature_df(
    date,
    usage_file=_USAGE_FILE,
    units_file=_UNITS_FILE,
    weather_file=_WEATHER_FILE,
):
    """
    Build a feature DataFrame for `date` — one row per unit per half-hour.

    Returns df with 12×48 rows tagged with 'unit_id' for downstream pivoting.
    """
    target        = pd.Timestamp(date).normalize()
    history_start = target - pd.Timedelta(days=_HISTORY_DAYS)
    history_end   = target + pd.Timedelta(days=1)

    all_cols = pd.read_csv(usage_file, nrows=0).columns.tolist()
    unit_cols = [c for c in all_cols if c.startswith('unit_')]
    if not unit_cols:
        raise ValueError(f"No unit_* columns found in {usage_file}")

    chunks = []
    for chunk in pd.read_csv(usage_file, parse_dates=['timestamp'], chunksize=10_000):
        chunk = chunk[
            (chunk['timestamp'] >= history_start)
            & (chunk['timestamp'] < history_end)
        ]
        if not chunk.empty:
            chunks.append(chunk)

    if chunks:
        usage_df = pd.concat(chunks, ignore_index=True)
    else:
        raise ValueError(f"No usage rows found for feature window ending {target.date()}")

    df = usage_df.melt(id_vars='timestamp', value_vars=unit_cols,
                       var_name='unit_id', value_name='unit_kw')
    df['unit_kw'] = df['unit_kw'].astype(np.float32)
    df = df.sort_values(['unit_id', 'timestamp']).reset_index(drop=True)

    weather = pd.read_csv(weather_file, parse_dates=['date'])
    weather['date'] = weather['date'].dt.normalize()
    weather = add_weather_daily_lags(weather)

    df['date'] = df['timestamp'].dt.normalize()
    df = df.merge(weather, on='date', how='left')

    df = add_time_features(df, 'timestamp')

    # Per-unit loop avoids rolling window crossing unit boundaries.
    for uid in df['unit_id'].unique():
        mask = df['unit_id'] == uid
        s = df.loc[mask, 'unit_kw']
        for lag in (1, 2, 3, 6, 12, 24, 48, 96, 336, 672):
            df.loc[mask, f'unit_lag_{lag}'] = s.shift(lag).values
        shifted = s.shift(1)
        df.loc[mask, 'unit_rolling_mean_48']  = shifted.rolling(48,  min_periods=1).mean().values
        df.loc[mask, 'unit_rolling_std_48']   = shifted.rolling(48,  min_periods=1).std().values
        df.loc[mask, 'unit_rolling_max_48']   = shifted.rolling(48,  min_periods=1).max().values
        df.loc[mask, 'unit_rolling_mean_336'] = shifted.rolling(336, min_periods=1).mean().values

    df = add_weather_features(df)
    df = add_holiday_features(df, 'timestamp', AU_HOLIDAYS_2013, NSW_SCHOOL_HOLIDAYS_2013)

    demo = _load_demographics(units_file)
    df   = df.merge(demo, left_on='unit_id', right_index=True, how='left')

    # Compute per-unit, per-half-hour-of-week mean — replicates
    # compute_customer_baseline() used during training so the model sees a real
    # value instead of NaN at inference time.
    baseline = (
        df.groupby(['unit_id', 'hour_of_week'])['unit_kw']
            .mean()
            .rename('customer_how_mean')
            .reset_index()
    )
    df = df.merge(baseline, on=['unit_id', 'hour_of_week'], how='left')

    df = df[df['date'] == target].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"No rows for {target.date()} after feature engineering. "
            f"Available range: 2013-01-01 – 2013-12-31"
        )

    return df


# ── XGBoost forecasts ─────────────────────────────────────────────────────────

def get_forecast(
    date,
    delta_t=0.5,
    sigma_scale=0.15,
    usage_file=_USAGE_FILE,
    units_file=_UNITS_FILE,
    weather_file=_WEATHER_FILE,
):
    """
    Predict per-unit demand for `date`.

    Parameters
    ----------
    date        : str | pd.Timestamp  target date in 2013 (YYYY-MM-DD)
    delta_t     : float               interval length [h]
    sigma_scale : float               uncertainty as fraction of point forecast (default 15%)

    Returns
    -------
    D_u_hat   : np.ndarray (12, 48)  point forecast [kW],  unit_N → index N-1
    D_u_sigma : np.ndarray (12, 48)  std dev estimate [kW]
    """
    cache_key = str(pd.Timestamp(date).date())
    cached = _forecast_cache.get(cache_key)
    if cached is not None:
        D_u_hat, _ = cached
        return D_u_hat.copy(), (sigma_scale * D_u_hat).astype(np.float32)

    model = _load_xgb()
    df    = _build_feature_df(date, usage_file, units_file, weather_file)

    X        = df[_FEATURE_COLS].values.astype('float32')
    dmat     = xgb.DMatrix(X, feature_names=_FEATURE_COLS)
    pred_kwh = np.clip(np.expm1(model.predict(dmat)), 0.0, None).astype('float32')

    D_u_hat   = np.zeros((12, _PERIODS_PER_DAY))
    D_u_sigma = np.zeros((12, _PERIODS_PER_DAY))

    for uid in sorted(df['unit_id'].unique(), key=lambda x: int(x.split('_')[1])):
        idx  = int(uid.split('_')[1]) - 1
        mask = (df['unit_id'] == uid).values
        D_u_hat[idx, :]   = pred_kwh[mask] / delta_t
        D_u_sigma[idx, :] = sigma_scale * D_u_hat[idx, :]

    if len(_forecast_cache) >= _CACHE_MAX:
        _forecast_cache.pop(next(iter(_forecast_cache)))
    _forecast_cache[cache_key] = (D_u_hat.copy(), D_u_sigma.copy())
    return D_u_hat, D_u_sigma


def _make_blocks(pi_block_edges, T=_PERIODS_PER_DAY):
    """
    Partition the T half-hour intervals into contiguous time-of-day blocks.

    pi_block_edges : None | sequence of interior cut points (interval indices).
        None (or empty) → a single block spanning the whole day, which makes the
        PI loop revert to the original scalar daily-energy correction.
        e.g. (12, 18, 32, 38) → blocks [0:12], [12:18], [18:32], [32:38], [38:48].

    Returns a list of np.ndarray, each holding the interval indices of one block.
    """
    if not pi_block_edges:
        return [np.arange(T)]
    cuts = [0] + [int(c) for c in pi_block_edges] + [T]
    return [np.arange(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def get_corrected_forecast(
    date,
    lookback_days=7,
    cf_min=0.5,
    cf_max=2.0,
    delta_t=0.5,
    sigma_scale=0.15,
    ew_decay=0.8,
    pi_block_edges=None,
    aggregate_units=None,
    aggregate_blend=0.0,
    usage_file=_USAGE_FILE,
    units_file=_UNITS_FILE,
    weather_file=_WEATHER_FILE,
):
    """
    XGBoost forecast with multiplicative per-unit bias correction derived from
    the previous `lookback_days` days of actual vs predicted demand.

    Correction uses exponential weighting so that day d-ago has weight ew_decay^d,
    giving recent errors more influence than older ones.

    The PI loop runs independently within each time-of-day block b (defined by
    `pi_block_edges`).  For each unit u and block b:
        α[u,b] = clip(
            Σ_d w_d * actual_kWh[u,b,d] / Σ_d w_d * predicted_kWh[u,b,d],
            cf_min, cf_max
        )
    With `pi_block_edges=None` this reduces exactly to the original single-scalar
    daily-energy correction; with tariff-aligned blocks it additionally corrects
    the forecaster's time-of-day *shape* bias (e.g. evening-peak under-prediction)
    that a single daily gain averages away.

    If `aggregate_units` is provided, those units' per-block gains are shrunk
    toward the aggregate gain for the same block:
        α'[u,b] = (1 - β) α[u,b] + β α[aggregate,b].
    This is useful for the MILP because dispatch depends on scheme-aggregate
    demand, while individual apartment ratios can be noisy.  β=0 recovers the
    original per-unit correction.

    Returns
    -------
    D_u_corr  : np.ndarray (12, 48)  bias-corrected forecast [kW]
    D_u_sigma : np.ndarray (12, 48)  uncertainty, also corrected [kW]
    correction: np.ndarray (12,)     effective per-unit daily gain (for display)
    """
    target = pd.Timestamp(date)
    blocks = _make_blocks(pi_block_edges)
    n_blocks = len(blocks)

    D_u_hat, D_u_sigma = get_forecast(
        date, delta_t=delta_t, sigma_scale=sigma_scale,
        usage_file=usage_file, units_file=units_file, weather_file=weather_file,
    )

    usage_full = _load_full_usage(usage_file)
    usage_days = usage_full['timestamp'].dt.normalize()
    unit_cols  = [c for c in usage_full.columns if c.startswith('unit_')]

    aggregate_blend = float(np.clip(aggregate_blend, 0.0, 1.0))
    if aggregate_units is not None:
        aggregate_units = [int(u) for u in aggregate_units]

    print(f"  [forecast] Computing bias correction from {lookback_days} past days "
          f"(ew_decay={ew_decay}, {n_blocks} block(s)"
          f"{', aggregate_blend=' + format(aggregate_blend, '.2f') if aggregate_units and aggregate_blend else ''}) …",
          flush=True)

    actual_blk   = np.zeros((12, n_blocks))   # EW actual kWh per unit per block
    pred_blk     = np.zeros((12, n_blocks))   # EW predicted kWh per unit per block
    total_weight = 0.0
    days_used    = 0

    for d in range(1, lookback_days + 1):
        past_date = target - pd.Timedelta(days=d)
        past_str  = past_date.strftime("%Y-%m-%d")

        day_rows = usage_full[usage_days == past_date.normalize()]
        if len(day_rows) != 48:
            continue

        try:
            D_past, _ = get_forecast(
                past_str, delta_t=delta_t, sigma_scale=0.0,
                usage_file=usage_file, units_file=units_file, weather_file=weather_file,
            )
        except Exception:
            continue

        w = ew_decay ** d
        day_sorted = day_rows.sort_values('timestamp')      # 00:00 → 23:30 order
        pred_kwh_interval = D_past * delta_t                 # (12, 48) kWh
        for uc in unit_cols:
            idx  = int(uc.split('_')[1]) - 1
            vals = day_sorted[uc].values                     # 48 actual kWh/interval
            for b, intervals in enumerate(blocks):
                actual_blk[idx, b] += w * float(vals[intervals].sum())
        for b, intervals in enumerate(blocks):
            pred_blk[:, b] += w * pred_kwh_interval[:, intervals].sum(axis=1)
        total_weight += w
        days_used    += 1

    # Per-block multiplicative gains → expanded to a (12, 48) per-interval matrix.
    alpha_blk = np.ones((12, n_blocks))
    gain = np.ones((12, _PERIODS_PER_DAY))
    if total_weight > 0:
        for u in range(12):
            for b in range(n_blocks):
                if pred_blk[u, b] > 1e-3:
                    alpha_blk[u, b] = float(np.clip(actual_blk[u, b] / pred_blk[u, b],
                                                    cf_min, cf_max))
        if aggregate_units and aggregate_blend > 0.0:
            aggregate_units = [u for u in aggregate_units if 0 <= u < 12]
            if aggregate_units:
                for b in range(n_blocks):
                    pred_sum = float(pred_blk[aggregate_units, b].sum())
                    if pred_sum <= 1e-3:
                        continue
                    agg_alpha = float(np.clip(
                        actual_blk[aggregate_units, b].sum() / pred_sum,
                        cf_min, cf_max,
                    ))
                    alpha_blk[aggregate_units, b] = (
                        (1.0 - aggregate_blend) * alpha_blk[aggregate_units, b]
                        + aggregate_blend * agg_alpha
                    )
        for b, intervals in enumerate(blocks):
            gain[:, intervals] = alpha_blk[:, b][:, np.newaxis]

    D_u_corr      = D_u_hat   * gain
    D_u_sigma_cor = D_u_sigma * gain

    # Effective per-unit daily gain (energy-weighted) — preserves the (12,) display API.
    denom      = np.maximum(D_u_hat.sum(axis=1), 1e-9)
    correction = D_u_corr.sum(axis=1) / denom

    print(f"  [forecast] Correction factors ({days_used}/{lookback_days} days used, "
          f"{n_blocks} block(s)):")
    for u in range(12):
        direction = "▲" if correction[u] > 1.01 else ("▼" if correction[u] < 0.99 else "─")
        if n_blocks > 1:
            blk_str = " ".join(f"{alpha_blk[u, b]:.2f}" for b in range(n_blocks))
            print(f"             unit_{u+1:02d}: {correction[u]:.3f} {direction}  [{blk_str}]")
        else:
            print(f"             unit_{u+1:02d}: {correction[u]:.3f} {direction}")

    return D_u_corr, D_u_sigma_cor, correction
