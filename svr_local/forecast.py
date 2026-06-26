"""
svr_local/forecast.py
─────────────────────
Drop-in SVR forecaster matching the public API of
`optimising_model.forecast` (XGBoost variant) so it can be selected at
runtime in `main.py` via `forecast_model="svr_linear"` or
`forecast_model="svr_nystroem"`.

Public API
----------
get_forecast(date, model="svr_nystroem", ...)         → (D_u_hat, D_u_sigma)
get_corrected_forecast(date, model="svr_nystroem", ...) → (D_u_hat, D_u_sigma, correction)
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

_DIR     = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.normpath(os.path.join(_DIR, '..'))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

# Reuse all feature engineering identical to the XGBoost path so the SVR is
# fed exactly the same features as the XGBoost forecaster.
sys.path.insert(0, os.path.join(_WS_ROOT, "optimising model"))
import forecast as _xgb_fc  # noqa: E402   (shared feature builder)

from xg_boost_local.config import FEATURE_COLS as _FEATURE_COLS

from .config import OUTPUT_DIR

_PERIODS_PER_DAY = 48

_svr_models    = {}  # key -> {imputer, scaler, [nystroem], model, feature_cols}
_forecast_cache = {}  # (date_str, model_key) -> (D_u_hat, D_u_sigma)
_CACHE_MAX = 64       # bounded LRU-ish; lookback uses ~14 entries per day


def _load_model(model_key):
    """Cache SVR models on first use; same idiom as the XGBoost forecaster."""
    if model_key in _svr_models:
        return _svr_models[model_key]

    fname = {
        "svr_linear":    "linear_svr.pkl",
        "svr_nystroem":  "nystroem_svr.pkl",
    }.get(model_key)
    if fname is None:
        raise ValueError(f"Unknown SVR model key: {model_key}")

    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SVR model not found: {path}\n"
            f"Run `cd /home/tran/trace_ws && python3 -m svr_local.train` to train it."
        )

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler_bundle = pickle.load(f)
    imputer = scaler_bundle["imputer"]
    scaler  = scaler_bundle["scaler"]

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    bundle["imputer"] = imputer
    bundle["scaler"]  = scaler
    _svr_models[model_key] = bundle
    print(f"  [svr_forecast] Loaded {model_key} ({len(bundle['feature_cols'])} features)")
    return bundle


def get_forecast(date, model="svr_nystroem", delta_t=0.5, sigma_scale=0.15,
                 usage_file=None, units_file=None, weather_file=None):
    """SVR demand forecast for `date`, returning (D_u_hat, D_u_sigma) in kW.

    Reuses the XGBoost path's feature builder to guarantee identical inputs.
    Caches the (date, model) result so the 7-day lookback inside
    `get_corrected_forecast` only pays the feature-engineering + prediction
    cost once per (date, model) across the rolling-horizon batch.
    """
    cache_key = (str(pd.Timestamp(date).date()), model)
    cached = _forecast_cache.get(cache_key)
    if cached is not None:
        D_u_hat, base_sigma = cached
        # honour the caller's sigma_scale even if the cached entry was built
        # with a different one (compute σ from the cached point forecast).
        D_u_sigma = (sigma_scale * D_u_hat).astype(np.float32)
        return D_u_hat.copy(), D_u_sigma

    df = _xgb_fc._build_feature_df(
        date,
        usage_file or _xgb_fc._USAGE_FILE,
        units_file or _xgb_fc._UNITS_FILE,
        weather_file or _xgb_fc._WEATHER_FILE,
    )

    bundle = _load_model(model)
    feat_cols = bundle["feature_cols"]
    X = df[feat_cols].values.astype(np.float32)

    X = bundle["imputer"].transform(X).astype(np.float32)
    X = bundle["scaler"].transform(X).astype(np.float32)

    if "nystroem" in bundle:
        Z = bundle["nystroem"].transform(X).astype(np.float32)
        y_log = bundle["model"].predict(Z)
    else:
        y_log = bundle["model"].predict(X)
    pred_kwh = np.clip(np.expm1(y_log), 0.0, None).astype(np.float32)

    D_u_hat   = np.zeros((12, _PERIODS_PER_DAY), dtype=np.float32)
    D_u_sigma = np.zeros((12, _PERIODS_PER_DAY), dtype=np.float32)

    for uid in sorted(df["unit_id"].unique(), key=lambda x: int(x.split("_")[1])):
        idx  = int(uid.split("_")[1]) - 1
        mask = (df["unit_id"] == uid).values
        D_u_hat[idx, :]   = pred_kwh[mask] / delta_t
        D_u_sigma[idx, :] = sigma_scale * D_u_hat[idx, :]

    if len(_forecast_cache) >= _CACHE_MAX:
        _forecast_cache.pop(next(iter(_forecast_cache)))
    _forecast_cache[cache_key] = (D_u_hat.copy(), D_u_sigma.copy())
    return D_u_hat, D_u_sigma


def get_corrected_forecast(date, model="svr_nystroem", lookback_days=7,
                            cf_min=0.5, cf_max=2.0, delta_t=0.5,
                            sigma_scale=0.15, ew_decay=0.8, pi_block_edges=None,
                            aggregate_units=None, aggregate_blend=0.0,
                            usage_file=None, units_file=None, weather_file=None):
    """SVR forecast with the same PI multiplicative bias-correction loop used
    for the XGBoost forecaster (Eq. 21 of the paper).  Returns
    (D_u_corr, D_u_sigma_corr, correction).

    The loop runs independently within each time-of-day block (`pi_block_edges`),
    identically to the XGBoost path; `pi_block_edges=None` reverts to the original
    single-scalar daily-energy correction.

    When `aggregate_units` is supplied, each listed unit's block gain is blended
    toward the aggregate gain for those units.  This regularises noisy apartment
    ratios toward the scheme demand profile actually used by the MILP.
    """
    target = pd.Timestamp(date)
    blocks = _xgb_fc._make_blocks(pi_block_edges)
    n_blocks = len(blocks)

    D_u_hat, D_u_sigma = get_forecast(
        date, model=model, delta_t=delta_t, sigma_scale=sigma_scale,
        usage_file=usage_file, units_file=units_file, weather_file=weather_file,
    )

    usage_full = _xgb_fc._load_full_usage(usage_file or _xgb_fc._USAGE_FILE)
    usage_days = usage_full["timestamp"].dt.normalize()
    unit_cols  = [c for c in usage_full.columns if c.startswith("unit_")]

    aggregate_blend = float(np.clip(aggregate_blend, 0.0, 1.0))
    if aggregate_units is not None:
        aggregate_units = [int(u) for u in aggregate_units]

    print(f"  [svr_forecast/{model}] PI bias correction from {lookback_days} past days"
          f" (ew_decay={ew_decay}, {n_blocks} block(s)"
          f"{', aggregate_blend=' + format(aggregate_blend, '.2f') if aggregate_units and aggregate_blend else ''}) …",
          flush=True)

    actual_blk   = np.zeros((12, n_blocks))
    pred_blk     = np.zeros((12, n_blocks))
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
                past_str, model=model, delta_t=delta_t, sigma_scale=0.0,
                usage_file=usage_file, units_file=units_file, weather_file=weather_file,
            )
        except Exception:
            continue

        w = ew_decay ** d
        day_sorted = day_rows.sort_values("timestamp")
        pred_kwh_interval = D_past * delta_t
        for uc in unit_cols:
            idx  = int(uc.split("_")[1]) - 1
            vals = day_sorted[uc].values
            for b, intervals in enumerate(blocks):
                actual_blk[idx, b] += w * float(vals[intervals].sum())
        for b, intervals in enumerate(blocks):
            pred_blk[:, b] += w * pred_kwh_interval[:, intervals].sum(axis=1)
        total_weight += w
        days_used    += 1

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
    denom      = np.maximum(D_u_hat.sum(axis=1), 1e-9)
    correction = D_u_corr.sum(axis=1) / denom
    return D_u_corr, D_u_sigma_cor, correction
