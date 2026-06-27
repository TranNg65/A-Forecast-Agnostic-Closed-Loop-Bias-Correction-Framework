"""
Input data and model parameters for the STRATA optimisation workflow.
"""

import os
import numpy as np
import pandas as pd

_USAGE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "strata", "unit_usage.csv")
_EXPORT_REBATE_CAP_KW = 50.0

# Lazily loaded; cached after first read so repeated calls don't re-parse the file
_usage_df: pd.DataFrame | None = None

def _load_usage() -> pd.DataFrame:
    global _usage_df
    if _usage_df is None:
        df = pd.read_csv(_USAGE_CSV, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        # Rename unit_N columns to zero-based integer indices (unit_1 → 0, etc.)
        df.columns = [int(c.split("_")[1]) - 1 for c in df.columns]
        # Sort columns so D_u[u] matches unit_{u+1} — same ordering as forecast.py
        df = df.sort_index(axis=1)
        _usage_df = df
    return _usage_df


def get_available_dates():
    """Return sorted list of all date strings available in unit_usage.csv."""
    df = _load_usage()
    return sorted(df.index.normalize().strftime("%Y-%m-%d").unique().tolist())


def build_input_data(demand_scale, date="2013-01-01",
                     use_forecast=False, gamma=0.3, soc_carry=None,
                     lookback_days=7, use_historical_avg=False,
                     use_persistence=False, forecast_model="xgboost",
                     ew_decay=0.8, cf_min=0.5, cf_max=2.0,
                     pi_block_edges=None, pi_aggregate_units=None,
                     pi_aggregate_blend=0.0):
    """
    Parameters
    ----------
    use_forecast       : bool   if True, use XGBoost predictions for D_u_forecast
                                (actual D_u is always loaded for post-hoc Equation E)
    gamma              : float  safety factor ∈ [0,1] for SOC reserve (Constraint C)
                                0 = trust forecast fully; 1 = fully risk-averse
    soc_carry          : float  initial SOC [kWh] for rolling carry-over (Constraint D1)
                                None → use SOC_target × E_bat_max (first day default)
    lookback_days      : int    days of past forecast errors used for bias correction
                                0 → use raw XGBoost forecast with no correction
    use_historical_avg : bool   if True, use per-unit historical mean demand as
                                D_u_forecast — represents the old BESS model that has
                                no day-ahead prediction, just assumes an average day.
                                Overrides use_forecast when True.
    """
    # ---- Time parameters ----
    T = 48                  # Number of half-hour time periods in one day
    delta_t = 0.5           # Time step duration [h]

    # ---- Unit / apartment sets ----
    # 12 apartments total: 6 × 1-bed, 3 × 2-bed, 3 × 3-bed
    U = list(range(12))     # Indices 0..11

    # Apartment type mapping
    #   Units 0-5:  1-bedroom  (6 units)
    #   Units 6-8:  2-bedroom  (3 units)
    #   Units 9-11: 3-bedroom  (3 units)
    unit_type = {}
    for u in range(0, 6):
        unit_type[u] = "1bed"
    for u in range(6, 9):
        unit_type[u] = "2bed"
    for u in range(9, 12):
        unit_type[u] = "3bed"

    # Scheme participants (S ⊆ U) — example: mix of apartment types
    #   Scheme: 1-bed units 0-3, 2-bed units 6-7, 3-bed unit 9  (7 units)
    S = [0, 1, 2, 3, 6, 7, 9]
    # Non-scheme units (N = U \ S)
    N = [u for u in U if u not in S]

    # ---- Demand scaling factor ----
    # Change this to simulate more demand for the building
    DEMAND_SCALE = demand_scale

    # ---- Solar PV capacity profile P_pv(t) [kW] ----
    # Source: demand.csv "200kW Solar PV (kWh)" column, converted to kW (÷ Δt)
    P_pv_kwh = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 2.00, 6.50, 12.00, 16.00,
        22.00, 30.00, 38.00, 32.00, 42.00, 43.00, 47.00, 48.00,
        49.00, 49.00, 48.00, 47.00, 43.00, 42.00, 38.00, 32.00,
        30.00, 22.00, 16.00, 12.00, 6.50, 2.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    ])
    P_pv = P_pv_kwh / delta_t  # Convert kWh per interval → kW

    # ---- Base electricity demand D_u(t) [kW] per unit — loaded from CSV ----
    df = _load_usage()
    target_day = pd.Timestamp(date).normalize()
    date_str = target_day.strftime("%Y-%m-%d")
    usage_days = df.index.normalize()
    day_data = df[usage_days == target_day]
    if len(day_data) != T:
        raise ValueError(
            f"Expected {T} intervals for {date_str}, got {len(day_data)}. "
            f"Available range: {df.index[0].date()} – {df.index[-1].date()}"
        )
    # day_data columns are 0..11; values are kWh per interval → convert to kW
    # D_u is always actual demand — used in post-hoc Equation E and unit cost billing
    D_u = (day_data.sort_index().values.T * DEMAND_SCALE) / delta_t  # shape (12, 48)

    # ---- Common area demand D_common(t) [kW] ----
    # Source: demand.csv "Common Area 12-Unit Building (kWh)" column → kW
    # Scaled by DEMAND_SCALE to account for lifts, pool pump, gym, HVAC in lobbies
    D_common_kwh = np.array([
        0.50, 0.45, 0.45, 0.45, 0.40, 0.40, 0.40, 0.40,
        0.40, 0.40, 0.45, 0.50, 0.65, 0.75, 0.85, 0.95,
        1.00, 0.95, 0.75, 0.70, 0.65, 0.60, 0.60, 0.60,
        0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65,
        0.75, 0.85, 0.95, 1.05, 1.15, 1.20, 1.15, 1.10,
        1.05, 0.95, 0.85, 0.75, 0.65, 0.60, 0.55, 0.50
    ]) * DEMAND_SCALE
    D_common = D_common_kwh / delta_t  # convert to kW

    # ---- Grid electricity price C_g(t) [$/kWh] — time-of-use tariff ----
    C_g = np.array([
        99.34, 95.51, 88.41, 87.90, 82.77, 80.29, 78.36, 75.29,
        81.26, 87.00, 92.83, 91.46, 122.65, 172.54, 153.32, 128.48,
        92.38, 76.01, 79.65, 72.44, 63.66, 60.05, 56.50, 59.83,
        56.57, 62.71, 74.45, 74.71, 76.89, 76.76, 76.67, 102.19,
        151.03, 175.66, 400.00, 400.00, 377.38, 196.43, 154.77, 162.63,
        154.79, 138.49, 126.93, 103.93, 109.45, 108.73, 107.64, 101.32
    ]) / 100.0  # Convert cents to $/kWh

    # ---- Export prices ----
    # C_excess(t): cost/penalty for exporting above threshold [$/kWh]
    C_excess = np.full(T, 0.02)  # Small penalty for excess export

    # C_rebate(t): rebate for exporting below threshold [$/kWh]
    C_rebate = np.full(T, 0.05)  # Feed-in tariff / rebate

    # ---- Scheme reduced rate C_scheme(t) [$/kWh] ----
    # Participants pay a reduced rate (e.g., 60-70% of grid rate)
    C_scheme = C_g * 0.65

    # ---- Export settings ----
    # Rebate only applies up to this export cap; any additional export is
    # charged via C_excess instead. Adjust this to match the site's agreement.
    P_export_threshold = _EXPORT_REBATE_CAP_KW

    # ---- Battery parameters ----
    E_bat_max = 500.0       # Maximum battery capacity [kWh]
    P_bat_max = 250.0       # Maximum charge/discharge power [kW]
    eta_c = 0.95            # Charging efficiency
    eta_d = 0.95            # Discharging efficiency
    SOC_min = 0.20          # Minimum SOC (20%)
    SOC_max = 0.80          # Maximum SOC (80%)
    SOC_target = 0.50       # Target SOC (50%)

    # ---- Cost parameters (from Pre-defined Parameters table) ----
    C_deg = 0.05            # Battery degradation cost [$/kWh]
    C_dis_inc = 0.50        # Battery discharge incentive [$/kWh]
    C_ch_inc = 0.10         # Battery charging incentive [$/kWh]
    lambda_SOC = 0.02       # Intraday SOC deviation penalty weight [$/kWh]
    lambda_term = 0.10      # Terminal SOC deviation penalty weight [$/kWh]
    SOC_term_floor = 0.30   # Hard lower floor on terminal SOC (fraction of E_bat_max)

    # ---- Constraint A: forecast demand seen by the optimizer ----
    correction_factors = np.ones(12)   # default: no correction applied
    if use_historical_avg:
        # Historical average benchmark — use only past days so the baseline is
        # causal and comparable to day-ahead forecast methods.
        past_days = df[(usage_days < target_day) &
                       (usage_days >= target_day - pd.Timedelta(days=14))]
        if past_days.empty:
            raise ValueError(
                f"Historical average benchmark: no past data available in the 14 days before {date_str}."
            )
        mean_profile = past_days.groupby(past_days.index.time).mean().sort_index()
        if len(mean_profile) != T:
            raise ValueError(
                f"Historical average benchmark: expected {T} time slots, got {len(mean_profile)}."
            )
        D_u_forecast = (mean_profile.values.T * DEMAND_SCALE) / delta_t  # (12,48) kW
        D_u_sigma_sc = np.zeros_like(D_u)
    elif use_persistence:
        # Persistence: use previous day's actual demand as the forecast.
        prev_day = target_day - pd.Timedelta(days=1)
        prev_date_str = prev_day.strftime("%Y-%m-%d")
        prev_data = df[usage_days == prev_day]
        if len(prev_data) != T:
            raise ValueError(
                f"Persistence forecast: no data for previous day {prev_date_str}. "
                f"Available range: {df.index[0].date()} – {df.index[-1].date()}"
            )
        D_u_forecast = (prev_data.sort_index().values.T * DEMAND_SCALE) / delta_t
        D_u_sigma_sc = np.zeros_like(D_u)
    elif use_forecast:
        if forecast_model == "xgboost":
            if lookback_days > 0:
                from forecast import get_corrected_forecast
                D_u_hat, D_u_sigma, correction_factors = get_corrected_forecast(
                    date_str, lookback_days=lookback_days, delta_t=delta_t,
                    ew_decay=ew_decay, cf_min=cf_min, cf_max=cf_max,
                    pi_block_edges=pi_block_edges,
                    aggregate_units=pi_aggregate_units,
                    aggregate_blend=pi_aggregate_blend,
                )
            else:
                from forecast import get_forecast
                D_u_hat, D_u_sigma = get_forecast(date_str, delta_t=delta_t)
        elif forecast_model in ("svr_linear", "svr_nystroem"):
            from svr_local.forecast import (
                get_forecast as svr_get_forecast,
                get_corrected_forecast as svr_get_corrected_forecast,
            )
            if lookback_days > 0:
                D_u_hat, D_u_sigma, correction_factors = svr_get_corrected_forecast(
                    date_str, model=forecast_model,
                    lookback_days=lookback_days, delta_t=delta_t,
                    ew_decay=ew_decay, cf_min=cf_min, cf_max=cf_max,
                    pi_block_edges=pi_block_edges,
                    aggregate_units=pi_aggregate_units,
                    aggregate_blend=pi_aggregate_blend,
                )
            else:
                D_u_hat, D_u_sigma = svr_get_forecast(
                    date_str, model=forecast_model, delta_t=delta_t,
                )
        else:
            raise ValueError(f"Unknown forecast_model: {forecast_model}")
        D_u_forecast = D_u_hat   * DEMAND_SCALE
        D_u_sigma_sc = D_u_sigma * DEMAND_SCALE
    else:
        D_u_forecast = D_u.copy()                 # optimizer sees actual demand
        D_u_sigma_sc = np.zeros_like(D_u)

    # ---- Constraint C: per-interval SOC reserve R_res(τ) [kWh] ----
    # R_res(τ) = γ · Σ_{u∈S} σ_u(τ) · Δt,  capped at ½(SOC_max−SOC_min)·E_bat_max
    R_res_cap = 0.5 * (SOC_max - SOC_min) * E_bat_max   # 150 kWh
    R_res = np.array([
        min(gamma * sum(D_u_sigma_sc[u, t] * delta_t for u in S), R_res_cap)
        for t in range(T)
    ])

    # ---- Constraint D1: initial SOC [kWh] ----
    # Use carried SOC from previous day if provided; otherwise use 50% target
    soc_init = soc_carry if soc_carry is not None else SOC_target * E_bat_max

    return {
        "T": T, "delta_t": delta_t, "DEMAND_SCALE": DEMAND_SCALE,
        "date": date_str, "use_forecast": use_forecast, "gamma": gamma,
        "use_historical_avg": use_historical_avg, "use_persistence": use_persistence,
        "correction_factors": correction_factors,
        "pi_aggregate_blend": pi_aggregate_blend,
        "U": U, "S": S, "N": N,
        "unit_type": unit_type,
        # D_u      = actual demand  (billing, post-hoc Equation E)
        # D_u_forecast = what the optimizer plans against (actual or predicted)
        "P_pv": P_pv, "D_u": D_u, "D_u_forecast": D_u_forecast,
        "D_u_sigma": D_u_sigma_sc, "R_res": R_res,
        "D_common": D_common,
        "C_g": C_g, "C_excess": C_excess, "C_rebate": C_rebate, "C_scheme": C_scheme,
        "P_export_threshold": P_export_threshold,
        "E_bat_max": E_bat_max, "P_bat_max": P_bat_max,
        "eta_c": eta_c, "eta_d": eta_d,
        "SOC_min": SOC_min, "SOC_max": SOC_max, "SOC_target": SOC_target,
        "soc_init": soc_init,
        "C_deg": C_deg, "C_dis_inc": C_dis_inc, "C_ch_inc": C_ch_inc,
        "lambda_SOC": lambda_SOC, "lambda_term": lambda_term,
        "SOC_term_floor": SOC_term_floor,
    }
