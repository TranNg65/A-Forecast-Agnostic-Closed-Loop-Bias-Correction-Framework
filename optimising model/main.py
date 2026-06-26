"""
Orchestrates the full pipeline: data loading, model solving,
results display, plotting, and CSV export.

Usage:
  python3 main.py                       # run default date (2013-03-19)
  python3 main.py --date 2013-03-19     # run one specific date
  python3 main.py --date all            # batch-run all dates, save CSV
  python3 main.py --date 2013-03-19 --ds 5   # scale demand by ×5
"""

import argparse
import os

import numpy as np
import pandas as pd
from gurobipy import GRB

from config import build_input_data, get_available_dates
from model import build_and_solve_model
from results import extract_and_display_results, compute_realised_cost, compute_metrics
from plotting import plot_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_config(data):
    unit_daily_kwh = data["D_u"] * data["delta_t"]
    print(f"\nProblem Configuration:")
    print(f"  Date:                   {data['date']}")
    print(f"  Time periods (T):       {data['T']} (half-hourly)")
    print(f"  Demand scale factor:    x{data['DEMAND_SCALE']}")
    print(f"  Total units (U):        {len(data['U'])}")
    for u in data['U']:
        print(f"    Unit {u:2d} ({data['unit_type'][u]}):  {unit_daily_kwh[u].sum():.2f} kWh/day")
    print(f"  Scheme units (S):       {len(data['S'])} — {data['S']}")
    print(f"  Non-scheme units (N):   {len(data['N'])} — {data['N']}")
    print(f"  Solar PV daily:         {sum(data['P_pv']) * data['delta_t']:.0f} kWh/day")
    print(f"  Battery capacity:       {data['E_bat_max']} kWh")
    print(f"  Battery max power:      {data['P_bat_max']} kW")
    print(f"  Export rebate cap:      {data['P_export_threshold']} kW")
    print(f"  SOC bounds:             [{data['SOC_min']*100:.0f}%, {data['SOC_max']*100:.0f}%]")
    if data.get("use_historical_avg"):
        print(f"  Demand mode:            Historical average (past-only baseline)")
    elif data["use_forecast"]:
        cf = data["correction_factors"]
        any_corrected = any(abs(cf[u] - 1.0) > 0.01 for u in range(12))
        print(f"  Forecast mode:          XGBoost  (γ={data['gamma']})")
        if any_corrected:
            print(f"  Bias correction:")
            if data.get("pi_aggregate_blend", 0.0):
                print(f"    Scheme aggregate blend: β={data['pi_aggregate_blend']:.2f}")
            for u in data['U']:
                bar = "▲" if cf[u] > 1.01 else ("▼" if cf[u] < 0.99 else "─")
                print(f"    Unit {u:2d}: ×{cf[u]:.3f} {bar}")
        else:
            print(f"  Bias correction:        none (lookback_days=0 or first day)")
    elif data.get("use_persistence"):
        print(f"  Demand mode:            Persistence (yesterday's actual demand)")
    else:
        print(f"  Forecast mode:          Actual demand (perfect foresight)")
    print()


def _export_results(v, data, unit_costs, total_cost, suffix=""):
    os.makedirs("results", exist_ok=True)
    ds   = data["DEMAND_SCALE"]
    date = data["date"].replace("-", "")
    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

    df_res = pd.DataFrame({
        "Time":              time_labels,
        "P_grid_S_kW":       v["P_grid_S"],
        "P_pv_gen_kW":       v["P_pv_gen"],
        "P_ch_kW":           v["P_ch"],
        "P_dis_kW":          v["P_dis"],
        "SOC_kWh":           v["P_soc"][:-1],
        "SOC_pct":           v["P_soc"][:-1] / data["E_bat_max"] * 100,
        "P_export_kW":       v["P_export"],
        "P_excess_kW":       v["P_excess"],
        "P_wasted_solar_kW": v["P_wasted_solar"],
        "C_g_per_kWh":       data["C_g"],
        "Total_STRATA_Cost": [total_cost] + [np.nan] * (data["T"] - 1),
    })
    df_res.to_csv(f"results/optimization_results_{date}_{ds}{suffix}.csv", index=False)

    unit_type = data["unit_type"]
    rows = []
    for u in sorted(unit_costs.keys()):
        info = unit_costs[u]
        row  = {"Unit": u, "Apt_Type": unit_type[u],
                "In_Scheme": info["in_scheme"]}
        if info["in_scheme"]:
            row["Scheme_Cost"]                  = info["scheme_cost"]
            row["Hypothetical_NonScheme_Cost"]  = info["hypothetical_non_scheme_cost"]
            row["Savings"]                      = info["savings"]
        else:
            row["NonScheme_Cost"] = info["non_scheme_cost"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"results/unit_costs_{date}_{ds}{suffix}.csv", index=False)


def _print_benchmark_table(results):
    """
    Four-way benchmark comparison table.
    All costs are realised on actual demand via Equation E.
    Savings % is relative to the historical-average baseline.
    """
    ref = next((r for r in results if "historical" in r["label"].lower()), results[-1])
    ref_cost = ref["realized_cost"]

    W = 104
    print("\n" + "=" * W)
    print("  BENCHMARK COMPARISON  (all costs realised on actual demand — Equation E)")
    print("=" * W)
    print(f"  {'Method':<28} {'Cost ($)':>9} {'vs Avg':>8} "
          f"{'SSR%':>6} {'SCR%':>6} {'Peak±%':>7} {'MAE (kW)':>9} {'RMSE (kW)':>10}")
    print(f"  {'-'*98}")

    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<28}  ERROR: {r['error']}")
            continue
        savings_pct = (ref_cost - r["realized_cost"]) / abs(ref_cost) * 100 if ref_cost else float("nan")
        mae_s  = f"{r['mae']:>8.3f}"  if not np.isnan(r["mae"])  else "       —"
        rmse_s = f"{r['rmse']:>9.3f}" if not np.isnan(r["rmse"]) else "        —"
        marker = "  ← baseline" if r is ref else ""
        print(f"  {r['label']:<28} {r['realized_cost']:>9.2f} {savings_pct:>+7.1f}% "
              f"{r['ssr']:>5.1f}% {r['scr']:>5.1f}% {r['peak_reduction']:>6.1f}% "
              f"{mae_s} {rmse_s}{marker}")

    print("=" * W)
    print("  SSR = self-sufficiency ratio  |  SCR = self-consumption ratio  "
          "|  Peak± = peak demand change vs no-BESS (negative = BESS increases peak via grid charging)")


def _export_comparison(data_f, v_f, v_hist):
    """Save per-interval comparison CSV (XGBoost vs historical-average baseline)."""
    os.makedirs("results", exist_ok=True)
    T    = data_f["T"]
    S    = data_f["S"]
    date = data_f["date"].replace("-", "")
    ds   = data_f["DEMAND_SCALE"]

    time_labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

    D_fc  = data_f["D_u_forecast"]
    D_act = data_f["D_u"]

    scheme_xgb = np.array([sum(D_fc[u, t] for u in S) for t in range(T)])
    scheme_act = np.array([sum(D_act[u, t] for u in S) for t in range(T)])

    P_grid_xgb, _, _ = compute_realised_cost(v_f, data_f)

    df = pd.DataFrame({
        "Time":                    time_labels,
        "C_g_per_kWh":             data_f["C_g"],
        "Demand_actual_kW":        scheme_act,
        "Demand_XGBoost_kW":       scheme_xgb,
        "Demand_error_kW":         scheme_xgb - scheme_act,
        # New model (XGBoost)
        "NEW_P_ch_kW":             v_f["P_ch"],
        "NEW_P_dis_kW":            v_f["P_dis"],
        "NEW_SOC_kWh":             v_f["P_soc"][:-1],
        "NEW_P_grid_planned_kW":   v_f["P_grid_S"],
        "NEW_P_grid_realised_kW":  P_grid_xgb,
        # Historical-average baseline
        "HISTAVG_P_ch_kW":         v_hist["P_ch"],
        "HISTAVG_P_dis_kW":        v_hist["P_dis"],
        "HISTAVG_SOC_kWh":         v_hist["P_soc"][:-1],
        "HISTAVG_P_grid_kW":       v_hist["P_grid_S"],
        # Per-interval battery schedule difference
        "Delta_P_ch_kW":           v_f["P_ch"]  - v_hist["P_ch"],
        "Delta_P_dis_kW":          v_f["P_dis"] - v_hist["P_dis"],
    })
    path = f"results/forecast_comparison_{date}_{ds}.csv"
    df.to_csv(path, index=False)
    print(f"  Comparison CSV → {path}")


# ── Default run parameters ────────────────────────────────────────────────────

DEMAND_SCALE  = 10
SOC_CARRY     = None
LOOKBACK_DAYS = 7
RESULTS_SUFFIX = "_aggblend50"

XGB_PI_GAMMA    = 0.25
SVR_PI_GAMMA    = 0.25
XGB_PI_RHO      = 0.70
SVR_PI_RHO      = 0.90
XGB_PI_CLIP_MIN = 0.70
XGB_PI_CLIP_MAX = 1.50
SVR_PI_CLIP_MIN = 0.70
SVR_PI_CLIP_MAX = 1.50
XGB_PI_AGGREGATE_BLEND = 0.50
SVR_PI_AGGREGATE_BLEND = 0.00

# Time-of-day blocks for the PI feedback loop (interior cut points, interval idx):
#   [0:12] overnight | [12:18] morning | [18:32] midday | [32:38] evening peak | [38:48] night
# The evening-peak block isolates the $6.00/kWh 17:00–18:30 spike so the PI loop
# corrects the forecaster's time-of-day shape bias there, not just daily level.
# Set to None to recover the original single-scalar daily-energy correction.
PI_BLOCK_EDGES  = (12, 18, 32, 38)
PI_AGGREGATE_UNITS = (0, 1, 2, 3, 6, 7, 9)


# ── Silent helper for batch runs ──────────────────────────────────────────────

def _extract_vars(variables):
    """Pull Gurobi variable values into plain numpy arrays."""
    result = {}
    for name, var in variables.items():
        if hasattr(var, 'X'):
            # scalar Var (not indexed)
            result[name] = float(var.X)
        else:
            result[name] = np.array([var[t].X for t in sorted(var.keys())])
    return result


def _benchmark_from_v(label, v, data):
    """Build a benchmark result dict from an already-extracted vars dict (no re-solve)."""
    P_grid_real, _, realized_cost = compute_realised_cost(v, data)
    m = compute_metrics(v, data, P_grid_real)

    is_perfect = not any([data.get("use_forecast"), data.get("use_historical_avg"),
                          data.get("use_persistence")])
    if is_perfect:
        mae = rmse = float("nan")
    else:
        S, T = data["S"], data["T"]
        D_f, D_a = data["D_u_forecast"], data["D_u"]
        errs = np.array([sum(D_f[u, t] - D_a[u, t] for u in S) for t in range(T)])
        mae  = float(np.mean(np.abs(errs)))
        rmse = float(np.sqrt(np.mean(errs ** 2)))

    return {
        "label":          label,
        "realized_cost":  round(realized_cost, 4),
        "ssr":            m["ssr"],
        "scr":            m["scr"],
        "peak_reduction": m["peak_reduction_pct"],
        "mae":            mae,
        "rmse":           rmse,
        "error":          "",
        "_v":             v,
        "_data":          data,
    }


def _run_benchmark(date, label, **build_kwargs):
    """Solve one benchmark silently and return a metrics dict."""
    try:
        data = build_input_data(DEMAND_SCALE, date, **build_kwargs)
    except Exception as e:
        return {"label": label, "error": str(e),
                "mae": float("nan"), "rmse": float("nan")}

    gm, vars_ = build_and_solve_model(data, silent=True)
    if gm.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return {"label": label, "error": f"solver status={gm.Status}",
                "mae": float("nan"), "rmse": float("nan")}

    v      = _extract_vars(vars_)
    result = _benchmark_from_v(label, v, data)
    # obj_val: planned cost (what optimizer expected with forecast demand)
    # final_soc: end-of-day SOC [kWh] — used for next-day carry-over
    result["obj_val"]   = float(gm.ObjVal)
    result["final_soc"] = float(v["P_soc"][-1])
    return result


def _run_date_silent(date, soc_carry=None):
    """
    Run all benchmarks for one date silently.
    soc_carry : end-of-previous-day SOC [kWh] for carry-over; None = 50% default
    Returns a flat result dict for CSV export, or a dict with 'error' key.
    """
    _soc   = soc_carry  # None is fine — build_input_data handles it

    bench_xgb = _run_benchmark(date, "XGBoost",
                               use_forecast=True, gamma=XGB_PI_GAMMA,
                               soc_carry=_soc, lookback_days=LOOKBACK_DAYS,
                               forecast_model="xgboost",
                               ew_decay=XGB_PI_RHO,
                               cf_min=XGB_PI_CLIP_MIN, cf_max=XGB_PI_CLIP_MAX,
                               pi_block_edges=PI_BLOCK_EDGES,
                               pi_aggregate_units=PI_AGGREGATE_UNITS,
                               pi_aggregate_blend=XGB_PI_AGGREGATE_BLEND)
    bench_xgb_no_pi = _run_benchmark(date, "XGBoostNoPI",
                                      use_forecast=True, gamma=XGB_PI_GAMMA,
                                      soc_carry=_soc, lookback_days=0,
                                      forecast_model="xgboost")
    bench_svr = _run_benchmark(date, "SVR",
                               use_forecast=True, gamma=SVR_PI_GAMMA,
                               soc_carry=_soc, lookback_days=LOOKBACK_DAYS,
                               forecast_model="svr_nystroem",
                               ew_decay=SVR_PI_RHO,
                               cf_min=SVR_PI_CLIP_MIN, cf_max=SVR_PI_CLIP_MAX,
                               pi_block_edges=PI_BLOCK_EDGES,
                               pi_aggregate_units=PI_AGGREGATE_UNITS,
                               pi_aggregate_blend=SVR_PI_AGGREGATE_BLEND)
    bench_svr_no_pi = _run_benchmark(date, "SVRNoPI",
                                      use_forecast=True, gamma=SVR_PI_GAMMA,
                                      soc_carry=_soc, lookback_days=0,
                                      forecast_model="svr_nystroem")
    bench_avg = _run_benchmark(date, "HistAvgPastOnly",
                               use_historical_avg=True, gamma=0.0,
                               soc_carry=_soc, lookback_days=0)
    bench_per = _run_benchmark(date, "Persistence",
                               use_persistence=True, gamma=0.0,
                               soc_carry=_soc, lookback_days=0)
    bench_pf  = _run_benchmark(date, "PerfectForesight",
                               use_forecast=False, gamma=0.0,
                               soc_carry=_soc, lookback_days=0)

    errors = {b["label"]: b["error"]
              for b in [bench_xgb, bench_xgb_no_pi,
                        bench_svr, bench_svr_no_pi,
                        bench_avg, bench_per, bench_pf]
              if b.get("error")}
    if errors:
        return {"date": date, "error": str(errors)}

    ref      = bench_avg["realized_cost"]
    savings  = ref - bench_xgb["realized_cost"]
    sav_pct  = savings / abs(ref) * 100 if ref else float("nan")

    forecast_err = bench_xgb["realized_cost"] - bench_xgb.get("obj_val",
                                                               bench_xgb["realized_cost"])

    pi_gain     = bench_xgb_no_pi["realized_cost"] - bench_xgb["realized_cost"]
    pi_gain_pct = (pi_gain / abs(bench_xgb_no_pi["realized_cost"]) * 100
                   if bench_xgb_no_pi["realized_cost"] else float("nan"))
    svr_pi_gain     = bench_svr_no_pi["realized_cost"] - bench_svr["realized_cost"]
    svr_pi_gain_pct = (svr_pi_gain / abs(bench_svr_no_pi["realized_cost"]) * 100
                      if bench_svr_no_pi["realized_cost"] else float("nan"))

    return {
        "date":                 date,
        "gamma_used":           round(XGB_PI_GAMMA, 4),
        "svr_gamma_used":       round(SVR_PI_GAMMA, 4),
        "soc_start_kwh":        round(_soc if _soc is not None else 250.0, 2),
        "soc_end_kwh":          round(bench_xgb.get("final_soc", float("nan")), 2),
        "forecast_err":         round(forecast_err, 4),
        "cost_xgboost":          bench_xgb["realized_cost"],
        "cost_xgboost_no_pi":    bench_xgb_no_pi["realized_cost"],
        "cost_svr":              bench_svr["realized_cost"],
        "cost_svr_no_pi":        bench_svr_no_pi["realized_cost"],
        "cost_hist_avg":         bench_avg["realized_cost"],
        "cost_persistence":      bench_per["realized_cost"],
        "cost_perfect":          bench_pf["realized_cost"],
        "savings_vs_avg":        round(savings, 4),
        "savings_pct":           round(sav_pct, 2),
        "pi_gain_vs_no_pi":      round(pi_gain, 4),
        "pi_gain_pct":           round(pi_gain_pct, 2),
        "svr_pi_gain":           round(svr_pi_gain, 4),
        "svr_pi_gain_pct":       round(svr_pi_gain_pct, 2),
        "ssr_xgb":               bench_xgb["ssr"],
        "scr_xgb":               bench_xgb["scr"],
        "peak_reduction_xgb":    bench_xgb["peak_reduction"],
        "ssr_xgb_no_pi":         bench_xgb_no_pi["ssr"],
        "scr_xgb_no_pi":         bench_xgb_no_pi["scr"],
        "peak_reduction_xgb_no_pi": bench_xgb_no_pi["peak_reduction"],
        "ssr_svr":               bench_svr["ssr"],
        "scr_svr":               bench_svr["scr"],
        "peak_reduction_svr":    bench_svr["peak_reduction"],
        "ssr_svr_no_pi":         bench_svr_no_pi["ssr"],
        "scr_svr_no_pi":         bench_svr_no_pi["scr"],
        "peak_reduction_svr_no_pi": bench_svr_no_pi["peak_reduction"],
        "mae_xgb":               round(bench_xgb["mae"], 4),
        "rmse_xgb":              round(bench_xgb["rmse"], 4),
        "mae_xgb_no_pi":         round(bench_xgb_no_pi["mae"], 4),
        "rmse_xgb_no_pi":        round(bench_xgb_no_pi["rmse"], 4),
        "mae_svr":               round(bench_svr["mae"], 4),
        "rmse_svr":              round(bench_svr["rmse"], 4),
        "mae_svr_no_pi":         round(bench_svr_no_pi["mae"], 4),
        "rmse_svr_no_pi":        round(bench_svr_no_pi["rmse"], 4),
        "mae_hist_avg":          round(bench_avg["mae"], 4),
        "rmse_hist_avg":         round(bench_avg["rmse"], 4),
        "mae_persistence":       round(bench_per["mae"], 4),
        "rmse_persistence":      round(bench_per["rmse"], 4),
        "error":                 "",
    }


# ── Single-date full run (with full output) ───────────────────────────────────

def _run_single(date):
    print("=" * 60)
    print("  STRATA Community Energy Sharing Optimization")
    print("=" * 60)

    data = build_input_data(DEMAND_SCALE, date,
                            use_forecast=True, gamma=XGB_PI_GAMMA,
                            soc_carry=SOC_CARRY, lookback_days=LOOKBACK_DAYS,
                            ew_decay=XGB_PI_RHO,
                            cf_min=XGB_PI_CLIP_MIN,
                            cf_max=XGB_PI_CLIP_MAX,
                            pi_block_edges=PI_BLOCK_EDGES,
                            pi_aggregate_units=PI_AGGREGATE_UNITS,
                            pi_aggregate_blend=XGB_PI_AGGREGATE_BLEND)
    _print_config(data)

    gurobi_model, variables = build_and_solve_model(data)
    v, unit_costs, total_cost = extract_and_display_results(gurobi_model, variables, data)
    if v is None:
        return

    plot_results(v, data)
    _export_results(v, data, unit_costs, total_cost, suffix="_forecast")

    print("\n" + "─" * 60)
    print("  Running benchmark models …")
    print("─" * 60)

    bench_xgb = _benchmark_from_v("XGBoost (new model)", v, data)
    bench_avg = _run_benchmark(date, "Historical average (past-only)",
                               use_historical_avg=True, gamma=0.0,
                               soc_carry=SOC_CARRY, lookback_days=0)
    bench_per = _run_benchmark(date, "Persistence (yesterday)",
                               use_persistence=True, gamma=0.0,
                               soc_carry=SOC_CARRY, lookback_days=0)
    bench_pf  = _run_benchmark(date, "Perfect foresight",
                               use_forecast=False, gamma=0.0,
                               soc_carry=SOC_CARRY, lookback_days=0)

    _print_benchmark_table([bench_pf, bench_xgb, bench_avg, bench_per])

    if not bench_avg.get("error") and "_v" in bench_avg:
        _export_comparison(data, v, bench_avg["_v"])

    print(f"\nResults exported to 'results/' directory (demand_scale={DEMAND_SCALE}).")


# ── Batch run over all dates ───────────────────────────────────────────────────

def _run_all():
    dates = get_available_dates()
    total = len(dates)
    print(f"Batch run: {total} dates, demand_scale={DEMAND_SCALE}")
    print(f"SOC carry-over: enabled")
    print(f"XGB PI: gamma={XGB_PI_GAMMA}, rho={XGB_PI_RHO}, "
          f"clip=[{XGB_PI_CLIP_MIN}, {XGB_PI_CLIP_MAX}], "
          f"aggregate_blend={XGB_PI_AGGREGATE_BLEND}")
    print(f"SVR PI: gamma={SVR_PI_GAMMA}, rho={SVR_PI_RHO}, "
          f"clip=[{SVR_PI_CLIP_MIN}, {SVR_PI_CLIP_MAX}], "
          f"aggregate_blend={SVR_PI_AGGREGATE_BLEND}")
    print("─" * 80)

    rows      = []
    soc_carry = None          # None → first day starts at SOC_target (50%)

    for i, date in enumerate(dates, 1):
        print(f"  [{i:3d}/{total}]  {date}  "
              f"SOC₀={'default' if soc_carry is None else f'{soc_carry:.1f} kWh'} … ",
              end="", flush=True)

        row = _run_date_silent(date, soc_carry=soc_carry)
        rows.append(row)

        if row.get("error"):
            print(f"ERROR: {row['error']}")
            soc_carry = None    # reset carry on solver failure
        else:
            pi_sign     = "+" if row["pi_gain_vs_no_pi"] >= 0 else ""
            svr_pi_sign = "+" if row["svr_pi_gain"]     >= 0 else ""
            print(f"xgb=${row['cost_xgboost']:.2f} (no-PI=${row['cost_xgboost_no_pi']:.2f}, PI gain {pi_sign}{row['pi_gain_vs_no_pi']:.2f})  "
                  f"svr=${row['cost_svr']:.2f} (no-PI=${row['cost_svr_no_pi']:.2f}, PI gain {svr_pi_sign}{row['svr_pi_gain']:.2f})  "
                  f"avg=${row['cost_hist_avg']:.2f}  pf=${row['cost_perfect']:.2f}")

            # ── SOC carry-over ────────────────────────────────────────────────
            soc_carry = row["soc_end_kwh"]

    os.makedirs("results", exist_ok=True)
    out_path = f"results/all_dates_comparison_scale{DEMAND_SCALE}{RESULTS_SUFFIX}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    successful = [r for r in rows if not r.get("error")]
    if successful:
        n_ok        = len(successful)
        avg_savings = np.mean([r["savings_vs_avg"] for r in successful])
        pct_positive = (sum(1 for r in successful if r["savings_vs_avg"] > 0)
                        / n_ok * 100)
        ann_pf      = sum(r["cost_perfect"]        for r in successful)
        ann_xgb     = sum(r["cost_xgboost"]        for r in successful)
        ann_xgb_no  = sum(r["cost_xgboost_no_pi"]  for r in successful)
        ann_svr     = sum(r["cost_svr"]            for r in successful)
        ann_svr_no  = sum(r["cost_svr_no_pi"]      for r in successful)
        ann_ha      = sum(r["cost_hist_avg"]       for r in successful)
        ann_pers    = sum(r["cost_persistence"]    for r in successful)
        print(f"\n{'─'*70}")
        print(f"  Dates run:              {n_ok}/{total}")
        print(f"  Avg savings/day vs HA:  ${avg_savings:.2f}")
        print(f"  Days XGBoost wins HA:   {pct_positive:.1f}%")
        print(f"\n  ── Annual realised cost ($, lower=better) ──")
        print(f"    PF        : {ann_pf:>12,.2f}")
        print(f"    XGB-PI    : {ann_xgb:>12,.2f}   (PI gain vs noPI: ${ann_xgb_no - ann_xgb:+,.2f})")
        print(f"    XGB-noPI  : {ann_xgb_no:>12,.2f}")
        print(f"    SVR-PI    : {ann_svr:>12,.2f}   (PI gain vs noPI: ${ann_svr_no - ann_svr:+,.2f})")
        print(f"    SVR-noPI  : {ann_svr_no:>12,.2f}")
        print(f"    HA        : {ann_ha:>12,.2f}")
        print(f"    Pers      : {ann_pers:>12,.2f}")
    print(f"\nSaved → {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STRATA energy optimisation — XGBoost model vs old model (no prediction)"
    )
    parser.add_argument(
        "--date",
        default="2013-03-19",
        help="Date to run (YYYY-MM-DD), or 'all' for every date in the dataset",
    )
    parser.add_argument(
        "--ds",
        type=float,
        default=DEMAND_SCALE,
        help=f"Demand scale multiplier (default: {DEMAND_SCALE})",
    )
    args = parser.parse_args()
    DEMAND_SCALE = args.ds

    if args.date.lower() == "all":
        _run_all()
    else:
        _run_single(args.date)
