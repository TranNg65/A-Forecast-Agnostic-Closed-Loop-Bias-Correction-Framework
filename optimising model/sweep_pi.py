"""
Sweep the PI feedback-loop hyperparameters (rho, L, gamma, clip bounds)
for XGB-PI and SVR-PI, looking for the configuration that minimises annual
realised settlement cost.

Strategy: coordinate descent.

  Stage 1: vary rho   in  {0.50, 0.70, 0.85, 0.90}  (L=7,    gamma=0.50)
  Stage 2: vary L     in  {3, 7, 14, 21}            (rho*,   gamma=0.50)
  Stage 3: vary gamma in  {0.25, 0.50, 0.75}        (rho*,   L*)

A second pass narrows clip bounds {[0.50,2.00], [0.70,1.50]} once
(rho*, L*, gamma*) are fixed.

The other five scenarios (PF, XGB-noPI, SVR-noPI, HA, Pers) are
hyperparameter-independent and read straight from the existing
all_dates_comparison_scale10.csv to avoid redundant solves.

Output:
  results/sweep_pi/sweep_results.csv       — annual cost per (config, forecaster)
  results/sweep_pi/best_config.json        — best config per forecaster
  results/sweep_pi/sweep_log.txt           — running log
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_input_data, get_available_dates
from model  import build_and_solve_model
from results import compute_realised_cost
from gurobipy import GRB

DEMAND_SCALE = 10
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "sweep_pi")
os.makedirs(OUT_DIR, exist_ok=True)

# Baseline anchor — XGB-PI's final SOC drives the shared SOC chain.
# We reuse that chain for the sweep so each config's run is comparable to
# the original 7-scenario batch.
ANCHOR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results", "all_dates_comparison_scale10.csv")


def _solve(date, forecast_model, lookback, gamma, rho, cf_min, cf_max,
           soc_carry):
    """One day, one forecaster, PI on. Returns (realised_cost, final_soc)."""
    data = build_input_data(
        DEMAND_SCALE, date,
        use_forecast=True, gamma=gamma,
        soc_carry=soc_carry, lookback_days=lookback,
        forecast_model=forecast_model,
        ew_decay=rho, cf_min=cf_min, cf_max=cf_max,
    )
    gm, vars_ = build_and_solve_model(data, silent=True)
    if gm.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return float("nan"), float("nan")
    v = {}
    for name, var in vars_.items():
        if hasattr(var, "X"):
            v[name] = float(var.X)
        else:
            v[name] = np.array([var[t].X for t in sorted(var.keys())])
    _, _, realised = compute_realised_cost(v, data)
    return float(realised), float(v["P_soc"][-1])


def _run_year(forecast_model, lookback, gamma, rho, cf_min, cf_max,
              dates, soc_carry_chain):
    """Run a full year for one PI configuration.  `soc_carry_chain` is a
    dict {date: SOC at start} — we use the XGB-PI baseline chain so all
    configurations share the same initial state each day."""
    annual = 0.0
    days   = 0
    for date in dates:
        soc = soc_carry_chain.get(date)
        cost, _ = _solve(date, forecast_model, lookback, gamma, rho,
                          cf_min, cf_max, soc)
        if np.isnan(cost):
            continue
        annual += cost
        days   += 1
    return annual, days


# ── Load baseline SOC chain (from existing batch) ──────────────────
def _load_anchor():
    df = pd.read_csv(ANCHOR_CSV)
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df.sort_values("date").reset_index(drop=True)

    # SOC chain: the start of day τ equals the soc_end of day τ-1 (XGB-PI's
    # rolling chain) — this is what the original 7-scenario batch uses.
    soc_chain = {}
    prev_end = None
    for _, r in df.iterrows():
        soc_chain[r["date"]] = prev_end  # None for first day → use default
        prev_end = r["soc_end_kwh"]
    return df, soc_chain


def main():
    anchor_df, soc_chain = _load_anchor()
    dates = list(anchor_df["date"])
    print(f"Sweep over {len(dates)} days, anchored SOC chain from {ANCHOR_CSV}")

    baseline = {
        "XGB-PI (current)":  float(anchor_df["cost_xgboost"].sum()),
        "XGB-noPI":          float(anchor_df["cost_xgboost_no_pi"].sum()),
        "SVR-PI (current)":  float(anchor_df["cost_svr"].sum()),
        "SVR-noPI":          float(anchor_df["cost_svr_no_pi"].sum()),
        "HA":                float(anchor_df["cost_hist_avg"].sum()),
        "Pers":              float(anchor_df["cost_persistence"].sum()),
        "PF":                float(anchor_df["cost_perfect"].sum()),
    }
    print("Baseline annual realised cost ($):")
    for k, v in baseline.items():
        print(f"  {k:<22} ${v:>12,.2f}")

    rows = []
    log_path  = os.path.join(OUT_DIR, "sweep_log.txt")
    csv_path  = os.path.join(OUT_DIR, "sweep_results.csv")
    json_path = os.path.join(OUT_DIR, "best_config.json")
    log = open(log_path, "w")

    def _try(forecast_model, label, lookback, gamma, rho, cf_min, cf_max,
              stage):
        t0 = time.perf_counter()
        cost, days = _run_year(forecast_model, lookback, gamma, rho,
                                cf_min, cf_max, dates, soc_chain)
        dt = time.perf_counter() - t0
        baseline_key = "XGB-PI (current)" if forecast_model == "xgboost" \
                        else "SVR-PI (current)"
        delta = baseline[baseline_key] - cost
        rec = dict(
            stage=stage, forecaster=forecast_model, label=label,
            rho=rho, L=lookback, gamma=gamma, cf_min=cf_min, cf_max=cf_max,
            annual_cost=round(cost, 4), days=days,
            delta_vs_current=round(delta, 4),
            seconds=round(dt, 1),
        )
        rows.append(rec)
        sign = "+" if delta >= 0 else ""
        msg = (f"[{stage}] {forecast_model:<14} ρ={rho:>4.2f} L={lookback:>2d} "
               f"γ={gamma:>4.2f} clip=[{cf_min:.2f},{cf_max:.2f}] "
               f"→ ${cost:>12,.2f}  Δ={sign}${delta:>10,.2f}  ({dt:>5.1f}s)")
        print(msg)
        log.write(msg + "\n"); log.flush()
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return cost, delta

    # ── Stage 1 — sweep ρ (L=7, γ=0.5, clip=[0.5,2.0]) ────────────
    best = {"xgboost": None, "svr_nystroem": None}
    for fc in ("xgboost", "svr_nystroem"):
        print(f"\n=== Stage 1 ({fc}): sweep ρ ===")
        log.write(f"\n=== Stage 1 ({fc}): sweep ρ ===\n")
        scores = {}
        for rho in (0.50, 0.70, 0.85, 0.90):
            cost, _ = _try(fc, f"S1.rho={rho}", 7, 0.50, rho, 0.50, 2.00,
                            "stage1_rho")
            scores[rho] = cost
        best_rho = min(scores, key=scores.get)
        best[fc] = {"rho": best_rho, "L": 7, "gamma": 0.50,
                    "cf_min": 0.50, "cf_max": 2.00,
                    "annual_cost": scores[best_rho]}
        print(f"  best ρ for {fc}: {best_rho}  →  ${scores[best_rho]:,.2f}")

    # ── Stage 2 — sweep L (ρ=best, γ=0.5) ─────────────────────────
    for fc in ("xgboost", "svr_nystroem"):
        print(f"\n=== Stage 2 ({fc}): sweep L ===")
        log.write(f"\n=== Stage 2 ({fc}): sweep L ===\n")
        scores = {}
        rho_star = best[fc]["rho"]
        for L in (3, 7, 14, 21):
            cost, _ = _try(fc, f"S2.L={L}", L, 0.50, rho_star, 0.50, 2.00,
                            "stage2_L")
            scores[L] = cost
        best_L = min(scores, key=scores.get)
        best[fc]["L"] = best_L
        best[fc]["annual_cost"] = scores[best_L]
        print(f"  best L for {fc}: {best_L}  →  ${scores[best_L]:,.2f}")

    # ── Stage 3 — sweep γ (ρ=best, L=best) ────────────────────────
    for fc in ("xgboost", "svr_nystroem"):
        print(f"\n=== Stage 3 ({fc}): sweep γ ===")
        log.write(f"\n=== Stage 3 ({fc}): sweep γ ===\n")
        scores = {}
        rho_star = best[fc]["rho"]
        L_star   = best[fc]["L"]
        for gamma in (0.25, 0.50, 0.75):
            cost, _ = _try(fc, f"S3.γ={gamma}", L_star, gamma, rho_star,
                            0.50, 2.00, "stage3_gamma")
            scores[gamma] = cost
        best_gamma = min(scores, key=scores.get)
        best[fc]["gamma"]       = best_gamma
        best[fc]["annual_cost"] = scores[best_gamma]
        print(f"  best γ for {fc}: {best_gamma}  →  ${scores[best_gamma]:,.2f}")

    # ── Stage 4 — narrow clip bound test [0.7, 1.5] vs current ───
    for fc in ("xgboost", "svr_nystroem"):
        print(f"\n=== Stage 4 ({fc}): try tighter clip ===")
        log.write(f"\n=== Stage 4 ({fc}): try tighter clip ===\n")
        cost, _ = _try(fc, "S4.tight_clip", best[fc]["L"], best[fc]["gamma"],
                        best[fc]["rho"], 0.70, 1.50, "stage4_clip")
        if cost < best[fc]["annual_cost"]:
            best[fc]["cf_min"] = 0.70
            best[fc]["cf_max"] = 1.50
            best[fc]["annual_cost"] = cost
            print(f"  tighter clip is better for {fc}")
        else:
            print(f"  current [0.5,2.0] clip remains best for {fc}")

    # ── Summary ────────────────────────────────────────────────────
    summary = {
        "baseline": baseline,
        "best_xgb": {**best["xgboost"],
                     "delta_vs_current_xgb_pi":
                        round(baseline["XGB-PI (current)"] - best["xgboost"]["annual_cost"], 4)},
        "best_svr": {**best["svr_nystroem"],
                     "delta_vs_current_svr_pi":
                        round(baseline["SVR-PI (current)"] - best["svr_nystroem"]["annual_cost"], 4)},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.write("\nBest configurations:\n" + json.dumps(summary, indent=2) + "\n")
    log.close()
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}\n      {json_path}\n      {log_path}")


if __name__ == "__main__":
    main()
