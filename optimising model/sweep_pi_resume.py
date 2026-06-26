"""Resume the PI sweep at Stage 3 (γ) and Stage 4 (clip).
Re-uses the existing baseline and SOC chain anchor from the original
sweep, so the resumed results are directly comparable.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sweep_pi import (
    DEMAND_SCALE, OUT_DIR, ANCHOR_CSV,
    _solve, _run_year, _load_anchor,
)

# Best (rho, L) per forecaster from stages 1 + 2 of the prior run
BEST = {
    "xgboost":      {"rho": 0.70, "L": 7,  "gamma": 0.50,
                     "cf_min": 0.50, "cf_max": 2.00,
                     "annual_cost": 173_084.84},
    "svr_nystroem": {"rho": 0.90, "L": 7,  "gamma": 0.50,
                     "cf_min": 0.50, "cf_max": 2.00,
                     "annual_cost": 180_325.49},
}


def main():
    anchor_df, soc_chain = _load_anchor()
    dates = list(anchor_df["date"])
    print(f"Resuming sweep over {len(dates)} days")

    baseline = {
        "XGB-PI (current)":  float(anchor_df["cost_xgboost"].sum()),
        "SVR-PI (current)":  float(anchor_df["cost_svr"].sum()),
    }
    print("Baselines: " + str({k: f"${v:,.2f}" for k, v in baseline.items()}))

    # Append to existing CSV to keep all stages together
    csv_path  = os.path.join(OUT_DIR, "sweep_results.csv")
    existing  = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    rows      = existing.to_dict("records")
    log_path  = os.path.join(OUT_DIR, "sweep_log.txt")
    json_path = os.path.join(OUT_DIR, "best_config.json")
    log = open(log_path, "a")

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
        return cost

    # ── Stage 3 — sweep γ ────────────────────────────────────────
    for fc in ("xgboost", "svr_nystroem"):
        print(f"\n=== Stage 3 ({fc}): sweep γ ===")
        log.write(f"\n=== Stage 3 ({fc}): sweep γ ===\n")
        scores = {}
        rho_star = BEST[fc]["rho"]
        L_star   = BEST[fc]["L"]
        for gamma in (0.25, 0.50, 0.75):
            cost = _try(fc, f"S3.γ={gamma}", L_star, gamma, rho_star,
                         0.50, 2.00, "stage3_gamma")
            scores[gamma] = cost
        best_gamma = min(scores, key=scores.get)
        BEST[fc]["gamma"]       = best_gamma
        BEST[fc]["annual_cost"] = scores[best_gamma]
        print(f"  best γ for {fc}: {best_gamma}  →  ${scores[best_gamma]:,.2f}")

    # ── Stage 4 — narrow clip ────────────────────────────────────
    for fc in ("xgboost", "svr_nystroem"):
        print(f"\n=== Stage 4 ({fc}): try tighter clip ===")
        log.write(f"\n=== Stage 4 ({fc}): try tighter clip ===\n")
        cost = _try(fc, "S4.tight_clip", BEST[fc]["L"], BEST[fc]["gamma"],
                     BEST[fc]["rho"], 0.70, 1.50, "stage4_clip")
        if cost < BEST[fc]["annual_cost"]:
            BEST[fc]["cf_min"] = 0.70
            BEST[fc]["cf_max"] = 1.50
            BEST[fc]["annual_cost"] = cost
            print(f"  tighter clip is better for {fc}")
        else:
            print(f"  current [0.5,2.0] clip remains best for {fc}")

    summary = {
        "baseline": baseline,
        "best_xgb": {**BEST["xgboost"],
                     "delta_vs_current_xgb_pi":
                        round(baseline["XGB-PI (current)"] - BEST["xgboost"]["annual_cost"], 4)},
        "best_svr": {**BEST["svr_nystroem"],
                     "delta_vs_current_svr_pi":
                        round(baseline["SVR-PI (current)"] - BEST["svr_nystroem"]["annual_cost"], 4)},
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.write("\nBest configurations:\n" + json.dumps(summary, indent=2) + "\n")
    log.close()
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
