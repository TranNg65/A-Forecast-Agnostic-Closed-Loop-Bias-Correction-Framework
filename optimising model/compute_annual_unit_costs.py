"""Aggregate per-unit annual costs for the seven scheme participants and the
five non-scheme units.  Per-unit costs depend only on actual demand and the
fixed tariffs, so they are scenario-independent across PF/XGB-PI/XGB-noPI/HA/Pers.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_input_data, get_available_dates
from costs import compute_unit_costs

DEMAND_SCALE = 10
UNIT_TYPE_LABEL = {"1bed": 1, "2bed": 2, "3bed": 3}


def main():
    dates = get_available_dates()

    annual = {}  # unit -> dict
    n_done = 0
    for date in dates:
        try:
            data = build_input_data(DEMAND_SCALE, date,
                                    use_forecast=False, gamma=0.0,
                                    soc_carry=None, lookback_days=0)
        except Exception as e:
            print(f"  skip {date}: {e}")
            continue
        unit_costs = compute_unit_costs(data)
        for u, info in unit_costs.items():
            row = annual.setdefault(u, {
                "unit": u,
                "type": data["unit_type"][u],
                "in_scheme": info["in_scheme"],
                "scheme_cost": 0.0,
                "non_scheme_cost": 0.0,
                "hypothetical_non_scheme_cost": 0.0,
                "hypothetical_scheme_cost": 0.0,
                "savings": 0.0,
            })
            if info["in_scheme"]:
                row["scheme_cost"]                  += info["scheme_cost"]
                row["hypothetical_non_scheme_cost"] += info["hypothetical_non_scheme_cost"]
                row["savings"]                      += info["savings"]
            else:
                row["non_scheme_cost"]              += info["non_scheme_cost"]
                row["hypothetical_scheme_cost"]     += info["hypothetical_scheme_cost"]
        n_done += 1

    print(f"\nDays aggregated: {n_done}")

    df = pd.DataFrame(annual.values()).sort_values("unit")
    out_path = os.path.join("results", "scale_10_pi_ablation", "annual_unit_costs.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  → {out_path}\n")

    print("Annual per-unit costs (scheme participants):")
    print(f"  {'Unit':<5} {'Type':<6} {'C_u^S ($)':>12} {'C_u^N hyp. ($)':>16} {'ΔC ($)':>12}")
    for _, r in df[df["in_scheme"]].iterrows():
        print(f"  {int(r['unit']):<5} {r['type']:<6} {r['scheme_cost']:>12,.2f} "
              f"{r['hypothetical_non_scheme_cost']:>16,.2f} {r['savings']:>12,.2f}")

    print("\nAnnual per-unit costs (non-scheme units):")
    print(f"  {'Unit':<5} {'Type':<6} {'C_u^N ($)':>12}")
    for _, r in df[~df["in_scheme"]].iterrows():
        print(f"  {int(r['unit']):<5} {r['type']:<6} {r['non_scheme_cost']:>12,.2f}")

    print("\nLaTeX rows for paper.tex (in_scheme units):")
    bedroom_map = {"1bed": "1", "2bed": "2", "3bed": "3"}
    for _, r in df[df["in_scheme"]].iterrows():
        print(f"{int(r['unit'])} & {bedroom_map[r['type']]} & "
              f"{r['scheme_cost']:,.2f} & {r['hypothetical_non_scheme_cost']:,.2f} & "
              f"{r['savings']:,.2f} \\\\")


if __name__ == "__main__":
    main()
