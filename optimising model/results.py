"""
Extracts optimized variable values from the solved Gurobi model
and prints summary statistics.
"""

import numpy as np
from gurobipy import GRB

from costs import compute_unit_costs


def _split_export_surplus(surplus_kw, threshold_kw):
    """Split surplus export into rebate-eligible and excess components."""
    if np.isinf(threshold_kw):
        return surplus_kw, 0.0
    rebate_eligible = min(surplus_kw, threshold_kw)
    excess = max(surplus_kw - threshold_kw, 0.0)
    return rebate_eligible, excess


def compute_metrics(v, data, P_grid_real):
    """
    System performance metrics for a realized battery schedule.

    SSR = self-sufficiency ratio  (% of actual demand met without grid import)
    SCR = self-consumption ratio  (% of available PV consumed on-site)
    Peak reduction compared to the no-BESS case (actual demand vs PV only).
    """
    T        = data["T"]
    delta_t  = data["delta_t"]
    S        = data["S"]
    D_u      = data["D_u"]
    D_common = data["D_common"]
    P_pv     = data["P_pv"]

    demand_per_t = np.array([
        sum(D_u[u, t] for u in S) + D_common[t]
        for t in range(T)
    ])

    total_demand_kwh = float(np.sum(demand_per_t) * delta_t)
    grid_kwh         = float(np.sum(P_grid_real)  * delta_t)
    ssr = (1.0 - grid_kwh / total_demand_kwh) * 100.0 if total_demand_kwh > 0 else 0.0

    pv_total_kwh = float(np.sum(P_pv)           * delta_t)
    pv_used_kwh  = float(np.sum(v["P_pv_gen"])  * delta_t)
    scr = (pv_used_kwh / pv_total_kwh) * 100.0 if pv_total_kwh > 0 else 0.0

    no_bess_per_t  = np.maximum(demand_per_t - P_pv, 0.0)
    peak_no_bess   = float(no_bess_per_t.max())
    peak_with_bess = float(np.max(P_grid_real))
    peak_reduction = (
        (peak_no_bess - peak_with_bess) / peak_no_bess * 100.0
        if peak_no_bess > 0 else 0.0
    )

    return {
        "ssr":                round(ssr, 2),
        "scr":                round(scr, 2),
        "peak_reduction_pct": round(peak_reduction, 2),
    }


def compute_realised_cost(v, data):
    """
    Equation E — post-hoc realised cost.

    The battery schedule (P_ch*, P_dis*, SOC*) is fixed from the optimizer.
    Only grid import/export adjusts to absorb forecast error:

      P_grid_real(t) = max(0, Σ_{u∈S} D_u_actual(t) + D_common(t)
                              + P_ch*(t) − P_dis*(t) − P_pv(t))

    The returned realised cost keeps the same controller-objective terms as
    the optimizer, but evaluates grid import/export against actual demand.
    """
    T          = data["T"]
    delta_t    = data["delta_t"]
    S          = data["S"]
    D_u        = data["D_u"]       # actual demand (always the real CSV values)
    D_common   = data["D_common"]
    P_pv       = data["P_pv"]
    C_g        = data["C_g"]
    C_rebate_p = data["C_rebate"]
    C_excess_p = data["C_excess"]
    C_deg       = data["C_deg"]
    C_dis_inc   = data["C_dis_inc"]
    C_ch_inc    = data["C_ch_inc"]
    lambda_SOC  = data["lambda_SOC"]
    lambda_term = data["lambda_term"]
    SOC_target  = data["SOC_target"]
    E_bat_max   = data["E_bat_max"]
    P_export_threshold = data["P_export_threshold"]

    # ── Grid-side terms (change with actual demand) ───────────────────────────
    P_grid_real   = np.zeros(T)
    P_export_real = np.zeros(T)
    P_excess_real = np.zeros(T)

    for t in range(T):
        actual_demand = sum(D_u[u, t] for u in S) + D_common[t]
        net = actual_demand + v["P_ch"][t] - v["P_dis"][t] - P_pv[t]
        if net > 0:
            P_grid_real[t] = net
        else:
            surplus = -net
            P_export_real[t], P_excess_real[t] = _split_export_surplus(
                surplus, P_export_threshold
            )

    C_grid_real   = sum(C_g[t]      * P_grid_real[t]   * delta_t for t in range(T))
    C_excess_real = sum(C_excess_p[t] * P_excess_real[t] * delta_t for t in range(T))
    C_rebate_real = sum(C_rebate_p[t] * P_export_real[t] * delta_t for t in range(T))

    # ── Battery / controller terms fixed by the optimized schedule ───────────────
    C_battery  = sum(C_deg     * v["P_ch"][t]  * delta_t for t in range(T))
    C_discharge = sum(C_dis_inc * v["P_dis"][t] * delta_t for t in range(T))
    C_charge    = sum(C_ch_inc  * v["P_ch"][t]  * delta_t for t in range(T))
    C_penalty = lambda_SOC * sum(
        abs(v["P_soc"][t] - SOC_target * E_bat_max) for t in range(T)
    )
    C_term_penalty = lambda_term * abs(v["P_soc"][-1] - SOC_target * E_bat_max)

    realised = (
        C_grid_real + C_battery + C_excess_real + C_penalty + C_term_penalty
        - C_discharge - C_charge - C_rebate_real
    )

    return P_grid_real, P_export_real, realised


def extract_and_display_results(model, variables, data):
    """Extract optimized values and display summary results."""

    T = data["T"]
    delta_t = data["delta_t"]
    E_bat_max = data["E_bat_max"]

    if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
        print("=" * 60)
        print(f"  Optimization Status: {model.Status}")
        print(f"  Optimal STRATA Cost: ${model.ObjVal:.2f}")
        print("=" * 60)

        # Extract variable values into numpy arrays (handle scalar and indexed vars)
        v = {}
        for name, var in variables.items():
            if hasattr(var, 'X'):
                v[name] = float(var.X)
            else:
                v[name] = np.array([var[t].X for t in sorted(var.keys())])

        # Print summary statistics
        print(f"\n--- Grid Import ---")
        print(f"  Total grid energy imported: {sum(v['P_grid_S'] * delta_t):.2f} kWh")
        print(f"  Peak grid import:           {max(v['P_grid_S']):.2f} kW")

        print(f"\n--- PV Generation ---")
        print(f"  Total PV energy used:       {sum(v['P_pv_gen'] * delta_t):.2f} kWh")
        print(f"  Total PV wasted:            {sum(v['P_wasted_solar'] * delta_t):.2f} kWh")

        print(f"\n--- Battery Operations ---")
        print(f"  Total energy charged:       {sum(v['P_ch'] * delta_t):.2f} kWh")
        print(f"  Total energy discharged:    {sum(v['P_dis'] * delta_t):.2f} kWh")
        print(f"  SOC range: [{min(v['P_soc']):.1f}, {max(v['P_soc']):.1f}] kWh "
              f"([{min(v['P_soc'])/E_bat_max*100:.1f}%, {max(v['P_soc'])/E_bat_max*100:.1f}%])")

        print(f"\n--- Grid Export ---")
        print(f"  Export (below threshold):   {sum(v['P_export'] * delta_t):.2f} kWh")
        print(f"  Excess (above threshold):   {sum(v['P_excess'] * delta_t):.2f} kWh")

        # Tariff comparison — this is independent of battery dispatch.
        print(f"\n--- Unit Tariff Comparison ---")
        unit_costs = compute_unit_costs(data)
        unit_type = data["unit_type"]
        for u in sorted(unit_costs.keys()):
            info = unit_costs[u]
            apt = unit_type[u]
            if info["in_scheme"]:
                print(f"  Unit {u:2d} ({apt}, Scheme):     "
                      f"Tariff=${info['scheme_cost']:8.2f}  |  "
                      f"Without scheme=${info['hypothetical_non_scheme_cost']:8.2f}  |  "
                      f"Savings=${info['savings']:8.2f}")
            else:
                print(f"  Unit {u:2d} ({apt}, Non-scheme): "
                      f"Tariff=${info['non_scheme_cost']:8.2f}")

        # Equation E — show realised cost only when forecast demand was used
        if data.get("use_forecast"):
            _, _, realised = compute_realised_cost(v, data)
            diagnostic_gap = realised - model.ObjVal
            print(f"\n--- Forecast vs Realised ---")
            print(f"  Optimised controller objective:   ${model.ObjVal:.2f}")
            print(f"  Realised settlement cost:         ${realised:.2f}")
            print(f"  Settlement - objective diagnostic:${diagnostic_gap:+.2f}")

        return v, unit_costs, model.ObjVal

    else:
        print(f"Optimization failed with status: {model.Status}")
        if model.Status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("infeasible.ilp")
            print("IIS written to infeasible.ilp")
        return None, None, None
