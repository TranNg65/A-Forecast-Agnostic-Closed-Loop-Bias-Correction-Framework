"""
Generates all result plots: energy balance, battery SOC,
charge/discharge schedule, unit cost comparison, and PV utilization.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from costs import compute_unit_costs


def plot_results(v, data, plots_dir="plots"):
    """Generate comprehensive visualization plots."""

    os.makedirs(plots_dir, exist_ok=True)

    T = data["T"]
    delta_t = data["delta_t"]
    C_g = data["C_g"]
    P_pv = data["P_pv"]
    E_bat_max = data["E_bat_max"]
    ds = data["DEMAND_SCALE"]

    time_labels = [f"{h:02d}:{m:02d}"
                   for h in range(24) for m in (0, 30)]

    # ---- Plot 1: Energy Balance Overview ----
    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.stackplot(range(T),
                  v["P_pv_gen"], v["P_grid_S"], v["P_dis"],
                  labels=["PV Generation", "Grid Import", "Battery Discharge"],
                  alpha=0.7)
    ax1.plot(range(T), v["P_ch"], "k--", linewidth=2, label="Battery Charge")

    # Total scheme demand line
    total_demand = np.array([
        sum(data["D_u"][u, t] for u in data["S"]) + data["D_common"][t]
        for t in range(T)
    ])
    ax1.plot(range(T), total_demand, "r-", linewidth=2.5, label="Total Demand")

    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("STRATA Energy Balance")
    ax1.set_xticks(range(0, T, 4))
    ax1.set_xticklabels([time_labels[i] for i in range(0, T, 4)], rotation=45)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Tariff on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(T), C_g, "g:", linewidth=2, label="Grid Price ($/kWh)")
    ax2.set_ylabel("Price ($/kWh)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"energy_balance_{ds}.pdf"), dpi=150)
    plt.close()

    # ---- Plot 2: Battery State of Charge ----
    fig, ax = plt.subplots(figsize=(16, 6))
    soc_pct = v["P_soc"] / E_bat_max * 100
    ax.plot(range(T + 1), soc_pct, "b-", linewidth=2.5, label="SOC")
    ax.axhline(y=20, color="r", linestyle="--", alpha=0.5, label="SOC Min (20%)")
    ax.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="SOC Max (80%)")
    ax.axhline(y=50, color="g", linestyle=":", alpha=0.5, label="SOC Target (50%)")
    ax.fill_between(range(T + 1), 20, 80, alpha=0.1, color="green")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("State of Charge (%)")
    ax.set_title("Battery State of Charge Over 24 Hours")
    ax.set_xticks(range(0, T + 1, 4))
    ax.set_xticklabels([time_labels[i] if i < T else "24:00"
                        for i in range(0, T + 1, 4)], rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"battery_soc_{ds}.pdf"), dpi=150)
    plt.close()

    # ---- Plot 3: Battery Charge/Discharge Profile ----
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(T), v["P_ch"], alpha=0.7, label="Charging", color="blue")
    ax.bar(range(T), -v["P_dis"], alpha=0.7, label="Discharging", color="orange")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Battery Charge/Discharge Schedule")
    ax.set_xticks(range(0, T, 4))
    ax.set_xticklabels([time_labels[i] for i in range(0, T, 4)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"battery_operations_{ds}.pdf"), dpi=150)
    plt.close()

    # ---- Plot 4: Tariff Comparison (Scheme vs Non-Scheme) ----
    unit_costs = compute_unit_costs(data)
    unit_type = data["unit_type"]
    fig, ax = plt.subplots(figsize=(14, 6))

    units = sorted(unit_costs.keys())
    scheme_costs = []
    non_scheme_costs = []

    for u in units:
        info = unit_costs[u]
        if info["in_scheme"]:
            scheme_costs.append(info["scheme_cost"])
            non_scheme_costs.append(info["hypothetical_non_scheme_cost"])
        else:
            scheme_costs.append(info["hypothetical_scheme_cost"])
            non_scheme_costs.append(info["non_scheme_cost"])

    x = np.arange(len(units))
    width = 0.35

    bars1 = ax.bar(x - width / 2, non_scheme_costs, width,
                   label="Non-Scheme / Full Rate", color="coral", alpha=0.8)
    bars2 = ax.bar(x + width / 2, scheme_costs, width,
                   label="Scheme / Reduced Rate", color="steelblue", alpha=0.8)

    ax.set_xlabel("Apartment Unit")
    ax.set_ylabel("Daily Cost ($)")
    ax.set_title("Tariff Comparison: Scheme vs Full Rate (by Apartment Type)")
    ax.set_xticks(x)
    # Label with apartment type
    x_labels = []
    for u in units:
        apt = unit_type[u]
        x_labels.append(f"U{u}\n{apt}")
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"unit_cost_comparison_{ds}.pdf"), dpi=150)
    plt.close()

    # ---- Plot 5: PV Utilization ----
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(range(T), P_pv, alpha=0.3, color="gold", label="PV Available")
    ax.plot(range(T), v["P_pv_gen"], "orange", linewidth=2.5, label="PV Used")
    ax.fill_between(range(T), v["P_wasted_solar"],
                    alpha=0.3, color="red", label="PV Wasted")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Power (kW)")
    ax.set_title("Solar PV Utilization")
    ax.set_xticks(range(0, T, 4))
    ax.set_xticklabels([time_labels[i] for i in range(0, T, 4)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"pv_utilization_{ds}.pdf"), dpi=150)
    plt.close()

    print(f"\nPlots saved to '{plots_dir}/' directory.")
