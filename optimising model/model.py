"""
Builds and solves the Gurobi MILP model with all decision variables,
constraints, and objective function components.
"""

import gurobipy as gp
from gurobipy import GRB


def build_and_solve_model(data, silent=False):
    # Unpack data
    T           = data["T"]
    delta_t     = data["delta_t"]
    S           = data["S"]
    P_pv         = data["P_pv"]
    D_u_forecast = data["D_u_forecast"]  # Constraint A: what optimizer plans against
    R_res        = data["R_res"]         # Constraint C: per-interval SOC reserve [kWh]
    soc_init     = data["soc_init"]      # Constraint D1: initial SOC [kWh]
    D_common    = data["D_common"]
    C_g         = data["C_g"]
    C_excess    = data["C_excess"]
    C_rebate_p  = data["C_rebate"]
    P_export_threshold = data["P_export_threshold"]
    E_bat_max   = data["E_bat_max"]
    P_bat_max   = data["P_bat_max"]
    eta_c       = data["eta_c"]
    eta_d       = data["eta_d"]
    SOC_min     = data["SOC_min"]
    SOC_max     = data["SOC_max"]
    SOC_target  = data["SOC_target"]
    C_deg         = data["C_deg"]
    C_dis_inc     = data["C_dis_inc"]
    C_ch_inc      = data["C_ch_inc"]
    lambda_SOC    = data["lambda_SOC"]
    lambda_term   = data["lambda_term"]
    SOC_term_floor = data["SOC_term_floor"]

    # Create Gurobi model
    model = gp.Model("STRATA_Energy_Optimization")

    # DECISION VARIABLES

    # P_grid_S(t): Total STRATA grid import at time t [kW]
    P_grid_S = model.addVars(T, lb=0.0, name="P_grid_S")

    # P_pv_generated(t): PV power actually used at time t [kW]
    P_pv_gen = model.addVars(T, lb=0.0, name="P_pv_gen")

    # P_ch(t): Battery charging power [kW]
    P_ch = model.addVars(T, lb=0.0, ub=P_bat_max, name="P_ch")

    # P_dis(t): Battery discharging power [kW]
    P_dis = model.addVars(T, lb=0.0, ub=P_bat_max, name="P_dis")

    # SOC(t): Battery state of charge [kWh] — T+1 values (0 to T)
    P_soc = model.addVars(T + 1,
                          lb=SOC_min * E_bat_max,
                          ub=SOC_max * E_bat_max,
                          name="SOC")

    # Binary variables for no-simultaneous charge/discharge
    u_ch = model.addVars(T, vtype=GRB.BINARY, name="u_ch")
    u_dis = model.addVars(T, vtype=GRB.BINARY, name="u_dis")

    # P_export(t): Power exported to grid below threshold [kW]
    P_export = model.addVars(T, lb=0.0, name="P_export")

    # P_excess(t): Power exported to grid above threshold [kW]
    P_excess = model.addVars(T, lb=0.0, name="P_excess")

    # P_wasted_solar(t): Wasted solar power [kW]
    P_wasted_solar = model.addVars(T, lb=0.0, name="P_wasted_solar")

    # P_wasted_grid(t): Grid energy not used [kW]
    P_wasted_grid = model.addVars(T, lb=0.0, name="P_wasted_grid")

    # SOC deviation variables (for linearized absolute-value penalty)
    SOC_dev_pos = model.addVars(T, lb=0.0, name="SOC_dev_pos")
    SOC_dev_neg = model.addVars(T, lb=0.0, name="SOC_dev_neg")

    # Terminal SOC deviation variables (linearize |P_soc[T] - SOC_target|)
    SOC_term_dev_pos = model.addVar(lb=0.0, name="SOC_term_dev_pos")
    SOC_term_dev_neg = model.addVar(lb=0.0, name="SOC_term_dev_neg")

    model.update()


    # CONSTRAINTS

    # Constraint A — Prediction-Based Energy Balance
    # Uses D_u_forecast (XGBoost predictions when use_forecast=True, else actual).
    # Lets the optimizer pre-charge/-discharge in anticipation of forecast load.
    for t in range(T):
        total_scheme_demand = sum(D_u_forecast[u, t] for u in S)
        model.addConstr(
            P_pv_gen[t] + P_grid_S[t] + P_dis[t]
            ==
            total_scheme_demand + D_common[t] + P_ch[t]
            + P_export[t] + P_excess[t] + P_wasted_grid[t],
            name=f"energy_balance_{t}"
        )

    # SOC dynamics: SOC(t+1) = SOC(t) + (eta_c * P_ch(t) - P_dis(t)/eta_d) * delta_t
    for t in range(T):
        model.addConstr(
            P_soc[t + 1] == P_soc[t]
            + (eta_c * P_ch[t] - P_dis[t] / eta_d) * delta_t,
            name=f"soc_dynamics_{t}"
        )

    # Constraint D1 — Rolling SOC Carry-over
    # soc_init = realised end-of-previous-day SOC, or SOC_target on the first day.
    model.addConstr(P_soc[0] == soc_init, name="soc_initial")

    # Constraint D2 — Horizon Terminal Condition (soft)
    # Hard floor prevents draining below SOC_term_floor; soft penalty on the
    # absolute deviation from SOC_target discourages large end-of-day swings
    # without locking the optimizer to exactly 50% every day.
    model.addConstr(P_soc[T] >= SOC_term_floor * E_bat_max, name="soc_terminal_floor")
    model.addConstr(
        SOC_term_dev_pos - SOC_term_dev_neg == P_soc[T] - SOC_target * E_bat_max,
        name="soc_terminal_dev"
    )

    # Constraint C — Uncertainty-Aware SOC Reserve
    # Tightens the effective SOC floor by R_res(t) at each interior step,
    # so the battery always holds enough reserve to absorb demand above forecast.
    # Applied for t=1..T-1 (t=0 fixed by D1; t=T governed by soft terminal constraint).
    for t in range(1, T):
        model.addConstr(
            P_soc[t] >= SOC_min * E_bat_max + R_res[t],
            name=f"soc_reserve_{t}"
        )

    # Power limits
    for t in range(T):
        model.addConstr(P_ch[t] <= P_bat_max * u_ch[t],
                        name=f"charge_binary_{t}")
        model.addConstr(P_dis[t] <= P_bat_max * u_dis[t],
                        name=f"discharge_binary_{t}")
        # No simultaneous charging and discharging
        model.addConstr(u_ch[t] + u_dis[t] <= 1,
                        name=f"no_simul_ch_dis_{t}")

    # P_pv_generated(t) <= P_pv(t)
    for t in range(T):
        model.addConstr(P_pv_gen[t] <= P_pv[t],
                        name=f"pv_capacity_{t}")

    # Rebate-eligible export is capped; any additional export must flow via P_excess.
    for t in range(T):
        model.addConstr(P_export[t] <= P_export_threshold,
                        name=f"export_rebate_cap_{t}")

    # Wasted solar: what PV could produce minus what's actually used
    for t in range(T):
        model.addConstr(P_wasted_solar[t] == P_pv[t] - P_pv_gen[t],
                        name=f"wasted_solar_{t}")

    # ---- SOC deviation constraints (linearization of |SOC - target|) ----
    for t in range(T):
        model.addConstr(
            SOC_dev_pos[t] - SOC_dev_neg[t] == P_soc[t] - SOC_target * E_bat_max,
            name=f"soc_dev_{t}"
        )

    # OBJECTIVE FUNCTION
    # minimize C_STRATA = C_grid + C_battery + C_excess_cost + C_penalty
    #                     - C_discharge - C_charge - C_rebate

    # Component 1: Grid Electricity Cost
    #   C_grid = sum_t [ C_g(t) * P_grid_S(t) * delta_t ]
    C_grid = gp.quicksum(
        C_g[t] * P_grid_S[t] * delta_t for t in range(T)
    )

    # Component 2: Battery Charging Cost (degradation)
    #   C_battery = sum_t [ 0.05 * P_ch(t) * delta_t ]
    C_battery = gp.quicksum(
        C_deg * P_ch[t] * delta_t for t in range(T)
    )

    # Component 3: Battery Discharge Incentive (subtracted)
    #   C_discharge = sum_t [ 0.50 * P_dis(t) * delta_t ]
    C_discharge = gp.quicksum(
        C_dis_inc * P_dis[t] * delta_t for t in range(T)
    )

    # Component 4: Battery Charging Incentive (subtracted)
    #   C_charge = sum_t [ 0.10 * P_ch(t) * delta_t ]
    C_charge = gp.quicksum(
        C_ch_inc * P_ch[t] * delta_t for t in range(T)
    )

    # Component 5: Intraday SOC Deviation Penalty
    #   C_penalty = 0.02 * sum_t [ SOC_dev_pos(t) + SOC_dev_neg(t) ]
    C_penalty = lambda_SOC * gp.quicksum(
        SOC_dev_pos[t] + SOC_dev_neg[t] for t in range(T)
    )

    # Component 5b: Terminal SOC Deviation Penalty
    #   C_term_penalty = lambda_term * |P_soc[T] - SOC_target|
    C_term_penalty = lambda_term * (SOC_term_dev_pos + SOC_term_dev_neg)

    # Component 6: Cost for excess export above threshold
    #   C_excess_cost = sum_t [ C_excess(t) * P_excess(t) * delta_t ]
    C_excess_cost = gp.quicksum(
        C_excess[t] * P_excess[t] * delta_t for t in range(T)
    )

    # Component 7: Rebate for export below threshold (subtracted)
    #   C_rebate = sum_t [ C_rebate(t) * P_export(t) * delta_t ]
    C_rebate_total = gp.quicksum(
        C_rebate_p[t] * P_export[t] * delta_t for t in range(T)
    )

    # Full objective
    objective = (C_grid + C_battery + C_excess_cost + C_penalty + C_term_penalty
                 - C_discharge - C_charge - C_rebate_total)

    model.setObjective(objective, GRB.MINIMIZE)

    # ------------------------------------------------------------------
    # SOLVER SETTINGS
    # ------------------------------------------------------------------
    if silent:
        model.Params.OutputFlag = 0  # must be set before other params to suppress all output
    model.Params.MIPGap = 0.01
    model.Params.Threads = 4
    model.Params.TimeLimit = 300

    # Solve
    model.optimize()

    return model, {
        "P_grid_S": P_grid_S, "P_pv_gen": P_pv_gen,
        "P_ch": P_ch, "P_dis": P_dis, "P_soc": P_soc,
        "u_ch": u_ch, "u_dis": u_dis,
        "P_export": P_export, "P_excess": P_excess,
        "P_wasted_solar": P_wasted_solar, "P_wasted_grid": P_wasted_grid,
        "SOC_dev_pos": SOC_dev_pos, "SOC_dev_neg": SOC_dev_neg,
        "SOC_term_dev_pos": SOC_term_dev_pos, "SOC_term_dev_neg": SOC_term_dev_neg,
    }
