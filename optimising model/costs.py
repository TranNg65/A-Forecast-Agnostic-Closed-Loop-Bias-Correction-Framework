"""
Computes tariff-based per-unit costs for scheme and non-scheme participants.

These figures compare membership pricing only; they do not allocate battery
dispatch outcomes back to individual apartments.
"""


def compute_unit_costs(data):
    """
    Compute per-unit tariff outcomes and savings from joining the scheme.
    """
    T       = data["T"]
    delta_t = data["delta_t"]
    S       = data["S"]
    N       = data["N"]
    D_u     = data["D_u"]
    C_g     = data["C_g"]
    C_scheme = data["C_scheme"]

    results = {}

    # 3.1 In-scheme unit cost:
    #   C_u^S = sum_t [ C_scheme(t) * D_u(t) * delta_t ]
    for u in S:
        cost = sum(
            C_scheme[t] * D_u[u, t] * delta_t
            for t in range(T)
        )
        results[u] = {"scheme_cost": cost, "in_scheme": True}

    # 3.2 Non-scheme unit cost:
    #   C_u^N = sum_t [ C_g(t) * D_u(t) * delta_t ]
    for u in N:
        cost = sum(
            C_g[t] * D_u[u, t] * delta_t
            for t in range(T)
        )
        results[u] = {"non_scheme_cost": cost, "in_scheme": False}

    # 3.2 also: compute what scheme units WOULD pay without the scheme
    # (for savings calculation)
    for u in S:
        hypothetical_cost = sum(
            C_g[t] * D_u[u, t] * delta_t
            for t in range(T)
        )
        results[u]["hypothetical_non_scheme_cost"] = hypothetical_cost

    # Also compute what non-scheme units WOULD pay under the scheme
    # (for comparison plotting)
    for u in N:
        hypothetical_scheme_cost = sum(
            C_scheme[t] * D_u[u, t] * delta_t
            for t in range(T)
        )
        results[u]["hypothetical_scheme_cost"] = hypothetical_scheme_cost

    # 4. Savings = C_u^N (hypothetical) - C_u^S
    for u in S:
        results[u]["savings"] = (
            results[u]["hypothetical_non_scheme_cost"]
            - results[u]["scheme_cost"]
        )

    return results
