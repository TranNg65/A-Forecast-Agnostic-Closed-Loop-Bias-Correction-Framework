"""
Demand-size robustness: regenerate figures/demand_scale_sensitivity.png and the
LaTeX rows for Table tab:demand_scale from the per-scale full-year batch CSVs
(all_dates_comparison_scale{s}_aggblend50.csv) for s in {1,5,10,15,20}.

Panel (a): absolute annual cost recovered by the ACLBC loop (XGB-ACLBC gain, $/yr) vs s.
Panel (b): share of the historical-average-to-perfect-foresight gap closed by
           XGB-ACLBC (%) vs s.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "results")
_FIG_OUT = os.path.abspath(os.path.join(_HERE, "..", "paper", "extracted",
                                        "figures", "demand_scale_sensitivity.png"))

SCALES = [1, 5, 10, 15, 20]


def _load(scale):
    path = os.path.join(_RES, f"all_dates_comparison_scale{scale}_aggblend50.csv")
    df = pd.read_csv(path)
    df = df[df["error"].isna() | (df["error"] == "")]
    return df


def _row(scale):
    df = _load(scale)
    pf  = float(df["cost_perfect"].sum())
    pi  = float(df["cost_xgboost"].sum())
    npi = float(df["cost_xgboost_no_pi"].sum())
    ha  = float(df["cost_hist_avg"].sum())
    svr_gain = float((df["cost_svr_no_pi"] - df["cost_svr"]).sum())
    xgb_gain = npi - pi
    gap = ha - pf
    gap_closed = (ha - pi) / gap * 100 if gap else float("nan")
    return {
        "s": scale, "C_real_ACLBC": pi, "xgb_gain": xgb_gain,
        "gap_closed_pct": gap_closed, "svr_gain": svr_gain,
    }


def main():
    rows = [_row(s) for s in SCALES]
    res = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): absolute annual XGB-ACLBC gain vs scale
    ax = axes[0]
    ax.plot(res["s"], res["xgb_gain"], "o-", color="#1565c0", linewidth=1.8,
            markersize=7, label="XGBoost + bias correction gain")
    for _, r in res.iterrows():
        ax.annotate(f"${r['xgb_gain']:,.0f}", (r["s"], r["xgb_gain"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xlabel("Clustering factor $s$ ($\\approx 12s$ dwellings)")
    ax.set_ylabel("Annual cost recovered by the correction (\\$/yr)")
    ax.set_title("(a) Absolute benefit of the correction")
    ax.grid(alpha=0.3)
    ax.set_xticks(SCALES)

    # Panel (b): gap closed % vs scale
    ax = axes[1]
    ax.plot(res["s"], res["gap_closed_pct"], "s-", color="#43a047",
            linewidth=1.8, markersize=7)
    for _, r in res.iterrows():
        ax.annotate(f"{r['gap_closed_pct']:.1f}%", (r["s"], r["gap_closed_pct"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.axhspan(20, 27, color="#43a047", alpha=0.08)
    ax.set_xlabel("Clustering factor $s$ ($\\approx 12s$ dwellings)")
    ax.set_ylabel("Perfect-foresight gap closed by XGBoost + bias correction (\\%)")
    ax.set_title("(b) Fraction of foresight gap closed")
    ax.grid(alpha=0.3)
    ax.set_xticks(SCALES)
    ax.set_ylim(0, max(35, res["gap_closed_pct"].max() + 8))

    fig.tight_layout()
    fig.savefig(_FIG_OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {_FIG_OUT}\n")

    # LaTeX rows for tab:demand_scale
    print("LaTeX rows for tab:demand_scale (s & C_real_ACLBC & XGB gain & Gap% & SVR gain):")
    def money(x):
        neg = x < 0
        s = f"{abs(x):,.0f}".replace(",", "{,}")
        return ("$-$" if neg else "") + s
    for _, r in res.iterrows():
        print(f"{int(r['s'])}  & {money(r['C_real_ACLBC'])} & {money(r['xgb_gain'])} "
              f"& {r['gap_closed_pct']:.1f} & {money(r['svr_gain'])} \\\\")

    res.to_csv(os.path.join(_RES, "demand_scale_sensitivity_summary.csv"), index=False)


if __name__ == "__main__":
    main()
