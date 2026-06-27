"""
Post-process the 7-scenario annual MPC results (PF, XGB-ACLBC, XGB-noACLBC,
SVR-ACLBC, SVR-noACLBC, HA, Pers) from the aggregate-blended ACLBC comparison CSV.

Produces:
  results/scale_10_all_scenarios/all_scenarios_summary.txt
  results/scale_10_all_scenarios/annual_cost_bars.png
  results/scale_10_all_scenarios/cumulative_cost.png
  results/scale_10_all_scenarios/aclbc_gains_xgb_vs_svr.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV  = os.path.join(_HERE, "results", "all_dates_comparison_scale10_aggblend50.csv")
_OUT  = os.path.join(_HERE, "results", "scale_10_all_scenarios")
os.makedirs(_OUT, exist_ok=True)

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# Display ordering used everywhere (cheapest to most expensive expected)
SCENARIOS = [
    ("Perfect Foresight",  "cost_perfect",       "#424242"),
    ("XGBoost + bias correction (proposed)", "cost_xgboost", "#1565c0"),
    ("XGBoost, no correction",           "cost_xgboost_no_pi", "#ff8a65"),
    ("SVR + bias correction",            "cost_svr",           "#43a047"),
    ("SVR, no correction",               "cost_svr_no_pi",     "#aed581"),
    ("Historical Average", "cost_hist_avg",      "#8e24aa"),
    ("Persistence",        "cost_persistence",   "#6d4c41"),
]

# Two-line x-tick labels for the bar chart (aligned with SCENARIOS order)
BAR_TICKS = [
    "Perfect\nForesight",
    "XGBoost +\nbias correction\n(proposed)",
    "XGBoost,\nno correction",
    "SVR +\nbias correction",
    "SVR,\nno correction",
    "Historical\nAverage",
    "Persistence",
]

MAE_COLS = [
    ("XGBoost + bias correction", "mae_xgb",         "rmse_xgb"),
    ("XGBoost, no correction",    "mae_xgb_no_pi",   "rmse_xgb_no_pi"),
    ("SVR + bias correction",     "mae_svr",         "rmse_svr"),
    ("SVR, no correction",        "mae_svr_no_pi",   "rmse_svr_no_pi"),
    ("Historical Average", "mae_hist_avg",    "rmse_hist_avg"),
    ("Persistence",        "mae_persistence", "rmse_persistence"),
]


def _load():
    df = pd.read_csv(_CSV)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    return df.sort_values("date").reset_index(drop=True)


def _fmt(x, nd=2):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "---"
    return f"{x:,.{nd}f}"


def _annual(df):
    ann = {label: float(df[col].sum()) for label, col, _ in SCENARIOS}
    return ann


def _bar_chart(ann, ha_baseline, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = [n for n, _, _ in SCENARIOS]
    colors = [c for _, _, c in SCENARIOS]
    vals   = [ann[n] for n in names]
    bars   = ax.bar(names, vals, color=colors, edgecolor="#222", linewidth=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:,.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(ha_baseline, color="#8e24aa", linestyle=":", alpha=0.6,
               label=f"HA baseline (${ha_baseline:,.0f})")
    ax.set_ylabel("Annual realised cost (\\$)")
    ax.set_title("Annual realised settlement cost — 7-scenario MPC comparison")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(BAR_TICKS, rotation=0, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _cumulative_plot(df, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    for label, col, color in SCENARIOS:
        ls = "--" if label == "Perfect Foresight" else "-"
        ax.plot(df["date"], df[col].cumsum(), label=label, color=color,
                linewidth=1.0, linestyle=ls)
    ax.set_xlabel("Date (2013)")
    ax.set_ylabel("Cumulative realised cost (\\$)")
    ax.set_title("Cumulative annual cost — 7-scenario comparison")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pi_gains_plot(df, path):
    """Two-panel: daily ACLBC saving for XGB and for SVR, plus cumulative gains."""
    xgb_gain = df["cost_xgboost_no_pi"] - df["cost_xgboost"]
    svr_gain = df["cost_svr_no_pi"]     - df["cost_svr"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    axes[0].bar(df["date"], xgb_gain, width=1.0, color="#1565c0", alpha=0.6, label="XGBoost: correction saving")
    axes[0].bar(df["date"], svr_gain, width=1.0, color="#43a047", alpha=0.6, label="SVR: correction saving")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Daily correction saving (\\$/day)")
    axes[0].set_title("Per-day correction saving")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["date"], xgb_gain.cumsum(), label="XGBoost: cumulative correction saving", color="#1565c0", linewidth=1.3)
    axes[1].plot(df["date"], svr_gain.cumsum(), label="SVR: cumulative correction saving", color="#43a047", linewidth=1.3)
    axes[1].set_ylabel("Cumulative correction saving (\\$)")
    axes[1].set_xlabel("Date (2013)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _wilcoxon_signed(gain_series):
    try:
        stat, p = wilcoxon(gain_series.values, alternative="greater")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def _latex_annual(ann, df):
    """Rows for the combined 7-scenario annual cost table."""
    ha = ann["Historical Average"]
    pf = ann["Perfect Foresight"]

    mae_map = {label: float(df[mae_col].mean()) for label, mae_col, _ in MAE_COLS}
    mae_map["Perfect Foresight"] = 0.0
    rmse_map = {label: float(df[rmse_col].mean()) for label, _, rmse_col in MAE_COLS}
    rmse_map["Perfect Foresight"] = 0.0

    def gap_ha(c): return (ha - c) / abs(ha) * 100 if ha else float("nan")
    def gap_pf(c): return (c - pf) / abs(pf) * 100 if pf else float("nan")

    rows = []
    for label, _, _ in SCENARIOS:
        c = ann[label]
        metric_label = "XGBoost + bias correction" if label == "XGBoost + bias correction (proposed)" else label
        if label == "Perfect Foresight":
            disp = "Perfect Foresight (PF)"
        elif label == "XGBoost + bias correction (proposed)":
            disp = "\\textbf{XGBoost + bias correction (proposed)}"
        else:
            disp = label
        mae = mae_map.get(metric_label, 0.0)
        rmse = rmse_map.get(metric_label, 0.0)
        rows.append(
            f"{disp} & {_fmt(c)} & {_fmt(gap_ha(c),2)} & "
            f"{_fmt(gap_pf(c),2)} & {_fmt(mae,3)} & {_fmt(rmse,3)} \\\\"
        )
    return "\n".join(rows)


def _latex_pi_ablation_combined(df):
    xgb_gain = df["cost_xgboost_no_pi"] - df["cost_xgboost"]
    svr_gain = df["cost_svr_no_pi"]     - df["cost_svr"]
    w_xgb, p_xgb = _wilcoxon_signed(xgb_gain)
    w_svr, p_svr = _wilcoxon_signed(svr_gain)

    return "\n".join([
        f"Annual cost saving from ACLBC & "
        f"\\${xgb_gain.sum():,.2f} & \\${svr_gain.sum():,.2f} \\\\",
        f"Mean daily ACLBC saving & "
        f"\\${xgb_gain.mean():,.2f} & \\${svr_gain.mean():,.2f} \\\\",
        f"Median daily ACLBC saving & "
        f"\\${xgb_gain.median():,.2f} & \\${svr_gain.median():,.2f} \\\\",
        f"Days ACLBC strictly improves cost (out of {len(df)}) & "
        f"{int((xgb_gain > 0).sum())} & {int((svr_gain > 0).sum())} \\\\",
        f"Wilcoxon signed-rank statistic ($W$) & "
        f"{_fmt(w_xgb,0)} & {_fmt(w_svr,0)} \\\\",
        f"$p$-value (alt.\\ $\\Delta C^{{\\mathrm{{ACLBC}}}}>0$) & "
        f"{p_xgb:.3e} & {p_svr:.3e} \\\\",
    ])


def main():
    if not os.path.exists(_CSV):
        print(f"ERROR: {_CSV} not found. Run main.py --date all first.")
        sys.exit(1)
    df = _load()
    print(f"Loaded {len(df)} successful days.")

    ann = _annual(df)
    ha  = ann["Historical Average"]

    _bar_chart(ann, ha, os.path.join(_OUT, "annual_cost_bars.png"))
    _cumulative_plot(df, os.path.join(_OUT, "cumulative_cost.png"))
    _pi_gains_plot(df, os.path.join(_OUT, "aclbc_gains_xgb_vs_svr.png"))

    summary_path = os.path.join(_OUT, "all_scenarios_summary.txt")
    with open(summary_path, "w") as f:
        w = f.write
        w("7-scenario annual MPC summary\n")
        w(f"  successful days: {len(df)}\n\n")
        w("Annual realised cost ($):\n")
        for name, _, _ in SCENARIOS:
            w(f"  {name:<22} {ann[name]:>14,.2f}\n")

        xgb_gain = df["cost_xgboost_no_pi"] - df["cost_xgboost"]
        svr_gain = df["cost_svr_no_pi"]     - df["cost_svr"]
        w("\nACLBC saving:\n")
        w(f"  XGB annual: ${xgb_gain.sum():,.2f}  "
          f"(mean ${xgb_gain.mean():.2f}/day, {(xgb_gain > 0).sum()}/{len(df)} wins)\n")
        w(f"  SVR annual: ${svr_gain.sum():,.2f}  "
          f"(mean ${svr_gain.mean():.2f}/day, {(svr_gain > 0).sum()}/{len(df)} wins)\n")

        w_xgb, p_xgb = _wilcoxon_signed(xgb_gain)
        w_svr, p_svr = _wilcoxon_signed(svr_gain)
        w(f"\n  Wilcoxon XGB: W={w_xgb:.1f}, p={p_xgb:.3e}\n")
        w(f"  Wilcoxon SVR: W={w_svr:.1f}, p={p_svr:.3e}\n")

        w("\nForecast accuracy (mean across the year):\n")
        for label, mae_col, rmse_col in MAE_COLS:
            w(f"  {label:<22} MAE={df[mae_col].mean():>7.3f}  RMSE={df[rmse_col].mean():>7.3f}\n")

        w("\n" + "=" * 78 + "\n")
        w("LaTeX rows — annual combined table\n")
        w("=" * 78 + "\n\n")
        w(_latex_annual(ann, df) + "\n\n")
        w("LaTeX rows — combined ACLBC ablation table\n")
        w("-" * 78 + "\n")
        w(_latex_pi_ablation_combined(df) + "\n")

    print(f"\n→ {summary_path}")
    print(f"→ {_OUT}/annual_cost_bars.png")
    print(f"→ {_OUT}/cumulative_cost.png")
    print(f"→ {_OUT}/aclbc_gains_xgb_vs_svr.png\n")

    # console summary
    print(f"{'Scenario':<22} {'Annual cost':>14} {'Δ vs HA (%)':>12}")
    print("-" * 50)
    for name, _, _ in SCENARIOS:
        gap = (ha - ann[name]) / abs(ha) * 100
        print(f"{name:<22} ${ann[name]:>12,.2f}  {gap:>+10.2f}")


if __name__ == "__main__":
    main()
