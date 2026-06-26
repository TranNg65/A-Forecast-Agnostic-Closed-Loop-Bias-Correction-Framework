"""
Post-process the PI-vs-no-PI annual batch results.

Reads results/all_dates_comparison_scale10_aggblend50.csv (produced by
`python3 main.py --date all --ds 10`), produces:

  1. results/scale_10_pi_ablation/pi_ablation_summary.txt
       — annual totals, Wilcoxon test, monthly breakdown, and ready-to-paste
         LaTeX cell contents for the paper tables in Section 5.
  2. results/scale_10_pi_ablation/pi_daily.png
       — Figure: daily realised cost (XGB-PI vs XGB-noPI vs PF vs HA) and
         the daily PI gain on a secondary axis.
  3. results/scale_10_pi_ablation/pi_cumulative.png
       — cumulative cost savings of PI over no-PI through the year.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_HERE  = os.path.dirname(os.path.abspath(__file__))
_CSV   = os.path.join(_HERE, "results", "all_dates_comparison_scale10_aggblend50.csv")
_OUT   = os.path.join(_HERE, "results", "scale_10_pi_ablation")

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _load():
    df = pd.read_csv(_CSV)
    df = df[df["error"].isna() | (df["error"] == "")]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.month
    return df


def _annual_summary(df):
    """Annual totals for each scenario plus PI ablation statistics."""
    cols = ["cost_perfect", "cost_xgboost", "cost_xgboost_no_pi",
            "cost_hist_avg", "cost_persistence"]
    annual = {c: float(df[c].sum()) for c in cols}

    annual["pi_gain"]      = annual["cost_xgboost_no_pi"] - annual["cost_xgboost"]
    annual["pi_gain_pct"]  = annual["pi_gain"] / abs(annual["cost_xgboost_no_pi"]) * 100

    # paired daily differences
    delta = (df["cost_xgboost_no_pi"] - df["cost_xgboost"]).values
    annual["days_pi_wins"]   = int((delta > 0).sum())
    annual["days_total"]     = int(len(delta))
    annual["mean_delta"]     = float(delta.mean())
    annual["median_delta"]   = float(np.median(delta))

    try:
        stat, pval = wilcoxon(delta, alternative="greater")
        annual["wilcoxon_W"] = float(stat)
        annual["wilcoxon_p"] = float(pval)
    except ValueError:
        annual["wilcoxon_W"] = float("nan")
        annual["wilcoxon_p"] = float("nan")

    return annual


def _monthly_summary(df):
    grp = df.groupby("month").agg(
        PF       = ("cost_perfect",        "sum"),
        XGB_PI   = ("cost_xgboost",        "sum"),
        XGB_noPI = ("cost_xgboost_no_pi",  "sum"),
        HA       = ("cost_hist_avg",       "sum"),
        Pers     = ("cost_persistence",    "sum"),
        days     = ("date",                "count"),
    )
    grp["PI_gain"]      = grp["XGB_noPI"] - grp["XGB_PI"]
    grp["PI_gain_pct"]  = grp["PI_gain"] / grp["XGB_noPI"].abs() * 100
    grp.index = [_MONTH_NAMES[m-1] for m in grp.index]
    return grp


def _forecast_accuracy(df):
    out = pd.DataFrame({
        "Scenario": ["XGB-PI", "XGB-noPI", "Historical Average", "Persistence"],
        "MAE_kW":   [df["mae_xgb"].mean(), df["mae_xgb_no_pi"].mean(),
                     df["mae_hist_avg"].mean(), df["mae_persistence"].mean()],
        "RMSE_kW":  [df["rmse_xgb"].mean(), df["rmse_xgb_no_pi"].mean(),
                     df["rmse_hist_avg"].mean(), df["rmse_persistence"].mean()],
    })
    return out


def _operational_metrics(df):
    out = pd.DataFrame({
        "Scenario": ["XGB-PI", "XGB-noPI"],
        "SSR_pct":  [df["ssr_xgb"].mean(), df["ssr_xgb_no_pi"].mean()],
        "SCR_pct":  [df["scr_xgb"].mean(), df["scr_xgb_no_pi"].mean()],
        "Peak_shaving_pct": [df["peak_reduction_xgb"].mean(),
                             df["peak_reduction_xgb_no_pi"].mean()],
    })
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_daily(df, path):
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(df["date"], df["cost_xgboost"],       label="XGB-PI (proposed)",  color="#1565c0", linewidth=1.1)
    ax1.plot(df["date"], df["cost_xgboost_no_pi"], label="XGB-noPI (ablation)", color="#e53935", linewidth=1.0, alpha=0.85)
    ax1.plot(df["date"], df["cost_hist_avg"],      label="Historical Average",  color="#43a047", linewidth=0.8, alpha=0.6)
    ax1.plot(df["date"], df["cost_perfect"],       label="Perfect Foresight",   color="#424242", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.set_ylabel("Daily realised cost (\\$)")
    ax1.set_xlabel("Date (2013)")
    ax1.set_title("Daily realised settlement cost — PI feedback ablation")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    delta = df["cost_xgboost_no_pi"] - df["cost_xgboost"]
    ax2.bar(df["date"], delta, width=1.0, color="#ffb300", alpha=0.5,
            label="PI feedback gain (no-PI minus PI)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("PI gain $\\Delta C_\\tau^{PI}$ (\\$/day)")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cumulative(df, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    cum_pi   = df["cost_xgboost"].cumsum()
    cum_no   = df["cost_xgboost_no_pi"].cumsum()
    cum_ha   = df["cost_hist_avg"].cumsum()
    cum_pf   = df["cost_perfect"].cumsum()
    ax.plot(df["date"], cum_pi, label="XGB-PI (proposed)",  color="#1565c0")
    ax.plot(df["date"], cum_no, label="XGB-noPI (ablation)", color="#e53935")
    ax.plot(df["date"], cum_ha, label="Historical Average",  color="#43a047", alpha=0.7)
    ax.plot(df["date"], cum_pf, label="Perfect Foresight",   color="#424242", linestyle="--", alpha=0.7)
    ax.set_xlabel("Date (2013)")
    ax.set_ylabel("Cumulative realised cost (\\$)")
    ax.set_title("Cumulative annual cost — PI feedback ablation")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── LaTeX table builders ──────────────────────────────────────────────────────

def _fmt(x, nd=2):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "---"
    return f"{x:,.{nd}f}"


def _latex_annual(ann, fa):
    pf_cost  = ann["cost_perfect"]
    ha_cost  = ann["cost_hist_avg"]
    mae_map  = dict(zip(fa["Scenario"], fa["MAE_kW"]))

    def row(name, cost, mae=None, ssr=None, scr=None):
        gap_ha = (ha_cost - cost) / abs(ha_cost) * 100 if ha_cost else float("nan")
        gap_pf = (cost - pf_cost) / abs(pf_cost) * 100 if pf_cost else float("nan")
        return (f"{name} & {_fmt(cost)} & {_fmt(gap_ha,2)} & {_fmt(gap_pf,2)} "
                f"& {_fmt(ssr,2) if ssr is not None else '---'} "
                f"& {_fmt(scr,2) if scr is not None else '---'} "
                f"& {_fmt(mae if mae is not None else 0.0,3) if mae is not None else '0.000'} \\\\")
    rows = [
        row("Perfect Foresight (PF)",  ann["cost_perfect"],          mae=0.0),
        row("\\textbf{XGB-PI (proposed)}",  ann["cost_xgboost"],         mae=mae_map.get("XGB-PI")),
        row("XGB-noPI (ablation)",     ann["cost_xgboost_no_pi"],    mae=mae_map.get("XGB-noPI")),
        row("Historical Average (HA)", ann["cost_hist_avg"],         mae=mae_map.get("Historical Average")),
        row("Persistence (Pers)",      ann["cost_persistence"],      mae=mae_map.get("Persistence")),
    ]
    return "\n".join(rows)


def _latex_ablation(ann, df):
    delta = df["cost_xgboost_no_pi"] - df["cost_xgboost"]
    mae_red = df["mae_xgb_no_pi"].mean() - df["mae_xgb"].mean()
    return "\n".join([
        f"Annual cost saving from PI feedback, $\\sum_\\tau\\Delta C_\\tau^{{\\mathrm{{PI}}}}$ & {_fmt(ann['pi_gain'])} \\$/yr \\\\",
        f"Mean daily PI gain, $\\overline{{\\Delta C^{{\\mathrm{{PI}}}}}}$ & {_fmt(ann['mean_delta'])} \\$/day \\\\",
        f"Median daily PI gain & {_fmt(ann['median_delta'])} \\$/day \\\\",
        f"Days on which PI strictly improves cost (out of {ann['days_total']}) & {ann['days_pi_wins']} \\\\",
        f"Mean MAE reduction on scheme-aggregate demand & {_fmt(mae_red,3)} kW \\\\",
        f"Wilcoxon signed-rank statistic ($W$) & {_fmt(ann['wilcoxon_W'])} \\\\",
        f"$p$-value & {ann['wilcoxon_p']:.3e} \\\\",
    ])


def _latex_monthly(monthly):
    out = []
    for mon, r in monthly.iterrows():
        out.append(
            f"{mon} & {_fmt(r['PF'])} & {_fmt(r['XGB_PI'])} & {_fmt(r['XGB_noPI'])} "
            f"& {_fmt(r['HA'])} & {_fmt(r['Pers'])} & {_fmt(r['PI_gain'])} & {_fmt(r['PI_gain_pct'])} \\\\"
        )
    totals = monthly.sum()
    pi_gain_total     = totals["XGB_noPI"] - totals["XGB_PI"]
    pi_gain_pct_total = pi_gain_total / abs(totals["XGB_noPI"]) * 100
    out.append(
        f"\\midrule\n\\textbf{{Annual}} & {_fmt(totals['PF'])} & {_fmt(totals['XGB_PI'])} "
        f"& {_fmt(totals['XGB_noPI'])} & {_fmt(totals['HA'])} & {_fmt(totals['Pers'])} "
        f"& {_fmt(pi_gain_total)} & {_fmt(pi_gain_pct_total)} \\\\"
    )
    return "\n".join(out)


def _latex_forecast_accuracy(fa, df):
    # also compute sMAPE on scheme-aggregate (per-day from MAE proxy not trivial;
    # use a stable estimator based on actual demand magnitude)
    out = []
    for _, r in fa.iterrows():
        out.append(f"{r['Scenario']} & {_fmt(r['MAE_kW'],3)} & {_fmt(r['RMSE_kW'],3)} & --- \\\\")
    return "\n".join(out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(_CSV):
        print(f"ERROR: {_CSV} not found. Run `python3 main.py --date all` first.")
        sys.exit(1)
    os.makedirs(_OUT, exist_ok=True)

    df = _load()
    if len(df) == 0:
        print("No successful runs in CSV; nothing to summarise.")
        sys.exit(1)

    ann      = _annual_summary(df)
    monthly  = _monthly_summary(df)
    fa       = _forecast_accuracy(df)
    ops      = _operational_metrics(df)

    monthly.to_csv(os.path.join(_OUT, "monthly_summary.csv"))

    _plot_daily(df,      os.path.join(_OUT, "pi_daily.png"))
    _plot_cumulative(df, os.path.join(_OUT, "pi_cumulative.png"))

    summary_path = os.path.join(_OUT, "pi_ablation_summary.txt")
    with open(summary_path, "w") as f:
        w = f.write
        w("PI feedback ablation — annual summary\n")
        w(f"  successful days: {len(df)}\n")
        w(f"  annual cost PF:        ${ann['cost_perfect']:.2f}\n")
        w(f"  annual cost XGB-PI:    ${ann['cost_xgboost']:.2f}\n")
        w(f"  annual cost XGB-noPI:  ${ann['cost_xgboost_no_pi']:.2f}\n")
        w(f"  annual cost HA:        ${ann['cost_hist_avg']:.2f}\n")
        w(f"  annual cost Pers:      ${ann['cost_persistence']:.2f}\n\n")
        w("PI feedback gain:\n")
        w(f"  annual:     ${ann['pi_gain']:.2f} ({ann['pi_gain_pct']:.2f}%)\n")
        w(f"  mean/day:   ${ann['mean_delta']:.2f}\n")
        w(f"  median/day: ${ann['median_delta']:.2f}\n")
        w(f"  days PI wins: {ann['days_pi_wins']}/{ann['days_total']}\n")
        w(f"  Wilcoxon W = {ann['wilcoxon_W']:.2f}, p = {ann['wilcoxon_p']:.3e}\n\n")

        w("operational metrics (mean across the year):\n")
        w(ops.to_string(index=False))
        w("\n\nforecast accuracy:\n")
        w(fa.to_string(index=False))

        w("\n\n" + "=" * 76 + "\n")
        w("LaTeX table cells — paste into paper.tex Section sec:results\n")
        w("=" * 76 + "\n\n")

        w("% Table 1: annual cost across scenarios\n")
        w(_latex_annual(ann, fa) + "\n\n")
        w("% Table 2: PI ablation summary\n")
        w(_latex_ablation(ann, df) + "\n\n")
        w("% Table 3: monthly decomposition\n")
        w(_latex_monthly(monthly) + "\n\n")
        w("% Table 4: forecast accuracy\n")
        w(_latex_forecast_accuracy(fa, df) + "\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {_OUT}/pi_daily.png and pi_cumulative.png")
    print(f"\nAnnual headlines:")
    print(f"  XGB-PI     ${ann['cost_xgboost']:,.0f}  vs  XGB-noPI ${ann['cost_xgboost_no_pi']:,.0f}")
    print(f"  PI annual gain: ${ann['pi_gain']:,.0f} ({ann['pi_gain_pct']:.1f}%)")
    print(f"  Wilcoxon W={ann['wilcoxon_W']:.1f}, p={ann['wilcoxon_p']:.2e}")


if __name__ == "__main__":
    main()
