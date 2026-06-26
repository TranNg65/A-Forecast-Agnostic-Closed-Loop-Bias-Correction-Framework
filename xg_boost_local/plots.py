import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error

from .config import OUTPUT_DIR, FEATURE_COLS


def plot_actual_vs_predicted(val_timestamps, y_val, y_pred_val, m_val):
    val_df_plot = pd.DataFrame({
        'date':   pd.to_datetime(val_timestamps).normalize(),
        'actual': y_val,
        'pred':   y_pred_val,
    })
    daily     = val_df_plot.groupby('date')[['actual', 'pred']].mean()
    daily_mae = val_df_plot.groupby('date').apply(
        lambda g: mean_absolute_error(g['actual'], g['pred'])
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [2, 1]}
    )
    fig.suptitle('Unit Demand — Actual vs Predicted (Validation Set)',
                 fontsize=14, fontweight='bold')
    dates = daily.index.to_numpy()   # pandas DatetimeIndex → numpy (fixes matplotlib compat)
    ax_top.plot(dates, daily['actual'].to_numpy(), color='#222222', lw=1.2, label='Actual (daily mean)')
    ax_top.plot(dates, daily['pred'].to_numpy(),   color='#d62728', lw=1.2, alpha=0.9,
                label='Predicted (daily mean)')
    ax_top.fill_between(dates, daily['actual'].to_numpy(), daily['pred'].to_numpy(),
                        alpha=0.15, color='#d62728', label='Error band')
    ax_top.set_ylabel('Mean Unit Demand (kWh / 30-min)', fontsize=10)
    ax_top.set_title(f"MAE={m_val['mae']:.4f}  RMSE={m_val['rmse']:.4f}  "
                     f"sMAPE={m_val['smape']:.2f}%  CV(RMSE)={m_val['cv_rmse']:.2f}%",
                     fontsize=10)
    ax_top.legend(fontsize=9)
    ax_top.grid(alpha=0.3)
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_top.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    ax_bot.bar(daily_mae.index.to_numpy(), daily_mae.values, color='#d62728', alpha=0.7, width=1)
    ax_bot.axhline(daily_mae.mean(), color='black', lw=1, ls='--',
                   label=f'Mean MAE = {daily_mae.mean():.4f}')
    ax_bot.set_ylabel('Daily MAE (kWh)', fontsize=9)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(alpha=0.3)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax_bot.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_actual_vs_predicted.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_actual_vs_predicted.png")


def plot_customer_traces(val_timestamps, val_customer_ids, y_val, y_pred_val):
    sample_customers = pd.Series(val_customer_ids).unique()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    fig.suptitle('Unit Demand — Sample Customer Traces (Validation, 1 week)',
                 fontsize=13, fontweight='bold')
    for i, cid in enumerate(sample_customers):
        ax      = axes[i // 3][i % 3]
        mask_c  = val_customer_ids == cid
        n       = 48 * 7
        sub_yw  = y_val[mask_c][:n]
        sub_yp  = y_pred_val[mask_c][:n]
        dates_c = val_timestamps[mask_c][:n]
        ax.fill_between(dates_c, sub_yw, sub_yp, alpha=0.15, color='#d62728', label='Error')
        ax.plot(dates_c, sub_yw, color='#222222', lw=1.0, alpha=0.85, label='Actual')
        ax.plot(dates_c, sub_yp, color='#d62728', lw=1.0, alpha=0.9,  label='Predicted')
        mae_c   = mean_absolute_error(sub_yw, sub_yp)
        smape_c = np.mean(np.abs(sub_yw - sub_yp) /
                          ((np.abs(sub_yw) + np.abs(sub_yp)) / 2 + 1e-6)) * 100
        r2_c    = 1 - np.sum((sub_yw - sub_yp) ** 2) / (
                      np.sum((sub_yw - sub_yw.mean()) ** 2) + 1e-9)
        ax.set_title(f"Customer {cid}  |  MAE={mae_c:.3f}  sMAPE={smape_c:.1f}%  R²={r2_c:.3f}",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('kWh / 30-min', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_customer_traces.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_customer_traces.png")


def plot_feature_importance(model, feature_cols_final):
    feat_category_map = {}
    for feat in feature_cols_final:
        if feat in {'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
                    'tod', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
                    'is_summer', 'is_winter', 'is_public_holiday',
                    'is_day_before_holiday', 'is_day_after_holiday', 'is_school_holiday'}:
            feat_category_map[feat] = 'time'
        elif feat.startswith('unit_lag') or feat.startswith('unit_rolling') or feat == 'customer_how_mean':
            feat_category_map[feat] = 'lag'
        elif feat in {'max_temp_c', 'min_temp_c', 'solar_exposure_mj_m2',
                      'rainfall_mm', 'heat_stress', 'cold_stress', 'temp_range', 'temp_x_tod',
                      'max_temp_c_lag1', 'min_temp_c_lag1',
                      'max_temp_c_roll3', 'max_temp_c_roll7',
                      'heat_stress_roll3', 'cold_stress_roll3'}:
            feat_category_map[feat] = 'weather'
        elif feat in {'aircon_x_heat', 'aircon_x_cold'}:
            feat_category_map[feat] = 'interaction'
        else:
            feat_category_map[feat] = 'demographic'

    cat_colors = {
        'lag':         '#1f77b4',
        'time':        '#2ca02c',
        'weather':     '#ff7f0e',
        'demographic': '#9467bd',
        'interaction': '#d62728',
    }

    if hasattr(model, 'feature_importances_'):
        imps = pd.Series(model.feature_importances_, index=feature_cols_final)
    else:
        scores = model.get_score(importance_type='weight')
        imps = pd.Series([scores.get(f, 0) for f in feature_cols_final],
                         index=feature_cols_final)
    imps_top = imps.sort_values(ascending=True).tail(25)
    bar_colors = [cat_colors[feat_category_map.get(f, 'demographic')] for f in imps_top.index]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.suptitle('Unit Demand — Feature Importances', fontsize=14, fontweight='bold')
    bars = ax.barh(range(len(imps_top)), imps_top.values,
                   color=bar_colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(imps_top)))
    ax.set_yticklabels(imps_top.index, fontsize=9)
    for bar, val in zip(bars, imps_top.values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7.5, color='#333333')
    ax.set_xlabel('Importance Score', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    legend_handles = [Patch(color=c, label=cat.capitalize()) for cat, c in cat_colors.items()]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_feature_importance.png")


def plot_residuals(val_timestamps, y_val, y_pred_val):
    res    = y_val - y_pred_val
    val_dt = pd.to_datetime(val_timestamps)
    tod_period = val_dt.hour * 2 + val_dt.minute // 30
    mae_by_tod = (pd.DataFrame({'tod': tod_period, 'abs_res': np.abs(res)})
                  .groupby('tod')['abs_res'].mean())

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Unit Demand — Residual Analysis (Validation)',
                 fontsize=14, fontweight='bold')

    ax_left.hist(res, bins=100, color='#d62728', alpha=0.75, edgecolor='white', density=True)
    ax_left.axvline(0,          color='black',   lw=1.5, ls='--', label='Zero')
    ax_left.axvline(res.mean(), color='#1f77b4', lw=1.2, ls=':',
                    label=f'Mean = {res.mean():.4f}')
    stats_text = (f'Mean:  {res.mean():.4f}\n'
                  f'Std:   {res.std():.4f}\n'
                  f'Skew:  {pd.Series(res).skew():.2f}\n'
                  f'P10:   {np.percentile(res, 10):.3f}\n'
                  f'P90:   {np.percentile(res, 90):.3f}')
    ax_left.text(0.97, 0.97, stats_text, transform=ax_left.transAxes,
                 va='top', ha='right', fontsize=8.5,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax_left.set_xlabel('Residual (kWh)', fontsize=9)
    ax_left.set_ylabel('Density', fontsize=9)
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)
    ax_left.set_title('Residual Distribution', fontsize=10)

    ax_right.bar(mae_by_tod.index / 2, mae_by_tod.values,
                 width=0.4, color='#d62728', alpha=0.8)
    ax_right.set_xlabel('Hour of Day', fontsize=9)
    ax_right.set_ylabel('Mean Absolute Error (kWh)', fontsize=9)
    ax_right.set_title('MAE by Time of Day', fontsize=10)
    ax_right.set_xticks(range(0, 25, 2))
    ax_right.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_residuals.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_residuals.png")


def plot_predicted_vs_actual_scatter(y_val, y_pred_val):
    r2_val = 1 - np.sum((y_val - y_pred_val) ** 2) / (
                 np.sum((y_val - y_val.mean()) ** 2) + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle('Unit Demand — Predicted vs Actual (Validation)',
                 fontsize=13, fontweight='bold')
    hb = ax.hexbin(y_val, y_pred_val, gridsize=60, cmap='Reds', mincnt=1, bins='log')
    plt.colorbar(hb, ax=ax, label='log₁₀(count)')
    lims = [0, max(float(y_val.max()), float(y_pred_val.max()))]
    ax.plot(lims, lims, 'k--', lw=1.2, label='Perfect prediction')
    ax.set_xlabel('Actual (kWh)', fontsize=10)
    ax.set_ylabel('Predicted (kWh)', fontsize=10)
    ax.text(0.05, 0.93, f'R² = {r2_val:.4f}', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_predicted_vs_actual_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_predicted_vs_actual_scatter.png")


def plot_per_customer_errors(val_customer_ids, y_val, y_pred_val):
    cids_unique    = np.unique(val_customer_ids)
    per_cust_mae   = []
    per_cust_smape = []
    for cid in cids_unique:
        mask = val_customer_ids == cid
        if mask.sum() < 10:
            continue
        yw = y_val[mask]
        yp = y_pred_val[mask]
        per_cust_mae.append(mean_absolute_error(yw, yp))
        per_cust_smape.append(
            np.mean(np.abs(yw - yp) / ((np.abs(yw) + np.abs(yp)) / 2 + 1e-6)) * 100
        )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Unit Demand — Per-Customer Error Distribution (Validation)',
                 fontsize=13, fontweight='bold')
    ax_left.hist(per_cust_mae, bins=40, color='#1f77b4', alpha=0.8, edgecolor='white')
    ax_left.axvline(np.median(per_cust_mae), color='black', lw=1.5, ls='--',
                    label=f'Median = {np.median(per_cust_mae):.3f}')
    ax_left.set_xlabel('MAE (kWh)', fontsize=9)
    ax_left.set_ylabel('Number of Customers', fontsize=9)
    ax_left.set_title('Per-Customer MAE', fontsize=10)
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)
    ax_right.hist(per_cust_smape, bins=40, color='#ff7f0e', alpha=0.8, edgecolor='white')
    ax_right.axvline(np.median(per_cust_smape), color='black', lw=1.5, ls='--',
                     label=f'Median = {np.median(per_cust_smape):.1f}%')
    ax_right.set_xlabel('sMAPE (%)', fontsize=9)
    ax_right.set_ylabel('Number of Customers', fontsize=9)
    ax_right.set_title('Per-Customer sMAPE', fontsize=10)
    ax_right.legend(fontsize=9)
    ax_right.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_per_customer_errors.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: plot_per_customer_errors.png")