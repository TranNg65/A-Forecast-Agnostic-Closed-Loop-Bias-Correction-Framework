import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# NSW public school 2013 term dates (Department of Education).
# School *holidays* are the dates between terms + the end-of-year break.
_NSW_2013_TERM_DATES = [
    ('2013-01-29', '2013-04-12'),  # Term 1
    ('2013-04-30', '2013-06-28'),  # Term 2
    ('2013-07-15', '2013-09-20'),  # Term 3
    ('2013-10-08', '2013-12-18'),  # Term 4
]


def build_nsw_school_holidays_2013():
    """Return a set of pd.Timestamp (midnight-normalised) for 2013 school holidays."""
    term_days = set()
    for start, end in _NSW_2013_TERM_DATES:
        term_days.update(
            pd.date_range(start, end, freq='D').normalize()
        )
    all_days = set(pd.date_range('2013-01-01', '2013-12-31', freq='D').normalize())
    return all_days - term_days


NSW_SCHOOL_HOLIDAYS_2013 = build_nsw_school_holidays_2013()


def add_time_features(df, dt_col):
    dt = pd.to_datetime(df[dt_col])
    df['hour']        = dt.dt.hour
    df['minute']      = dt.dt.minute
    df['tod']         = (df['hour'] + df['minute'] / 60).astype('float32')
    df['day_of_week'] = dt.dt.dayofweek.astype('int8')
    df['month']       = dt.dt.month.astype('int8')
    df['day_of_year'] = dt.dt.dayofyear.astype('int16')
    df['is_weekend']  = (df['day_of_week'] >= 5).astype('int8')
    df['is_summer']   = dt.dt.month.isin([12, 1, 2]).astype('int8')
    df['is_winter']   = dt.dt.month.isin([6, 7, 8]).astype('int8')
    df['hour_sin']    = np.sin(2 * np.pi * df['tod'] / 24).astype('float32')
    df['hour_cos']    = np.cos(2 * np.pi * df['tod'] / 24).astype('float32')
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12).astype('float32')
    df['dow_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7).astype('float32')
    df['dow_cos']     = np.cos(2 * np.pi * df['day_of_week'] / 7).astype('float32')
    # Half-hour-of-week index (0..335) — granularity of the customer baseline merge
    df['hour_of_week'] = (df['day_of_week'].astype('int16') * 48
                          + (df['tod'] * 2).astype('int16')).astype('int16')
    return df


def add_holiday_features(df, dt_col, public_holidays, school_holidays):
    # Callers must pass pre-converted sets of pd.Timestamp (see config.py /
    # NSW_SCHOOL_HOLIDAYS_2013) so isin() never does per-call type coercion.
    dates = pd.to_datetime(df[dt_col]).dt.normalize()
    df['is_public_holiday']     = dates.isin(public_holidays).astype('int8')
    df['is_day_before_holiday'] = (dates + pd.Timedelta(days=1)).isin(public_holidays).astype('int8')
    df['is_day_after_holiday']  = (dates - pd.Timedelta(days=1)).isin(public_holidays).astype('int8')
    df['is_school_holiday']     = dates.isin(school_holidays).astype('int8')
    return df


def add_weather_features(df):
    df['heat_stress'] = (df['max_temp_c'] - 26).clip(lower=0)
    df['cold_stress'] = (18 - df['min_temp_c']).clip(lower=0)
    df['temp_range']  = df['max_temp_c'] - df['min_temp_c']
    # Interaction: a hot day at 3pm differs from a hot day at 3am.
    df['temp_x_tod']  = df['max_temp_c'] * df['tod']
    return df


def add_weather_daily_lags(weather_daily):
    """Add lag/rolling columns to a daily weather table (one row per date)."""
    w = weather_daily.sort_values('date').copy()
    w['max_temp_c_lag1']   = w['max_temp_c'].shift(1)
    w['min_temp_c_lag1']   = w['min_temp_c'].shift(1)
    w['max_temp_c_roll3']  = w['max_temp_c'].rolling(3, min_periods=1).mean()
    w['max_temp_c_roll7']  = w['max_temp_c'].rolling(7, min_periods=1).mean()
    heat = (w['max_temp_c'] - 26).clip(lower=0)
    cold = (18 - w['min_temp_c']).clip(lower=0)
    w['heat_stress_roll3'] = heat.rolling(3, min_periods=1).mean()
    w['cold_stress_roll3'] = cold.rolling(3, min_periods=1).mean()
    return w


def add_unit_lag_features(df):
    """Attach lag and rolling stats to a single customer's sorted timeseries."""
    # Short-horizon lags (30 min .. 12 hr) — typically the strongest signal
    # for 30-min demand. Absent from the original config.
    for lag in (1, 2, 3, 6, 12, 24):
        df[f'unit_lag_{lag}'] = df['unit_kw'].shift(lag)
    # Day-scale lags (same time yesterday, 2 days, 1 week, 2 weeks)
    df['unit_lag_48']           = df['unit_kw'].shift(48)
    df['unit_lag_96']           = df['unit_kw'].shift(96)
    df['unit_lag_336']          = df['unit_kw'].shift(336)
    df['unit_lag_672']          = df['unit_kw'].shift(672)
    df['unit_rolling_mean_48']  = df['unit_kw'].shift(1).rolling(48,  min_periods=1).mean()
    df['unit_rolling_std_48']   = df['unit_kw'].shift(1).rolling(48,  min_periods=1).std()
    df['unit_rolling_max_48']   = df['unit_kw'].shift(1).rolling(48,  min_periods=1).max()
    df['unit_rolling_mean_336'] = df['unit_kw'].shift(1).rolling(336, min_periods=1).mean()
    return df


def compute_metrics(y_true, y_pred, label=''):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(np.abs(y_true - y_pred) /
                    ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-6)) * 100
    cv    = rmse / (y_true.mean() + 1e-6) * 100
    print(f"  {label:<30} MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"sMAPE={smape:.2f}%  CV(RMSE)={cv:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'smape': smape, 'cv_rmse': cv}
