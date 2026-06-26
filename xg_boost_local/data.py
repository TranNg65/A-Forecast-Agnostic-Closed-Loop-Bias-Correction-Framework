import gc
import numpy as np
import pandas as pd

from .config import UNIT_FILE, SGSC_WEATHER_FILE, AU_HOLIDAYS_2013
from .features import (
    add_time_features,
    add_holiday_features,
    add_weather_features,
    add_weather_daily_lags,
    add_unit_lag_features,
    NSW_SCHOOL_HOLIDAYS_2013,
)

BATCH_SIZE = 200


def load_weather():
    weather = pd.read_csv(SGSC_WEATHER_FILE, parse_dates=['date'])
    weather['date'] = weather['date'].dt.normalize()

    required = ['date', 'max_temp_c', 'min_temp_c',
                'solar_exposure_mj_m2', 'rainfall_mm']
    missing_cols = [c for c in required if c not in weather.columns]
    if missing_cols:
        raise ValueError(
            f'SGSC_WEATHER_FILE missing columns: {missing_cols}\n'
            f'Found: {list(weather.columns)}\n'
            f'Rename your BOM columns to match: {required}'
        )

    weather = (weather[weather['date'].dt.year == 2013]
               .drop_duplicates(subset='date')
               .sort_values('date')
               .reset_index(drop=True))

    if len(weather) == 0:
        raise ValueError('No 2013 dates found in SGSC_WEATHER_FILE.')

    missing_days = 365 - len(weather)
    if missing_days > 0:
        print(f'  WARNING: {missing_days} days missing from weather.')

    # Add daily lag + rolling temperature features (thermal mass proxies)
    weather = add_weather_daily_lags(weather)

    print(f'  Weather: {len(weather)} days  '
          f'({weather["date"].min().date()} → {weather["date"].max().date()})')
    return weather


def encode_demographics(unit_df):
    u = unit_df.copy()
    lmh = {'LOW': 0, 'MED': 1, 'HI': 2}

    u['electricity_usage_enc'] = u['electricity_usage_group'].map(lmh).astype('int8')
    u['gas_usage_enc']         = u['gas_usage_group'].map(lmh).astype('int8')
    u['income_enc']            = u['income_group'].map(lmh).astype('int8')

    yn_map = {
        'HAS_AIRCON':        'has_aircon',
        'HAS_GAS':           'has_gas',
        'HAS_GAS_HEATING':   'has_gas_heating',
        'HAS_GAS_HOT_WATER': 'has_gas_hot_water',
        'HAS_GAS_COOKING':   'has_gas_cooking',
        'IS_RENTING':        'is_renting',
    }
    for src, dst in yn_map.items():
        u[dst] = (u[src] == 'Y').astype('int8')

    # The source column is lowercase `has_children` — earlier versions checked
    # for `HAS_CHILDREN`, so has_children_enc was silently 100% NaN.
    if 'has_children' in u.columns:
        u['has_children_enc'] = u['has_children'].map({'Y': 1, 'N': 0})
    else:
        u['has_children_enc'] = np.nan

    for col in ['num_occupants', 'num_children_0_10',
                'num_children_11_17', 'num_occupants_70plus']:
        if col in u.columns:
            u[col] = u[col].astype('int8')

    keep = [
        'customer_ID',
        'electricity_usage_enc', 'gas_usage_enc', 'income_enc',
        'has_aircon', 'has_gas', 'has_gas_heating',
        'has_gas_hot_water', 'has_gas_cooking', 'is_renting',
        'eDaily', 'peakTime',
        'num_occupants',
        'num_children_0_10',
        'num_children_11_17',
        'num_occupants_70plus',
        'has_children_enc',
    ]
    return u[keep]


def engineer_one_customer(cid, timestamps, values, unit_row, weather):
    """Feature-engineer a single customer's timeseries.

    `timestamps` and `values` must already be sorted by time and cover the
    FULL period (train + val combined) so that lag features computed via
    shift() correspond to real 30-min offsets.
    """
    df = pd.DataFrame({'timestamp': timestamps, 'unit_kw': values.astype('float32')})
    df['customer_ID'] = np.int32(cid)

    df['date'] = df['timestamp'].dt.normalize()
    df = df.merge(weather, on='date', how='left')
    df = df.drop(columns='date')

    df = add_time_features(df, 'timestamp')
    df = add_unit_lag_features(df)
    df = add_weather_features(df)
    df = add_holiday_features(df, 'timestamp',
                              AU_HOLIDAYS_2013, NSW_SCHOOL_HOLIDAYS_2013)

    for col in unit_row.index:
        if col != 'customer_ID':
            df[col] = unit_row[col]

    df = df.drop(columns=['hour', 'minute'], errors='ignore')
    # Drop rows where core lag / weather features are missing (warm-up period)
    df = df.dropna(subset=['unit_lag_48', 'max_temp_c'])

    float_cols = df.select_dtypes('float64').columns
    df[float_cols] = df[float_cols].astype('float32')

    return df


def load_train_val_unit_data(weather, train_file, val_file):
    """Load train+val as one combined timeseries per customer so lag features
    are computed across the whole year, then split back into train / val based
    on which file each timestamp came from.

    Returns (train_df, val_df).
    """
    print("  Loading unit demographics...")
    unit_meta = encode_demographics(pd.read_csv(UNIT_FILE))
    unit_meta = unit_meta.set_index('customer_ID')
    unit_ids  = set(unit_meta.index.astype(str))
    print(f"  Unit customers: {len(unit_ids)}")

    print("  Scanning usage file columns...")
    train_all_cols = pd.read_csv(train_file, nrows=0).columns.tolist()
    customer_cols  = [c for c in train_all_cols if c != 'timestamp' and c in unit_ids]
    print(f"  Customers matched in usage file: {len(customer_cols)}")

    train_ts = pd.to_datetime(pd.read_csv(train_file, usecols=['timestamp'])['timestamp'])
    val_ts   = pd.to_datetime(pd.read_csv(val_file,   usecols=['timestamp'])['timestamp'])

    # Combined, sorted timestamps used once per customer; keep a boolean
    # mask to split the engineered frame back into train / val.
    combined_ts  = pd.concat([train_ts, val_ts], ignore_index=True)
    is_train_src = np.concatenate([
        np.ones(len(train_ts),  dtype=bool),
        np.zeros(len(val_ts),   dtype=bool),
    ])
    sort_idx       = np.argsort(combined_ts.values, kind='stable')
    combined_ts    = pd.to_datetime(combined_ts.values[sort_idx])
    is_train_src   = is_train_src[sort_idx]
    train_ts_pos   = np.where(is_train_src)[0]   # positions of train rows
    val_ts_pos     = np.where(~is_train_src)[0]  # positions of val rows

    # Built once; reused per customer to split the engineered frame back.
    mask_train = pd.Series(is_train_src, index=combined_ts)

    n_batches = (len(customer_cols) + BATCH_SIZE - 1) // BATCH_SIZE
    train_batches, val_batches = [], []

    for batch_idx in range(n_batches):
        batch_cols = customer_cols[batch_idx * BATCH_SIZE
                                   : (batch_idx + 1) * BATCH_SIZE]
        print(f"  Batch {batch_idx + 1}/{n_batches}  "
              f"({len(batch_cols)} customers)...", end='\r')

        train_batch = pd.read_csv(train_file, usecols=batch_cols)
        val_batch   = pd.read_csv(val_file,   usecols=batch_cols)

        train_frames, val_frames = [], []

        for col in batch_cols:
            cid = int(col)
            if cid not in unit_meta.index:
                continue

            combined_values = np.empty(len(combined_ts), dtype='float32')
            combined_values[train_ts_pos] = train_batch[col].values.astype('float32')
            combined_values[val_ts_pos]   = val_batch[col].values.astype('float32')

            customer_df = engineer_one_customer(
                cid, combined_ts, combined_values,
                unit_meta.loc[cid], weather,
            )

            in_train = customer_df['timestamp'].map(mask_train)
            train_frames.append(customer_df[in_train.values].copy())
            val_frames.append(customer_df[~in_train.values].copy())

        if train_frames:
            train_batches.append(pd.concat(train_frames, ignore_index=True))
        if val_frames:
            val_batches.append(pd.concat(val_frames, ignore_index=True))

        del train_batch, val_batch, train_frames, val_frames
        gc.collect()

    print()
    train_df = pd.concat(train_batches, ignore_index=True); train_batches.clear()
    val_df   = pd.concat(val_batches,   ignore_index=True); val_batches.clear()

    print(f"  Train dataset: {len(train_df):,} rows  |  "
          f"{train_df['customer_ID'].nunique()} customers  |  "
          f"~{train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"  Val   dataset: {len(val_df):,} rows  |  "
          f"{val_df['customer_ID'].nunique()} customers  |  "
          f"~{val_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    return train_df, val_df


def compute_customer_baseline(train_df):
    """Per-customer, per-half-hour-of-week mean of unit_kw, computed on TRAIN
    only. Used as a hierarchical prior feature (not as a target transform)
    so XGBoost can lean on it where useful and override it where needed.
    Returns a DataFrame keyed on (customer_ID, hour_of_week).
    """
    base = (train_df.groupby(['customer_ID', 'hour_of_week'])['unit_kw']
                    .mean()
                    .reset_index()
                    .rename(columns={'unit_kw': 'customer_how_mean'}))
    base['customer_how_mean'] = base['customer_how_mean'].astype('float32')
    return base


def attach_customer_baseline(df, baseline):
    # Use index-based lookup instead of merge to avoid copying the full DataFrame
    lookup = baseline.set_index(['customer_ID', 'hour_of_week'])['customer_how_mean']
    keys = pd.MultiIndex.from_arrays([df['customer_ID'], df['hour_of_week']])
    df['customer_how_mean'] = lookup.reindex(keys).values
    return df
