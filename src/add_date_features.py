import numpy as np
import pandas as pd
import holidays

def add_date_features(df, date_col, hour_col):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # === Basic calendar parts ===
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['hour'] = df[hour_col]
    df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int8')

    # === Cyclical encodings (optional for non-tree models) ===
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # === Trend proxy ===
    df['days_since_start'] = (df[date_col] - df[date_col].min()).dt.days

    # === Part-of-month group ===
    df['day_group'] = pd.cut(df['day'], bins=[0, 10, 20, 31], labels=['early', 'mid', 'late'])

    # === UK Bank Holiday flag (as 0/1) ===
    uk_holidays = holidays.UnitedKingdom()
    df['is_uk_holiday'] = df[date_col].dt.date.astype('O').isin(uk_holidays).astype('int8')

    # === Prophet-inspired time bins ===

    def bin_hour(h):
        if 2 <= h <= 5:
            return 'early_am_spike'
        elif 6 <= h <= 11:
            return 'morning'
        elif 12 <= h <= 17:
            return 'afternoon'
        elif 18 <= h <= 21:
            return 'evening'
        else:
            return 'night'
    df['hour_bin'] = df['hour'].apply(bin_hour)

    def bin_weekday(dow):
        if dow == 0:
            return 'monday_low'
        elif dow in [1, 2, 3, 4]:
            return 'midweek'
        elif dow == 5:
            return 'saturday_rising'
        else:
            return 'sunday_peak'
    df['dow_bin'] = df['dayofweek'].apply(bin_weekday)

    def bin_month(m):
        if m in [1, 2, 3]:
            return 'q1_low'
        elif m in [4, 5, 6]:
            return 'q2_neutral'
        elif m in [7, 8, 9, 10, 11]:
            return 'q3_q4_high'
        else:
            return 'dec_dip'
    df['month_bin'] = df['month'].apply(bin_month)

    def bin_day(day):
        if day <= 10:
            return 'early_month'
        elif day <= 20:
            return 'mid_month'
        else:
            return 'late_month'
    df['day_bin'] = df['day'].apply(bin_day)

    return df