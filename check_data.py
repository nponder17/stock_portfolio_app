import pandas as pd
from datetime import date

def read_df():
    df = pd.read_csv('one_day_data.csv')
    return df

def convert_timestamp(df):
    ts = pd.to_datetime(df['timestamp'])
    df['date'] = ts.dt.date
    df['time'] = ts.dt.time
    return df

def drop_frequency_and_timestamp_columns(df):
    df = df.drop(columns=['frequency'])
    df = df.drop(columns=['timestamp'])
    return df

def check_col_names(df):
    print(df.columns)


def check_symbols(df):
    symbol_count = df['symbol'].nunique()
    symbol_list = df['symbol'].unique()
    #print(f"Total symbols: {symbol_count}")
    #print(f"Symbols: {symbol_list}")
    return symbol_count, symbol_list


def check_dates(df):
    # DataFrame of min/max timestamp per symbol (one row per symbol)
    date_ranges = df.groupby('symbol')['date'].agg(['min', 'max']).reset_index()
    date_ranges.columns = ['symbol', 'min_date', 'max_date']
    #print(date_ranges)
    incorrect_start = (date_ranges[date_ranges['min_date'] != date(2015, 11, 19)])
    incorrect_start_count = len(incorrect_start)
    
    incorrect_end = (date_ranges[date_ranges['max_date'] != date(2025, 11, 14)])
    incorrect_end_count = len(incorrect_end)

    """All data ends on same date, dropping rows with incorrect start date
    as they dont have a sufficient amount of data points"""
    return date_ranges, incorrect_start


def drop_incorrect_start_rows(df, incorrect_start):
    """Drop all rows for symbols that have an incorrect start date."""
    return df[~df['symbol'].isin(incorrect_start['symbol'])]

def check_new_dates(df):
    date_ranges = df.groupby('symbol')['date'].agg(['min', 'max']).reset_index()
    date_ranges.columns = ['symbol', 'min_date', 'max_date']
    """Verified all other start dates are correct with 10 years of data"""

def check_nans(df):
    print(df.isnull().sum())
    """No NaNs in the data"""


def check_duplicates(df):
    print(df.duplicated().sum())
    """No duplicates in the data"""


def drop_duplicates(df):
    """Drop duplicate rows so each (symbol, date) is unique for ML."""
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} duplicate rows.")
    return df


def drop_rows_with_nans(df):
    """Drop rows with any NaN (safe for ML; avoids imputation bias)."""
    n_before = len(df)
    df = df.dropna()
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with NaNs.")
    return df


def drop_rows_with_inf(df, numeric_cols=None):
    """Drop rows where any numeric column has inf/-inf."""
    if numeric_cols is None:
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    n_before = len(df)
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=numeric_cols)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with inf.")
    return df


def validate_ohlc(df):
    """Drop rows with invalid OHLC: non-positive prices or high < low, etc."""
    n_before = len(df)
    df = df[
        (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0) &
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) & (df['high'] >= df['close']) &
        (df['low'] <= df['open']) & (df['low'] <= df['close'])
    ]
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with invalid OHLC.")
    return df


def validate_volume(df):
    """Drop rows with invalid volume (e.g. negative). Zero volume optional."""
    n_before = len(df)
    df = df[df['volume'] >= 0]
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with negative volume.")
    return df


def ensure_numeric_dtypes(df):
    """Ensure OHLC and volume are numeric for ML."""
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def sort_by_symbol_and_date(df):
    """Stable sort for time-series ML: symbol then date."""
    return df.sort_values(by=['symbol', 'date']).reset_index(drop=True)


def display_shape(df):
    print("Shape of dataframe: ", df.shape)

def sort_by_date(df):
    df = df.sort_values(by='date')
    return df

def group_by_stock_and_date(df):
    """Return a DataFrameGroupBy object (use .agg(), .sum(), etc. to get a DataFrame)."""
    return df.groupby(['symbol', 'date'])

def count_symbols_after_drops(df):
    new_symbol_count = df['symbol'].nunique()
    print(new_symbol_count)

def write_to_csv(df):
    df.to_csv('cleaned_data.csv', index=False)


if __name__ == "__main__":
    df = read_df()
    df = convert_timestamp(df)
    df = drop_frequency_and_timestamp_columns(df)
    check_col_names(df)
    symbol_count, symbol_list = check_symbols(df)
    date_ranges, incorrect_start = check_dates(df)
    df = drop_incorrect_start_rows(df, incorrect_start)
    check_new_dates(df)

    # Data cleaning for ML (no feature engineering)
    df = ensure_numeric_dtypes(df)
    df = validate_ohlc(df)
    df = validate_volume(df)
    df = drop_rows_with_nans(df)
    df = drop_rows_with_inf(df)
    df = drop_duplicates(df)
    df = sort_by_symbol_and_date(df)

    check_nans(df)
    check_duplicates(df)
    display_shape(df)
    count_symbols_after_drops(df)
    write_to_csv(df)