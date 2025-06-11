# utils.py

import pandas as pd
from scipy.stats import skew, kurtosis

def load_data(file, filename):
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(file)
        elif filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return None
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def prepare_time_series(df, date_col, value_col):
    df = df[[date_col, value_col]].dropna()
    df = df.rename(columns={date_col: 'ds', value_col: 'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds', 'y'])
    df = df.sort_values('ds')
    return df


def detect_frequency(df):
    if len(df) < 2:
        return "D"  # Default daily jika tidak cukup data
    df_sorted = df.sort_values('ds')
    diffs = df_sorted['ds'].diff().dropna()
    if diffs.empty:
        return "D"
    most_common = diffs.mode()[0]

    if pd.Timedelta(days=27) < most_common < pd.Timedelta(days=32):
        return "M"  # Monthly
    elif pd.Timedelta(days=6) < most_common < pd.Timedelta(days=8):
        return "W"  # Weekly
    elif pd.Timedelta(days=0) < most_common < pd.Timedelta(days=2):
        return "D"  # Daily
    elif pd.Timedelta(days=89) < most_common < pd.Timedelta(days=93):
        return "Q"  # Quarterly
    elif pd.Timedelta(days=364) < most_common < pd.Timedelta(days=367):
        return "Y"  # Yearly
    else:
        return "D"  # Default to daily


# ===== FUNGSI BARU DITAMBAHKAN ===== #

def validate_univariate_time_series(df: pd.DataFrame) -> bool:
    """
    Memvalidasi bahwa DataFrame memiliki kolom 'ds' dan 'y',
    serta cukup data untuk forecasting (minimal 10 baris).
    """
    required_columns = {'ds', 'y'}
    if not required_columns.issubset(df.columns):
        return False
    if len(df) < 10:
        return False
    return True