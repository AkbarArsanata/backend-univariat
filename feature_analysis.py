from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy.stats import boxcox, kurtosis, skew
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from stored_data import stored_data
import math
import traceback

router = APIRouter(prefix="/features", tags=["Features"])

# ======================
# HELPER FUNCTIONS
# ======================

def sanitize_float(value):
    """Mengganti NaN/inf menjadi None agar JSON valid"""
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def sanitize_dict(data):
    """Rekursif membersihkan dictionary dari NaN/inf"""
    if isinstance(data, dict):
        return {k: sanitize_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    else:
        return sanitize_float(data)


# ======================
# FEATURE ENGINEERING
# ======================

def create_time_features(df):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['day'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df


def create_cyclical_features(df):
    """Encode bulan, hari dalam minggu sebagai sin/cos"""
    df = df.copy()
    month_in_year = df['month']
    day_in_week = df['day_of_week']

    df['month_sin'] = np.sin(2 * np.pi * month_in_year / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_in_year / 12)
    df['dow_sin'] = np.sin(2 * np.pi * day_in_week / 7)
    df['dow_cos'] = np.cos(2 * np.pi * day_in_week / 7)
    return df


def create_lag_features(df, lags=[1, 2, 7, 14]):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df


def create_rolling_features(df, windows=[3, 7, 14, 30]):
    df = df.copy()
    for w in windows:
        df[f"rolling_mean_{w}"] = df["y"].rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df["y"].rolling(window=w).std()
        df[f"rolling_min_{w}"] = df["y"].rolling(window=w).min()
        df[f"rolling_max_{w}"] = df["y"].rolling(window=w).max()
        df[f"rolling_skew_{w}"] = df["y"].rolling(window=w).skew()
    return df


def create_expanding_features(df):
    df = df.copy()
    df["expanding_mean"] = df["y"].expanding().mean()
    df["expanding_std"] = df["y"].expanding().std()
    df["expanding_sum"] = df["y"].expanding().sum()
    return df


def transform_target(df):
    df = df.copy()

    unique_values = df["y"].nunique()
    if unique_values == 1:
        # Skip jika semua nilai sama
        return df

    if (df["y"] <= 0).any():
        df["y_log"] = np.log1p(df["y"])
    else:
        try:
            df["y_boxcox"], _ = boxcox(df["y"])
        except Exception as e:
            print("Box-Cox failed:", str(e))
            df["y_log"] = np.log(df["y"])
    return df


# ======================
# STATISTIK LANJUT
# ======================

def check_stationarity(df_col):
    result = {}
    try:
        adf_test = adfuller(df_col)
        kpss_test = kpss(df_col)

        result["adf"] = {
            "statistic": adf_test[0],
            "pvalue": adf_test[1],
            "critical_values": adf_test[4]
        }

        result["kpss"] = {
            "statistic": kpss_test[0],
            "pvalue": kpss_test[1],
            "critical_values": kpss_test[2]
        }
    except Exception as e:
        result = {"error": f"Cannot perform stationarity test: {str(e)}"}
    return result


def calculate_distribution_stats(df_col):
    if not pd.api.types.is_numeric_dtype(df_col):
        return {}

    try:
        return {
            "skewness": float(skew(df_col)),
            "kurtosis": float(kurtosis(df_col)),
            "mean": float(df_col.mean()),
            "std": float(df_col.std())
        }
    except Exception as e:
        return {"error": f"Stat calculation error: {str(e)}"}


# ======================
# VISUALISASI
# ======================

def generate_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"


def generate_lag_plot(df, lag=1):
    plt.figure(figsize=(6, 6))
    pd.plotting.lag_plot(df['y'], lag=lag)
    plt.title(f"Lag {lag} Plot")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"


def generate_rolling_vs_actual(df):
    df = df.tail(60)
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='Actual')
    for col in df.columns:
        if col.startswith("rolling_mean"):
            plt.plot(df['ds'], df[col], label=col)
    plt.legend()
    plt.title("Rolling Mean vs Actual")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{data}"


# ======================
# UTILITY DAN ANALISIS
# ======================

def auto_select_features(df, target="y", k=10):
    X = df.drop(columns=[target, 'ds']).select_dtypes(include=[np.number])
    y = df[target]

    if X.empty:
        return [], {}

    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X, y)
    scores = dict(zip(X.columns, selector.scores_))
    selected = sorted(scores, key=scores.get, reverse=True)[:k]
    return selected, scores


# ======================
# ENDPOINT
# ======================

@router.get("/")
async def get_features():
    if 'df' not in stored_data:
        raise HTTPException(status_code=400, detail="No data uploaded yet")

    try:
        df = pd.DataFrame(stored_data['df'])
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)

        # Generate features
        df = create_time_features(df)
        df = create_cyclical_features(df)
        df = create_lag_features(df)
        df = create_rolling_features(df)
        df = create_expanding_features(df)
        df = transform_target(df)

        # Drop missing values
        df = df.dropna()

        # Simpan hasil ke stored_data
        stored_data['engineered_df'] = df.to_dict(orient="records")

        # Hitung statistik
        correlations = {
            k: v for k, v in df.corr(numeric_only=True)["y"].sort_values(ascending=False).to_dict().items()
            if abs(v) > 0.01 and not math.isnan(v)
        }

        stats_per_column = {
            col: calculate_distribution_stats(df[col]) for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        }

        stationarity = check_stationarity(df["y"])

        lag_plots = []
        for l in [1, 7, 14]:
            try:
                lag_plots.append(generate_lag_plot(df, lag=l))
            except:
                lag_plots.append(None)

        # Pemilihan fitur otomatis
        selected_features, feature_scores = auto_select_features(df)

        # Visualisasi
        heatmap = generate_correlation_heatmap(df)
        rolling_plot = generate_rolling_vs_actual(df)

        # Bersihkan data sebelum dikirim
        clean_response = {
            "message": "Feature analysis completed",
            "feature_count": len(df.columns),
            "sample_data": df.head().to_dict(orient="records"),
            "correlations_with_target": correlations,
            "distribution_stats": stats_per_column,
            "stationarity": stationarity,
            "selected_features": selected_features,
            "feature_importance_scores": feature_scores,
            "visualizations": {
                "correlation_matrix": heatmap,
                "rolling_vs_actual": rolling_plot,
                "lag_plots": lag_plots[:3]
            }
        }

        return sanitize_dict(clean_response)

    except Exception as e:
        print("Traceback:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")