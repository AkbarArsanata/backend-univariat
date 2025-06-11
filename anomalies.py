#anomalies.py
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from typing import List, Dict, Optional

# Algoritma ML
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Data penyimpanan sementara
from stored_data import stored_data

router = APIRouter(prefix="/anomalies", tags=["Anomaly Detection"])


# =======================
# Helper Functions
# =======================

def load_stored_dataframe() -> pd.DataFrame:
    if 'df' not in stored_data:
        raise HTTPException(status_code=400, detail="No data uploaded yet")
    df = pd.DataFrame(stored_data['df'])
    df['ds'] = pd.to_datetime(df['ds'])
    return df.sort_values('ds').reset_index(drop=True)


def generate_base64_plot(df: pd.DataFrame, title: str) -> str:
    """Generate static plot of anomalies using Matplotlib"""
    plt.figure(figsize=(12, 5))
    plt.plot(df['ds'], df['y'], label='Time Series', color='blue')
    anomalies = df[df['anomaly'] == -1] if 'anomaly' in df.columns and (df['anomaly'] == -1).any() else df[df['anomaly']]
    plt.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies')
    plt.title(title)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')


def get_anomaly_stats(df: pd.DataFrame) -> dict:
    """Get statistical summary for anomalies vs normal points"""
    anomalies = df[df['anomaly'] == -1] if 'anomaly' in df.columns and (df['anomaly'] == -1).any() else df[df['anomaly']]
    normal = df[~df.index.isin(anomalies.index)]

    return {
        "anomalies": {
            "count": len(anomalies),
            "mean": float(anomalies['y'].mean()) if not anomalies.empty else None,
            "min": float(anomalies['y'].min()) if not anomalies.empty else None,
            "max": float(anomalies['y'].max()) if not anomalies.empty else None,
            "std": float(anomalies['y'].std()) if not anomalies.empty else None
        },
        "normal": {
            "count": len(normal),
            "mean": float(normal['y'].mean()),
            "min": float(normal['y'].min()),
            "max": float(normal['y'].max()),
            "std": float(normal['y'].std())
        }
    }


def get_monthly_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Count anomalies per month"""
    df['month_year'] = df['ds'].dt.to_period('M').astype(str)
    dist = df.groupby('month_year').size().reset_index(name='count')
    return dict(zip(dist['month_year'], dist['count']))


# =======================
# Anomaly Detection Methods
# =======================

def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df[['y']])
    df['confidence'] = -model.score_samples(df[['y']])
    return df


def detect_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    mean = df['y'].mean()
    std = df['y'].std()
    df['z_score'] = (df['y'] - mean) / std
    df['anomaly'] = df['z_score'].abs() > threshold
    df['confidence'] = df['z_score'].abs()
    return df


def detect_local_outlier_factor(df: pd.DataFrame, n_neighbors: int = 20) -> pd.DataFrame:
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    df['anomaly'] = model.fit_predict(df[['y']])
    df['confidence'] = -model.negative_outlier_factor_
    return df


def detect_sarimax_residuals(df: pd.DataFrame, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) -> pd.DataFrame:
    try:
        model = SARIMAX(df['y'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        residuals = results.resid
        threshold = residuals.std() * 2
        df['residual'] = residuals
        df['anomaly'] = abs(df['residual']) > threshold
        df['confidence'] = abs(df['residual']) / threshold
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SARIMAX error: {str(e)}")
    return df


# =======================
# Endpoints
# =======================

@router.get("/isolation-forest")
async def isolation_forest_anomalies(contamination: float = 0.05):
    df = load_stored_dataframe()
    result_df = detect_isolation_forest(df.copy(), contamination=contamination)
    stats = get_anomaly_stats(result_df)
    plot = generate_base64_plot(result_df, "Isolation Forest Anomalies")

    return {
        "method": "Isolation Forest",
        "all_points": result_df[['ds', 'y', 'anomaly', 'confidence']].to_dict(orient="records"),
        "anomalies_only": result_df[result_df['anomaly'] == -1][['ds', 'y']].to_dict(orient="records"),
        "stats": stats,
        "plot_base64": plot,
        "monthly_distribution": get_monthly_distribution(result_df[result_df['anomaly'] == -1])
    }


@router.get("/zscore")
async def zscore_anomalies(threshold: float = 3.0):
    df = load_stored_dataframe()
    result_df = detect_zscore(df.copy(), threshold=threshold)
    stats = get_anomaly_stats(result_df)
    plot = generate_base64_plot(result_df, "Z-Score Anomalies")

    return {
        "method": "Z-Score",
        "all_points": result_df[['ds', 'y', 'anomaly', 'confidence']].to_dict(orient="records"),
        "anomalies_only": result_df[result_df['anomaly']][['ds', 'y']].to_dict(orient="records"),
        "stats": stats,
        "plot_base64": plot,
        "monthly_distribution": get_monthly_distribution(result_df[result_df['anomaly']])
    }


@router.get("/local-outlier-factor")
async def local_outlier_factor_anomalies(n_neighbors: int = 20):
    df = load_stored_dataframe()
    result_df = detect_local_outlier_factor(df.copy(), n_neighbors=n_neighbors)
    stats = get_anomaly_stats(result_df)
    plot = generate_base64_plot(result_df, "Local Outlier Factor Anomalies")

    return {
        "method": "Local Outlier Factor",
        "all_points": result_df[['ds', 'y', 'anomaly', 'confidence']].to_dict(orient="records"),
        "anomalies_only": result_df[result_df['anomaly'] == -1][['ds', 'y']].to_dict(orient="records"),
        "stats": stats,
        "plot_base64": plot,
        "monthly_distribution": get_monthly_distribution(result_df[result_df['anomaly'] == -1])
    }


@router.get("/sarimax-residuals")
async def sarimax_anomalies(order: str = "1,1,1", seasonal_order: str = "1,1,1,12"):
    df = load_stored_dataframe()

    try:
        order_tuple = tuple(map(int, order.split(',')))
        seasonal_order_tuple = tuple(map(int, seasonal_order.split(',')))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid order or seasonal_order format. Use comma-separated integers.")

    result_df = detect_sarimax_residuals(df.copy(), order=order_tuple, seasonal_order=seasonal_order_tuple)
    stats = get_anomaly_stats(result_df)
    plot = generate_base64_plot(result_df, "SARIMAX Residual Anomalies")

    return {
        "method": "SARIMAX Residuals",
        "all_points": result_df[['ds', 'y', 'anomaly', 'confidence']].to_dict(orient="records"),
        "anomalies_only": result_df[result_df['anomaly']][['ds', 'y']].to_dict(orient="records"),
        "stats": stats,
        "plot_base64": plot,
        "monthly_distribution": get_monthly_distribution(result_df[result_df['anomaly']])
    }


@router.get("/filtered")
async def filtered_anomalies(
    start_date: str,
    end_date: str,
    method: str = Query("isolation-forest", enum=["isolation-forest", "zscore", "local-outlier-factor", "sarimax-residuals"]),
    contamination: float = 0.05,
    threshold: float = 3.0,
    n_neighbors: int = 20,
    order: str = "1,1,1",
    seasonal_order: str = "1,1,1,12"
):
    df = load_stored_dataframe()
    filtered_df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

    if method == "isolation-forest":
        result_df = detect_isolation_forest(filtered_df, contamination=contamination)
    elif method == "zscore":
        result_df = detect_zscore(filtered_df, threshold=threshold)
    elif method == "local-outlier-factor":
        result_df = detect_local_outlier_factor(filtered_df, n_neighbors=n_neighbors)
    elif method == "sarimax-residuals":
        try:
            order_tuple = tuple(map(int, order.split(',')))
            seasonal_order_tuple = tuple(map(int, seasonal_order.split(',')))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid order or seasonal_order format.")
        result_df = detect_sarimax_residuals(filtered_df, order=order_tuple, seasonal_order=seasonal_order_tuple)
    else:
        raise HTTPException(status_code=400, detail="Unknown anomaly detection method")

    return {
        "filtered_result": result_df[['ds', 'y', 'anomaly', 'confidence']].to_dict(orient="records"),
        "start_date": start_date,
        "end_date": end_date,
        "method_used": method
    }