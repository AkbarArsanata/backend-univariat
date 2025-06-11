from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from prophet import Prophet
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Union, Optional
import logging
import numpy as np

# Local imports
from stored_data import stored_data
from utils import validate_univariate_time_series

router = APIRouter(prefix="/forecast", tags=["Forecasting"])

# Setup logger
logger = logging.getLogger("forecasting")
logger.setLevel(logging.INFO)

# =================== Enums & Configs =================== #

class ForecastModel(str, Enum):
    PROPHET = "prophet"
    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"

# =================== Utility Functions =================== #

def load_and_prepare_data() -> pd.DataFrame:
    """Memuat dan mempersiapkan data time series."""
    if 'df' not in stored_data:
        raise ValueError("Belum ada data yang diproses. Gunakan /set-columns terlebih dahulu.")
    
    df = pd.DataFrame(stored_data['df'])
    df['ds'] = pd.to_datetime(df['ds'])

    if not validate_univariate_time_series(df):
        raise ValueError("Data time series tidak valid atau jumlah data kurang dari 10.")

    return df.sort_values('ds').reset_index(drop=True)

def generate_future_dates(last_date, periods, freq="D"):
    """Menghasilkan daftar tanggal masa depan berdasarkan frekuensi"""
    future_dates = []
    current = last_date
    for _ in range(periods):
        if freq == "D":
            current += timedelta(days=1)
        elif freq == "W":
            current += timedelta(weeks=1)
        elif freq == "M":
            # Bulan berikutnya
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        future_dates.append(current.strftime("%Y-%m-%d"))
    return future_dates

# Fungsi utilitas untuk konversi timestamp ke string agar bisa di-serialize ke JSON
def convert_timestamp_to_str(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df

# =================== Forecasting Methods =================== #

def prophet_forecast(df: pd.DataFrame, periods: int) -> dict:
    """Forecast menggunakan Prophet"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=stored_data.get("freq") == "D",
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        n_changepoints=int(len(df) / 5),
        interval_width=0.95
    )
    model.add_country_holidays(country_name='ID')
    model.fit(df[['ds', 'y']])
    
    future = model.make_future_dataframe(periods=periods, freq=stored_data.get("freq", "D"))
    forecast = model.predict(future)

    # Pastikan semua kolom datetime dikonversi ke string
    forecast = convert_timestamp_to_str(forecast)

    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

    components = {
        "trend": forecast[['ds', 'trend']].to_dict(orient='records'),
        "yearly": forecast[['ds', 'yearly']].to_dict(orient='records') if 'yearly' in forecast.columns else [],
        "weekly": forecast[['ds', 'weekly']].to_dict(orient='records') if 'weekly' in forecast.columns else []
    }

    return {"forecast": prediction, "components": components}

def naive_forecast(df: pd.DataFrame, periods: int) -> dict:
    """Naive Forecast: Menggunakan nilai terakhir sebagai prediksi"""
    last_value = df.iloc[-1]["y"]
    last_date = df.iloc[-1]["ds"].strftime("%Y-%m-%d")
    dates = generate_future_dates(datetime.strptime(last_date, "%Y-%m-%d"), periods, stored_data.get("freq", "D"))

    prediction = [
        {"ds": date, "yhat": float(last_value), "yhat_lower": float(last_value), "yhat_upper": float(last_value)}
        for date in dates
    ]

    return {"forecast": prediction, "components": {"trend": [], "yearly": [], "weekly": []}}

def moving_average_forecast(df: pd.DataFrame, periods: int) -> dict:
    """Simple Moving Average Forecast"""
    window = min(7, len(df))
    avg = df["y"].rolling(window=window).mean().iloc[-1]
    last_date = df.iloc[-1]["ds"].strftime("%Y-%m-%d")
    dates = generate_future_dates(datetime.strptime(last_date, "%Y-%m-%d"), periods, stored_data.get("freq", "D"))

    prediction = [{"ds": date, "yhat": float(avg), "yhat_lower": float(avg * 0.95), "yhat_upper": float(avg * 1.05)} for date in dates]

    return {"forecast": prediction, "components": {"trend": [], "yearly": [], "weekly": []}}

def exponential_smoothing_forecast(df: pd.DataFrame, periods: int) -> dict:
    """Exponential Smoothing Forecast"""
    alpha = 0.2
    smoothed = df["y"].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    last_date = df.iloc[-1]["ds"].strftime("%Y-%m-%d")
    dates = generate_future_dates(datetime.strptime(last_date, "%Y-%m-%d"), periods, stored_data.get("freq", "D"))

    prediction = [{"ds": date, "yhat": float(smoothed), "yhat_lower": float(smoothed * 0.93), "yhat_upper": float(smoothed * 1.07)} for date in dates]

    return {"forecast": prediction, "components": {"trend": [], "yearly": [], "weekly": []}}

# =================== Main Forecast Runner =================== #

def run_forecast(model_type: ForecastModel, df: pd.DataFrame, periods: int) -> dict:
    """Menjalankan forecasting sesuai model yang dipilih"""
    if model_type == ForecastModel.PROPHET:
        return prophet_forecast(df, periods)
    elif model_type == ForecastModel.NAIVE:
        return naive_forecast(df, periods)
    elif model_type == ForecastModel.MOVING_AVERAGE:
        return moving_average_forecast(df, periods)
    elif model_type == ForecastModel.EXPONENTIAL_SMOOTHING:
        return exponential_smoothing_forecast(df, periods)
    else:
        raise ValueError(f"Model {model_type} belum tersedia")

# =================== FastAPI Endpoint =================== #

@router.get("/predict")
async def get_forecast(
    model: ForecastModel = Query(ForecastModel.PROPHET),
    periods: int = Query(30, ge=1, le=365)
):
    """
    Endpoint utama untuk melakukan forecasting.
    
    Parameters:
    - model: Jenis model forecasting (prophet, naive, moving_average, exponential_smoothing)
    - periods: Jumlah hari ke depan untuk forecast (min 1, max 365)
    """

    try:
        df = load_and_prepare_data()
        result = run_forecast(model, df, periods)

        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_forecast_days": periods,
            "time_granularity": stored_data.get("freq", "unknown"),
            "forecast_model": model.value,
            "trend_direction": "increasing" if result["forecast"][-1]["yhat"] > result["forecast"][0]["yhat"] else "decreasing"
        }

        logger.info(f"Menggunakan model '{model}' untuk forecasting {periods} hari ke depan.")
        return JSONResponse(content={
            "status": "success",
            "data": {
                "forecast": result["forecast"],
                "components": result["components"],
                "metadata": metadata
            }
        })

    except ValueError as ve:
        logger.warning(str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error saat memproses forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")