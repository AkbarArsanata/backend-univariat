# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from utils import load_data, prepare_time_series, detect_frequency
from stored_data import stored_data

# Import routers dari modul terpisah
from overview import router as overview_router
from ai_insights import router as ai_insights_router
from decomposition import router as decomposition_router
from anomalies import router as anomalies_router
from calendar_view import router as calendar_router
from feature_analysis import router as feature_router
from forecasting import router as forecast_router

from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model untuk /set-columns
class ColumnSelection(BaseModel):
    date_col: str
    value_col: str

# Register routers
app.include_router(overview_router)
app.include_router(ai_insights_router)
app.include_router(decomposition_router)
app.include_router(anomalies_router)
app.include_router(calendar_router)
app.include_router(feature_router)
app.include_router(forecast_router)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = load_data(file.file, file.filename)
    if df is None:
        return {"error": "Unsupported file format or invalid data"}

    # Simpan semua kolom ke stored_data
    stored_data['all_columns'] = df.columns.tolist()
    stored_data['raw_df'] = df.to_dict(orient='records')  # Untuk nanti pilih kolom

    return {
        "message": "File berhasil diupload. Silakan pilih kolom.",
        "available_columns": df.columns.tolist()
    }


@app.post("/set-columns")
async def set_columns(data: ColumnSelection):
    if 'raw_df' not in stored_data:
        raise HTTPException(status_code=400, detail="No data uploaded yet")

    df = pd.DataFrame(stored_data['raw_df'])

    if data.date_col not in df.columns or data.value_col not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column names")

    # Prepare time series
    try:
        ts_df = prepare_time_series(df, data.date_col, data.value_col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preparing time series: {str(e)}")

    freq = detect_frequency(ts_df)

    # Simpan ke stored_data
    stored_data.update({
        'df': ts_df.to_dict(orient='records'),
        'date_col': data.date_col,
        'value_col': data.value_col,
        'freq': freq,
        'time_range': [str(ts_df["ds"].min()), str(ts_df["ds"].max())],
        'mean_value': float(ts_df["y"].mean()),
        'min_value': float(ts_df["y"].min()),
        'max_value': float(ts_df["y"].max()),
        'std_value': float(ts_df["y"].std()),
        'range_value': float(ts_df["y"].max() - ts_df["y"].min())
    })

    return {
        "message": "Kolom berhasil dikonfigurasi",
        "date_col": data.date_col,
        "value_col": data.value_col,
        "freq": freq,
        "data_points": len(ts_df),
        "time_range": stored_data["time_range"],
        "mean_value": stored_data["mean_value"]
    }