from fastapi import APIRouter
from fastapi.responses import JSONResponse
from stored_data import stored_data
import numpy as np
from typing import Any, Dict, Optional
import pandas as pd

router = APIRouter(prefix="/overview", tags=["Overview"])


def safe_stat(fn, data, dtype: type = float) -> Optional[Any]:
    """Fungsi bantu untuk menghitung statistik secara aman."""
    try:
        result = fn(data)
        return dtype(result) if not np.isnan(result) else None
    except Exception as e:
        print(f"[safe_stat] Error calculating statistic: {e}")
        return None


def get_column_stats(col_data):
    """Hitung statistik dasar untuk satu kolom numerik."""
    return {
        "mean": safe_stat(np.mean, col_data),
        "median": safe_stat(np.median, col_data),
        "std": safe_stat(np.std, col_data),
        "min": safe_stat(np.min, col_data),
        "max": safe_stat(np.max, col_data),
        "range": safe_stat(lambda x: np.max(x) - np.min(x), col_data),
        "missing_values": int(col_data.isnull().sum()),
        "unique_values": int(col_data.nunique()),
        "dtype": str(col_data.dtype)
    }


@router.get("/")
async def get_overview():
    """
    Endpoint untuk mendapatkan ringkasan lengkap tentang dataset time series.

    Output mencakup:
    - Metadata time series (rentang waktu, frekuensi, jumlah titik data)
    - Informasi kolom tanggal dan nilai target
    - Statistik deskriptif dari kolom target (numerik)
    - Informasi tambahan per kolom dalam dataframe
    """

    if 'df' not in stored_data:
        return JSONResponse(
            content={"error": "No valid time series data available. Please upload and set columns first."},
            status_code=400
        )

    try:
        # Ambil data time series yang sudah disiapkan
        df = pd.DataFrame(stored_data['df'])
        date_col = stored_data.get('date_col')
        value_col = stored_data.get('value_col')

        # Debugging: Cek isi dataframe
        print("[DEBUG] Columns in df:", df.columns.tolist())
        print("[DEBUG] Target column:", value_col)

        # Ringkasan umum
        summary = {
            "time_series_info": {
                "data_points": len(df),
                "start_date": stored_data["time_range"][0] if "time_range" in stored_data else "N/A",
                "end_date": stored_data["time_range"][1] if "time_range" in stored_data else "N/A",
                "frequency": stored_data.get("freq", "Unknown"),
                "date_column": date_col or "Not specified",
                "target_column": value_col or "Not specified"
            },
            "target_statistics": {},
            "columns_info": {}
        }

        # Jika kolom target tersedia, hitung statistiknya
        if value_col and value_col in df.columns:
            y_data = df[value_col]

            # Debugging: Tipe data & sampel data
            print(f"[DEBUG] Dtype of '{value_col}':", y_data.dtype)
            print(f"[DEBUG] Sample data from '{value_col}':", y_data.head().tolist())

            # Pastikan kolom bertipe numerik
            if not np.issubdtype(y_data.dtype, np.number):
                print(f"[WARNING] Column '{value_col}' is not numeric. Attempting to convert...")
                y_data = pd.to_numeric(y_data, errors='coerce')

            # Hitung statistik
            summary["target_statistics"] = {
                "mean": safe_stat(np.mean, y_data),
                "median": safe_stat(np.median, y_data),
                "std": safe_stat(np.std, y_data),
                "min": safe_stat(np.min, y_data),
                "max": safe_stat(np.max, y_data),
                "range": safe_stat(lambda x: np.max(x) - np.min(x), y_data),
                "missing_values": int(y_data.isnull().sum()),
                "unique_values": int(y_data.nunique())
            }

        # Statistik per kolom
        for col in df.columns:
            col_data = df[col]
            is_numeric = np.issubdtype(col_data.dtype, np.number)

            col_info = {
                "is_numeric": is_numeric,
                "total_non_null": int(col_data.count()),
                "missing_ratio": round(float(col_data.isna().mean()), 4)
            }

            if is_numeric:
                col_info.update({
                    "statistics": get_column_stats(col_data)
                })
            else:
                mode_result = col_data.mode()
                top_value = str(mode_result[0]) if not mode_result.empty else None
                count = int(col_data.value_counts().iloc[0]) if not col_data.value_counts().empty else 0

                col_info.update({
                    "top_value": top_value,
                    "top_value_count": count,
                    "distinct_values": int(col_data.nunique())
                })

            summary["columns_info"][col] = col_info

        return JSONResponse(content={
            "status": "success",
            "message": "Time series overview successfully retrieved.",
            "summary": summary
        })

    except Exception as e:
        print(f"[ERROR] Internal server error: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": f"Internal server error: {str(e)}"},
            status_code=500
        )