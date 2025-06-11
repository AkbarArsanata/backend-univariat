# decomposition.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.signal import find_peaks
from scipy.fftpack import fft
from stored_data import stored_data

router = APIRouter(prefix="/decomposition", tags=["Decomposition"])


def replace_inf_nan(lst):
    """Ganti inf, -inf, dan NaN dengan 0"""
    return [0.0 if x == np.inf or x == -np.inf or pd.isna(x) else float(x) for x in lst]


def safe_float(value):
    """Pastikan nilai adalah float valid"""
    try:
        if pd.isna(value) or np.isinf(value):
            return 0.0
        return float(value)
    except:
        return 0.0


def compute_fft_period(y_values, fs=1.0):
    """Gunakan FFT untuk mencari periode dominan"""
    y_values = pd.to_numeric(y_values, errors='coerce')
    y_values = y_values[~np.isnan(y_values)]
    n = len(y_values)

    if n < 4:
        return None

    try:
        y_fft = fft(y_values - np.mean(y_values))
        freqs = np.fft.fftfreq(n, d=1/fs)
        power = np.abs(y_fft[:n//2]) ** 2
        peaks, _ = find_peaks(power, height=np.mean(power)+2*np.std(power))

        if len(peaks) > 0:
            dominant_freq = freqs[peaks[np.argmax(power[peaks])]]
            period = int(np.round(1 / dominant_freq)) if dominant_freq != 0 else None
            if period and period >= 2 and period < n:
                return period
    except Exception as e:
        print(f"[compute_fft_period] Error during FFT: {e}")
        return None

    return None


def decompose_timeseries(df: pd.DataFrame, period: int, model_type: str):
    try:
        df = df.set_index("ds")
        df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)

        if model_type == "additive" or model_type == "multiplicative":
            result = seasonal_decompose(df["y"], period=period, model=model_type, extrapolate_trend="freq")
        elif model_type == "stl":
            result = STL(df["y"], period=period).fit()
        else:
            raise ValueError(f"Model {model_type} tidak didukung")

        components = {
            "observed": replace_inf_nan(result.observed.tolist()),
            "trend": replace_inf_nan(result.trend.tolist()),
            "seasonal": replace_inf_nan(result.seasonal.tolist()),
            "residual": replace_inf_nan(result.resid.tolist())
        }

        # Hitung statistik dengan proteksi error
        try:
            trend_strength = 1 - (np.var(result.resid) / np.var(result.trend + result.resid))
        except:
            trend_strength = 0.0

        try:
            seasonality_strength = 1 - (np.var(result.resid) / np.var(result.seasonal + result.resid))
        except:
            seasonality_strength = 0.0

        try:
            noise_strength = np.var(result.resid)
        except:
            noise_strength = 0.0

        try:
            trend_stable = abs(np.corrcoef(result.trend[~np.isnan(result.trend)])).mean()
        except:
            trend_stable = 0.0

        try:
            seasonal_consistent = abs(np.corrcoef(result.seasonal[~np.isnan(result.seasonal)])).mean()
        except:
            seasonal_consistent = 0.0

        stats = {
            "trend_strength": safe_float(trend_strength),
            "seasonality_strength": safe_float(seasonality_strength),
            "noise_strength": safe_float(noise_strength),
            "stability": {
                "trend_stable": safe_float(trend_stable),
                "seasonal_consistent": safe_float(seasonal_consistent)
            }
        }

        return {"components": components, "stats": stats, "model_used": model_type}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decomposition failed: {str(e)}")


def generate_component_plots(decomposition_result):
    """Generate plot per komponen sebagai base64 image"""
    plt.figure(figsize=(12, 8))
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    titles = ["Observed", "Trend", "Seasonal", "Residual"]
    values = [
        decomposition_result["observed"],
        decomposition_result["trend"],
        decomposition_result["seasonal"],
        decomposition_result["residual"]
    ]

    for ax, title, val in zip(axes, titles, values):
        ax.plot(val)
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    data_uri = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data_uri}"


def export_to_json(decomposition_result, metadata):
    """Export hasil dekomposisi lengkap dengan metadata"""
    return {
        "metadata": metadata,
        "components": {
            "time": metadata["time_range"],
            "frequency": metadata["freq"],
            "model_used": metadata["model_used"],
            "period_used": metadata["period_used"]
        },
        "series": {
            "observed": decomposition_result["components"]["observed"],
            "trend": decomposition_result["components"]["trend"],
            "seasonal": decomposition_result["components"]["seasonal"],
            "residual": decomposition_result["components"]["residual"]
        },
        "statistics": decomposition_result["stats"]
    }


@router.get("/")
async def get_decomposition(
    period: Optional[int] = Query(None, description="Periode musiman (default: otomatis dari frekuensi atau FFT)"),
    model_type: str = Query("auto", enum=["auto", "additive", "multiplicative", "stl"], description="Pilih model dekomposisi")
):
    if 'df' not in stored_data:
        raise HTTPException(status_code=400, detail="No data uploaded yet")

    df = pd.DataFrame(stored_data['df'])
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)

    # Validasi kolom y harus numerik
    df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
    if df['y'].isna().all():
        raise HTTPException(status_code=400, detail="Kolom target tidak memiliki nilai numerik valid.")

    # Jika tidak disediakan, gunakan freq dari stored_data
    freq = stored_data.get("freq", "M")[0]  # Ambil huruf pertama seperti M dari MS
    freq_map = {'Y': 365, 'Q': 90, 'M': 30, 'W': 7, 'D': 1}
    default_period = freq_map.get(freq, 12)

    # Coba deteksi menggunakan FFT jika tidak ditentukan
    detected_period = compute_fft_period(df["y"].values)
    final_period = period if period is not None else detected_period or default_period

    # Validasi periode
    if final_period < 2 or final_period >= len(df):
        final_period = default_period

    # Tentukan model jika auto
    selected_model = model_type
    if model_type == "auto":
        if np.any(df["y"] <= 0) or np.ptp(df["y"]) < 3 * df["y"].std():
            selected_model = "additive"
        else:
            selected_model = "multiplicative"

    # Lakukan dekomposisi
    decomposition_result = decompose_timeseries(df.copy(), final_period, selected_model)

    # Generate plot
    plot_url = generate_component_plots(decomposition_result["components"])

    # Metadata tambahan
    metadata = {
        "time_range": [str(df["ds"].min()), str(df["ds"].max())],
        "freq": freq,
        "model_used": decomposition_result["model_used"],
        "period_used": final_period,
        "data_points": len(df),
        "mean_value": float(df["y"].mean()),
        "std_value": float(df["y"].std())
    }

    # Export sebagai JSON siap pakai
    export_data = export_to_json(decomposition_result, metadata)

    return {
        "summary": {
            "success": True,
            "period_used": final_period,
            "model_used": selected_model,
            "freq_detected": freq,
            "trend_strength": decomposition_result["stats"]["trend_strength"],
            "seasonality_strength": decomposition_result["stats"]["seasonality_strength"]
        },
        "plot": plot_url,
        "export": export_data
    }