from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from stored_data import stored_data

router = APIRouter(prefix="/calendar", tags=["Calendar"])


def generate_yearly_heatmap(df):
    """
    Menghasilkan data heatmap untuk visualisasi kalender tahunan.
    Format: {tahun: [{date, value, day_name}, ...]}
    """
    df = df[['ds', 'y']].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['year'] = df['ds'].dt.year
    df['day_name'] = df['ds'].dt.day_name()

    result = {}
    for year, group in df.groupby('year'):
        result[int(year)] = group[["ds", "y", "day_name"]].rename(columns={"ds": "date"}).to_dict(orient="records")
    return result


def analyze_repeating_day_values(df):
    """
    Hitung rata-rata nilai berdasarkan tanggal tetap (misal: setiap 01-01)
    """
    df = df[['ds', 'y']].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['month_day'] = df['ds'].dt.strftime('%m-%d')
    grouped = df.groupby('month_day')['y'].mean().round(2)
    return grouped.to_dict()


def detect_month_start_end_pattern(df):
    """
    Analisis apakah awal/akhir bulan memiliki pola tertentu
    """
    df = df[['ds', 'y']].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['is_month_start'] = df['ds'].dt.is_month_start
    df['is_month_end'] = df['ds'].dt.is_month_end

    start_avg = df[df['is_month_start']]['y'].mean().round(2)
    end_avg = df[df['is_month_end']]['y'].mean().round(2)

    return {
        "avg_at_month_start": float(start_avg),
        "avg_at_month_end": float(end_avg),
        "percent_diff": round(((end_avg - start_avg) / start_avg * 100), 2) if start_avg != 0 else None
    }


def detect_weekday_behavior(df):
    """
    Cari tahu rata-rata per hari dalam seminggu
    Contoh: Senin selalu rendah?
    """
    df = df.copy()
    df['day_name'] = pd.to_datetime(df['ds']).dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_by_day = df.groupby('day_name')['y'].mean().reindex(order).round(2)
    best_day = avg_by_day.idxmax()
    worst_day = avg_by_day.idxmin()

    return {
        "avg_by_day_of_week": avg_by_day.to_dict(),
        "best_day": best_day,
        "best_value": float(avg_by_day[best_day]),
        "worst_day": worst_day,
        "worst_value": float(avg_by_day[worst_day])
    }


def detect_holiday_impact(df, holidays=None):
    """
    Deteksi dampak tanggal spesifik/libur terhadap nilai
    """
    if not holidays:
        return {"warning": "No holiday list provided."}

    df = df[['ds', 'y']].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['month_day'] = df['ds'].dt.strftime('%m-%d')
    df['is_special_date'] = df['month_day'].isin(holidays)

    special_avg = df[df['is_special_date']]['y'].mean().round(2)
    normal_avg = df[~df['is_special_date']]['y'].mean().round(2)

    return {
        "avg_on_special_dates": float(special_avg),
        "avg_on_normal_days": float(normal_avg),
        "impact_percent": round(((special_avg - normal_avg) / normal_avg * 100), 2)
    }


def compare_years_by_day_of_year(df):
    """
    Bandingkan nilai antar tahun berdasarkan urutan hari dalam tahun (1-366)
    Cocok untuk melihat pola musiman antar-tahun
    """
    df = df[['ds', 'y']].copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['year'] = df['ds'].dt.year
    df['day_of_year'] = df['ds'].dt.dayofyear

    pivot = df.pivot_table(index='day_of_year', columns='year', values='y', aggfunc='mean').round(2)
    return pivot.to_dict()


@router.get("/")
async def get_calendar_insights():
    if 'df' not in stored_data:
        raise HTTPException(status_code=400, detail="No data uploaded yet")

    df = pd.DataFrame(stored_data['df'])
    df['ds'] = pd.to_datetime(df['ds'])

    # Daftar contoh libur nasional (bisa diganti/ditambah sesuai negara)
    holiday_list = ['01-01', '08-17', '12-25']

    return {
        "heatmap": generate_yearly_heatmap(df),
        "patterns": {
            "repeating_day_values": analyze_repeating_day_values(df),
            "month_start_end": detect_month_start_end_pattern(df),
            "weekday_behavior": detect_weekday_behavior(df),
            "holiday_impact": detect_holiday_impact(df, holidays=holiday_list),
            "year_comparison": compare_years_by_day_of_year(df)
        },
        "summary": {
            "min_date": str(df['ds'].min()),
            "max_date": str(df['ds'].max()),
            "total_years": len(df['ds'].dt.year.unique())
        }
    }