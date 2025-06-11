# AI Insights.py
from fastapi import APIRouter, HTTPException, Query
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import traceback  # Untuk logging error lengkap

# Import data dan fungsi bantu
from stored_data import stored_data


router = APIRouter(prefix="/ai-insights", tags=["AI Insights"])


def robust_create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur waktu dari kolom 'ds' dengan penanganan error.
    Harus menerima DataFrame dengan kolom 'ds' dan 'y'
    """
    print("[robust_create_time_features] Memulai...")
    print("Columns input:", df.columns.tolist())

    if not {'ds', 'y'}.issubset(df.columns):
        raise ValueError(f"DataFrame harus memiliki kolom 'ds' dan 'y'. Kolom tersedia: {df.columns.tolist()}")

    df = df[['ds', 'y']].copy()
    print("Sample data sebelum parsing datetime:\n", df.head().to_string())

    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    if df['ds'].isnull().all():
        raise ValueError("Semua nilai di kolom 'ds' tidak valid setelah konversi ke datetime.")
    elif df['ds'].isnull().any():
        print("Peringatan: Beberapa nilai di 'ds' null. Menghapus baris null...")
        df = df.dropna(subset=['ds'])

    df['day_of_week'] = df['ds'].dt.dayofweek  # 0=Monday, ..., 6=Sunday
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['quarter'] = df['ds'].dt.quarter
    df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)

    print("[robust_create_time_features] Hasil akhir:")
    print(df[['ds', 'y', 'day_of_week', 'month', 'is_weekend']].head().to_string())
    return df


def analyze_clusters(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """
    Klusterisasi berdasarkan fitur waktu dan menghasilkan analisis lengkap
    """
    print("[analyze_clusters] Memulai proses clustering...")

    try:
        print("DataFrame input columns:", df.columns.tolist())
        print("Sample data:\n", df[['ds', 'y']].head().to_string())

        features_df = robust_create_time_features(df)

        # Pastikan y bertipe numerik
        features_df['y'] = pd.to_numeric(features_df['y'], errors='coerce')
        if features_df['y'].isnull().all():
            raise ValueError("Kolom 'y' penuh dengan NaN setelah dikonversi ke numerik")
        elif features_df['y'].isnull().any():
            print("Ada NaN di kolom 'y', akan diisi dengan rata-rata")
            mean_y = features_df['y'].mean()
            features_df['y'] = features_df['y'].fillna(mean_y)

        # Fitur untuk clustering
        X = features_df[['day_of_week', 'month', 'is_weekend', 'quarter', 'week_of_year']]
        print("Fitur clustering:\n", X.describe().to_string())

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        features_df['cluster'] = kmeans.fit_predict(X_scaled)

        # Statistik cluster
        mean_features = features_df.groupby('cluster')[X.columns].mean().round(2)
        mean_features.index = mean_features.index.astype(str)  # ubah index ke string agar aman
        cluster_analysis = {
            "count_per_cluster": features_df['cluster'].value_counts().astype(int).to_dict(),
            "mean_features_per_cluster": mean_features.to_dict(),
            "centers": kmeans.cluster_centers_.tolist(),
        }

        # Label otomatis
        labels = []
        for i in range(n_clusters):
            cluster_key = str(i)
            day_of_week = cluster_analysis["mean_features_per_cluster"]["day_of_week"][cluster_key]
            is_weekend = cluster_analysis["mean_features_per_cluster"]["is_weekend"][cluster_key]

            if is_weekend > 0.7:
                label = "Weekend-Dominant"
            elif day_of_week < 2:
                label = "Early Week"
            elif 2 <= day_of_week < 4:
                label = "Mid Week"
            elif day_of_week >= 4:
                label = "Late Week"
            else:
                label = "Unknown"

            labels.append({
                "cluster_id": i,
                "label": label,
                "avg_day_of_week": round(day_of_week, 2),
                "avg_is_weekend": round(is_weekend, 2)
            })

        cluster_analysis["cluster_labels"] = labels

        # Korelasi antara fitur waktu dan y
        correlation_with_y = {}
        for col in X.columns:
            corr = features_df[col].corr(features_df['y'])
            correlation_with_y[col] = round(corr, 2)

        # Tren rata-rata y per cluster
        trend_insight = []
        y_mean_by_cluster = features_df.groupby('cluster')['y'].mean().sort_index()
        prev_avg = None
        for cluster_id in y_mean_by_cluster.index:
            curr_avg = y_mean_by_cluster.loc[cluster_id]
            trend = "Awal" if prev_avg is None else ("Naik" if curr_avg > prev_avg else "Turun")
            trend_insight.append({
                "cluster_id": int(cluster_id),
                "trend": trend,
                "prev_avg_y": round(float(prev_avg), 2) if prev_avg is not None else None,
                "curr_avg_y": round(float(curr_avg), 2)
            })
            prev_avg = curr_avg

        # Distribusi y per cluster
        y_distribution = {}
        for cluster_id in range(n_clusters):
            y_distribution[str(cluster_id)] = features_df[features_df['cluster'] == cluster_id]['y'].round(2).tolist()

        print("[analyze_clusters] Selesai tanpa error")
        return {
            "clusters": features_df[['ds', 'cluster']].to_dict(orient="records"),
            "analysis": cluster_analysis,
            "correlation_with_y": correlation_with_y,
            "trend_insight": trend_insight,
            "y_distribution_per_cluster": y_distribution
        }

    except Exception as e:
        print("[analyze_clusters] Error terjadi:")
        print(traceback.format_exc())  # Log error lengkap
        raise e


@router.get("/clusters")
async def get_clusters(n_clusters: int = Query(3, ge=2, le=10)):
    if 'df' not in stored_data:
        raise HTTPException(status_code=400, detail="No data uploaded yet")

    try:
        df = pd.DataFrame(stored_data['df'])

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded data is empty")

        print("Data berhasil dimuat dari stored_data")
        result = analyze_clusters(df, n_clusters=n_clusters)

        return {
            "result": result
        }

    except KeyError as ke:
        raise HTTPException(
            status_code=500,
            detail=f"Missing key in stored_data: {str(ke)}"
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data format: {str(ve)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )