# Gunakan base image Python 3.11 (atau versi lain sesuai lingkungan lokal kamu)
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt dulu, biar caching pip efisien
COPY requirements.txt .

# Install semua dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode sumber aplikasi
COPY . .

# Expose port (sesuaikan jika platform memerlukan port tertentu)
EXPOSE 80

# Jalankan aplikasi menggunakan gunicorn + Uvicorn worker
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app"]