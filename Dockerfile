FROM python:3.11-slim

WORKDIR /app

# Instalamos dependencias del sistema para PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos el puerto del worker
EXPOSE 8000

CMD ["uvicorn", "worker_ia:app", "--host", "0.0.0.0", "--port", "8000"]
