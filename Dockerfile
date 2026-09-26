# ==============================================================================
# SIH 26077: AI-Driven Hyper-Local Severe Weather Nowcasting System
# Containerized Deployment (FastAPI Backend + Streamlit UI)
# ==============================================================================
FROM python:3.10-slim

# Avoid prompts from debian and ensure output is printed directly
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    API_PORT=8000 \
    API_HOST=0.0.0.0

WORKDIR /app

# Install minimal OS dependencies required for GDAL/rasterio and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports: 8501 (Streamlit UI), 8000 (FastAPI Backend)
EXPOSE 8501 8000

# Make entrypoint executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Default command: launch FastAPI in background and bind Streamlit to Render's dynamic $PORT (fallback 8501)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/main.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false"]
