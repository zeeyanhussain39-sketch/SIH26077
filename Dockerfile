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
    libexpat1 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose ports: 10000 (Render default), 8501 (Local default)
EXPOSE 10000 8501

# Make entrypoint executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Default command: launch Streamlit command dashboard bound to dynamic $PORT
CMD ["sh", "-c", "streamlit run app/main.py --server.port ${PORT:-10000} --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --server.headless true"]
