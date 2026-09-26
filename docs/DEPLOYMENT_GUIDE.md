# 🚀 Comprehensive Deployment & Hosting Guide
### AI-Driven Hyper-Local Severe Weather Nowcasting System (SIH Problem Statement 26077)

---

## 📌 Architecture Overview & Port Bindings

The platform is designed as a modular, containerized multi-tier system:

| Service | Technology | Internal Port | External Port | URL / Endpoint | Purpose |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Command Dashboard** | Streamlit + Folium | `8501` | `8501` | `http://localhost:8501` | Interactive operator UI, map viewer, scripted replay tour, SHAP panel |
| **Alert Engine API** | FastAPI + Uvicorn | `8000` | `8000` | `http://localhost:8000` | Asynchronous CAP-v1.2 alert feed, evaluation, and `smtplib` dispatch |
| **Interactive Docs** | OpenAPI (Swagger) | `8000` | `8000` | `http://localhost:8000/docs` | Live API testbed for disaster management integration |

---

## 💻 Method 1: Local Bare-Metal Deployment (Recommended for Hackathon Judging)

Running locally guarantees zero latency and eliminates risks of venue Wi-Fi drops or cloud bandwidth limits during live evaluation.

### Option A: Windows One-Click Script (Fastest)
Double-click [`scripts/start_local.bat`](file:///c:/Users/zeeya/Desktop/SIH26077/scripts/start_local.bat) or run from PowerShell:
```powershell
.\scripts\start_local.bat
```
*This automatically launches the FastAPI backend in a background console and opens the Streamlit dashboard in your default browser.*

### Option B: Manual Command-Line Startup

#### Terminal 1 — Start the FastAPI Alert Engine:
```bash
# Activate your virtual environment
.\venv\Scripts\Activate.ps1    # On Windows
# source venv/bin/activate     # On Linux/macOS

# Start FastAPI on port 8000
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2 — Start the Streamlit Dashboard:
```bash
# In a separate terminal window with virtual environment active:
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Method 2: Docker & Docker Compose Deployment

Containerized deployment bundles all system packages (`gdal`, `libgomp`, `rasterio`) and isolates dependencies across any Linux, macOS, or Windows host.

### Step 1: Ensure Docker is Running
Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

### Step 2: Build & Launch with Docker Compose
Run from the repository root:
```bash
# Build images and start all services in detached mode
docker compose up --build -d

# View real-time container logs
docker compose logs -f
```

### Step 3: Access Containers
- **Streamlit Dashboard:** `http://localhost:8501`
- **FastAPI Alert API:** `http://localhost:8000/docs`

### Step 4: Stop Containers
```bash
docker compose down
```

### Alternative: Single Container Build
If you prefer running a single self-contained image:
```bash
# Build single combined image
docker build -t sih26077-nowcaster:latest .

# Run with port forwarding
docker run -p 8501:8501 -p 8000:8000 sih26077-nowcaster:latest
```

---

## ☁️ Method 3: 100% Free Cloud Deployment (Zero Cloud Bills)

To share a live public link with judges so they can interact on their smartphones or laptops without installing anything:

### 1. Streamlit Community Cloud (Recommended — Free & 1-Click)
1. Push this repository to your GitHub account (public or private):
   ```bash
   git remote add origin https://github.com/your-username/SIH26077.git
   git push -u origin master
   ```
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and log in with GitHub.
3. Click **"New app"** &rarr; Select your repository &rarr; Set **Main file path** to `app/main.py`.
4. Under **Advanced Settings**, Python version: `3.10` or `3.11`.
5. Click **"Deploy!"**. In ~2 minutes, you will receive a free public URL (e.g. `https://sih26077-nowcaster.streamlit.app`).

### 2. Hugging Face Spaces (Free Docker / Streamlit Space)
1. Create a free account at **[huggingface.co](https://huggingface.co/)**.
2. Click **"New Space"** &rarr; Space SDK: **Docker** (or **Streamlit**).
3. Set Space visibility to **Public**.
4. Clone the HF Space repo and push this codebase:
   ```bash
   git remote add hf https://huggingface.co/spaces/your-username/sih26077-weather-nowcaster
   git push hf master
   ```
5. Hugging Face automatically builds the `Dockerfile` and hosts it permanently on free CPU hardware.

### 3. Render / Railway Free Tier (For FastAPI Alert Backend)
1. Sign up for free at **[render.com](https://render.com/)**.
2. Click **New Web Service** &rarr; Connect GitHub repository.
3. Environment: `Python` (or `Docker`).
4. Build Command: `pip install -r requirements.txt`.
5. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port 10000`.
6. You will get a live public REST API URL (e.g. `https://sih26077-api.onrender.com`).

---

## 🏢 Method 4: Production Linux Edge Server (SDMA / State EOC)

For deployment on on-premise servers in State Disaster Management Authorities (SDMAs) or District Emergency Operations Centers (DEOCs):

### 1. Systemd Service: FastAPI Backend (`/etc/systemd/system/sih-api.service`)
```ini
[Unit]
Description=SIH 26077 Severe Weather Alert Engine (FastAPI)
After=network.target

[Service]
Type=simple
User=weatherops
WorkingDirectory=/opt/sih26077
EnvironmentFile=/opt/sih26077/.env
ExecStart=/opt/sih26077/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 2. Systemd Service: Streamlit Dashboard (`/etc/systemd/system/sih-dashboard.service`)
```ini
[Unit]
Description=SIH 26077 Severe Weather Interactive Dashboard (Streamlit)
After=network.target sih-api.service

[Service]
Type=simple
User=weatherops
WorkingDirectory=/opt/sih26077
EnvironmentFile=/opt/sih26077/.env
ExecStart=/opt/sih26077/venv/bin/streamlit run app/main.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 3. Nginx Reverse Proxy (`/etc/nginx/sites-available/sih26077`)
```nginx
server {
    listen 80;
    server_name nowcast.sdma.gov.in;

    # Streamlit UI Proxy
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # FastAPI REST Backend Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable services:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sih-api sih-dashboard
sudo systemctl restart nginx
```

---

## 🔒 Pre-Deployment Security & Verification Checklist

Before presenting or publishing to the cloud:
- [x] **No Secrets Committed:** Verified that `.env` is listed in [`.gitignore`](file:///c:/Users/zeeya/Desktop/SIH26077/.gitignore). Only [`.env.example`](file:///c:/Users/zeeya/Desktop/SIH26077/.env.example) is tracked.
- [x] **Offline Benchmark Verified:** Verified that synthetic benchmark NetCDF generation works without active internet if required:
  ```bash
  python -m src.data_ingestion.case_studies_manager --case all
  ```
- [x] **Headless CLI Verified:** Verified that the 3 historical scripted replay scenarios run cleanly in headless mode:
  ```bash
  python scripts/run_scripted_replay.py --all
  ```
- [x] **Port Availability:** Confirmed ports `8501` and `8000` are free on the host machine.
