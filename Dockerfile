# ============================================================
# BrainVerse — Brain Tumor Classification App
# Single-container Docker image (FastAPI + Frontend)
# CPU-only build (GPU pass-through via docker-compose override)
# ============================================================

# Use slim Python 3.12 to match the development environment
FROM python:3.12-slim

# ---------- labels -----------------------------------------------------------
LABEL maintainer="BrainVerse"
LABEL description="BrainVerse Brain MRI Tumor Classifier – FastAPI + DenseNet121 + MONAI"

# ---------- system deps ------------------------------------------------------
# libgomp1 is needed by PyTorch for OpenMP parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---------- working directory -------------------------------------------------
WORKDIR /app

# ---------- Python deps (cached layer) ----------------------------------------
# Copy requirements first to leverage Docker layer cache
COPY requirements.txt .

# Upgrade pip, then install Python dependencies
# Use --extra-index-url so PyPI packages (fastapi, etc.) and PyTorch CPU wheels can both be found
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi==0.142.2 \
        uvicorn==0.54.0 \
        starlette==1.7.0 \
        python-multipart==0.0.32 \
        Pillow==12.3.0 \
        numpy==2.5.3 \
        monai==1.6.1 \
        scikit-learn==1.9.1

# ---------- copy application source ------------------------------------------
# model.py and related inference code
COPY model.py .

# Backend API
COPY backend/ ./backend/

# Frontend (HTML/CSS/JS + assets served as static files)
COPY frontend/ ./frontend/

# Image carousel and logo assets
COPY assets/ ./assets/

# Trained model weights (28 MB .pth file)
COPY best_brain_tumor_monai.pth .

# ---------- runtime config ---------------------------------------------------
# Expose the port Uvicorn will listen on
EXPOSE 8000

# Environment: keep Python output unbuffered so logs appear in real-time
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------- entrypoint -------------------------------------------------------
# Run Uvicorn from the project root so that sys.path.append in main.py can
# find model.py correctly. The --host 0.0.0.0 is required inside Docker.
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

