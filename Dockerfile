# ============================================================
# Dockerfile — Video Ad Classifier Backend (CPU baseline)
# ============================================================
#
# Build:
#   docker build -t ad-classifier-backend .
#
# CUDA variant:
#   Change FROM line to pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
#   and install torch with CUDA extras:
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#
# Apple Silicon (MPS):
#   MPS is not available inside Docker; use direct venv setup on host.
#   See docs/technical/09-deployment-and-runbooks.md → "Local Development".
# ============================================================

FROM python:3.11-slim AS base

# System deps for OpenCV, EasyOCR, yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt requirements-service.txt* ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn[standard] fastapi httpx

# Copy application source
COPY video_service/ ./video_service/
COPY cluster_config.json* ./
COPY categories.csv* ./

# Runtime configuration (all overridable at docker run / compose)
ENV NODE_NAME=node-a \
    PORT=8000 \
    DATABASE_PATH=/data/video_service.db \
    UPLOAD_DIR=/tmp/video_service_uploads \
    ARTIFACTS_DIR=/data/artifacts \
    LOG_LEVEL=INFO \
    CORS_ORIGINS=http://localhost:5173

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn video_service.app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level ${LOG_LEVEL,,} \
    --workers 1
