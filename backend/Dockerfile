FROM python:3.10.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    WEB_CONCURRENCY=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --root-user-action=ignore --upgrade pip setuptools wheel \
    && pip install --root-user-action=ignore -r /app/backend/requirements.txt \
    && pip install --root-user-action=ignore --no-deps face_recognition==1.3.0 \
    && pip uninstall -y opencv-python opencv-contrib-python || true

RUN python -m uvicorn --version

COPY backend /app/backend

WORKDIR /app/backend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:api --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
