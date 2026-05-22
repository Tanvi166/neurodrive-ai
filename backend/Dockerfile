FROM python:3.10.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        liblapack-dev \
        libx11-6 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libpng-dev \
        libjpeg-dev \
        libatlas-base-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --root-user-action=ignore --upgrade pip setuptools wheel \
    && pip install --root-user-action=ignore cmake==3.22.6

# install dlib ONLY ONCE
RUN pip install --root-user-action=ignore --no-build-isolation dlib==19.24.6

COPY requirements.txt /app/requirements.txt

RUN pip install --root-user-action=ignore --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/requirements.txt

# install face_recognition WITHOUT dlib dependency
RUN pip install --root-user-action=ignore --no-deps face_recognition==1.3.0

RUN pip uninstall -y opencv-python opencv-contrib-python \
                     opencv-python-headless 2>/dev/null || true \
    && pip install --root-user-action=ignore --no-cache-dir opencv-python-headless==4.8.1.78

RUN pip install --root-user-action=ignore --no-cache-dir "uvicorn[standard]==0.35.0"

RUN python -m uvicorn --version

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]