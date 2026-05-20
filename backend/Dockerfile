FROM python:3.10.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install cmake==3.22.6

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir -r /app/requirements.txt

RUN python -m pip install --no-build-isolation dlib==19.24.2

RUN python -m pip install uvicorn[standard]==0.35.0

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]