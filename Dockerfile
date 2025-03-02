FROM python:3.9-slim

WORKDIR /tmp

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Set environment variables
ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    PYTHONDONTWRITEBYTECODE=1

# Install Python dependencies
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -U setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy application code
COPY . /app/

# Run gunicorn
CMD exec gunicorn \
    --preload \
    --bind :$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 0 \
    --access-logfile - \
    --error-logfile - \
    _wsgi:app
