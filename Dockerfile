# Stage 1: Base Image with Python 3.12
FROM python:3.12-slim-bullseye AS base

# Metadata and maintainer information
LABEL maintainer="Digital Twin Research Team"
LABEL version="1.0.0"
LABEL description="Digital Twin Drag Prediction Application"

# Prevent Python from writing bytecode and buffering outputs
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Stage 2: System Dependencies
FROM base AS system-deps

# Install system dependencies with minimal layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Python Dependencies
FROM system-deps AS python-deps

# Upgrade pip and install core dependencies
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

# Create a non-root user for security
RUN addgroup --system --gid 1001 appuser \
    && adduser \
    --system \
    --disabled-password \
    --home /app \
    --shell /bin/bash \
    --uid 1001 \
    --ingroup appuser \
    appuser

# Copy only requirements to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies with optimized PyTorch and other requirements
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.1 \
    torchvision==0.17.1 \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt \
    --no-cache-dir \
    && pip check

# Stage 4: Application Image
FROM python-deps AS app

# Copy project files with proper ownership
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models \
    && chown -R appuser:appuser /app/logs /app/data /app/models

# Set permissions and switch to non-root user
RUN chmod +x start.sh && \
    chown appuser:appuser start.sh

# Switch to non-root user for enhanced security
USER appuser

# Expose application ports
EXPOSE 8501 5000

# Performance and resource optimizations
ENV PYTHONHASHSEED=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Healthcheck to monitor container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:5000', timeout=5)" || exit 1

# Default command
CMD ["./start.sh"]

# Optional build-time arguments
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"