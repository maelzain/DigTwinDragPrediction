# Stage 1: Base Image
FROM python:3.12-slim-bullseye AS base

# Prevent Python from writing bytecode and buffering outputs
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Stage 2: System Dependencies
FROM base AS system-deps

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Python Dependencies
FROM system-deps AS python-deps

# Upgrade pip and install core dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.1 \
    torchvision==0.17.1 && \
    pip install \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Stage 4: Application
FROM python-deps AS app

# Copy project files
COPY . .

# Create non-root user for security
RUN addgroup --system appuser && \
    adduser --system --ingroup appuser appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports for Streamlit and Flask
EXPOSE 8501 5000

# Default command (can be overridden in docker-compose)
CMD ["./start.sh"]