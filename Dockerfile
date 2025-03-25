# Use a slim Python 3.10 image on Bullseye
FROM python:3.10-slim-bullseye

# Prevent Python from writing .pyc files and buffering outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (including python3-distutils and python3-setuptools)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    python3-distutils \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt .

# Upgrade pip and install torch and torchvision (CPU-only) from the official wheels,
# then install the remaining dependencies from requirements.txt using only binary wheels.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.1+cpu torchvision==0.17.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose necessary ports for Streamlit (8501) and Flask API (5000)
EXPOSE 8501 5000

# Copy and grant execution permission for the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Start both the API and the Streamlit app concurrently
CMD ["./start.sh"]
