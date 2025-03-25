# Use a slim Python 3.12 image on Bullseye
FROM python:3.12-slim-bullseye

# Prevent Python from writing .pyc files and buffering outputs
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies 
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

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch @ https://download.pytorch.org/whl/cpu/torch-2.2.1+cpu-cp312-cp312-linux_x86_64.whl \
    torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.17.1+cpu-cp312-cp312-linux_x86_64.whl && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose necessary ports for Streamlit (8501) and Flask API (5000)
EXPOSE 8501 5000

# Copy and grant execution permission for the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Start both the API and the Streamlit app concurrently
CMD ["./start.sh"]