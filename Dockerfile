# syntax=docker/dockerfile:1

# Base stage: use an official Python 3.12 slim image
FROM python:3.12-slim-bullseye AS base
LABEL maintainer="Your Name <youremail@example.com>"
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:${PATH}"
WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Builder stage: install Python dependencies
FROM base AS builder
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App stage: final image for running the application
FROM base AS app
# Copy installed packages from the builder stage (global installs)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy the rest of your application code
COPY . .

# Disable Streamlit's file watcher via environment variable
ENV STREAMLIT_WATCHDOG=false

# Expose the necessary ports
EXPOSE 5000
EXPOSE 8501

# Default command: run the API (this will be overridden by docker-compose for the Streamlit service)
CMD ["gunicorn", "--workers=4", "--threads=2", "--bind=0.0.0.0:5000", "api_server:app"]
