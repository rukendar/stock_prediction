# Use official lightweight Python runtime
FROM python:3.12-slim

# Install system dependencies needed for compiling numeric/ML packages like LightGBM & XGBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy dependencies manifest
COPY requirements.txt .

# Install dependencies (disable caching to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source code
COPY . .

# Ensure data raw and models registry directories exist
RUN mkdir -p data/raw models

# Expose default Flask port
EXPOSE 5000

# Set environment defaults for production
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Setup symlinks to the mounted persistent volume and run Flask using Gunicorn
CMD sh -c "mkdir -p /app/storage/data/raw /app/storage/models && rm -rf /app/data /app/models && ln -sf /app/storage/data /app/data && ln -sf /app/storage/models /app/models && exec gunicorn --workers 2 --threads 4 --bind 0.0.0.0:5000 server:app"
