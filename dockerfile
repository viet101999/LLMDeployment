# Use an official Python image as the base
FROM nvidia/cuda:12.5.0-base-ubuntu20.04

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install required system libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

    # Upgrade pip and set up Python environment
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir psutil
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir -U transformers
RUN pip install --no-cache-dir evaluate
RUN pip install --no-cache-dir prometheus-client
RUN pip install --no-cache-dir accelerate
RUN pip install --no-cache-dir -U bitsandbytes


# Copy application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
