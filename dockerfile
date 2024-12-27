# Base image with Python and CUDA
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git && \
    apt-get clean

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
