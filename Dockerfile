FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (simplified - no pytorch-forecasting)
RUN pip install --no-cache-dir \
    torch \
    pygad \
    psutil \
    matplotlib \
    pandas \
    numpy \
    scipy

# Copy the application code
COPY . /app

# Default: train the model first
CMD ["python", "train_deepar.py"]
