FROM python:3.10-slim

# Install system dependencies for psutil, git, and potential build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# amazon-chronos-forecasting depends on torch, transformers, etc.
# pygad for Genetic Algorithm
# psutil for system monitoring
# matplotlib, pandas, numpy for data handling and visualization
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    "git+https://github.com/amazon-science/chronos-forecasting.git" \
    pygad \
    psutil \
    matplotlib \
    pandas \
    numpy \
    scipy \
    autogluon.timeseries \
    captum

# Copy the application code
COPY . /app

# default command (can be overridden)
CMD ["python", "attack_ga.py"]
