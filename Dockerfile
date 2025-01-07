# Use the NVIDIA CUDA base image with the full toolkit (devel)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies for Haystack and Elasticsearch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt

# Install nltk and download the stopwords during the build process
RUN python3 -m nltk.downloader stopwords

# Install PyTorch with CUDA support (make sure to match the CUDA version)
RUN pip3 install torch==2.5.0+cu124 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Install gensim for word embeddings
RUN pip3 install gensim

# Install kaggle and kagglehub
RUN pip3 install kaggle kagglehub

# Install pytest
RUN pip3 install pytest

# Temporary debug commands to check directory structure
RUN echo "Checking /usr/local/ contents..." && \
    ls /usr/local/ && \
    echo "Checking /usr/local/cuda-12.4 contents..." && \
    ls /usr/local/cuda-12.4

# Copy the application code to the container
COPY . .

# Set the entry point to keep the container alive
CMD ["tail", "-f", "/dev/null"]
