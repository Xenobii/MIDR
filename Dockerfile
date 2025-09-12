# Use an official PyTorch image as base
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (including git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libsndfile1 \
    ffmpeg \
    tzdata \
    && ln -sf /usr/share/zoneinfo/UTC /etc/localtime \
    && echo "UTC" > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire repository (avoids needing git inside container)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for data
RUN mkdir /data

COPY . . 

# Reset DEBIAN_FRONTEND to default (recommended)
ENV DEBIAN_FRONTEND=

# Set the default command
CMD ["/bin/bash"]