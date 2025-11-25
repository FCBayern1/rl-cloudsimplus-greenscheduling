# ==============================================================================
# Multi-Datacenter RL Training Environment
# ==============================================================================
# This Dockerfile creates a complete environment for training hierarchical
# multi-agent RL for green cloud scheduling with CloudSim Plus.
#
# Features:
# - CUDA 12.8 + cuDNN 9 for GPU acceleration
# - Java 17 for CloudSim Plus
# - Python 3.12 with Ray RLlib
# - Multi-process support (num_workers > 0)
# - Optimized for training performance
# ==============================================================================

# Stage 1: Base image with CUDA support
# Using CUDA 12.2 with cuDNN 8 (stable and widely available)
# Compatible with PyTorch 2.x CUDA builds
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Java 17
    openjdk-17-jdk \
    # Python 3.12
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
# Python 3.12 doesn't include distutils, so we use ensurepip
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --upgrade pip setuptools wheel

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ==============================================================================
# Stage 2: Build Java CloudSim Plus Gateway
# ==============================================================================
FROM base AS java-builder

WORKDIR /build

# Copy Gradle files first (for caching)
COPY cloudsimplus-gateway/build.gradle \
     cloudsimplus-gateway/settings.gradle \
     cloudsimplus-gateway/gradlew \
     cloudsimplus-gateway/
COPY cloudsimplus-gateway/gradle/ cloudsimplus-gateway/gradle/

# Copy source code
COPY cloudsimplus-gateway/src/ cloudsimplus-gateway/src/

# Build Java project (skip tests for faster build)
# NOTE: On Windows hosts, 'gradlew' may have CRLF line endings, which breaks
# execution inside the Linux container ("./gradlew: not found").
# We normalize line endings before making it executable and running it.
WORKDIR /build/cloudsimplus-gateway
RUN sed -i 's/\r$//' gradlew && chmod +x gradlew && ./gradlew build -x test

# ==============================================================================
# Stage 3: Python dependencies
# ==============================================================================
FROM base AS python-builder

WORKDIR /build

# Copy Python requirements
COPY drl-manager/requirements_rllib.txt drl-manager/

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN python3 -m pip install --no-cache-dir \
    -r drl-manager/requirements_rllib.txt

# ==============================================================================
# Stage 4: Final runtime image
# ==============================================================================
FROM base AS runtime

# Set working directory
WORKDIR /workspace

# Copy Java build from java-builder
COPY --from=java-builder /build/cloudsimplus-gateway/ /workspace/cloudsimplus-gateway/

# Copy Python packages from python-builder
COPY --from=python-builder /usr/local/lib/python3.12/ /usr/local/lib/python3.12/
COPY --from=python-builder /usr/local/bin/ /usr/local/bin/

# Copy project files
COPY config.yml /workspace/
COPY drl-manager/ /workspace/drl-manager/

# Create necessary directories
RUN mkdir -p /workspace/logs \
    /workspace/checkpoints \
    /workspace/tensorboard

# Set permissions
RUN chmod -R 755 /workspace

# Verify installations
RUN echo "=== Verifying installations ===" \
    && java -version \
    && python3 --version \
    && python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" \
    && python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" \
    && python3 -c "import ray; print(f'Ray: {ray.__version__}')" \
    && echo "=== All verifications passed ==="

# Expose ports
# 25333: Py4J gateway
# 6006: TensorBoard
EXPOSE 25333
EXPOSE 6006

# Default environment variables (can be overridden at runtime)
ENV EXPERIMENT_ID=experiment_multi_dc_3
ENV NUM_WORKERS=4
ENV NUM_GPUS=1
ENV TOTAL_TIMESTEPS=100000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

# Start script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["train"]
