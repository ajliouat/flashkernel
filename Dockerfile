# ══════════════════════════════════════════════════════════════════════════════
# FlashKernel — Reproducible CUDA build environment
#
# Build:   docker build -t flashkernel .
# Run:     docker run --gpus all flashkernel pytest tests/
# Dev:     docker run --gpus all -v $(pwd):/workspace -it flashkernel bash
# ══════════════════════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ─── System packages ─────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ─── Python packages ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Triton (ships with PyTorch but pin explicitly)
RUN pip install --no-cache-dir \
        triton>=2.1 \
        numpy>=1.24 \
        pytest>=7.0 \
        pytest-benchmark>=4.0 \
        pandas>=2.0

# ─── Nsight Compute (for profiling) ─────────────────────────────────────────
# Already included in nvidia/cuda:*-devel images at /usr/local/cuda/bin/ncu

# ─── Project ─────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace/

# Build the CUDA extension
RUN pip install --no-cache-dir -e ".[dev]"

# ─── Default command ─────────────────────────────────────────────────────────
CMD ["pytest", "tests/", "-v"]
