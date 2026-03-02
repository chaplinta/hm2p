# GPU Docker image — Suite2p + DeepLabCut + CUDA
# Used for Stage 1 (2P extraction) and Stage 2 (pose estimation) on EC2 g4dn.
#
# Build:
#   docker build -f docker/gpu.Dockerfile -t hm2p-gpu .
#
# Run (Stage 1 example):
#   docker run --gpus all -v /data:/data hm2p-gpu python -m hm2p.cli extraction --session <id>

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── System dependencies ────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

# ── Python dependencies ────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

# Install core + GPU-dependent extras
RUN pip install --no-cache-dir ".[suite2p,dlc]"

# ── Pipeline code ──────────────────────────────────────────────────────────
COPY config/ config/
COPY metadata/ metadata/

ENV HM2P_COMPUTE_PROFILE=aws-batch

ENTRYPOINT ["python", "-m", "hm2p.cli"]
