# CPU Docker image — movement + CASCADE + kinematics/calcium/sync stages
# Used for Stages 0, 3, 4, 5 on EC2 c5 or locally.
#
# Build:
#   docker build -f docker/cpu.Dockerfile -t hm2p-cpu .
#
# Run (Stage 3 example):
#   docker run -v /data:/data hm2p-cpu python -m hm2p.cli kinematics --session <id>

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    libhdf5-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

# Install core CPU-only pipeline (no suite2p, no dlc)
RUN pip install --no-cache-dir -e "."

# ── Pipeline code ──────────────────────────────────────────────────────────
COPY config/ config/
COPY metadata/ metadata/

ENV HM2P_COMPUTE_PROFILE=aws-batch

ENTRYPOINT ["python", "-m", "hm2p.cli"]
