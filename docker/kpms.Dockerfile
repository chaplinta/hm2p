# keypoint-MoSeq Docker image — isolated environment for syllable discovery.
#
# keypoint-MoSeq pins numpy<=1.26 and JAX dependencies that conflict with
# the main hm2p environment. This container runs kpms in isolation, reading
# DLC .h5 files from S3 and writing syllable arrays back.
#
# Build:
#   docker build -f docker/kpms.Dockerfile -t hm2p-kpms .
#
# Run:
#   docker run -v /data:/data hm2p-kpms python scripts/run_kpms.py \
#       --project-dir /data/kpms --output-dir /data/syllables \
#       --dlc-dir /data/pose
#
# On EC2 with S3:
#   docker run hm2p-kpms python scripts/run_kpms.py \
#       --s3-bucket hm2p-derivatives --all-sessions

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────
WORKDIR /app

# Install keypoint-MoSeq and its dependencies in isolation
# Pin numpy to what kpms requires
RUN pip install --no-cache-dir \
    "keypoint-moseq>=0.6" \
    "numpy<1.27" \
    "h5py>=3.0" \
    "boto3>=1.26" \
    "pandas>=1.5" \
    "tables>=3.8"

# ── Pipeline code (only what kpms needs) ──────────────────────────────────
COPY scripts/run_kpms.py scripts/run_kpms.py
COPY metadata/ metadata/

ENTRYPOINT ["python", "scripts/run_kpms.py"]
