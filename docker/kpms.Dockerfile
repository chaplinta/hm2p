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

# Force JAX to CPU-only at the environment level so the CUDA plugin is never
# initialised, even if jax[cuda12] ends up installed as a transitive dep.
ENV JAX_PLATFORMS=cpu

# Install keypoint-MoSeq and all its dependencies in one step.
# kpms 0.6.8 pins jax~=0.6.2 and tensorflow-probability~=0.25.0 which must
# stay in sync — do NOT upgrade JAX independently.
RUN pip install --no-cache-dir \
    "keypoint-moseq>=0.6,<0.7" \
    "numpy<1.27" \
    "h5py>=3.0" \
    "boto3>=1.26" \
    "pandas>=1.5" \
    "tables>=3.8" \
    "pyyaml>=6.0"

# Remove CUDA JAX plugins (pulled in by kpms as hard deps) — not needed on CPU.
# Keep the kpms-compatible JAX version intact; do NOT force-reinstall jax[cpu]
# which would upgrade to an incompatible version.
RUN pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true

# ── Pipeline code (only what kpms needs) ──────────────────────────────────
COPY scripts/run_kpms.py scripts/run_kpms.py
COPY metadata/ metadata/

ENTRYPOINT ["python", "scripts/run_kpms.py"]
