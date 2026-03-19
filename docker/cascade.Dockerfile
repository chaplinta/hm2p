# CASCADE spike inference — requires Python 3.8 + TensorFlow 2.3
#
# CASCADE (Rupprecht et al. 2021, Nature Neuroscience) infers calibrated
# spike rates (spikes/s) from dF/F0 traces using pre-trained deep learning
# models matched to GCaMP indicator and frame rate.
#
# This runs as a standalone container because CASCADE's TensorFlow 2.3
# dependency is incompatible with the main project environment.
#
# Usage:
#   docker build -f docker/cascade.Dockerfile -t hm2p-cascade .
#   docker run -v ~/.aws:/root/.aws:ro hm2p-cascade \
#       python scripts/run_cascade.py --all
#
# Reference:
#   Rupprecht et al. 2021. "A database and deep learning toolbox for
#   noise-optimized, generalized spike inference from calcium imaging."
#   Nature Neuroscience 24:1324-1337. doi:10.1038/s41593-021-00895-5
#   https://github.com/HelmchenLabSoftware/Cascade

FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install CASCADE and dependencies
RUN pip install --no-cache-dir \
    cascade2p \
    tensorflow==2.3.4 \
    numpy==1.21.6 \
    scipy==1.7.3 \
    h5py==3.7.0 \
    boto3 \
    tqdm

WORKDIR /app

# Copy only the scripts needed
COPY scripts/run_cascade.py /app/scripts/run_cascade.py
COPY metadata/ /app/metadata/

CMD ["python", "scripts/run_cascade.py", "--all"]
