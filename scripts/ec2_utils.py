"""Shared utilities for EC2 launch scripts.

Provides the GPU guard, hard timeout, PyTorch CUDA install, and
AWS credential handling used by all DLC/Suite2p/CASCADE launch scripts.
"""

from __future__ import annotations

# ── Bash snippets for user-data scripts ──────────────────────────────────

HARD_TIMEOUT_SNIPPET = """
# === HARD TIMEOUT ===
MAX_HOURS={max_hours}
echo "Hard timeout: ${{MAX_HOURS}}h from $(date -u)"
(sleep $((MAX_HOURS * 3600)); echo "TIMEOUT: ${{MAX_HOURS}}h reached. Terminating."; shutdown -h now) &
TIMEOUT_PID=$!
"""

GPU_GUARD_SNIPPET = """
# === GPU GUARD: continuous monitoring + abort on CPU fallback ===
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu \\
    --format=csv -l 30 > /var/log/gpu_monitor.csv 2>&1 &
GPU_CSV_PID=$!

# Upload logs to S3 every 5 minutes
(while true; do
    sleep 300
    aws s3 cp /var/log/gpu_monitor.csv s3://{bucket}/{log_prefix}/_gpu_monitor.csv 2>/dev/null || true
    aws s3 cp /var/log/hm2p-*.log s3://{bucket}/{log_prefix}/_run_log.txt 2>/dev/null || true
done) &
UPLOAD_PID=$!

# GPU watchdog: abort if 0% utilization for 5 min during processing
(while true; do
    sleep 300
    [ ! -f /tmp/gpu_processing_active ] && continue
    ZERO_COUNT=$(tail -10 /var/log/gpu_monitor.csv 2>/dev/null | grep -c ', 0 %' || echo 0)
    if [ "$ZERO_COUNT" -ge 10 ]; then
        echo "FATAL: GPU utilization 0% for 5+ minutes during processing. DLC running on CPU."
        echo "Aborting to save money."
        aws s3 cp /var/log/gpu_monitor.csv s3://{bucket}/{log_prefix}/_gpu_monitor.csv 2>/dev/null || true
        shutdown -h now
    fi
done) &
WATCHDOG_PID=$!
echo "GPU guard started (monitor=$GPU_CSV_PID, upload=$UPLOAD_PID, watchdog=$WATCHDOG_PID)"
"""

PYTORCH_CUDA_INSTALL_SNIPPET = """
# === PyTorch + DLC install (CUDA verified) ===
# Step 1: CUDA PyTorch from official index (NOT pip default which pulls CPU-only)
pip3 install --break-system-packages \\
    torch torchvision torchaudio \\
    --index-url https://download.pytorch.org/whl/cu121

# Step 2: DLC without deps (avoids overwriting CUDA PyTorch)
pip3 install --break-system-packages --quiet --pre --no-deps deeplabcut

# Step 3: Remaining DLC deps (excluding torch*)
pip3 install --break-system-packages --quiet \\
    pandas numpy scipy matplotlib \\
    timm dlclibrary tables huggingface_hub scikit-image scikit-learn \\
    filterpy numba imgaug segment-anything pyyaml 2>/dev/null || true

# Step 4: HARD VERIFY — abort if CUDA not working
python3 -c "
import torch
assert torch.cuda.is_available(), 'FATAL: CUDA not available'
t = torch.randn(100, 100, device='cuda')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA tensor test: OK')
import deeplabcut
print(f'DLC {deeplabcut.__version__}')
" || { echo "FATAL: PyTorch/CUDA/DLC verification failed. Aborting."; exit 1; }
"""

DPKG_WAIT_SNIPPET = """
# === Wait for apt locks (DL AMI runs unattended-upgrades on boot) ===
export DEBIAN_FRONTEND=noninteractive
echo "Waiting for apt locks..."
for i in $(seq 1 60); do
    if apt-get check >/dev/null 2>&1; then
        echo "Locks free"
        break
    fi
    echo "  attempt $i: locked, waiting 15s..."
    sleep 15
done
"""

APT_INSTALL_SNIPPET = """
apt-get update -qq
apt-get install -y -qq awscli ffmpeg git
"""


def format_gpu_guard(bucket: str, log_prefix: str) -> str:
    """Format the GPU guard snippet with bucket and prefix."""
    return GPU_GUARD_SNIPPET.format(bucket=bucket, log_prefix=log_prefix)


def format_hard_timeout(max_hours: int) -> str:
    """Format the hard timeout snippet."""
    return HARD_TIMEOUT_SNIPPET.format(max_hours=max_hours)


def build_creds_block(key_id: str, secret: str, region: str = "ap-southeast-2") -> str:
    """Build the AWS credentials block for user-data."""
    return f"""
mkdir -p /root/.aws
cat > /root/.aws/credentials << 'CREDS'
[default]
aws_access_key_id = {key_id}
aws_secret_access_key = {secret}
CREDS
cat > /root/.aws/config << 'CONF'
[default]
region = {region}
output = json
CONF
"""


def get_s3_credentials() -> tuple[str, str, str]:
    """Read AWS credentials from ~/.aws/credentials."""
    import configparser
    config = configparser.ConfigParser()
    config.read("/home/node/.aws/credentials")
    key_id = config["default"]["aws_access_key_id"]
    secret = config["default"]["aws_secret_access_key"]
    return key_id, secret, "ap-southeast-2"
