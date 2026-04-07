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

# GPU watchdog: abort if 0% utilization for 10 min during processing.
# Threshold is 20 readings (at 30s intervals = 10 min) to allow for
# inter-session gaps (video download ~2-3 min between inference runs).
(while true; do
    sleep 300
    [ ! -f /tmp/gpu_processing_active ] && continue
    ZERO_COUNT=$(tail -20 /var/log/gpu_monitor.csv 2>/dev/null | grep -c ', 0 %' || echo 0)
    if [ "$ZERO_COUNT" -ge 20 ]; then
        echo "FATAL: GPU utilization 0% for 10+ minutes during processing. DLC likely running on CPU."
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

# Step 2: Install DLC with all deps (--pre for 3.0rc)
# PyTorch CUDA was installed in Step 1 with --index-url pinning.
# DLC's pip install may pull CPU torch — we re-verify CUDA afterwards.
pip3 install --break-system-packages --quiet --pre deeplabcut

# Step 3: Re-install CUDA PyTorch if DLC overwrote it
pip3 install --break-system-packages \\
    torch torchvision torchaudio \\
    --index-url https://download.pytorch.org/whl/cu121

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
    """Read AWS credentials from ~/.aws/credentials.

    Checks ``hm2p-agent`` profile first, then ``default``. Works on both
    macOS and Linux (including the devcontainer) by using ``Path.home()``.

    Returns
    -------
    tuple[str, str, str]
        ``(aws_access_key_id, aws_secret_access_key, region)``

    Raises
    ------
    SystemExit
        If the credentials file is missing or no usable profile is found.
    """
    import configparser
    from pathlib import Path

    creds_path = Path.home() / ".aws" / "credentials"
    if not creds_path.exists():
        raise SystemExit(
            f"AWS credentials file not found at {creds_path}. "
            "Run 'aws configure' or create ~/.aws/credentials manually."
        )

    config = configparser.ConfigParser()
    config.read(creds_path)

    for profile in ("hm2p-agent", "default"):
        if profile in config:
            key_id = config[profile].get("aws_access_key_id", "")
            secret = config[profile].get("aws_secret_access_key", "")
            if key_id and secret:
                return key_id, secret, "ap-southeast-2"

    raise SystemExit(
        f"No usable AWS credentials found in {creds_path} "
        "(checked profiles: hm2p-agent, default)."
    )
