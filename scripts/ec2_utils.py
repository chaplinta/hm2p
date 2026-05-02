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
    aws s3 cp /var/log/hm2p-*.log s3://{bucket}/{log_prefix}/_gpu_run_log.txt 2>/dev/null || true
done) &
UPLOAD_PID=$!

# GPU watchdog: abort if 0% utilization for 30 min during processing.
# Threshold is 60 readings (at 30s intervals = 30 min) to tolerate the
# SA-finetune setup phase (HuggingFace SA-TVM weight download +
# memory-replay teacher build can take 10+ min CPU-only) and inter-session
# video downloads in the inference path.
(while true; do
    sleep 300
    [ ! -f /tmp/gpu_processing_active ] && continue
    ZERO_COUNT=$(tail -60 /var/log/gpu_monitor.csv 2>/dev/null | grep -c ', 0 %' || echo 0)
    if [ "$ZERO_COUNT" -ge 60 ]; then
        echo "FATAL: GPU utilization 0% for 30+ minutes during processing. DLC likely running on CPU."
        echo "Aborting to save money."
        aws s3 cp /var/log/gpu_monitor.csv s3://{bucket}/{log_prefix}/_gpu_monitor.csv 2>/dev/null || true
        shutdown -h now
    fi
done) &
WATCHDOG_PID=$!
echo "GPU guard started (monitor=$GPU_CSV_PID, upload=$UPLOAD_PID, watchdog=$WATCHDOG_PID)"

# === DIAGNOSTIC CAPTURE: dump py-spy stack + network + HF cache every 60s ===
# Installs py-spy lazily; on each iteration finds the run_dlc_retrain
# python pid and writes a stack dump + nvidia-smi + ss + HF cache size.
# Uploaded to S3 each iteration so we can debug hangs without SSH.
pip3 install --break-system-packages py-spy 2>&1 | tail -2 || true
apt-get install -y -qq iproute2 procps 2>/dev/null || true

(
    while true; do
        sleep 60
        PID=$(pgrep -f run_dlc_retrain.py | head -1 || true)
        {{
            echo "=== $(date -u) ==="
            if [ -n "$PID" ]; then
                echo "[pid] $PID"
                echo "[ps]"
                ps -o pid,stat,etime,pcpu,pmem,cmd -p $PID 2>/dev/null || true
                echo "[py-spy dump]"
                py-spy dump --pid $PID 2>&1 | head -80 || true
            else
                echo "[pid] (no run_dlc_retrain.py process found)"
            fi
            echo "[nvidia-smi]"
            nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
            echo "[network: established to non-loopback]"
            ss -tn state established 2>/dev/null | head -10 || true
            echo "[hf cache size]"
            du -sh /root/.cache/huggingface 2>/dev/null || true
            ls -la /root/.cache/huggingface/hub 2>/dev/null | head -10 || true
            echo
        }} >> /var/log/diagnostics.log 2>&1
        # Upload (best-effort).
        aws s3 cp /var/log/diagnostics.log s3://{bucket}/{log_prefix}/_diagnostics.log 2>/dev/null || true
    done
) &
DIAG_PID=$!
echo "Diagnostic capture started (diag=$DIAG_PID)"
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
# Disable xet Rust download backend — it stalls indefinitely on
# HuggingFace downloads from ap-southeast-2 (three consecutive instances
# hung at xet_get for 15-30 min with zero network activity).
# HF_HUB_DISABLE_XET=1 falls back to Python requests (slower but reliable).
# HF_HUB_ENABLE_HF_TRANSFER=0 disables the older hf_transfer backend too.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

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

CPU_LOG_UPLOAD_SNIPPET = """
# === Periodic log upload (CPU instance) ===
(while true; do
    sleep 300
    aws s3 cp /var/log/hm2p-downstream.log \\
        s3://{bucket}/{log_prefix}/_cpu_run_log.txt 2>/dev/null || true
done) &
CPU_UPLOAD_PID=$!
echo "Periodic log upload started (PID=$CPU_UPLOAD_PID, interval=5min)"
"""

IMDS_HELPER_SNIPPET = """
# === IMDSv2-compatible metadata fetch ===
_ec2_metadata() {
    local _TOKEN
    _TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \\
        -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null) || true
    if [ -n "$_TOKEN" ]; then
        curl -s -H "X-aws-ec2-metadata-token: $_TOKEN" \\
            "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null
    else
        curl -s "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null
    fi
}
"""

HEARTBEAT_SNIPPET = """
# === Instance heartbeat (60s interval) ===
INSTANCE_ID=$(_ec2_metadata instance-id)
LAUNCH_TIME=$(date +%s)
(while true; do
    UPTIME=$(( $(date +%s) - LAUNCH_TIME ))
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    DISK=$(df / --output=avail -BG | tail -1 | tr -d 'G ')
    MEM=$(awk '/MemFree/ {{print int($2/1024)}}' /proc/meminfo)
    printf '{{\\n  "instance_id": "%s",\\n  "instance_type": "{instance_type}",\\n  "uptime_s": %s,\\n  "timestamp": "%s",\\n  "load_avg_1m": %s,\\n  "disk_free_gb": %s,\\n  "memory_free_mb": %s\\n}}' \\
        "$INSTANCE_ID" "$UPTIME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOAD" "$DISK" "$MEM" \\
        | aws s3 cp - s3://{bucket}/{log_prefix}/{heartbeat_key} 2>/dev/null || true
    sleep 60
done) &
HEARTBEAT_PID=$!
echo "Heartbeat started (PID=$HEARTBEAT_PID)"
"""

COST_RECORD_LAUNCH_SNIPPET = """
# === Cost record — launch event ===
_CR_INSTANCE_ID=$(_ec2_metadata instance-id)
_CR_LAUNCH_TIME_EPOCH=$(date +%s)
_CR_LAUNCH_TIME_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
_CR_GIT_SHA=$(git -C /home/ubuntu/hm2p rev-parse --short HEAD 2>/dev/null || echo "unknown")
printf '{{\\n  "event": "launch",\\n  "instance_id": "%s",\\n  "instance_type": "{instance_type}",\\n  "region": "{region}",\\n  "launch_time": "%s",\\n  "pipeline_step": "{pipeline_step}",\\n  "git_sha": "%s",\\n  "mode": "{mode}"\\n}}' \\
    "$_CR_INSTANCE_ID" "$_CR_LAUNCH_TIME_ISO" "$_CR_GIT_SHA" \\
    | aws s3 cp - s3://{bucket}/{log_prefix}/{launch_key} 2>/dev/null || true
echo "Cost record launch written"
"""

COST_RECORD_SHUTDOWN_SNIPPET = """
# === Cost record — shutdown event (written in EXIT trap) ===
_CR_SHUTDOWN_TIME_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
_CR_RUNTIME_S=$(( $(date +%s) - $_CR_LAUNCH_TIME_EPOCH ))
printf '{{\\n  "event": "shutdown",\\n  "instance_id": "%s",\\n  "shutdown_time": "%s",\\n  "runtime_s": %s\\n}}' \\
    "$_CR_INSTANCE_ID" "$_CR_SHUTDOWN_TIME_ISO" "$_CR_RUNTIME_S" \\
    | aws s3 cp - s3://{bucket}/{log_prefix}/{shutdown_key} 2>/dev/null || true
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


def format_cpu_log_upload(bucket: str, log_prefix: str) -> str:
    """Format the periodic CPU log upload snippet with bucket and prefix."""
    return CPU_LOG_UPLOAD_SNIPPET.format(bucket=bucket, log_prefix=log_prefix)


def format_heartbeat(
    bucket: str,
    log_prefix: str,
    instance_type: str,
    heartbeat_key: str = "_heartbeat.json",
) -> str:
    """Format the instance heartbeat snippet.

    The returned bash snippet starts a background loop that uploads a small
    JSON to S3 every 60 seconds with instance health metrics (uptime, load,
    disk free, memory free). Failures are silenced so the loop never kills
    the main script.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    log_prefix:
        S3 key prefix (e.g. ``"dlc-retrain"``).
    instance_type:
        EC2 instance type string, embedded in the JSON (e.g. ``"c5.xlarge"``).
    heartbeat_key:
        S3 object name within ``log_prefix`` (default ``"_heartbeat.json"``).
        Pass ``"_downstream_heartbeat.json"`` for the CPU instance.
    """
    return HEARTBEAT_SNIPPET.format(
        bucket=bucket,
        log_prefix=log_prefix,
        instance_type=instance_type,
        heartbeat_key=heartbeat_key,
    )


def format_cost_record_launch(
    bucket: str,
    log_prefix: str,
    instance_type: str,
    pipeline_step: str,
    mode: str,
    region: str = "ap-southeast-2",
    launch_key: str = "_cost_record_launch.json",
) -> str:
    """Format the cost record launch snippet.

    The returned bash snippet writes a launch-event JSON to S3 immediately at
    startup, before the main processing work begins. It captures the instance
    ID from the EC2 metadata service and the git SHA from the cloned repo.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    log_prefix:
        S3 key prefix (e.g. ``"dlc-retrain"``).
    instance_type:
        EC2 instance type string (e.g. ``"g4dn.xlarge"``).
    pipeline_step:
        Human-readable step label (e.g. ``"dlc-retrain-gpu"``).
    mode:
        Run mode string (e.g. ``"train+infer"``, ``"inference only"``).
    region:
        AWS region, written into the JSON record.
    launch_key:
        S3 object name within ``log_prefix`` (default
        ``"_cost_record_launch.json"``). Pass
        ``"_downstream_cost_record_launch.json"`` for the CPU instance.
    """
    return COST_RECORD_LAUNCH_SNIPPET.format(
        bucket=bucket,
        log_prefix=log_prefix,
        instance_type=instance_type,
        pipeline_step=pipeline_step,
        mode=mode,
        region=region,
        launch_key=launch_key,
    )


def format_cost_record_shutdown(
    bucket: str,
    log_prefix: str,
    shutdown_key: str = "_cost_record_shutdown.json",
) -> str:
    """Format the cost record shutdown snippet.

    The returned bash snippet is intended for inclusion in the EXIT trap.
    It writes a shutdown-event JSON to S3 containing the instance ID,
    shutdown timestamp, and computed runtime in seconds (derived from the
    ``_CR_LAUNCH_TIME_EPOCH`` variable set by the launch snippet).

    Parameters
    ----------
    bucket:
        S3 bucket name.
    log_prefix:
        S3 key prefix (e.g. ``"dlc-retrain"``).
    shutdown_key:
        S3 object name within ``log_prefix`` (default
        ``"_cost_record_shutdown.json"``). Pass
        ``"_downstream_cost_record_shutdown.json"`` for the CPU instance.
    """
    return COST_RECORD_SHUTDOWN_SNIPPET.format(
        bucket=bucket,
        log_prefix=log_prefix,
        shutdown_key=shutdown_key,
    )


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
