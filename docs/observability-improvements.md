# Observability Improvements — DLC Pipeline

**Date:** 2026-04-02
**Scope:** `launch_dlc_finetune_ec2.py`, `launch_downstream_cpu.py`,
`run_dlc_retrain.py`, `run_downstream_pipeline.py`, `ec2_utils.py`,
`render_dlc_videos.py`
**Context:** Seven categories of reliability and observability failure have been
identified in production runs: silent log upload failures, no mid-run visibility
into the CPU instance, undetected S3 upload failures after kinematics/sync/analysis,
watchdog kills during inter-session gaps (since fixed in ec2_utils.py), coarse
progress tracking, inability to diagnose crashes, and no post-hoc cost attribution.
This document describes the improvements needed. None are implemented yet.

---

## 1. Periodic Log Uploads — Both Instances

**Priority:** Critical  
**Effort:** ~30 min

### Problem

The GPU instance (`launch_dlc_finetune_ec2.py`) already has a periodic upload
loop via `GPU_GUARD_SNIPPET` in `ec2_utils.py` (lines 27–30). This uploads the
GPU monitor CSV and the run log every 5 minutes while the instance is alive.

The CPU instance (`launch_downstream_cpu.py`) has no periodic upload. The `trap`
on EXIT uploads the log only when the instance shuts down. For a CPU run that
processes 26 sessions of kinematics + sync + analysis + video rendering, this
means no log visibility for potentially 4–6 hours. When an instance crashes or
hangs, the only diagnostic output is whatever was already in the console at
launch time.

### Fix

**File:** `ec2_utils.py`  
Add a new `CPU_LOG_UPLOAD_SNIPPET` constant alongside the existing
`GPU_GUARD_SNIPPET`. This snippet starts a background loop that uploads the log
file every 5 minutes:

```bash
# === Periodic log upload (CPU instance) ===
(while true; do
    sleep 300
    aws s3 cp /var/log/hm2p-downstream.log \
        s3://{bucket}/{log_prefix}/_downstream_log.txt 2>/dev/null || true
done) &
CPU_UPLOAD_PID=$!
echo "Periodic log upload started (PID=$CPU_UPLOAD_PID, interval=5min)"
```

**File:** `ec2_utils.py`  
Add a `format_cpu_log_upload(bucket: str, log_prefix: str) -> str` function
analogous to `format_gpu_guard()`.

**File:** `launch_downstream_cpu.py`, `build_user_data()`  
Import and call `format_cpu_log_upload(DERIVATIVES_BUCKET, "dlc-retrain")` in
the user-data script, placed immediately after the credentials block and before
`set -ex`. Place it before the main processing work so the upload daemon is
running even if the apt install step fails.

**Verification:** After deploying, run the CPU instance and confirm that
`_downstream_log.txt` on S3 is updated at 5-minute intervals by checking
`LastModified` via `aws s3 ls`.

---

## 2. Per-Session S3 Upload Verification

**Priority:** Critical  
**Effort:** ~45 min

### Problem

`run_downstream_pipeline.py` calls `run_stage3()`, `run_stage5()`, `run_stage6()`
as subprocesses. Each stage script uploads its output (kinematics.h5, sync.h5,
analysis.h5) to S3. If the upload fails — for example due to an SSL handshake
error or a temporary S3 outage — the subprocess still exits with return code 0
(the upload failure is caught by a `try/except` inside the stage script that
logs the error but does not re-raise it). `run_downstream_pipeline.py` marks the
stage as successful, progress JSON is updated, and no error is reported.

This is the exact pattern that caused Stage 3 kinematics to appear "succeeded"
while the kinematics.h5 file never made it to S3.

### Fix

**File:** `run_downstream_pipeline.py`, functions `run_stage3()`, `run_stage5()`,
`run_stage6()`  
After each stage subprocess completes successfully (return code 0), verify that
the expected S3 key exists using `s3.head_object()`. If the key is missing, retry
the stage subprocess once before marking it failed. Do not catch exceptions from
the verification call — let them propagate as a hard failure.

The verification logic should be a shared helper:

```python
def _verify_s3_upload(
    s3,
    sub: str,
    ses: str,
    stage_prefix: str,
    expected_filename: str,
    *,
    retries: int = 2,
    retry_delay_s: int = 30,
) -> bool:
    """Verify expected_filename exists in stage_prefix/sub/ses/ on S3.
    Returns True if found within retries attempts, False otherwise.
    """
```

The retry loop should wait `retry_delay_s` seconds between checks, since S3
list consistency can lag by a few seconds after a fresh upload.

**File:** `run_stage3_kinematics.py` (when implemented)  
The stage script itself should also verify the S3 key immediately after calling
`s3.upload_file()` and raise a `RuntimeError` if the verify fails, so the
subprocess exits with a non-zero code. This means upload failures are caught
both inside the stage and outside in the orchestrator.

**Affected keys to verify:**

| Stage | Expected S3 key pattern |
|-------|------------------------|
| Stage 3 | `kinematics/{sub}/{ses}/kinematics.h5` |
| Stage 5 | `sync/{sub}/{ses}/sync.h5` |
| Stage 6 | `analysis/{sub}/{ses}/analysis.h5` |

**Files:** `render_dlc_videos.py`, `render_session()`  
After `s3.upload_file(str(out_path), DERIV_BUCKET, upload_key)` (line 559),
add a `s3.head_object()` call to confirm the key landed. If it fails, log a
warning and append the session to a failed list rather than silently continuing.

---

## 3. Progress Granularity

**Priority:** High  
**Effort:** ~40 min

### Problem

`run_dlc_retrain.py`'s `update_progress()` (line 33–44) writes a single
`_retrain_progress.json` to S3 and is called at coarse-grained points:
- "Training: creating dataset"
- "Training: HRNet-W32 (N epochs)"
- "Training complete"
- "Inference N/26: sub/ses"
- "Inference complete"
- "Promoted to pose/"

The per-session progress update (`"Inference N/26"`) fires at the *start* of each
session, not at completion. A session that takes 15 minutes to process provides no
intermediate signal — the operator sees "Inference 5/26" and cannot tell if session
5 is running, hanging, or whether it already completed and moved to 6.

`run_downstream_pipeline.py` has no `update_progress()` at all. For the CPU
downstream run, the only progress signal is the periodic log upload (once that is
implemented). There is no structured progress JSON showing per-stage per-session
completion.

### Fix

**File:** `run_dlc_retrain.py`, `infer()`  
Change the `update_progress()` call to fire at session *completion* rather than
at the start of each session. Add a second call at the start for "started" state
so the operator knows the session is running:

```python
# At start of session loop iteration:
update_progress(s3, f"Inference {i}/{total}: starting {sub}/{ses}",
                completed=len(completed), failed=len(failed), total=total,
                current_session=exp_id)

# At end, after upload:
update_progress(s3, f"Inference {i}/{total}: done {sub}/{ses}",
                completed=len(completed), failed=len(failed), total=total,
                current_session=exp_id, stage="inference_done")
```

Also add stage-level granularity within training: add `update_progress()` calls
after `create_training_dataset()`, after `train_network()`, and after
`evaluate_network()` so each training sub-phase is visible.

**File:** `run_downstream_pipeline.py`  
Add progress updates at each stage boundary using an `update_downstream_progress()`
function that writes a separate key `dlc-retrain/_downstream_progress.json`:

```python
def update_downstream_progress(s3, session_idx: int, total: int, exp_id: str,
                                stage: str, status: str, results: dict) -> None:
    """Write per-stage progress to S3."""
```

Call this after each of `run_stage3()`, `run_stage5()`, `run_stage6()` returns,
passing the stage name and boolean success. This gives a structured view of
exactly where each session is in the pipeline.

**File:** `launch_downstream_cpu.py`  
The inline Python one-liner that writes the final progress JSON (lines 83–94)
should be replaced with a call to an importable function so it can be unit-tested
and extended. The inline approach is fragile and hard to read.

---

## 4. Error Aggregation

**Priority:** High  
**Effort:** ~50 min

### Problem

Errors from failed sessions are currently collected in a `failed` list in
`run_dlc_retrain.py` and written to the final progress JSON as `failed_sessions`.
This is good. However:

1. There are no stack traces — only `print(f"  ERROR: {e}")` before
   `failed.append(exp_id)` (line 396). The exception type and full traceback are
   not captured.
2. The CPU downstream pipeline (`run_downstream_pipeline.py`) has no error
   collection at all. `run_stage3()` captures `result.stderr[:500]` and prints
   it, but that truncated stderr is never written to S3. There is no summary
   JSON for the downstream run.
3. `render_dlc_videos.py` calls `log.exception("Failed to process %s", exp_id)`
   (line 629) which logs the traceback locally but does not write a failure
   summary to S3.

After a run, diagnosing failures requires SSHing to the instance (if it is still
alive) or reading the uploaded log line by line to find per-session errors.

### Fix

**File:** `run_dlc_retrain.py`, `infer()`  
Replace bare `except Exception as e: print(...)` with a structured error record:

```python
except Exception as e:
    import traceback
    error_record = {
        "session": exp_id,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "traceback": traceback.format_exc(),
        "stage": "inference",
    }
    error_records.append(error_record)
    failed.append(exp_id)
    print(f"  ERROR [{type(e).__name__}]: {e}")
```

At the end of `infer()`, upload `error_records` to
`dlc-retrain/_inference_errors.json` via `s3.put_object()` (not `upload_file`,
to avoid needing a temp file). This JSON is always written, even if empty, so
the frontend can distinguish "no errors" from "error file missing".

**File:** `run_downstream_pipeline.py`  
Add an `error_records: list[dict]` to `main()` that collects failures from each
`process_session()` call, including the partial stderr from the failed subprocess.
Write these to `dlc-retrain/_downstream_errors.json` at the end of the run,
using the same schema as `_inference_errors.json`.

**File:** `render_dlc_videos.py`, `main()`  
After the session loop, write a `_render_errors.json` to S3 listing all sessions
where `result is None`, including the exception message. The `log.exception()`
call already has the traceback — capture it with `traceback.format_exc()` in the
except block.

**Common error JSON schema:**

```json
{
  "run_id": "ISO8601 timestamp of run start",
  "instance_id": "i-0abcdef1234567890",
  "errors": [
    {
      "session": "20210823_16_59_50_1114353",
      "stage": "inference | stage3 | stage5 | stage6 | render",
      "error_type": "SSLError",
      "error_message": "...",
      "traceback": "Traceback (most recent call last):\n  ...",
      "timestamp": "2026-04-02T14:23:01Z"
    }
  ]
}
```

The instance ID should be retrieved at startup via the EC2 metadata service:
`http://169.254.169.254/latest/meta-data/instance-id` using `urllib.request`.
This links error reports back to the specific instance that produced them.

---

## 5. Instance Health Monitoring — Heartbeat

**Priority:** High  
**Effort:** ~25 min

### Problem

There is no way to tell from outside an instance whether it is alive and making
progress, vs hanging, vs crashed silently. The GPU instance uploads logs every
5 minutes, but a log upload succeeding does not mean the Python process is still
running — the upload daemon is a separate background process. The CPU instance
uploads nothing until EXIT.

The frontend shows a "last updated" timestamp from the progress JSON, but this
only updates at session boundaries. Between sessions, or during a long stage like
video rendering, the timestamp can be 10–30 minutes stale without indicating a
failure.

### Fix

**File:** `ec2_utils.py`  
Add a `HEARTBEAT_SNIPPET` that uploads a small heartbeat JSON to S3 every 60
seconds, independent of the log upload loop. The heartbeat JSON contains:

```json
{
  "instance_id": "i-0abc...",
  "instance_type": "c5.xlarge",
  "uptime_s": 3421,
  "timestamp": "2026-04-02T14:23:01Z",
  "load_avg_1m": 3.2,
  "disk_free_gb": 72.4,
  "memory_free_mb": 2048
}
```

The snippet is a background bash loop:

```bash
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
LAUNCH_TIME=$(date +%s)
(while true; do
    UPTIME=$(( $(date +%s) - LAUNCH_TIME ))
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    DISK=$(df / --output=avail -BG | tail -1 | tr -d 'G ')
    MEM=$(awk '/MemFree/ {print int($2/1024)}' /proc/meminfo)
    printf '{
  "instance_id": "%s",
  "instance_type": "{instance_type}",
  "uptime_s": %s,
  "timestamp": "%s",
  "load_avg_1m": %s,
  "disk_free_gb": %s,
  "memory_free_mb": %s
}' "$INSTANCE_ID" "$UPTIME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOAD" "$DISK" "$MEM" \
    | aws s3 cp - s3://{bucket}/{log_prefix}/_heartbeat.json 2>/dev/null || true
    sleep 60
done) &
HEARTBEAT_PID=$!
echo "Heartbeat started (PID=$HEARTBEAT_PID)"
```

Add `format_heartbeat(bucket: str, log_prefix: str, instance_type: str) -> str`
to `ec2_utils.py`.

**Files:** `launch_dlc_finetune_ec2.py`, `launch_downstream_cpu.py`  
Call `format_heartbeat()` in `build_user_data()` for both instances, using
appropriate `log_prefix` values (`"dlc-retrain"` for GPU, `"dlc-retrain"` for
CPU with a different key like `"_downstream_heartbeat.json"`).

**Frontend:** Add a check in the pipeline status page. If the heartbeat JSON
exists and its `timestamp` is more than 3 minutes old, show "Instance stale or
dead". If the heartbeat JSON does not exist, show "Not started". If it is fresh,
show "Instance alive" with uptime and disk free. The threshold should be 3 minutes
(3x the 60-second update interval) to allow for S3 eventual consistency and brief
network hiccups.

---

## 6. Cost Tracking

**Priority:** Medium  
**Effort:** ~30 min

### Problem

There is no record of which instance type ran which pipeline step, when it started,
or when it terminated. Post-hoc cost analysis requires querying AWS Cost Explorer
or CloudWatch, which requires correlating instance IDs and time ranges without any
pipeline-level metadata. The current EC2 tags (`Name`, `Project`) allow filtering
by project but not by pipeline run.

### Fix

**File:** `ec2_utils.py` or a new `run_cost_record.py`  
Add a function that writes a `_cost_record.json` to S3 at two points in the
run: immediately at startup (with launch time and instance metadata), and at
shutdown (with termination time and computed runtime seconds).

Startup record (written before `set -ex`):

```json
{
  "event": "launch",
  "instance_id": "i-0abc...",
  "instance_type": "g4dn.xlarge",
  "region": "ap-southeast-2",
  "launch_time": "2026-04-02T14:00:00Z",
  "pipeline_step": "dlc-retrain-gpu",
  "git_sha": "abc1234",
  "mode": "train+infer"
}
```

Shutdown record (appended via `trap`):

```json
{
  "event": "shutdown",
  "instance_id": "i-0abc...",
  "shutdown_time": "2026-04-02T18:32:00Z",
  "runtime_s": 16320,
  "sessions_completed": 24,
  "sessions_failed": 2
}
```

These are stored at a stable key per run
(`dlc-retrain/_cost_record_launch.json` and
`dlc-retrain/_cost_record_shutdown.json`). The frontend cost page or a local
script can read these to compute instance-hours and estimated cost without
querying Cost Explorer.

**File:** `launch_dlc_finetune_ec2.py`, `build_user_data()`  
Embed the EC2 instance type in the cost record. The instance type is known at
build time (`INSTANCE_TYPE = "g4dn.xlarge"`) and can be interpolated directly
into the user-data snippet.

**File:** `launch_downstream_cpu.py`, `build_user_data()`  
Same pattern for the CPU instance (`INSTANCE_TYPE = "c5.xlarge"`), writing to
`_downstream_cost_record_launch.json` and `_downstream_cost_record_shutdown.json`.

**Pricing table for reference:** Current on-demand prices in `ap-southeast-2`:

| Instance | USD/hr |
|----------|--------|
| g4dn.xlarge | 0.736 |
| g5.xlarge | 1.408 |
| c5.xlarge | 0.214 |

These should be stored in `ec2_constants.py` as `INSTANCE_PRICES: dict[str, float]`
so the cost page can multiply `runtime_s / 3600 * price` without hardcoding.

---

## 7. Alerting — What Should Trigger a Notification

**Priority:** Medium  
**Effort:** ~2 hr (depending on notification mechanism chosen)

### Conditions That Should Trigger an Alert

The following conditions indicate a run has failed or is in an unrecoverable
state, and require human attention:

**Immediate/critical:**

| Condition | Detection mechanism | Threshold |
|-----------|-------------------|-----------|
| All sessions failed | `_inference_errors.json` has `len(errors) == 26` | At run end |
| Instance heartbeat stale | `_heartbeat.json` `timestamp` > 5 min old | Continuous |
| GPU watchdog abort | `_run_log.txt` contains "FATAL: GPU utilization 0% for 10+ minutes" | At log upload |
| Hard timeout triggered | `_run_log.txt` contains "TIMEOUT: 24h reached" | At log upload |
| No sessions started after 30 min | `_retrain_progress.json` `updated` timestamp unchanged for 30 min | Polling |

**Post-run review required:**

| Condition | Detection mechanism |
|-----------|-------------------|
| Any sessions failed (not all) | `_retrain_progress.json` `failed > 0` at "Inference complete" |
| S3 upload verification failed | `_inference_errors.json` error_type contains "upload" |
| Render failures > 0 | `_render_errors.json` exists and `errors` non-empty |
| CPU instance stale | `_downstream_heartbeat.json` timestamp > 10 min old during known CPU run |
| Downstream stage failed | `_downstream_errors.json` non-empty |

### Implementation Options

**Option A — Email via SNS (recommended for this project):**  
Create an SNS topic `hm2p-pipeline-alerts`. A lightweight polling Lambda (or a
local `scripts/poll_pipeline_health.py` run with `uv run python ... --watch`)
checks the heartbeat and progress JSON every 2 minutes and publishes to SNS on
the conditions above. SNS delivers to email. Lambda cost is negligible (~$0.01/run).

**Option B — Local poller + terminal bell:**  
Add a `--watch` mode to `launch_dlc_finetune_ec2.py` that polls S3 every 2
minutes and prints a terminal alert (with `\a` bell character) when any critical
condition is detected. No AWS dependencies beyond the existing S3 access.
Simpler to implement, but only works while the terminal is open.

**Option C — Streamlit frontend alert banner:**  
The pipeline status page already polls for progress. Add a red alert banner at
the top of the page when critical conditions are detected. This requires the
frontend to be open and refreshed, but no additional infrastructure.

**Recommendation:** Implement Option B first (low effort, immediate value) and
Option C alongside it. Defer Option A until the pipeline is running reliably.

### Specific Alert Implementation — Local Poller

**File:** `scripts/poll_pipeline_health.py` (new file)

```python
def check_health(s3, bucket: str, prefix: str) -> list[str]:
    """Return a list of alert messages, or empty list if healthy."""
    alerts = []
    # Check heartbeat age
    # Check progress JSON staleness
    # Check error counts
    # Return list of human-readable alert strings
    ...
```

The poller should accept `--once` (print and exit) and `--watch` (poll every
N seconds) modes. The `--watch` mode should print a summary line every poll
cycle and highlight changes.

---

## Cross-Cutting Issues

### S3 Upload Retry Pattern

Several improvements above require uploading to S3 with retry on failure. A
shared utility should be added to `ec2_utils.py` (or a new `s3_utils.py`)
rather than duplicating retry logic:

**File:** `ec2_utils.py` or `s3_utils.py` (new)

```python
def s3_upload_with_verify(
    s3,
    local_path: str,
    bucket: str,
    key: str,
    *,
    retries: int = 3,
    retry_delay_s: int = 15,
) -> None:
    """Upload local_path to S3 and verify with head_object. Raises on failure."""
```

This function should be used everywhere a file is uploaded and must be confirmed
present (kinematics.h5, sync.h5, analysis.h5, model weights). It should not be
used for best-effort uploads (heartbeat, log files) where failure is acceptable.

### Log File Naming Consistency

Currently the GPU instance log file is `/var/log/hm2p-dlc-retrain.log` and the
CPU instance log is `/var/log/hm2p-downstream.log`. The S3 keys differ:

- GPU: `dlc-retrain/_retrain_log.txt` (EXIT trap) and `dlc-retrain/_run_log.txt`
  (uploaded once after setup)
- CPU: `dlc-retrain/_downstream_log.txt`

There are two S3 keys for the GPU log (`_retrain_log.txt` vs `_run_log.txt`)
written by different code paths, which is redundant and confusing. Consolidate
to one key per instance:

| Instance | Log file | S3 key |
|----------|----------|--------|
| GPU | `/var/log/hm2p-dlc-retrain.log` | `dlc-retrain/_gpu_run_log.txt` |
| CPU | `/var/log/hm2p-downstream.log` | `dlc-retrain/_cpu_run_log.txt` |

The GPU `_retrain_log.txt` and `_run_log.txt` should be consolidated to
`_gpu_run_log.txt`. Update the EXIT trap, the mid-run upload loop, and the
manual upload after setup to all use `_gpu_run_log.txt`.

---

## Dependency on Missing Stage Scripts

`run_downstream_pipeline.py` calls `run_stage3_kinematics.py`, `run_stage5_sync.py`,
and `run_stage6_analysis.py` as subprocesses. None of these scripts exist yet.
Until they are implemented, all of the per-stage S3 upload verification (section 2)
and progress updates (section 3) for the CPU downstream run are moot. The
observability improvements in sections 2 and 3 for downstream stages should be
built into those stage scripts when they are written, not retrofitted to
`run_downstream_pipeline.py` alone.

Add a pre-flight check to `run_downstream_pipeline.py` `main()` that verifies
the required scripts exist before processing any session, rather than failing
silently per session with a truncated subprocess error.

---

## Summary Table

| # | Improvement | Files affected | Priority | Effort |
|---|-------------|----------------|----------|--------|
| 1 | Periodic log uploads — CPU instance | `ec2_utils.py`, `launch_downstream_cpu.py` | Critical | 30 min |
| 2 | Per-session S3 upload verification with retry | `run_downstream_pipeline.py`, `render_dlc_videos.py`, stage scripts | Critical | 45 min |
| 3 | Per-stage progress updates (not just session boundaries) | `run_dlc_retrain.py`, `run_downstream_pipeline.py`, `launch_downstream_cpu.py` | High | 40 min |
| 4 | Error aggregation JSON with stack traces and instance ID | `run_dlc_retrain.py`, `run_downstream_pipeline.py`, `render_dlc_videos.py` | High | 50 min |
| 5 | Instance heartbeat (60s, disk/mem/load) | `ec2_utils.py`, `launch_dlc_finetune_ec2.py`, `launch_downstream_cpu.py`, frontend | High | 25 min |
| 6 | Cost tracking — launch/shutdown records with instance type | `ec2_utils.py`, `launch_dlc_finetune_ec2.py`, `launch_downstream_cpu.py`, `ec2_constants.py` | Medium | 30 min |
| 7 | Alerting — heartbeat staleness, all-sessions-failed, watchdog abort | `scripts/poll_pipeline_health.py` (new), optional SNS or frontend banner | Medium | 2 hr |

Critical items (1, 2) should be implemented before the next GPU run. High items
(3, 4, 5) provide the diagnostic depth needed to resolve any future failures
quickly. Medium items (6, 7) can follow once the critical and high items are in.
