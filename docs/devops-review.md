# DevOps Review — hm2p Pipeline

**Date:** 2026-04-11  
**Scope:** All pipeline scripts in `scripts/` and relevant source modules in `src/hm2p/`  
**Predecessor:** `docs/observability-improvements.md` (architect's review, 2026-04-02)  
**Approach:** Hands-on review of actual code — specific files, line numbers, and concrete fixes

This review covers what the architect's document identified at a high level but examines
the actual code. It adds new findings not in that document and provides implementation-ready
fixes. Items are ordered by operational impact.

---

## Priority Index

| # | Issue | File(s) | Impact |
|---|-------|---------|--------|
| 1 | Upload errors silently pass → stage marked "done" | `run_stage3`, `run_stage5`, `run_stage6` | Critical |
| 2 | No CPU log visibility for up to ~6 hours | `launch_downstream_cpu.py`, `ec2_utils.py` | Critical |
| 3 | `run_dlc_retrain.py` uses bare `except` with no traceback in `infer()` | `run_dlc_retrain.py:395` | Critical |
| 4 | numpy used but never imported in `run_dlc_retrain.py` train() | `run_dlc_retrain.py:214` | Critical |
| 5 | CPU instance has no hard timeout | `launch_downstream_cpu.py` | High |
| 6 | `run_downstream_pipeline.py` has no pre-flight script existence check | `run_downstream_pipeline.py` | High |
| 7 | `render_dlc_videos.py` ffmpeg subprocess errors swallowed | `render_dlc_videos.py:493` | High |
| 8 | Temp directory leaked when stage script crashes before `finally` | `run_stage3_kinematics.py`, `run_stage5_sync.py` | High |
| 9 | `run_dlc_retrain.py` S3 promotion uses non-paginated `list_objects_v2` | `run_dlc_retrain.py:430` | Medium |
| 10 | AWS credentials written to disk as plaintext in user-data | `ec2_utils.py:110` | Medium |
| 11 | `update_progress()` upload failure is silent | `run_dlc_retrain.py:44` | Medium |
| 12 | No per-stage exit code from stage scripts when called as subprocess | `run_downstream_pipeline.py:84` | Medium |
| 13 | `render_dlc_videos.py` reads all sessions including excluded ones inconsistently | `render_dlc_videos.py:111` | Low |
| 14 | Log naming inconsistency: two keys for the same GPU log | `ec2_utils.py:29`, `launch_dlc_finetune_ec2.py:95` | Low |

Items 1–4 are bugs, not just operational gaps. Items 5–10 are reliability gaps.
Items 11–14 are maintenance issues.

---

## 1. Upload errors silently pass — stages falsely marked "done"

**Files:** `run_stage3_kinematics.py:286`, `run_stage5_sync.py:151`,
`run_stage6_analysis.py:289`  
**Confirmed by architect:** Yes (doc section 2)

### The bug

In all three stage scripts, the pattern is:

```python
# run_stage3_kinematics.py lines 284-289
print(f"  Uploading to s3://{DERIVATIVES_BUCKET}/{kin_key}")
s3.upload_file(str(output_path), DERIVATIVES_BUCKET, kin_key)
print(f"  DONE")
return "ok"
```

If `upload_file()` raises (SSL error, S3 outage, credential expiry), the exception
propagates to the outer `except Exception as e:` block at line 291, which catches it,
prints a truncated error string, and returns `"error: <msg>"`. The subprocess then exits
with return code 0 because the exception was caught and the function returned normally.

`run_downstream_pipeline.py` only checks `result.returncode`, so return code 0 =
success even though nothing was uploaded.

This is the pattern the architect flagged as having already caused a production failure.

### Fix — two lines per stage script

After `upload_file()`, add a `head_object()` to verify the key landed. If the object
is not found, raise immediately so the subprocess exits non-zero.

```python
# After upload_file() in each stage script:
s3.upload_file(str(output_path), DERIVATIVES_BUCKET, kin_key)

# Verify the upload landed — raises if not found, ensuring non-zero exit on failure
try:
    s3.head_object(Bucket=DERIVATIVES_BUCKET, Key=kin_key)
except Exception as verify_err:
    raise RuntimeError(
        f"Upload verification failed for {kin_key}: {verify_err}"
    ) from verify_err

print(f"  DONE")
return "ok"
```

Apply the same pattern in `run_stage5_sync.py:151` and `run_stage6_analysis.py:289`.

In `run_stage6_analysis.py`, the upload is inside a `try/except` block at line 288
that catches all exceptions with `log.exception()`. Move the `head_object()` call
inside the same try block so a failed verify also goes to `failed.append(exp_id)`.

This addresses the root cause. The `_verify_s3_upload()` shared helper the architect
proposed is worth adding for DRY purposes, but the two-liner above is sufficient and
adds no new dependencies.

---

## 2. No CPU log visibility for up to ~6 hours

**File:** `launch_downstream_cpu.py:48-55`, `ec2_utils.py`  
**Confirmed by architect:** Yes (doc section 1)

The CPU user-data script:

```bash
trap 'aws s3 cp /var/log/hm2p-downstream.log \
      s3://.../dlc-retrain/_downstream_log.txt || true; \
      shutdown -h now' EXIT
```

The EXIT trap only fires on instance shutdown. For a ~6-hour downstream run the log
is invisible until the instance terminates. If the instance crashes mid-run (OOM,
kernel panic, Spot interruption), the trap may not run at all.

### Fix

The architect's proposed `CPU_LOG_UPLOAD_SNIPPET` in `ec2_utils.py` is exactly right.
Concrete implementation:

```python
# ec2_utils.py — add alongside GPU_GUARD_SNIPPET

CPU_LOG_UPLOAD_SNIPPET = """
# === Periodic log upload (CPU instance) ===
(while true; do
    sleep 300
    aws s3 cp /var/log/hm2p-downstream.log \\
        s3://{bucket}/{log_prefix}/_downstream_log.txt 2>/dev/null || true
done) &
CPU_UPLOAD_PID=$!
echo "Periodic log upload started (PID=$CPU_UPLOAD_PID, interval=5min)"
"""

def format_cpu_log_upload(bucket: str, log_prefix: str) -> str:
    return CPU_LOG_UPLOAD_SNIPPET.format(bucket=bucket, log_prefix=log_prefix)
```

In `launch_downstream_cpu.py build_user_data()`, place the call immediately after
`{creds}` and before `{DPKG_WAIT_SNIPPET}`:

```python
from ec2_utils import format_cpu_log_upload, ...

cpu_upload = format_cpu_log_upload(DERIVATIVES_BUCKET, "dlc-retrain")

return f"""#!/bin/bash
exec > >(tee /var/log/hm2p-downstream.log) 2>&1
...
{creds}
{cpu_upload}      # <-- add here, before set -ex
{DPKG_WAIT_SNIPPET}
...
```

Placing the upload loop before `set -ex` means it starts even if the apt install
step fails, which is a common early failure mode.

---

## 3. Bare `except` with no traceback in `infer()` — failures invisible

**File:** `run_dlc_retrain.py:395-397`

```python
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(exp_id)
```

`str(e)` for most boto3/DLC exceptions gives a message like `"An error occurred ..."`.
The full traceback, which tells you which line in `analyze_videos()` or
`download_file()` raised, is discarded. When inference fails at session 18 of 26, you
cannot diagnose the failure without SSHing to the instance.

### Fix

```python
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  ERROR [{type(e).__name__}]: {e}")
            print(tb)
            failed.append(exp_id)
```

This is a two-line change. The traceback is written to the instance log (visible via
the periodic GPU log upload from `GPU_GUARD_SNIPPET`), so it is recoverable from S3
without SSH.

The full error JSON approach the architect proposed (doc section 4) is a good
follow-up once the basic traceback is captured.

---

## 4. `numpy` used but never imported in `train()` — NameError crash

**File:** `run_dlc_retrain.py:214`

```python
valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
```

`numpy` is not imported at the top of this file. There is no `import numpy as np`
anywhere in `run_dlc_retrain.py`. This line will raise `NameError: name 'np' is not
defined` the first time DLC training finishes and the per-bodypart metrics block runs
(lines 186–226). The exception is caught by the outer `try/except` at line 224:

```python
    except Exception as e:
        print(f"  Per-bodypart metrics failed: {e}")
        import traceback; traceback.print_exc()
```

So the per-bodypart summary silently fails and training continues. But it means
`_per_bodypart_summary.json` is never uploaded, and the error is only visible in the
log if you look.

### Fix

Add `import numpy as np` to the top-level imports in `run_dlc_retrain.py` (after the
existing imports, line 24).

---

## 5. CPU instance has no hard timeout

**File:** `launch_downstream_cpu.py`

The GPU instance has `format_hard_timeout(24)` which schedules `shutdown -h now` after
24 hours. The CPU user-data script has no equivalent. A hung downstream run (e.g. a
`run_stage3_kinematics.py` subprocess that blocks on a malformed HDF5 file) will keep
the `c5.xlarge` running indefinitely at ~$0.214/hr.

The downstream + render pipeline for 26 sessions should complete in 3–5 hours on a
`c5.xlarge`. A 12-hour hard timeout is a safe ceiling.

### Fix

Import and use `format_hard_timeout` in `launch_downstream_cpu.py`:

```python
# launch_downstream_cpu.py
from ec2_utils import (
    DPKG_WAIT_SNIPPET,
    build_creds_block,
    format_cpu_log_upload,
    format_hard_timeout,  # <-- add
    get_s3_credentials,
)

def build_user_data(render_only: bool = False) -> str:
    ...
    timeout = format_hard_timeout(12)

    return f"""...
    {creds}
    {cpu_upload}
    {timeout}       # <-- add here
    {DPKG_WAIT_SNIPPET}
    ...
```

---

## 6. No pre-flight check for required stage scripts

**File:** `run_downstream_pipeline.py:84-135`  
**Confirmed by architect:** Yes (doc "Dependency on Missing Stage Scripts")

`run_downstream_pipeline.py` calls `run_stage3_kinematics.py`, `run_stage5_sync.py`,
and `run_stage6_analysis.py` as subprocesses. If any of those scripts don't exist
(e.g. after a fresh `git clone` without all branches merged), the subprocess exits with
return code 127, which `run_downstream_pipeline.py` treats as a stage failure and
continues processing all 26 sessions, logging 26 identical "Stage 3 FAILED" lines with
`/usr/bin/python3: can't open file 'scripts/run_stage3_kinematics.py'`.

### Fix

Add a pre-flight check at the start of `main()`:

```python
# run_downstream_pipeline.py, inside main(), before the session loop

REQUIRED_SCRIPTS = [
    "scripts/run_stage3_kinematics.py",
    "scripts/run_stage5_sync.py",
    "scripts/run_stage6_analysis.py",
]
missing = [s for s in REQUIRED_SCRIPTS if not Path(s).exists()]
if missing:
    print(f"ERROR: required scripts not found: {missing}")
    print("Run from the repo root (cd /home/ubuntu/hm2p).")
    sys.exit(1)
```

This fails immediately with one clear error instead of 26 truncated subprocess errors.

---

## 7. ffmpeg subprocess errors swallowed during render

**File:** `render_dlc_videos.py:493-505`

The ffmpeg process is started with:

```python
ffproc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", ...],
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
```

Both `stdout` and `stderr` are sent to `/dev/null`. After the render loop, `ffproc.wait()`
is called (line 545) but its return code is not checked. If ffmpeg fails (out of disk,
codec error, interrupted), the output `.mp4` either does not exist or is truncated.
The script then proceeds to `s3.upload_file()` on the empty/corrupted file. The upload
succeeds because S3 will accept any byte sequence. The viewer page later shows a broken
video.

### Fix

```python
# render_dlc_videos.py, replace the ffproc.wait() block at lines 543-547

for m in modes:
    out_path, ffproc, writer = pipes[m]
    if ffproc is not None:
        ffproc.stdin.close()
        rc = ffproc.wait()
        if rc != 0:
            log.error(
                "  ffmpeg exited with code %d for mode=%s session=%s",
                rc, m, exp_id,
            )
            # Do not upload a corrupt output; remove the partial file
            out_path.unlink(missing_ok=True)
            pipes[m] = (None, ffproc, writer)   # mark as failed
    if writer is not None:
        writer.release()
```

Then in the upload block (lines 551-562), skip modes where `pipes[m][0] is None`.

Also capture ffmpeg stderr for logging:

```python
ffproc = subprocess.Popen(
    ["ffmpeg", ...],
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,   # capture for error logging
)
```

And read it at wait time:

```python
_, stderr_bytes = ffproc.communicate()
if rc != 0:
    log.error("  ffmpeg stderr: %s", stderr_bytes[-500:].decode(errors="replace"))
```

Note: `communicate()` reads all stdout/stderr and waits for completion — use it instead
of `stdin.close() + wait()` when you also need stderr.

---

## 8. Temp directory leaked on crash before `finally`

**Files:** `run_stage3_kinematics.py:211-299`, `run_stage5_sync.py:86-164`

The pattern in both scripts is:

```python
session_dir = work_dir / sub / ses
session_dir.mkdir(parents=True, exist_ok=True)

try:
    ...
    return "ok"

except Exception as e:
    ...
    return f"error: {e}"

finally:
    shutil.rmtree(session_dir, ignore_errors=True)
```

This is correct for normal exits. However, the outer `work_dir` is created by
`tempfile.mkdtemp()` at the top of `main()`:

```python
work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage3-"))
```

At the bottom of `main()`:

```python
shutil.rmtree(work_dir, ignore_errors=True)
```

If `main()` is killed (SIGKILL, OOM, hard timeout triggering `shutdown -h now`), the
outer `work_dir` cleanup never runs. On a 100 GB EBS volume with 26 sessions each
downloading a ~300 MB DLC `.h5`, a partial run can leave up to 7.8 GB in `/tmp`.
For the `c5.xlarge` with a 100 GB root volume this is recoverable, but it can cause
disk-full failures mid-run if the volume fills with partial downloads.

### Fix

Use a `try/finally` around the session loop in `main()`:

```python
work_dir = Path(tempfile.mkdtemp(prefix="hm2p-stage3-"))
try:
    for ses in sessions:
        ...
finally:
    shutil.rmtree(work_dir, ignore_errors=True)
```

This ensures cleanup even if `main()` exits abnormally due to an unhandled exception.
It does not protect against SIGKILL, but that is acceptable — the instance terminates
shortly after anyway.

Also add a disk-space preflight check at the start of processing:

```python
import shutil as _shutil
free_gb = _shutil.disk_usage("/tmp").free / 1e9
if free_gb < 5.0:
    print(f"WARNING: only {free_gb:.1f} GB free in /tmp")
```

---

## 9. S3 pagination bug in `infer()` auto-promote

**File:** `run_dlc_retrain.py:430-438`

```python
resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=src_prefix)
for obj in resp.get("Contents", []):
    src_key = obj["Key"]
    ...
    s3.copy_object(...)
```

`list_objects_v2` returns at most 1,000 objects per call. A pose `.h5` file plus
labelled videos for 26 sessions is well under 1,000 keys, so this does not currently
trigger. However, if `pose-finetuned/sub/ses/` ever contains over 1,000 objects
(e.g. multi-animal DLC output with per-individual files), the promote will silently
miss objects beyond the first page.

The same pattern appears at `run_dlc_retrain.py:500` (model file download).

### Fix

Use the paginator everywhere you are iterating over S3 objects where the count is
unbounded:

```python
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=DERIVATIVES_BUCKET, Prefix=src_prefix):
    for obj in page.get("Contents", []):
        src_key = obj["Key"]
        ...
```

For the model file download at lines 498-512, the same paginator fix applies.

---

## 10. AWS credentials written as plaintext in EC2 user-data

**File:** `ec2_utils.py:110-124`

`build_creds_block()` embeds the AWS access key ID and secret access key directly into
the EC2 user-data script as a heredoc:

```bash
cat > /root/.aws/credentials << 'CREDS'
[default]
aws_access_key_id = AKIA...
aws_secret_access_key = ...
CREDS
```

EC2 user-data is retrievable from within the instance via the instance metadata service
(`http://169.254.169.254/latest/user-data`) without authentication. Any process running
on the instance — including the cloned repo if it contained malicious code — can read it.

The IAM instance profile (`hm2p-ec2-role`) already provides S3 and CloudWatch access
to the instance without static credentials. The `build_creds_block()` approach was
likely carried over from before the instance profile was configured.

### Fix

Remove `build_creds_block()` and `get_s3_credentials()` from both
`launch_dlc_finetune_ec2.py` and `launch_downstream_cpu.py`. The instance profile
provides credentials automatically via the metadata service, which boto3 queries
by default when no explicit credentials are configured.

**Verify first:** confirm that the instance profile `hm2p-ec2-role` has the required
S3 permissions for both `hm2p-rawdata` and `hm2p-derivatives`:

```bash
aws iam list-attached-role-policies --role-name hm2p-ec2-role
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::390897005556:role/hm2p-ec2-role \
    --action-names s3:GetObject s3:PutObject s3:ListBucket \
    --resource-arns "arn:aws:s3:::hm2p-rawdata/*" "arn:aws:s3:::hm2p-derivatives/*"
```

Once confirmed, remove the `{creds}` block from `build_user_data()` in both launchers
and delete `build_creds_block` and `get_s3_credentials` from `ec2_utils.py`. The
instance will use the role credentials automatically.

This is a security improvement, not just an operational one.

---

## 11. `update_progress()` upload failure is silent

**File:** `run_dlc_retrain.py:33-44`

```python
def update_progress(s3, status: str, **extra: object) -> None:
    ...
    s3.upload_file(str(tmp), DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_retrain_progress.json")
```

If this upload fails (S3 hiccup, permission error), the exception propagates up to the
caller — `train()` or `infer()`. In `train()` the call at line 84 is not inside a try
block, so a failed progress update will abort the entire training run. In `infer()` the
call at line 331 is inside the session loop's try block, so a failed progress update at
session start will cause the session to be marked as failed even though no processing
was attempted.

### Fix

Wrap the upload in `update_progress()` in a try/except that logs a warning rather than
re-raising. Progress updates are best-effort — a failed update should not abort
processing:

```python
def update_progress(s3, status: str, **extra: object) -> None:
    import datetime
    progress = {
        "status": status,
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    tmp = Path("/tmp/_retrain_progress.json")
    tmp.write_text(json.dumps(progress, indent=2))
    try:
        s3.upload_file(str(tmp), DERIVATIVES_BUCKET,
                       f"{RETRAIN_PREFIX}/_retrain_progress.json")
    except Exception as e:
        print(f"  WARNING: progress update failed (non-fatal): {e}")
```

---

## 12. Stage scripts exit with 0 even on total failure

**File:** `run_downstream_pipeline.py:94-99`, `run_stage3_kinematics.py`,
`run_stage5_sync.py`

`run_downstream_pipeline.py` runs stage scripts as subprocesses and checks
`result.returncode != 0` to detect failure. However, `run_stage3_kinematics.py` always
exits with code 0 (no `sys.exit(1)` on error). If all sessions fail with `"error: ..."`,
the script exits 0 with an error-filled summary. `run_downstream_pipeline.py` sees
`returncode == 0` and marks all stages as succeeded.

```python
# run_stage3_kinematics.py main() — exits 0 even if all sessions failed
for i, ses in enumerate(sessions):
    status = run_session(...)
    results[ses["exp_id"]] = status
# No sys.exit(1) if errors > 0
```

### Fix

In `main()` of each stage script, exit non-zero if any session had an error:

```python
# run_stage3_kinematics.py — after the summary block
if err > 0:
    sys.exit(1)
```

Apply the same to `run_stage5_sync.py` and `run_stage6_analysis.py` (where `failed`
is the equivalent counter).

---

## 13. `render_dlc_videos.py` session filter inconsistency

**File:** `render_dlc_videos.py:111-119`

`load_sessions()` filters out sessions with `exclude == "1"`. This means render skips
excluded sessions. But `run_stage3_kinematics.py` and `run_stage5_sync.py` process all
sessions including excluded ones (per the CLAUDE.md policy: "Process ALL sessions
regardless of exclude flag"). If a session is excluded but has kinematics and sync
outputs, it will not get a labelled video.

This is inconsistent. The frontier downstream scripts follow the "all sessions" policy;
render does not. This is a minor policy gap but worth noting.

### Fix

Remove the `exclude` filter from `load_sessions()` in `render_dlc_videos.py` to be
consistent with the other pipeline stages:

```python
def load_sessions(metadata_path: Path) -> list[dict]:
    sessions = []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(row)   # remove exclude filter
    return sessions
```

---

## 14. Two S3 keys for the same GPU run log

**File:** `ec2_utils.py:29`, `launch_dlc_finetune_ec2.py:95`

The GPU instance uploads the log to two different S3 keys:

1. `dlc-retrain/_retrain_log.txt` — EXIT trap in `build_user_data()` line 80
2. `dlc-retrain/_run_log.txt` — one-shot upload immediately after setup, line 95

The `GPU_GUARD_SNIPPET` upload loop also targets `_run_log.txt` (line 29 of
`ec2_utils.py`). So `_run_log.txt` is the periodic mid-run log, and `_retrain_log.txt`
is the final log on exit. They are the same file (`/var/log/hm2p-dlc-retrain.log`). An
operator checking the wrong key will see stale data.

### Fix

Consolidate to one key as the architect proposed. Use `_gpu_run_log.txt` for both.
Change line 29 of `ec2_utils.py` and line 80 and 95 of `launch_dlc_finetune_ec2.py`
to use the same key. The frontend status page (which reads `_retrain_log.txt`) also
needs updating.

---

## Operational Runbooks

The following are brief procedures for common failure scenarios.

### Scenario A: GPU instance launches but inference fails at session N

1. Check progress: `aws s3 cp s3://hm2p-derivatives/dlc-retrain/_retrain_progress.json -`
2. Check the log: `aws s3 cp s3://hm2p-derivatives/dlc-retrain/_run_log.txt -`
   (search for `ERROR` and the full traceback — present after fix #3)
3. If the instance is still alive:
   `ssh -i ~/.ssh/hm2p-suite2p.pem ubuntu@$(aws ec2 describe-instances ...)`
   `tail -f /var/log/hm2p-dlc-retrain.log`
4. If you want to re-run inference only without re-training:
   `python scripts/launch_dlc_finetune_ec2.py --infer-only`
   (the pre-flight check will confirm model weights exist before launching)

### Scenario B: CPU downstream instance hung — no log updates

1. Check when log was last updated:
   `aws s3 ls s3://hm2p-derivatives/dlc-retrain/_downstream_log.txt`
2. If `LastModified` is more than 5 minutes ago while instance is running,
   the instance is likely stuck.
3. SSH to the instance and check what is running:
   `ps aux | grep python`
   `tail -100 /var/log/hm2p-downstream.log`
4. If a stage script is blocked on S3 download, kill it and re-run with
   `--force --session <exp_id>` for the specific session.
5. Force-terminate if unresponsive:
   `aws ec2 terminate-instances --instance-ids <iid>`

Without fix #2 (periodic CPU log upload), step 3 requires SSH and the log is
inaccessible remotely. With fix #2, step 1 alone answers whether the instance is
alive and making progress.

### Scenario C: Stage 3 appears complete but kinematics.h5 missing from S3

This is the exact failure mode the architect documented. Before fix #1 this was
silent. After fix #1:

1. The stage script exits non-zero.
2. `run_downstream_pipeline.py` prints `"Stage 3 FAILED: ..."` with the full
   subprocess stderr.
3. The session is not promoted to Stage 5.
4. Re-run with: `python scripts/run_stage3_kinematics.py --session <exp_id> --force`

### Scenario D: Instance terminates but progress JSON shows partial run

The EXIT trap runs `shutdown -h now` after the EXIT trap uploads the final log.
Check:
- `_retrain_progress.json` — last status, completed/failed counts
- `_inference_errors.json` — once fix #3 from the architect's doc is implemented
- GPU monitor: `aws s3 cp s3://hm2p-derivatives/dlc-retrain/_gpu_monitor.csv -`

If some sessions completed, re-run with `--infer-only --skip-failed` to process
only the sessions that failed. The pre-flight check on model weights ensures this
is safe.

### Scenario E: Disk full mid-run on CPU instance

Signs: `OSError: [Errno 28] No space left on device` in stage script output.
This most commonly happens when kinematics processing for 26 sessions accumulates
temp files faster than they are cleaned up.

Fix #8 wraps the session loop in try/finally so individual session dirs are cleaned.
If disk is already full on a running instance:
1. SSH to instance
2. `du -sh /tmp/hm2p-stage3-*/` to find large temp dirs
3. `rm -rf /tmp/hm2p-stage3-*/` to clear (safe — data is on S3)
4. Resume: `python scripts/run_downstream_pipeline.py --force` (skips already-done sessions)

---

## What Is Already Working Well

The following were checked and are in good shape:

- **GPU watchdog** (`ec2_utils.py:33-48`): The 20-consecutive-zero-reading threshold is
  appropriate for the inter-session gap pattern. The `/tmp/gpu_processing_active`
  sentinel correctly disables the watchdog during video download gaps.
- **Hard timeout on GPU** (`ec2_utils.py:11-17`): Present and correctly placed before
  `set -ex`.
- **CUDA verification** (`ec2_utils.py:67-78`): Hard abort if CUDA fails, not just a
  warning. This prevents silent CPU fallback.
- **Pre-flight model weight check** (`launch_dlc_finetune_ec2.py:128-147`): `--infer-only`
  mode correctly verifies model weights exist on S3 before launching an instance.
- **`InstanceInitiatedShutdownBehavior=terminate`** (both launchers): Instances
  self-terminate on shutdown — no stopped-instance EBS accumulation from these runs.
- **`run_stage3_kinematics.py` skip logic** (lines 165-195): Correctly checks for
  existing outputs before downloading, avoiding redundant S3 operations.
- **`render_dlc_videos.py` logging** (line 589): Uses Python `logging` module, not
  `print`. This is good practice and is not followed in the other scripts.
- **`run_stage6_analysis.py` logging** (lines 30-35): Also uses the `logging` module
  properly. Both of these should be the pattern for the other scripts to adopt.

---

## Summary of Changes Required

The critical items (1–4) are bugs that either cause incorrect pipeline state or
crash on every training run:

1. Add `head_object()` verification after `upload_file()` in `run_stage3_kinematics.py:286`,
   `run_stage5_sync.py:151`, `run_stage6_analysis.py:289`
2. Add `format_cpu_log_upload()` to `ec2_utils.py`; call it in `launch_downstream_cpu.py`
3. Add `traceback.format_exc()` to the `except` block in `run_dlc_retrain.py:395`
4. Add `import numpy as np` to `run_dlc_retrain.py` top-level imports

The high items (5–8) prevent runaway costs or data loss:

5. Add `format_hard_timeout(12)` to `launch_downstream_cpu.py`
6. Add pre-flight script existence check to `run_downstream_pipeline.py main()`
7. Check `ffproc.wait()` return code in `render_dlc_videos.py`; capture stderr
8. Wrap session loop in `try/finally` in `run_stage3_kinematics.py main()` and
   `run_stage5_sync.py main()`

The medium items (9–12) address correctness and security, and should follow once the
above are done.
