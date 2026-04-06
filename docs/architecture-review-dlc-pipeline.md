# Architecture Review — DLC Training and Inference Pipeline

**Date:** 2026-04-02  
**Scope:** Scripts, EC2 user-data, and frontend covering Stage 2a (DLC training) and
Stage 2b (DLC inference), including the finetuned-pose promotion path and downstream
pipeline dependency handling.  
**Trigger:** Debugging session that cost multiple EC2 instances and several hours due
to a cluster of design flaws: coupled train/infer, missing model weights in `--infer-only`
mode, watchdog killing instances during inter-session gaps, batch_size=1 default,
unnecessary manual promotion step, and wrong S3 path checked by the frontend.

---

## Issues Found

### Critical

**1. `promote_finetuned_pose.py` still uses `aws s3 sync` (the original failure mode)**

`promote_finetuned_pose.py` (line 64) calls `subprocess.run(["aws", "s3", "sync", ...])`.
This is the same pattern that caused weights to not be uploaded during training — `aws s3 sync`
is a separate CLI tool and is not guaranteed to be installed in every environment. The retrain
script fixed this by using boto3 directly for the copy loop, but `promote_finetuned_pose.py`
was not updated. If `awscli` is not installed locally, promotion will fail with a subprocess
error, but only after the dry-run check passes (which never touches S3).

Fix: Replace the `subprocess.run(["aws", "s3", "sync", ...])` block with a boto3
`list_objects_v2` + `copy_object` loop, matching the pattern already used in
`run_dlc_retrain.py`'s auto-promote block.

---

**2. `run_dlc_retrain.py` inference loop uploads results with `aws s3 sync` (subprocess), not boto3**

Lines 204–209 of `run_dlc_retrain.py`:

```python
subprocess.run(
    ["aws", "s3", "sync", str(out_dir), s3_dest],
    check=True, capture_output=True,
)
```

The model weight upload was fixed to use boto3, but the per-session result upload in
`infer()` still uses `aws s3 sync`. On the EC2 instance this works (awscli is installed
via `apt-get` in `APT_INSTALL_SNIPPET`), but it is inconsistent — if the APT install
ever fails silently (possible under network pressure during instance bootstrap), inference
results will silently fail to upload and the session will be added to `failed`.

Fix: Replace both `aws s3 sync` calls in `infer()` — the video download (line 163–170)
and the result upload (lines 204–209) — with boto3 equivalents. The download can use
`s3.list_objects_v2` + `s3.download_file`, and the upload can use `s3.upload_file` in
a loop. This removes the hard dependency on `awscli` from the inference loop.

Note: `launch_dlc_ec2.py` and `launch_dlc_parallel.py` are inference-only SuperAnimal
scripts. They also rely on `aws s3 sync` for video download and result upload, but they
install `awscli` reliably via `apt-get`. This is an inconsistency with the retrain script
but acceptable for those scripts specifically, since they were not affected by the original
bug.

---

**3. Auto-promotion skips silently if any session failed, with no partial-session option**

In `run_dlc_retrain.py` lines 229–231:

```python
if failed:
    print(f"\nSkipping auto-promote: {len(failed)} sessions failed.")
    return
```

If 25 sessions succeed and 1 fails (e.g., a missing video), the entire promotion is
skipped. There is no way to promote the 25 successful sessions from the EC2 run, and
the only recovery path is to re-run `promote_finetuned_pose.py` manually after checking
what failed. The user won't know this happened unless they read the progress JSON.

Fix: Add a `--skip-failed` flag to `promote_finetuned_pose.py` to allow partial promotion
after review. Update the auto-promote logic to promote all completed sessions and log
the skipped ones, rather than aborting entirely. The progress JSON should clearly list
which sessions were promoted and which were not.

---

**4. `get_ec2_instances()` filters by `Project` tag values `"hm2p-suite2p"` and `"hm2p-dlc"` only**

In `frontend/data.py` line 568:

```python
{"Name": "tag:Project", "Values": ["hm2p-suite2p", "hm2p-dlc"]},
```

The retrain instance (`launch_dlc_finetune_ec2.py`) sets:

```python
{"Key": "Project", "Value": "hm2p"},
```

Not `"hm2p-dlc"`. So the frontend's `get_ec2_instances()` will never find a running
retrain instance. This means:
- The "Active Instances" panel on the DLC Training page shows nothing when training is running.
- The `_get_rerun_status()` auto-detection (which looks for `"dlc-retrain"` in the
  `project` field) will also miss the instance because `project` is populated from
  the `Project` tag (`"hm2p"`), not the `Name` tag (`"hm2p-dlc-retrain"`).

Fix: Either (a) change `launch_dlc_finetune_ec2.py` to use `"Project": "hm2p-dlc"` as
the tag value, or (b) extend `get_ec2_instances()` to also filter by
`"tag:Name", "Values": ["hm2p-dlc-retrain"]`. The `_get_rerun_status()` auto-detection
also needs updating to inspect the `Name` tag, not the `Project` tag.

---

**5. `_get_rerun_status()` inspects the wrong tag field to identify instance type**

`_get_rerun_status()` (data.py lines 248–254) checks `inst.get("project", "").lower()`
for the string `"dlc-retrain"`. But `project` is populated from the EC2 `Project` tag.
The retrain instance has `Project = "hm2p"` (no mention of "dlc-retrain") and the
distinguishing identifier is in the `Name` tag (`"hm2p-dlc-retrain"`). The name field
is not returned by `get_ec2_instances()` at all. This means the auto-detection of
active re-runs will never trigger for the retrain instance, so stale downstream stages
will not be flagged while training is running.

Fix: Extend the `instances` dict returned by `get_ec2_instances()` to include the `Name`
tag, and update `_get_rerun_status()` to check both `name` and `project` fields.

---

### Medium

**6. No pre-launch check that model weights exist before launching `--infer-only`**

`launch_dlc_finetune_ec2.py` validates that `config.yaml` exists on S3 before launching
(line 111), but does not check for model weights at `dlc-retrain/models/`. In `--infer-only`
mode, the instance will download `config.yaml`, then call `list_objects_v2` on
`dlc-retrain/models/` and find nothing, then call `sys.exit(1)` — but this happens
inside the EC2 user-data script, so the instance immediately shuts down with no
visible error from the local machine. The log is uploaded on `EXIT` trap but the
failure message is not surfaced back to the local operator.

Fix: Add an S3 check for at least one file under `dlc-retrain/models/` in the `launch()`
function before submitting the run, similar to the `config.yaml` check. Fail fast
locally with a clear message rather than spending instance time.

---

**7. Session ID parsing is duplicated across five scripts with no shared utility**

`parse_session_id()` logic (splitting `exp_id` on `_`, building `sub-` and `ses-` strings)
is duplicated in:

- `run_dlc_retrain.py` (lines 140–142)
- `launch_dlc_ec2.py` (lines 72–75)
- `launch_dlc_parallel.py` (lines 84–88)
- `run_downstream_pipeline.py` (lines 38–42)
- `frontend/data.py` (lines 494–499, this one is tested)

Four of the five copies are separate implementations. If the session ID format ever
changes, all five need updating. `frontend/data.py` has the canonical version but the
scripts cannot import from frontend. `ec2_utils.py` would be the right home for a
shared `parse_session_id()` in the scripts layer.

Fix: Move the canonical implementation to `ec2_utils.py` and import it in all four
other scripts. Requires no code changes to `frontend/data.py` (which imports from
`src/hm2p/`, not scripts).

---

**8. `launch_dlc_ec2.py` has its own `get_s3_credentials()` that reads `~/.aws/credentials`
differently from `ec2_utils.get_s3_credentials()`**

`launch_dlc_ec2.py` (lines 48–59) iterates profiles looking for `"hm2p-agent"` then
`"default"`. `ec2_utils.get_s3_credentials()` (lines 127–134) only reads `"default"`.
`launch_dlc_parallel.py` (lines 64–74) also has its own copy that matches the
`launch_dlc_ec2.py` version (checking both profiles). So there are three slightly
different implementations of credential reading across these files.

Fix: Consolidate into `ec2_utils.get_s3_credentials()`, updating it to check
`"hm2p-agent"` then `"default"` (the more robust version), and remove the duplicate
implementations in the other two scripts.

---

**9. Hard-coded AWS credentials path in `ec2_utils.get_s3_credentials()` is
container-specific**

Line 131 of `ec2_utils.py`:

```python
config.read("/home/node/.aws/credentials")
```

This path is specific to the devcontainer. On macOS it should be
`~/.aws/credentials`. The function silently returns wrong keys if run outside the
container, and the resulting EC2 launch will fail with an authentication error, not
with a clear "wrong credentials path" message.

Fix: Use `Path.home() / ".aws" / "credentials"` (as `launch_dlc_ec2.py` and
`launch_dlc_parallel.py` already do). Add a check that the file exists and that the
parsed key/secret are non-empty before returning.

---

**10. `launch_dlc_parallel.py` DLC install snippet does not use `PYTORCH_CUDA_INSTALL_SNIPPET`**

The parallel script (line 153) installs DLC with a one-liner:

```bash
pip3 install --break-system-packages --quiet --pre deeplabcut
```

This is the exact pattern that causes CPU-only PyTorch, which was the root cause of
the original GPU-at-5% failure. The single-instance `launch_dlc_ec2.py` has the same
issue. Both scripts rely on the Deep Learning AMI's pre-installed PyTorch rather than
reinstalling from the CUDA index — this works only if the AMI's PyTorch is not
overwritten by DLC's pip dependencies.

The retrain script (via `ec2_utils.PYTORCH_CUDA_INSTALL_SNIPPET`) does this correctly:
installs CUDA PyTorch first, then DLC, then reinstalls CUDA PyTorch again, then hard-
verifies CUDA availability. The parallel and single-instance scripts should do the same.

Fix: Replace the bare `pip3 install --pre deeplabcut` line in both `launch_dlc_ec2.py`
and `launch_dlc_parallel.py` with `PYTORCH_CUDA_INSTALL_SNIPPET` from `ec2_utils.py`.
This is the same fix that was already applied to the retrain script.

---

**11. `launch_dlc_parallel.py` does not have the GPU watchdog from `ec2_utils`**

The parallel script has no `format_gpu_guard()` call. It has `nvidia-smi` output
piped to stdout but no watchdog that aborts on sustained 0% GPU utilization. If DLC
falls back to CPU on any shard, that instance will run for the full 24h producing
garbage-speed output and billing at full GPU price.

Fix: Add `format_gpu_guard(DERIVATIVES_BUCKET, f"pose/_gpu_shard{shard_id}")` to
the shard user-data. The shard's Python processing loop should touch
`/tmp/gpu_processing_active` before the DLC call and remove it after, as the retrain
script does.

---

**12. `run_downstream_pipeline.py` calls scripts that may not exist yet**

`run_downstream_pipeline.py` builds `sys.executable + "scripts/run_stage3_kinematics.py"`
etc. as subprocess commands (lines 87–89, 105–107, 123–125). None of these stage scripts
appear to exist in the repository yet — the downstream pipeline runner is a stub that
assumes they will be written. If called now, every `process_session()` call will fail
with a "file not found" subprocess error, but `run_stage3()` catches this in
`capture_output=True` and prints a truncated stderr, making the failure non-obvious.

This is not a bug to fix immediately (the scripts are planned), but it is a documentation
gap: the script should state explicitly in its docstring which stage scripts it requires
and that they are not yet implemented.

Fix (documentation): Add a `REQUIRES` block to the `run_downstream_pipeline.py`
docstring listing the not-yet-implemented scripts. Add a pre-flight check at startup
that warns if the required scripts are missing, rather than failing silently per session.

---

**13. `_count_cascade_outputs()` uses a hardcoded sample session key**

`frontend/data.py` line 399:

```python
sample_key = "calcium/sub-1114353/ses-20210823T165950/ca.h5"
```

This hardcodes one specific session to check for CASCADE output. If this session
happened to fail during CASCADE processing (or was re-processed and the key changed),
the status will be wrong for all 26 sessions. It also silently returns `26` or `0`
with no intermediate state.

Fix: Sample 2–3 sessions using `experiments.csv` rather than a hardcoded key. Or
add a `cascade_complete` marker file written by the CASCADE stage, and check for
that instead.

---

**14. `get_pipeline_status()` checks `STAGE_PREFIXES` but `PIPELINE_STAGES` has more keys**

`STAGE_PREFIXES` (data.py lines 125–133) has 7 entries. `PIPELINE_STAGES` has 11 entries.
`get_pipeline_status()` (line 523) iterates over `STAGE_PREFIXES` only, so it does not
populate status for `kpms`, `cascade`, `pose_finetuned`, `ingest`, or `dlc_training`
per-session. `get_stage_summary()` handles some of these with separate counting functions,
but the mismatch between the two dicts is a maintenance trap: adding a stage to
`PIPELINE_STAGES` without a corresponding entry in `STAGE_PREFIXES` (or a special-case
counter) silently shows 0/26 for that stage.

Fix: Consolidate these into one data structure, or add an explicit comment on
`PIPELINE_STAGES` entries that do not use the `STAGE_PREFIXES`/`get_pipeline_status()`
path, with a pointer to their counting function.

---

**15. No timeout guard on the `--train-only` + `--infer-only` split workflow**

When running `--train-only` followed by a separate `--infer-only` run, the second
instance has a 24h timeout from its own launch, not from when training started. This
is correct behaviour. However, there is no check that the model weights on S3 are
recent (e.g., from the current training run vs a stale run). If an operator runs
`--infer-only` against old model weights by mistake, the pipeline will silently use
the wrong model. The progress JSON for the training step does contain a timestamp
(`updated` field), but nothing compares this to the inference launch time.

Fix: In `--infer-only` mode, check the `LastModified` timestamp of the newest file
in `dlc-retrain/models/` against the `updated` timestamp in `_retrain_progress.json`.
If the weights are older than the last successful training report, warn before
proceeding. This should be a pre-flight check in `launch_dlc_finetune_ec2.py`'s
`launch()` function, not on the instance.

---

### Low

**16. `launch_dlc_ec2.py` uses a local state file (`~/.hm2p-dlc-instance.json`) for `--status` and `--terminate`**

This state file is machine-local. If the operator switches machines or the container is
rebuilt, `--status` and `--terminate` will report "No active instance" even if an instance
is running. `launch_dlc_finetune_ec2.py` correctly queries EC2 tags (`describe_instances`
with tag filters) rather than relying on a local file. The single-instance script should
be updated to match.

---

**17. `dlc_training_page.py` caption says "promote fine-tuned results" as a manual step**

The caption at the bottom of the training page (line 357) reads:

> "After training completes, run `scripts/promote_finetuned_pose.py` to QC and promote
> results, then re-run Stage 2b."

Since auto-promotion now happens inside `run_dlc_retrain.py` when all sessions succeed,
this caption is misleading. The workflow has changed: manual promotion is only needed
when there are failures. The caption and the "How to start training" expander (lines
379–384) should be updated to reflect that manual promotion is the exception, not the
default step.

---

**18. `dlc-pipeline.md` GPU watchdog docs say "5 consecutive minutes" but implementation is 10 minutes**

The docs (line 11) say:

> "GPU watchdog — if GPU utilization stays at 0% for 5 consecutive minutes during
> processing, instance terminates"

`ec2_utils.py` GPU_GUARD_SNIPPET (lines 36–47) checks `tail -20` (20 readings at 30s
intervals = 10 minutes). The original threshold was 5 minutes (10 readings). The threshold
was doubled after the watchdog killed instances during inter-session download gaps, but
the documentation was not updated.

Fix: Update `dlc-pipeline.md` line 11 to say "10 consecutive minutes" and clarify that
the watchdog runs every 5 minutes (checks the last 10 minutes of readings).

---

**19. `run_dlc_retrain.py` has a `--batch-size` argument that is accepted but not forwarded to `deeplabcut.train_network()`**

The `main()` parser accepts `--batch-size` (line 258) and passes it to `train()` (line
267). Inside `train()`, `deeplabcut.train_network()` (line 94) does not receive the
`batch_size` argument:

```python
deeplabcut.train_network(
    str(config_path),
    maxiters=maxiters,
    displayiters=100,
    saveiters=5000,
)
```

The `batch_size` parameter exists in the function signature (`train(s3, maxiters, batch_size=8)`)
but is silently dropped. DLC will use whatever batch size is in `config.yaml`.

Fix: Add `batch_size=batch_size` to the `deeplabcut.train_network()` call. Verify that
the DLC 3.x PyTorch API accepts this parameter at this call site (it may need to go into
`pose_cfg.yaml` instead, in which case add a config patch step).

---

**20. `run_dlc_retrain.py` does not clean up `/tmp/dlc-retrain` after training**

After training, the full DLC project directory (`/tmp/dlc-retrain`, including videos
used during training and potentially large intermediate files) is left on the instance's
100 GB EBS volume. The inference loop then creates additional per-session directories
under `/tmp/dlc-infer/` and cleans those up per session. The training directory is never
cleaned. On a 100 GB volume this is unlikely to cause a disk-full failure during a single
run, but it is unnecessary disk use.

Fix: After the model weights have been uploaded to S3, delete large subdirectories from
`/tmp/dlc-retrain` (keeping only the config and model weights for inference use). A
`shutil.rmtree(work / "labeled-data", ignore_errors=True)` call after upload is
sufficient.

---

## Paths and Constants That Should Be Centralised

The following values are currently hardcoded in multiple places. A `constants.py` or
top-level config block in `ec2_utils.py` would eliminate duplication and drift.

| Value | Where it appears |
|-------|-----------------|
| `REGION = "ap-southeast-2"` | 5 scripts + data.py |
| `DERIVATIVES_BUCKET = "hm2p-derivatives"` | 5 scripts + data.py |
| `RAWDATA_BUCKET = "hm2p-rawdata"` | 3 scripts + data.py |
| `RETRAIN_PREFIX = "dlc-retrain"` | retrain script + training page + data.py |
| `FINETUNED_PREFIX = "pose-finetuned"` | retrain script + promote script + data.py |
| `AMI = "ami-05186a30469f66913"` | finetune launch + parallel launch + superanimal launch |
| `KEY_NAME = "hm2p-suite2p"` | finetune launch + parallel launch + superanimal launch |
| `SG_ID / SG_NAME = "sg-020161fb424325e6b" / "hm2p-suite2p-sg"` | all launch scripts |
| `IAM_PROFILE = "hm2p-ec2-role"` | all launch scripts |
| Session parsing logic | 5 scripts |

The AMI ID in particular is risky: `launch_dlc_finetune_ec2.py` uses the variable `AMI`
while `launch_dlc_ec2.py` and `launch_dlc_parallel.py` use `AMI_ID`, both set to the
same literal string. If the AMI is updated for one script it is easy to forget the others.

---

## Correct Workflow (How the Pipeline Should Work)

The diagram below shows the intended end-to-end flow after fixing the issues above.
Dashed boxes are optional human decision points.

```
Local machine
┌──────────────────────────────────────────────────────────────────────┐
│ 1. Select frames                                                     │
│    uv run python scripts/prepare_retrain_frames.py sub/ses F1 F2 ...│
│    → opens napari; labels saved to sourcedata/trackers/dlc/          │
│                                                                      │
│ 2. Upload labels                                                     │
│    uv run python scripts/upload_dlc_labels.py                        │
│    → copies config.yaml + CollectedData to s3://hm2p-derivatives/   │
│      dlc-retrain/                                                    │
│                                                                      │
│ 3. Launch training (or train + infer in one step)                    │
│    uv run python scripts/launch_dlc_finetune_ec2.py                  │
│    [or --train-only, then later --infer-only]                        │
│    → pre-flight: checks config.yaml AND models/ exist (infer-only)  │
└──────────────────────────────────────────────────────────────────────┘

EC2 g5.xlarge (user-data)
┌──────────────────────────────────────────────────────────────────────┐
│ 4. Bootstrap                                                         │
│    • wait for apt locks (DPKG_WAIT_SNIPPET)                          │
│    • apt install awscli ffmpeg git                                   │
│    • install PyTorch (CUDA index) → DLC → reinstall PyTorch          │
│    • hard verify: CUDA tensor test + DLC import                      │
│    • start GPU monitor (30s readings → /var/log/gpu_monitor.csv)     │
│    • start S3 upload loop (every 5 min)                              │
│    • start GPU watchdog (abort if 0% for 10 min during processing)  │
│    • start 24h hard timeout                                          │
│                                                                      │
│ 5a. Training (if not --infer-only)                                  │
│    • download dlc-retrain/ from S3 (config + labels) via boto3      │
│    • fix project_path in config.yaml                                 │
│    • deeplabcut.create_training_dataset()                            │
│    • deeplabcut.train_network() [with explicit batch_size]           │
│    • deeplabcut.evaluate_network()                                   │
│    • upload dlc-models-pytorch/ to dlc-retrain/models/ via boto3    │
│    • write _retrain_progress.json: "Training complete"               │
│                                                                      │
│ 5b. Inference (if not --train-only)                                  │
│    • if --infer-only: download config + weights from S3 via boto3   │
│    • for each session (26 total):                                    │
│        • download overhead .mp4 from hm2p-rawdata via boto3         │
│        • ffmpeg subsample to 30fps                                   │
│        • touch /tmp/gpu_processing_active                            │
│        • deeplabcut.analyze_videos(..., batch_size=64)               │
│        • rm /tmp/gpu_processing_active                               │
│        • upload .h5/.csv/.json to pose-finetuned/{sub}/{ses}/ boto3 │
│        • shutil.rmtree(local session dir)                            │
│        • update _retrain_progress.json                               │
│                                                                      │
│ 6. Auto-promote (if ALL sessions completed successfully)             │
│    • boto3 copy_object: pose-finetuned/{sub}/{ses}/ → pose/          │
│    • write _retrain_progress.json: "Promoted to pose/"               │
│    • write pipeline_rerun.json: pose stage re-ran                   │
│                                                                      │
│ 7. Upload final logs → S3; shutdown -h now                           │
└──────────────────────────────────────────────────────────────────────┘

Local machine (post-run)
┌──────────────────────────────────────────────────────────────────────┐
│ 8. Review (always, even after auto-promote)                          │
│    • frontend Tracking QC page: compare SuperAnimal vs finetuned    │
│    • if partial failures: inspect progress JSON for which sessions   │
│                                                                      │
│ 9. Manual promote (only if sessions failed in step 5b)              │
│    uv run python scripts/promote_finetuned_pose.py --dry-run        │
│    uv run python scripts/promote_finetuned_pose.py                   │
│    [uses boto3 copy_object, not aws s3 sync]                         │
│                                                                      │
│ 10. Re-run downstream stages                                         │
│    uv run python scripts/run_stage3_kinematics.py --force            │
│    [Stages 3 → 3b → 5 → 6 must all re-run; Stage 4 is independent] │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Documentation Gaps

The following items are missing or incorrect in current docs:

1. `docs/dlc-pipeline.md` — watchdog threshold says 5 minutes, should be 10 minutes
   (issue 18).
2. `docs/dlc-pipeline.md` — the "Workflow" section still lists step 6 as
   "Promote fine-tuned results: `scripts/promote_finetuned_pose.py`" as a required step.
   It should clarify that auto-promotion happens when all sessions succeed, and manual
   promotion is only for partial runs.
3. `docs/dlc-pipeline.md` — does not document the `--train-only` / `--infer-only` split
   workflow, including the pre-flight check for model weights that should exist before
   `--infer-only` is safe to use.
4. `docs/dlc-retraining.md` — does not mention the risk of running `--infer-only` against
   stale model weights.
5. `frontend/pages/dlc_training_page.py` — caption and "How to start training" expander
   describe promotion as a manual required step (issue 17).
6. Neither doc explains what happens when partial inference fails: which sessions land
   in `pose-finetuned/` (successful ones), what the `failed_sessions` list in the
   progress JSON means, and how to promote only the successful subset.

---

## Summary of Prioritised Actions

| Priority | Issue | Fix effort |
|----------|-------|-----------|
| Critical | Issue 1: `promote_finetuned_pose.py` uses `aws s3 sync` | ~15 min |
| Critical | Issue 4: EC2 tag mismatch hides retrain instance from frontend | ~10 min |
| Critical | Issue 5: `_get_rerun_status()` checks wrong tag field | ~10 min |
| Critical | Issue 6: No pre-launch check for model weights in `--infer-only` | ~20 min |
| Medium | Issue 2: `infer()` upload/download uses subprocess `aws s3 sync` | ~30 min |
| Medium | Issue 3: Auto-promotion skips all-or-nothing, no partial path | ~30 min |
| Medium | Issue 7: Session ID parsing duplicated across 5 scripts | ~20 min |
| Medium | Issue 8/9: Multiple `get_s3_credentials()` implementations | ~15 min |
| Medium | Issue 10: Parallel + single-instance scripts lack CUDA reinstall guard | ~20 min |
| Medium | Issue 11: Parallel script lacks GPU watchdog | ~20 min |
| Medium | Issue 13: CASCADE status uses hardcoded session key | ~20 min |
| Medium | Issue 14: `STAGE_PREFIXES` / `PIPELINE_STAGES` mismatch | ~20 min |
| Medium | Issue 15: No stale-weights check for `--infer-only` | ~25 min |
| Medium | Issue 19: `--batch-size` dropped before `train_network()` | ~5 min |
| Low | Issue 12: Missing stage scripts not detected early | ~10 min |
| Low | Issue 16: `launch_dlc_ec2.py` local state file for status | ~15 min |
| Low | Issue 17: Frontend caption describes promotion as always-required | ~5 min |
| Low | Issue 18: Watchdog docs say 5 min, code is 10 min | ~5 min |
| Low | Issue 20: `/tmp/dlc-retrain` not cleaned after weight upload | ~10 min |
