# DLC Pipeline — Training (Stage 2a) and Inference (Stage 2b)

## Hard Requirements

1. **GPU enforced** — CUDA verified at startup. If not available, instance
   terminates immediately. GPU utilization monitored every 30s.
2. **GPU watchdog** — if GPU utilization stays at 0% for 5 consecutive
   minutes during processing, instance terminates (DLC fell back to CPU).
3. **24-hour hard timeout** — instance terminates after 24h regardless.
4. **Auto-terminate on completion** — `InstanceInitiatedShutdownBehavior=terminate`
   on all launch scripts. No stopped instances accruing EBS costs.
5. **Continuous monitoring** — GPU utilization CSV and run logs uploaded to
   S3 every 5 minutes. Training progress (iteration, loss) visible via
   `--progress` CLI.

## Instance Types

| Task | Instance | GPU | Cost (On-Demand) | Est. time | Est. cost |
|------|----------|-----|-------------------|-----------|-----------|
| Training (50k iters) | g5.xlarge | A10G 24GB | $1.01/hr | ~2h | ~$2 |
| Inference (26 sessions) | g5.xlarge | A10G 24GB | $1.01/hr | ~8h | ~$8 |
| Combined (train + infer) | g5.xlarge | A10G 24GB | $1.01/hr | ~10h | ~$10 |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ec2_utils.py` | Shared bash snippets: GPU guard, timeout, PyTorch install, credentials |
| `scripts/launch_dlc_finetune_ec2.py` | Launch training + inference on EC2 |
| `scripts/run_dlc_retrain.py` | On-instance: training + inference logic |
| `scripts/launch_dlc_ec2.py` | Launch inference-only (SuperAnimal, no fine-tuning) |
| `scripts/prepare_retrain_frames.py` | Local: download video, extract frames, open labeling GUI |
| `scripts/upload_dlc_labels.py` | Local: upload labeled data to S3 |
| `scripts/promote_finetuned_pose.py` | Local: copy pose-finetuned → pose after QC |

## PyTorch CUDA Installation

DLC 3.0 with PyTorch must be installed in a specific order to avoid
CPU-only fallback:

1. Install PyTorch from CUDA index: `pip install torch --index-url .../cu121`
2. Install DLC with `--no-deps` (avoids pulling CPU PyTorch)
3. Install remaining DLC deps separately
4. **Hard verify**: run a CUDA tensor test. Abort if it fails.

This is the `PYTORCH_CUDA_INSTALL_SNIPPET` in `ec2_utils.py`.

## GPU Monitoring

All launch scripts include:

- `nvidia-smi --format=csv -l 30` logging to `/var/log/gpu_monitor.csv`
- S3 upload every 5 minutes to `{stage}/_gpu_monitor.csv`
- Watchdog process: if GPU utilization = 0% for 10 consecutive readings
  (5 min) during active processing, terminate instance

Processing code touches `/tmp/gpu_processing_active` before DLC calls
and removes it after, so the watchdog doesn't trigger during
download/upload phases.

## Frame Rate and Median Filter

DLC inference runs on video subsampled to **30fps** (from ~100fps raw).
The downstream median filter window is set to **3 frames**, giving
~100ms temporal smoothing. This approximates the old pipeline which
used 5 frames at 100fps (50ms).

**If the inference frame rate changes**, update the median filter window
to maintain approximately 100ms smoothing (`window = round(0.1 * fps)`):

| Location | Parameter |
|----------|-----------|
| `src/hm2p/kinematics/compute.py` → `median_filter_dataset()` | `window` default |
| `src/hm2p/kinematics/compute.py` → `compute_kinematics()` | `median_filter_dataset(ds, window=...)` call |
| `scripts/render_dlc_videos.py` → `_apply_median_filter()` | `window` default |
| `scripts/render_dlc_videos.py` → `_apply_pipeline_filter()` | `window` default |
| `frontend/pages/dlc_viewer_page.py` → `get_median_filtered()` | `rolling_filter(..., window=...)` call |

## Workflow

```
1. Label frames locally:
   uv run python scripts/prepare_retrain_frames.py sub/ses 606 2093 ...

2. Upload labels to S3:
   uv run python scripts/upload_dlc_labels.py

3. Launch training on AWS:
   uv run python scripts/launch_dlc_finetune_ec2.py

4. Monitor progress:
   uv run python scripts/launch_dlc_finetune_ec2.py --progress

5. After completion, review tracking in frontend:
   Tracking QC page > compare SuperAnimal vs fine-tuned

6. Promote fine-tuned results:
   uv run python scripts/promote_finetuned_pose.py
```

## Troubleshooting

**Instance terminated with "GPU utilization 0%":**
DLC's FasterRCNN detector fell back to CPU. Check `_gpu_monitor.csv`
on S3. Likely cause: PyTorch was installed without CUDA. Verify
the install log shows `CUDA tensor test: OK`.

**Instance terminated after 24h:**
Training or inference took too long. Check `_run_log.txt` on S3 for
the last processed session. Re-run with `--infer-only` to skip
completed sessions.

**"No labeled data on S3":**
Run `upload_dlc_labels.py` first. Check that `config.yaml` exists at
`s3://hm2p-derivatives/dlc-retrain/config.yaml`.
