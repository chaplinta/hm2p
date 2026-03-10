# Relaunch DLC for Remaining 13 Sessions

13/26 sessions are complete on S3. The launch script auto-skips sessions that
already have `.h5` or `.csv` outputs in `s3://hm2p-derivatives/pose/<sub>/<ses>/`.

## Prerequisites

- Run from your **local Mac** (the devcontainer cannot call EC2 APIs)
- AWS credentials configured (`~/.aws/credentials` with `hm2p-agent` or `default` profile)
- `uv` installed and project dependencies available

## Option A: Single instance (simplest)

All 26 sessions are assigned to one instance; the 13 already-done sessions are
skipped automatically (~10s each for the S3 check). Only the remaining 13 run.

```bash
cd ~/Neuro/hm2p-v2
uv run scripts/launch_dlc_parallel.py -n 1 --on-demand --use-profile
```

Estimated time: ~20h (13 sessions x ~1.5h each on g5.xlarge A10G).

## Option B: Two parallel instances (faster)

Splits 26 sessions into 2 shards of 13 each. Each shard skips its already-done
sessions and processes only the remaining ones.

```bash
cd ~/Neuro/hm2p-v2
uv run scripts/launch_dlc_parallel.py -n 2 --on-demand --use-profile
```

Estimated time: ~10h (each instance handles ~6-7 remaining sessions).

Requires 8 On-Demand G/VT vCPUs (2 x g5.xlarge x 4 vCPUs). You already have
this quota approved.

## Flag explanations

| Flag | Why |
|------|-----|
| `-n 1` or `-n 2` | Number of parallel EC2 instances |
| `--on-demand` | Spot quota may still be 0; On-Demand avoids that issue |
| `--use-profile` | Uses the `hm2p-ec2-role` IAM instance profile (no creds baked into user-data) |

## Dry run (preview without launching)

```bash
uv run scripts/launch_dlc_parallel.py -n 1 --on-demand --use-profile --dry-run
```

This prints the cloud-init user-data script without launching anything.

## Monitoring

Check progress from either local Mac or devcontainer:

```bash
# S3 progress files (works from devcontainer too)
uv run scripts/launch_dlc_parallel.py --progress

# Instance status (local Mac only — needs EC2 API)
uv run scripts/launch_dlc_parallel.py --status
```

SSH into the instance to watch the log live:

```bash
ssh -i ~/.ssh/hm2p-suite2p.pem ubuntu@<IP>
sudo tail -f /var/log/hm2p-dlc.log
```

The IP address is printed when the instance launches, or use `--status` to retrieve it.

## After completion

The instances self-terminate 60 seconds after the last session finishes (or fails).
Progress is uploaded to `s3://hm2p-derivatives/pose/_progress_shard<N>.json` after
each session. Logs are at `s3://hm2p-derivatives/pose/_dlc_log_shard<N>.txt`.

To manually terminate early:

```bash
uv run scripts/launch_dlc_parallel.py --terminate
```

## Expected output (launch)

```
Sessions: 26 total -> 1 shards
  Shard 0: 26 sessions
Instance type: g5.xlarge
Spot: False
Batch config: pose=96, detector=24

Waiting for 1 instances to start... all running!
  Shard 0: i-0xxxxxxxxxxxxxxxxx @ 13.xxx.xxx.xxx

Total: 1 instances, 4 vCPUs
Estimated time: ~20h

Monitor: python scripts/launch_dlc_parallel.py --progress
```

On the instance, you will see 13 "SKIP: already processed" messages followed by
13 sessions being downloaded, subsampled to 30fps, and processed by DLC.
