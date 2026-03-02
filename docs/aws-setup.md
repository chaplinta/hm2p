# AWS Setup Guide

## 1. Create an AWS Account

1. Go to <https://aws.amazon.com> and click **Create an AWS Account**
2. Enter your email address and choose an account name (e.g. `hm2p-lab`)
3. Provide billing details (credit/debit card required — you will not be charged until you
   use paid services; the free tier covers small experiments)
4. Complete phone verification
5. Choose the **Basic (free)** support plan
6. Sign in to the **AWS Management Console** at <https://console.aws.amazon.com>

> Keep the account root email and password somewhere secure (e.g. a password manager).
> The root account should only be used for initial setup — all day-to-day work uses IAM users.

---

## 2. Authentication Model

AWS has two layers of identity:

### 2.1 Root Account

The email/password you used to create the account. Has unlimited access to everything.
**Do not use for day-to-day work** — only for billing and initial IAM setup.

### 2.2 IAM (Identity and Access Management)

IAM lets you create users, roles, and permissions. For this project you need:

| Identity | Purpose |
| --- | --- |
| **IAM user** (you) | Human access — Console + CLI |
| **IAM role** (EC2) | EC2 instances assume this role to read/write S3 |
| **IAM role** (Batch) | AWS Batch job role |

### 2.3 Access Keys vs IAM Roles

| Method | When to use |
| --- | --- |
| **Access keys** (key ID + secret) | CLI from your laptop; CI/CD secrets |
| **IAM roles** | EC2 instances and Batch jobs — no keys needed, role is assumed automatically |

Never hardcode access keys in code or commit them to git.

---

## 3. Initial IAM Setup (one-time)

### 3.1 Create an IAM user for yourself

1. In the Console, go to **IAM → Users → Create user**
2. Username: e.g. `tristan`
3. Enable **Console access** (set a password)
4. Attach policy: `AdministratorAccess` (for initial setup; restrict later if needed)
5. Click through and save the user

### 3.2 Enable MFA on the root account

1. **IAM → Dashboard → Add MFA for root user**
2. Use an authenticator app (e.g. Google Authenticator, 1Password)

### 3.3 Create access keys for CLI use

1. **IAM → Users → tristan → Security credentials → Create access key**
2. Choose **CLI** as the use case
3. Download the `.csv` — store it securely, you cannot retrieve the secret again

---

## 4. Configure the AWS CLI

Install:

```bash
# macOS
brew install awscli

# or via uv/pip
uv tool install awscli
```

Configure:

```bash
aws configure
# AWS Access Key ID: <your key ID>
# AWS Secret Access Key: <your secret>
# Default region name: ap-southeast-2   # Sydney — closest to Perth
# Default output format: json
```

This writes credentials to `~/.aws/credentials` and config to `~/.aws/config`.

Verify:

```bash
aws sts get-caller-identity
# Should return your account ID and IAM user ARN
```

---

## 5. S3 Bucket Setup

Create the two project buckets (do this once):

```bash
aws s3 mb s3://hm2p-rawdata    --region ap-southeast-2
aws s3 mb s3://hm2p-derivatives --region ap-southeast-2
```

Enable versioning on the rawdata bucket (protects against accidental deletion):

```bash
aws s3api put-bucket-versioning \
  --bucket hm2p-rawdata \
  --versioning-configuration Status=Enabled
```

Set lifecycle policy to move rawdata to Infrequent Access after 30 days:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket hm2p-rawdata \
  --lifecycle-configuration file://docs/s3-lifecycle-rawdata.json
```

---

## 6. EC2 IAM Role (for instances to access S3)

Create a role that EC2 instances assume automatically — no keys needed on the instance:

1. **IAM → Roles → Create role**
2. Trusted entity: **AWS service → EC2**
3. Attach policies:
   - `AmazonS3FullAccess` (or a custom policy scoped to `hm2p-*` buckets)
   - `AmazonEC2ContainerRegistryReadOnly` (if pulling Docker images from ECR)
4. Name: `hm2p-ec2-role`

When launching an EC2 instance, assign this role under **IAM instance profile**.
The instance can then run `aws s3 cp ...` without any credentials configured.

---

## 7. AWS Batch Setup (for Snakemake cloud jobs)

AWS Batch is the recommended executor for the Snakemake `aws-batch` profile.

### 7.1 Compute Environment

1. **AWS Batch → Compute environments → Create**
2. Type: **Managed**
3. Instance type: `g4dn.xlarge` (GPU) and `c5.4xlarge` (CPU) — use **Spot**
4. Min/max vCPUs: 0 / 256
5. Attach IAM role: `hm2p-ec2-role`

### 7.2 Job Queue

1. **Batch → Job queues → Create**
2. Name: `hm2p-gpu-queue` and `hm2p-cpu-queue`
3. Priority: 1
4. Attach the compute environment

### 7.3 Snakemake profile

Configure `workflow/profiles/aws-batch/config.yaml` with the queue ARNs.
See the [Snakemake AWS Batch executor docs](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/aws-batch.html).

---

## 7.5 ECR — Container Image Registry

The pipeline uses two Docker images pushed to Amazon ECR (Elastic Container Registry):

- **hm2p-cpu** — Stages 0, 3, 4, 5 (CPU only, `python:3.11-slim` base)
- **hm2p-gpu** — Stages 1, 2 (CUDA 12.1 + Suite2p + DLC, `nvidia/cuda` base)

### One-time setup + push

```bash
# Build images, create ECR repos, push (interactive — prompts for confirmation)
./scripts/ecr_push.sh --region ap-southeast-2

# Or with a named profile:
./scripts/ecr_push.sh --region ap-southeast-2 --profile hm2p-agent
```

The script will print the ECR prefix. Copy it into `config/pipeline.yaml`:

```yaml
ecr_prefix: "123456789012.dkr.ecr.ap-southeast-2.amazonaws.com/hm2p"
```

### Rebuilding after code changes

```bash
# Rebuild and push both images
./scripts/ecr_push.sh

# Or build locally without pushing (for testing)
docker build -f docker/cpu.Dockerfile -t hm2p-cpu .
docker build -f docker/gpu.Dockerfile -t hm2p-gpu .
```

### Batch job execution role

Ensure the Batch job execution role has `ecr:GetAuthorizationToken` and
`ecr:BatchGetImage` permissions. The managed policy
`AmazonEC2ContainerRegistryReadOnly` covers this.

---

## 8. Cost Controls

Always set a billing alarm to avoid unexpected charges:

1. **Billing → Budgets → Create budget**
2. Type: **Cost budget**
3. Amount: e.g. £100/month
4. Alert at 80% and 100%
5. Email: your address

Also enable **AWS Cost Explorer** to see spending by service.

Spot Instance interruptions: Snakemake handles these automatically by resubmitting failed
jobs. Suite2p and DLC both support checkpointing.

---

## 9. Verify Upload Integrity

After uploading data to S3, verify integrity with:

```bash
./scripts/verify_s3_upload.sh
```

This runs five checks against the live S3 bucket (single API call, ~10 seconds):

1. **Session completeness** — all 26 sessions from `experiments.csv` exist in S3
2. **Directory structure** — each session has `funcimg/` and `behav/` subdirectories
3. **Object counts + sizes** — per-session file count and total size summary
4. **MD5 checksums** — `behav/meta.txt` local MD5 vs S3 ETag (single-part uploads)
5. **TDMS file sizes** — byte-exact local vs S3 size comparison (multipart uploads)

Override defaults with `--profile` or `--bucket`:

```bash
./scripts/verify_s3_upload.sh --profile my-profile --bucket my-bucket
```

---

## 10. Current Status

| Item | Status |
| --- | --- |
| S3 buckets (`hm2p-rawdata`, `hm2p-derivatives`) | Created in `ap-southeast-2` |
| `hm2p-rawdata` versioning | Enabled |
| `hm2p-rawdata` lifecycle | STANDARD → IA after 30 days |
| Data upload (26 sessions, 91.4 GiB, 503 objects) | Complete — verified |
| `hm2p-agent` IAM user | S3 access only (no IAM/Batch) |
| AWS Batch (compute envs + job queues) | Not yet created (needs admin) |

---

## 11. Quick Reference

| Task | Command |
| --- | --- |
| List S3 buckets | `aws s3 ls` |
| Upload a file | `aws s3 cp file.tif s3://hm2p-rawdata/path/` |
| Sync a folder | `aws s3 sync ./local/ s3://hm2p-rawdata/remote/` |
| Verify upload | `./scripts/verify_s3_upload.sh` |
| Launch EC2 spot | `aws ec2 run-instances --instance-type g4dn.xlarge --instance-market-options MarketType=spot ...` |
| Check Batch jobs | `aws batch list-jobs --job-queue hm2p-gpu-queue` |
| Check current IAM identity | `aws sts get-caller-identity` |
