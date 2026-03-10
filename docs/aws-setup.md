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

## 6. EC2 IAM Role + Instance Profile

EC2 instances need an IAM role to access S3 and CloudWatch without embedded credentials.
The `launch_suite2p_ec2.py` script auto-detects the instance profile and uses it.

### What was created

| Resource | Name | Purpose |
| --- | --- | --- |
| IAM Policy | `hm2p-ec2-policy` | S3 read (`hm2p-rawdata`), S3 read/write (`hm2p-derivatives`), CloudWatch Logs (`/hm2p/suite2p`) |
| IAM Role | `hm2p-ec2-role` | EC2 trusted entity, `hm2p-ec2-policy` attached |
| Instance Profile | `hm2p-ec2-role` | Auto-created with the role (console creates both with same name) |
| CloudWatch Log Group | `/hm2p/suite2p` | Receives logs from EC2 instances |

### Setup (one-time, already done)

**Option A — Python script** (from a machine with admin IAM access):

```bash
python3 scripts/setup_ec2_iam.py
```

**Option B — AWS Console** (as root or admin user):

1. IAM → Policies → Create policy → JSON → paste the policy from `scripts/setup_ec2_iam.py`
   → name it `hm2p-ec2-policy`
2. IAM → Roles → Create role → AWS service → EC2 → attach `hm2p-ec2-policy`
   → name it `hm2p-ec2-role` (this also creates an instance profile with the same name)
3. CloudWatch → Log groups → Create → name `/hm2p/suite2p` (region: ap-southeast-2)

**Option C — AWS CLI / CloudShell:**

```bash
python3 scripts/setup_ec2_iam.py --dry-run
# Prints all 6 aws CLI commands to copy-paste
```

### How it works

When `launch_suite2p_ec2.py` runs:
1. It tries to detect the instance profile via IAM API
2. If found (or if `--use-profile` is passed), it attaches the profile to the instance
   and skips embedding credentials in user-data
3. If not found, it falls back to embedding S3 credentials from `~/.aws/credentials`

The IAM endpoint is blocked from the devcontainer, so use `--use-profile` to force it:

```bash
python scripts/launch_suite2p_ec2.py --use-profile
```

### Monitoring with CloudWatch

When the instance profile is used, logs stream to CloudWatch automatically:

```bash
# From devcontainer
python scripts/launch_suite2p_ec2.py --logs
```

Or view in the AWS Console: CloudWatch → Log groups → `/hm2p/suite2p`.

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

## 8. Running Scripts from macOS

Several setup scripts must be run from **macOS** (not the devcontainer) because
the container firewall blocks AWS IAM/EC2 API calls.

**Important:** By default, `uv run` uses `.venv/` in the project directory — the
same venv the devcontainer uses. Running `uv run` on macOS will rebuild `.venv`
for macOS Python, breaking the container's venv.

**Fix:** Set `UV_PROJECT_ENVIRONMENT` on your Mac to use a separate venv:

```bash
# Add to ~/.zshrc (one-time)
export UV_PROJECT_ENVIRONMENT="$HOME/.venv-hm2p"
source ~/.zshrc

# One-time setup
cd ~/Neuro/hm2p-v2
uv sync --extra dev
```

From then on, `uv run` on Mac uses `~/.venv-hm2p` and never touches `.venv/` in
the project. The devcontainer ignores this env var (it's not set there).

---

## 9. Frontend Read-Only Access

The Streamlit frontend only needs read access to S3. Two approaches are
available — an **EC2 instance role** (recommended, no keys) or an
**IAM user with access keys** (fallback for local dev).

### 8.1 Option A — EC2 Instance Role (recommended)

When hosting the frontend on EC2, attach an IAM role directly. The instance
gets temporary credentials automatically via the metadata service — no keys
to manage, rotate, or leak.

#### Resources created

| Resource | Name | Purpose |
| --- | --- | --- |
| IAM Policy | `hm2p-frontend-readonly` | `s3:GetObject` + `s3:ListBucket` on both hm2p buckets |
| IAM Role | `hm2p-frontend-role` | EC2 trusted entity, readonly policy attached |
| Instance Profile | `hm2p-frontend-role` | Attaches the role to an EC2 instance |

#### Setup

Run from your **local machine** (not the devcontainer — IAM is blocked there):

```bash
# Automated setup
uv run scripts/setup_frontend_iam.py

# Or dry-run to see the AWS CLI commands first
uv run scripts/setup_frontend_iam.py --dry-run
```

The script creates the role, instance profile, and attaches the policy.

#### Attach to an EC2 instance

```bash
# New instance — include at launch time
aws ec2 run-instances \
  --image-id ami-0c5204531f799e0c6 \
  --instance-type t3.micro \
  --key-name hm2p-suite2p \
  --security-group-ids sg-020161fb424325e6b \
  --iam-instance-profile Name=hm2p-frontend-role \
  --region ap-southeast-2 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hm2p-frontend}]'

# Existing instance — attach the profile
aws ec2 associate-iam-instance-profile \
  --instance-id i-xxxxxxxxxxxx \
  --iam-instance-profile Name=hm2p-frontend-role
```

Once attached, `boto3` on the instance picks up credentials automatically —
no environment variables, no `~/.aws/credentials`, no configuration needed.

#### Teardown

```bash
uv run scripts/setup_frontend_iam.py --teardown
```

#### How it works

1. EC2 metadata service (`169.254.169.254`) provides temporary credentials
2. Credentials rotate automatically every few hours
3. `boto3` checks the metadata service by default (no config needed)
4. If the instance is terminated, credentials are immediately revoked

#### Cost

`t3.micro` is ~$9.50/month (~$0.013/hr), or free-tier eligible for the
first year. Sufficient for Streamlit with a few concurrent users.

### 8.2 Option B — IAM User with Access Keys (fallback)

For local development or environments where an instance role isn't available.
Less secure than Option A — keys are long-lived and must be rotated manually.

| Resource | Name | Purpose |
| --- | --- | --- |
| IAM Policy | `hm2p-frontend-readonly` | Same policy as Option A |
| IAM User | `hm2p-frontend` | Frontend-only identity |

#### Generate access keys

```bash
aws iam create-access-key --user-name hm2p-frontend
```

Save the `AccessKeyId` and `SecretAccessKey` — the secret cannot be
retrieved again.

#### Configure

```bash
# Environment variables
export AWS_ACCESS_KEY_ID=<access key>
export AWS_SECRET_ACCESS_KEY=<secret>
export AWS_DEFAULT_REGION=ap-southeast-2

# Or a named profile
aws configure --profile hm2p-frontend
export AWS_PROFILE=hm2p-frontend
```

#### Security notes

- **No write access** — cannot upload, delete, or modify S3 objects
- **No access** to EC2, IAM, Batch, or any other AWS service
- **Rotate keys regularly**: IAM → Users → hm2p-frontend → Security credentials
- Prefer Option A (instance role) whenever possible

### Policy document (shared by both options)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::hm2p-rawdata",
        "arn:aws:s3:::hm2p-rawdata/*",
        "arn:aws:s3:::hm2p-derivatives",
        "arn:aws:s3:::hm2p-derivatives/*"
      ]
    }
  ]
}
```

> Note: `s3:GetObject` also authorises the `HeadObject` API call — there is
> no separate `s3:HeadObject` IAM action.

---

## 10. SSM Session Manager

AWS Systems Manager (SSM) Session Manager lets you connect to EC2 instances
from the AWS Console or CLI — no SSH keys, no open ports, no bastion hosts.

### Why SSM over SSH

| | SSH | SSM |
| --- | --- | --- |
| Open port 22 | Required | Not needed |
| Key pair management | `.pem` files | None |
| Audit logging | Manual | Built-in (CloudTrail) |
| Works from Console | No | Yes (browser shell) |
| IAM-based access | No | Yes |

### Prerequisites

1. The EC2 instance must have an IAM role with `AmazonSSMManagedInstanceCore`
2. The SSM agent must be running (pre-installed on Amazon Linux 2, Ubuntu 20.04+,
   and all Deep Learning AMIs)

### Setup

Run from your **local machine** (not the devcontainer — IAM is blocked there):

```bash
# Attach SSM policy to the existing hm2p-ec2-role
uv run scripts/setup_ssm.py

# Or dry-run to see the AWS CLI commands first
uv run scripts/setup_ssm.py --dry-run
```

### Connecting

```bash
# CLI (requires the Session Manager plugin for AWS CLI)
aws ssm start-session --target i-xxxxxxxxxxxx --region ap-southeast-2

# Or use the AWS Console: EC2 → Instances → select instance → Connect → Session Manager
```

Install the Session Manager plugin:

```bash
# macOS
brew install --cask session-manager-plugin
```

### Teardown

```bash
uv run scripts/setup_ssm.py --teardown
```

---

## 11. Auto-Shutdown (EventBridge + Lambda)

Automatically stop and start EC2 instances on a daily schedule to avoid
overnight charges. Targets instances tagged with `Project=hm2p-dlc`,
`Project=hm2p`, or `Name=hm2p-frontend`.

### What it creates

| Resource | Name | Purpose |
| --- | --- | --- |
| IAM Role | `hm2p-lambda-scheduler` | Lambda execution role (EC2 + CloudWatch Logs) |
| IAM Policy | `hm2p-lambda-scheduler-policy` | ec2:Stop/Start/Describe + logs permissions |
| Lambda | `hm2p-stop-instances` | Stops matching running instances |
| Lambda | `hm2p-start-instances` | Starts matching stopped instances |
| EventBridge Rule | `hm2p-nightly-stop` | Triggers stop Lambda on schedule |
| EventBridge Rule | `hm2p-morning-start` | Triggers start Lambda on schedule |

### Default schedule

- **Stop**: 22:00 AWST (14:00 UTC) — end of work day
- **Start**: 08:00 AWST (00:00 UTC) — start of work day

### Setup

Run from your **local machine** (not the devcontainer — IAM is blocked there):

```bash
# Create everything with default hours
uv run scripts/setup_auto_shutdown.py

# Dry-run to see what would be created
uv run scripts/setup_auto_shutdown.py --dry-run

# Custom hours (AWST)
uv run scripts/setup_auto_shutdown.py --stop-hour 23 --start-hour 9
```

### Teardown

```bash
uv run scripts/setup_auto_shutdown.py --teardown
```

### How it works

1. EventBridge fires a cron rule at the scheduled time
2. The rule invokes a Lambda function
3. Lambda queries EC2 for instances matching the target tags
4. Lambda stops (or starts) the matching instances
5. Execution logs go to CloudWatch Logs

If you don't want an instance managed by the scheduler, remove or change its
`Project` / `Name` tag.

---

## 12. Security Group Lockdown

Restrict the `hm2p-suite2p-sg` security group so that only a specific IP can
reach the frontend (port 8501) and SSH (port 22). Also removes any existing
wide-open (`0.0.0.0/0` or `::/0`) rules on port 8501.

Run from **macOS** (not the devcontainer -- the container firewall blocks AWS
API calls):

```bash
# Apply lockdown (default IP: 2001:4860:7:801::e1)
uv run scripts/setup_sg_lockdown.py

# Use a different IP
uv run scripts/setup_sg_lockdown.py --ip 2001:db8::1

# Use a different security group
uv run scripts/setup_sg_lockdown.py --sg-id sg-0123456789abcdef0

# Dry-run -- show what would change without doing anything
uv run scripts/setup_sg_lockdown.py --dry-run

# Remove the rules added by this script
uv run scripts/setup_sg_lockdown.py --teardown
```

### What it does

1. Describes the security group to find existing rules
2. Revokes any wide-open inbound rules on port 8501
3. Adds TCP port 22 (SSH) and 8501 (Streamlit) from the specified IPv6 /128
4. Skips rules that already exist (idempotent)

---

## 13. S3 Access Logging

Enable server access logging on `hm2p-rawdata` and `hm2p-derivatives`. Logs
go to a dedicated `hm2p-access-logs` bucket with a 90-day expiration lifecycle.

Run from **macOS**:

```bash
# Enable logging
uv run scripts/setup_s3_logging.py

# Dry-run
uv run scripts/setup_s3_logging.py --dry-run

# Disable logging (keep the logging bucket)
uv run scripts/setup_s3_logging.py --teardown

# Disable logging and delete the logging bucket
uv run scripts/setup_s3_logging.py --teardown --delete-logging-bucket
```

### What it creates

| Resource | Name | Purpose |
| --- | --- | --- |
| S3 Bucket | `hm2p-access-logs` | Stores access logs |
| Lifecycle Rule | `expire-logs-90-days` | Deletes log objects after 90 days |
| Logging Config | on `hm2p-rawdata` | Writes logs to `rawdata-logs/` prefix |
| Logging Config | on `hm2p-derivatives` | Writes logs to `derivatives-logs/` prefix |

The logging bucket has all public access blocked.

---

## 14. Cost Controls

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

## 15. Verify Upload Integrity

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

## 16. Current Status

| Item | Status |
| --- | --- |
| S3 buckets (`hm2p-rawdata`, `hm2p-derivatives`) | Created in `ap-southeast-2` |
| `hm2p-rawdata` versioning | Enabled |
| `hm2p-rawdata` lifecycle | STANDARD → IA after 30 days |
| Data upload (26 sessions, 91.4 GiB, 503 objects) | Complete — verified |
| `hm2p-agent` IAM user | S3 + EC2 access (no IAM/Batch) |
| `hm2p-frontend` IAM user | Read-only S3 access (fallback for local dev) |
| `hm2p-frontend-role` IAM role + instance profile | EC2 instance role for frontend (recommended, no keys) |
| `hm2p-frontend-readonly` IAM policy | `s3:GetObject` + `s3:ListBucket` on both hm2p buckets |
| `hm2p-ec2-role` IAM role + instance profile | S3 + CloudWatch Logs (for EC2 instances) |
| `hm2p-ec2-policy` IAM policy | Scoped to hm2p S3 buckets + `/hm2p/suite2p` log group |
| `/hm2p/suite2p` CloudWatch log group | Created in ap-southeast-2 |
| Suite2p cloud run (26 sessions) | Complete — all outputs in `s3://hm2p-derivatives/ca_extraction/` |
| SSM Session Manager on `hm2p-ec2-role` | Ready — run `setup_ssm.py` to attach policy |
| Auto-shutdown (EventBridge + Lambda) | Ready — run `setup_auto_shutdown.py` to create |
| Security group lockdown (`hm2p-suite2p-sg`) | Ready -- run `setup_sg_lockdown.py` from macOS |
| S3 access logging (`hm2p-access-logs`) | Ready -- run `setup_s3_logging.py` from macOS |
| AWS Batch (compute envs + job queues) | Not yet created (needs admin) |

---

## 17. Google OAuth Authentication (Frontend)

The Streamlit frontend supports Google OAuth login to restrict access to
authorised users only. Authentication is **optional** — when the required
environment variables are absent, it is skipped entirely (local dev mode).

### 17.1 Create OAuth credentials (one-time)

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or use an existing one)
3. Navigate to **APIs & Services > Credentials**
4. Click **Create Credentials > OAuth 2.0 Client ID**
5. Application type: **Web application**
6. Name: e.g. `hm2p-dashboard`
7. Authorised redirect URIs: add the URL where the frontend is hosted, e.g.
   `http://<EC2-IP>:8501` or `http://localhost:8501` for local testing
8. Click **Create** and note the **Client ID** and **Client Secret**

You must also configure the **OAuth consent screen** (APIs & Services >
OAuth consent screen):
- User type: **External** (or Internal if using Google Workspace)
- Add your email as a test user during development
- No scopes need to be added beyond the defaults (email, profile, openid)

### 17.2 Set environment variables on EC2

```bash
export GOOGLE_CLIENT_ID="your-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-client-secret"
export STREAMLIT_REDIRECT_URI="http://<EC2-PUBLIC-IP>:8501"
```

To persist across reboots, add these to `/etc/environment` or a systemd
service file.

### 17.3 Environment variables reference

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GOOGLE_CLIENT_ID` | No | (empty) | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | No | (empty) | Google OAuth client secret |
| `STREAMLIT_REDIRECT_URI` | No | `http://localhost:8501` | OAuth callback URL |

### 17.4 Behaviour

- **Both `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` set**: authentication
  is enforced. Only email addresses in the allowed list (currently
  `tristan.chaplin@gmail.com`) can access the dashboard.
- **Either variable missing or empty**: authentication is skipped entirely.
  The dashboard loads without any login gate. This is the default for local
  development.
- The allowed email list is defined in `frontend/app.py` (`_ALLOWED_EMAILS`).
  To add more users, append their Gmail address to that list.

### 17.5 Security notes

- OAuth credentials are read from environment variables — **never hardcode
  them** in source code or commit them to git.
- The temporary credentials JSON is written to `/tmp` at runtime and cached
  for the lifetime of the Streamlit process.
- A login cookie (`hm2p_auth`) is stored in the browser for 30 days to avoid
  re-authentication on every page load.

---

## 18. Quick Reference

| Task | Command |
| --- | --- |
| List S3 buckets | `aws s3 ls` |
| Upload a file | `aws s3 cp file.tif s3://hm2p-rawdata/path/` |
| Sync a folder | `aws s3 sync ./local/ s3://hm2p-rawdata/remote/` |
| Verify upload | `./scripts/verify_s3_upload.sh` |
| Launch EC2 spot | `aws ec2 run-instances --instance-type g4dn.xlarge --instance-market-options MarketType=spot ...` |
| Check Batch jobs | `aws batch list-jobs --job-queue hm2p-gpu-queue` |
| Check current IAM identity | `aws sts get-caller-identity` |
