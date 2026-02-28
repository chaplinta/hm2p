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

## 9. Quick Reference

| Task | Command |
| --- | --- |
| List S3 buckets | `aws s3 ls` |
| Upload a file | `aws s3 cp file.tif s3://hm2p-rawdata/path/` |
| Sync a folder | `aws s3 sync ./local/ s3://hm2p-rawdata/remote/` |
| Launch EC2 spot | `aws ec2 run-instances --instance-type g4dn.xlarge --instance-market-options MarketType=spot ...` |
| Check Batch jobs | `aws batch list-jobs --job-queue hm2p-gpu-queue` |
| Check current IAM identity | `aws sts get-caller-identity` |
