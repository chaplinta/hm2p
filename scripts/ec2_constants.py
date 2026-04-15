"""Shared AWS/EC2/S3 constants for all DLC and Suite2p launch scripts.

Import these in launch scripts instead of defining them locally to avoid
drift when values change (e.g. AMI updates, bucket renames).
"""

from __future__ import annotations

# AWS region
REGION = "ap-southeast-2"

# S3 buckets
RAWDATA_BUCKET = "hm2p-rawdata"
DERIVATIVES_BUCKET = "hm2p-derivatives"

# S3 prefixes
RETRAIN_PREFIX = "dlc-retrain"
FINETUNED_PREFIX = "pose-finetuned"

# EC2 instance config — shared across all launch scripts
AMI_ID = "ami-05186a30469f66913"  # Deep Learning Base OSS Nvidia Driver (Ubuntu 22.04)
KEY_NAME = "hm2p-suite2p"
SG_ID = "sg-020161fb424325e6b"
SG_NAME = "hm2p-suite2p-sg"
IAM_PROFILE = "hm2p-ec2-role"

# On-demand pricing in ap-southeast-2 (USD/hr) — used by cost page to compute
# estimated cost from runtime_s without querying Cost Explorer.
INSTANCE_PRICES: dict[str, float] = {
    "g4dn.xlarge": 0.736,
    "g5.xlarge": 1.408,
    "c5.xlarge": 0.214,
}
