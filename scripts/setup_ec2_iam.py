#!/usr/bin/env python3
"""Create IAM role + instance profile for hm2p EC2 instances.

This gives EC2 instances S3 + CloudWatch Logs access via an IAM role,
eliminating the need to embed AWS credentials in user-data scripts.

Usage:
    # From a machine with IAM permissions (NOT the devcontainer — IAM endpoint blocked):
    python scripts/setup_ec2_iam.py

    # Or use the AWS CLI directly:
    aws iam create-role ...   (see --dry-run output)

    # Verify it was created:
    python scripts/setup_ec2_iam.py --check

What it creates:
    - IAM Policy: hm2p-ec2-policy (S3 + CloudWatch Logs access)
    - IAM Role: hm2p-ec2-role (EC2 trusted entity)
    - Instance Profile: hm2p-ec2-profile (used by run_instances)

After running this, launch_suite2p_ec2.py will automatically detect and use
the instance profile, and the user-data script will skip embedding credentials.
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3

REGION = "ap-southeast-2"
ACCOUNT_ID = "390897005556"

ROLE_NAME = "hm2p-ec2-role"
POLICY_NAME = "hm2p-ec2-policy"
PROFILE_NAME = "hm2p-ec2-profile"
LOG_GROUP = "/hm2p/suite2p"

# Policy: S3 access to hm2p buckets + CloudWatch Logs
POLICY_DOCUMENT = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3ReadRawData",
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [
                "arn:aws:s3:::hm2p-rawdata",
                "arn:aws:s3:::hm2p-rawdata/*",
            ],
        },
        {
            "Sid": "S3ReadWriteDerivatives",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket",
                "s3:DeleteObject",
            ],
            "Resource": [
                "arn:aws:s3:::hm2p-derivatives",
                "arn:aws:s3:::hm2p-derivatives/*",
            ],
        },
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
            ],
            "Resource": [
                f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:{LOG_GROUP}",
                f"arn:aws:logs:{REGION}:{ACCOUNT_ID}:log-group:{LOG_GROUP}:*",
            ],
        },
    ],
}

# Trust policy: allow EC2 to assume this role
TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def create_all() -> None:
    """Create IAM policy, role, and instance profile."""
    iam = boto3.client("iam")

    # 1. Create policy
    print(f"Creating IAM policy: {POLICY_NAME}")
    try:
        resp = iam.create_policy(
            PolicyName=POLICY_NAME,
            PolicyDocument=json.dumps(POLICY_DOCUMENT),
            Description="hm2p EC2 instances: S3 buckets + CloudWatch Logs",
        )
        policy_arn = resp["Policy"]["Arn"]
        print(f"  Created: {policy_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        policy_arn = f"arn:aws:iam::{ACCOUNT_ID}:policy/{POLICY_NAME}"
        print(f"  Already exists: {policy_arn}")

    # 2. Create role
    print(f"Creating IAM role: {ROLE_NAME}")
    try:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="hm2p EC2 compute role (S3 + CloudWatch)",
        )
        print(f"  Created")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"  Already exists")

    # 3. Attach policy to role
    print(f"Attaching policy to role")
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
    print(f"  Done")

    # 4. Create instance profile
    print(f"Creating instance profile: {PROFILE_NAME}")
    try:
        iam.create_instance_profile(InstanceProfileName=PROFILE_NAME)
        print(f"  Created")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"  Already exists")

    # 5. Add role to instance profile
    print(f"Adding role to instance profile")
    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=PROFILE_NAME,
            RoleName=ROLE_NAME,
        )
        print(f"  Done")
    except iam.exceptions.LimitExceededException:
        print(f"  Role already attached")

    print(f"\nSetup complete. Instance profile '{PROFILE_NAME}' is ready.")
    print(f"launch_suite2p_ec2.py will auto-detect it on next launch.")

    # 6. Create CloudWatch log group
    logs = boto3.client("logs", region_name=REGION)
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        print(f"Created CloudWatch log group: {LOG_GROUP}")
    except logs.exceptions.ResourceAlreadyExistsException:
        print(f"CloudWatch log group already exists: {LOG_GROUP}")


def check() -> None:
    """Check if the IAM resources exist."""
    iam = boto3.client("iam")

    try:
        resp = iam.get_instance_profile(InstanceProfileName=PROFILE_NAME)
        roles = resp["InstanceProfile"]["Roles"]
        print(f"Instance profile: {PROFILE_NAME}")
        print(f"  ARN: {resp['InstanceProfile']['Arn']}")
        if roles:
            print(f"  Role: {roles[0]['RoleName']}")
        else:
            print(f"  WARNING: No role attached!")
    except iam.exceptions.NoSuchEntityException:
        print(f"Instance profile '{PROFILE_NAME}' does not exist.")
        print(f"Run: python scripts/setup_ec2_iam.py")
        return

    # Check policy
    policy_arn = f"arn:aws:iam::{ACCOUNT_ID}:policy/{POLICY_NAME}"
    try:
        iam.get_policy(PolicyArn=policy_arn)
        print(f"  Policy: {POLICY_NAME} (attached)")
    except iam.exceptions.NoSuchEntityException:
        print(f"  WARNING: Policy '{POLICY_NAME}' not found!")


def dry_run() -> None:
    """Print the equivalent AWS CLI commands."""
    print("# Run these commands from a machine with IAM permissions:")
    print()
    policy_json = json.dumps(POLICY_DOCUMENT)
    trust_json = json.dumps(TRUST_POLICY)
    print(f"# 1. Create policy")
    print(f"aws iam create-policy \\")
    print(f"  --policy-name {POLICY_NAME} \\")
    print(f"  --policy-document '{policy_json}'")
    print()
    print(f"# 2. Create role")
    print(f"aws iam create-role \\")
    print(f"  --role-name {ROLE_NAME} \\")
    print(f"  --assume-role-policy-document '{trust_json}'")
    print()
    print(f"# 3. Attach policy to role")
    print(f"aws iam attach-role-policy \\")
    print(f"  --role-name {ROLE_NAME} \\")
    print(f"  --policy-arn arn:aws:iam::{ACCOUNT_ID}:policy/{POLICY_NAME}")
    print()
    print(f"# 4. Create instance profile")
    print(f"aws iam create-instance-profile \\")
    print(f"  --instance-profile-name {PROFILE_NAME}")
    print()
    print(f"# 5. Add role to instance profile")
    print(f"aws iam add-role-to-instance-profile \\")
    print(f"  --instance-profile-name {PROFILE_NAME} \\")
    print(f"  --role-name {ROLE_NAME}")
    print()
    print(f"# 6. Create CloudWatch log group")
    print(f"aws logs create-log-group \\")
    print(f"  --log-group-name {LOG_GROUP} \\")
    print(f"  --region {REGION}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup IAM for hm2p EC2 instances")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check if resources exist")
    group.add_argument("--dry-run", action="store_true", help="Print AWS CLI commands")
    args = parser.parse_args()

    if args.check:
        check()
    elif args.dry_run:
        dry_run()
    else:
        create_all()


if __name__ == "__main__":
    main()
