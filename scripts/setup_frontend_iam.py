#!/usr/bin/env python3
"""Set up IAM role + instance profile for the Streamlit frontend.

Creates a read-only IAM role that an EC2 instance can assume to access S3.
No long-lived access keys needed — the instance gets temporary credentials
automatically via the instance metadata service.

Run this from your local machine (not the devcontainer — IAM is blocked there).

Usage:
    python scripts/setup_frontend_iam.py              # create everything
    python scripts/setup_frontend_iam.py --dry-run     # print AWS CLI commands only
    python scripts/setup_frontend_iam.py --teardown    # delete everything created
"""

from __future__ import annotations

import argparse
import json
import sys

ROLE_NAME = "hm2p-frontend-role"
POLICY_NAME = "hm2p-frontend-readonly"
ACCOUNT_ID = "390897005556"
POLICY_ARN = f"arn:aws:iam::{ACCOUNT_ID}:policy/{POLICY_NAME}"
REGION = "ap-southeast-2"

TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
})

READONLY_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "s3:GetObject",
            "s3:ListBucket",
        ],
        "Resource": [
            "arn:aws:s3:::hm2p-rawdata",
            "arn:aws:s3:::hm2p-rawdata/*",
            "arn:aws:s3:::hm2p-derivatives",
            "arn:aws:s3:::hm2p-derivatives/*",
        ],
    }],
}, indent=2)


def dry_run():
    """Print AWS CLI commands without executing."""
    print("# Run these commands from your local terminal:\n")

    print(f"# 1. Create IAM role (trusts EC2)")
    print(f"aws iam create-role \\")
    print(f"  --role-name {ROLE_NAME} \\")
    print(f"  --assume-role-policy-document '{TRUST_POLICY}'")
    print()

    print(f"# 2. Attach the read-only policy")
    print(f"aws iam attach-role-policy \\")
    print(f"  --role-name {ROLE_NAME} \\")
    print(f"  --policy-arn {POLICY_ARN}")
    print()

    print(f"# 3. Create instance profile")
    print(f"aws iam create-instance-profile \\")
    print(f"  --instance-profile-name {ROLE_NAME}")
    print()

    print(f"# 4. Add role to instance profile")
    print(f"aws iam add-role-to-instance-profile \\")
    print(f"  --instance-profile-name {ROLE_NAME} \\")
    print(f"  --role-name {ROLE_NAME}")
    print()

    print(f"# 5. Wait ~10 seconds for propagation, then launch or attach:")
    print(f"#    New instance:")
    print(f"#      aws ec2 run-instances --iam-instance-profile Name={ROLE_NAME} ...")
    print(f"#    Existing instance:")
    print(f"#      aws ec2 associate-iam-instance-profile \\")
    print(f"#        --instance-id i-xxxxxxxxxxxx \\")
    print(f"#        --iam-instance-profile Name={ROLE_NAME}")


def create():
    """Create role + instance profile."""
    import boto3

    iam = boto3.client("iam")

    # 1. Create or check policy
    try:
        iam.get_policy(PolicyArn=POLICY_ARN)
        print(f"Policy {POLICY_NAME} already exists")
    except iam.exceptions.NoSuchEntityException:
        iam.create_policy(
            PolicyName=POLICY_NAME,
            PolicyDocument=READONLY_POLICY,
            Description="Read-only S3 access to hm2p buckets (frontend)",
        )
        print(f"Created policy: {POLICY_NAME}")

    # 2. Create role
    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"Role {ROLE_NAME} already exists")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=TRUST_POLICY,
            Description="EC2 role for hm2p Streamlit frontend (read-only S3)",
        )
        print(f"Created role: {ROLE_NAME}")

    # 3. Attach policy to role
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=POLICY_ARN)
    print(f"Attached {POLICY_NAME} to {ROLE_NAME}")

    # 4. Create instance profile
    try:
        iam.get_instance_profile(InstanceProfileName=ROLE_NAME)
        print(f"Instance profile {ROLE_NAME} already exists")
    except iam.exceptions.NoSuchEntityException:
        iam.create_instance_profile(InstanceProfileName=ROLE_NAME)
        print(f"Created instance profile: {ROLE_NAME}")

    # 5. Add role to instance profile
    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME,
        )
        print(f"Added {ROLE_NAME} to instance profile")
    except iam.exceptions.LimitExceededException:
        print(f"Role already in instance profile")

    print(f"\nDone! Attach to an EC2 instance with:")
    print(f"  aws ec2 associate-iam-instance-profile \\")
    print(f"    --instance-id i-xxxxxxxxxxxx \\")
    print(f"    --iam-instance-profile Name={ROLE_NAME}")


def teardown():
    """Remove everything created by this script."""
    import boto3

    iam = boto3.client("iam")

    try:
        iam.remove_role_from_instance_profile(
            InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME,
        )
        print(f"Removed role from instance profile")
    except Exception:
        pass

    try:
        iam.delete_instance_profile(InstanceProfileName=ROLE_NAME)
        print(f"Deleted instance profile: {ROLE_NAME}")
    except Exception:
        pass

    try:
        iam.detach_role_policy(RoleName=ROLE_NAME, PolicyArn=POLICY_ARN)
        print(f"Detached policy from role")
    except Exception:
        pass

    try:
        iam.delete_role(RoleName=ROLE_NAME)
        print(f"Deleted role: {ROLE_NAME}")
    except Exception:
        pass

    # Don't delete the policy — it may be attached to the hm2p-frontend user too
    print(f"\nNote: Policy {POLICY_NAME} was NOT deleted (may be used by other users)")


def main():
    parser = argparse.ArgumentParser(
        description="Set up IAM role for hm2p frontend EC2 instance",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="Print AWS CLI commands without executing")
    group.add_argument("--teardown", action="store_true",
                       help="Delete the role and instance profile")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    elif args.teardown:
        teardown()
    else:
        create()


if __name__ == "__main__":
    main()
