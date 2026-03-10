#!/usr/bin/env python3
"""Attach the SSM managed policy to hm2p-ec2-role for Session Manager access.

AWS Systems Manager Session Manager lets you connect to EC2 instances via the
AWS Console or CLI without opening SSH ports, managing key pairs, or using
bastion hosts. All you need is the SSM agent (pre-installed on Amazon Linux 2,
Ubuntu 20.04+, and the Deep Learning AMIs) and the AmazonSSMManagedInstanceCore
policy attached to the instance's IAM role.

Run this from your local machine (not the devcontainer — IAM is blocked there).

Usage:
    uv run scripts/setup_ssm.py              # attach the policy
    uv run scripts/setup_ssm.py --dry-run    # print AWS CLI commands only
    uv run scripts/setup_ssm.py --teardown   # detach the policy
"""

from __future__ import annotations

import argparse
import sys

ROLE_NAME = "hm2p-ec2-role"
SSM_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
ACCOUNT_ID = "390897005556"
REGION = "ap-southeast-2"


def dry_run() -> None:
    """Print AWS CLI commands without executing."""
    print("# Run these commands from your local terminal:\n")

    print("# 1. Attach the SSM managed policy to the EC2 role")
    print(f"aws iam attach-role-policy \\")
    print(f"  --role-name {ROLE_NAME} \\")
    print(f"  --policy-arn {SSM_POLICY_ARN}")
    print()

    print("# 2. Verify the policy is attached")
    print(f"aws iam list-attached-role-policies --role-name {ROLE_NAME}")
    print()

    print("# 3. Connect to an instance via Session Manager")
    print(f"aws ssm start-session --target i-xxxxxxxxxxxx --region {REGION}")


def create() -> None:
    """Attach the SSM managed policy to the EC2 role."""
    import boto3

    iam = boto3.client("iam")

    # Verify the role exists
    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"Role {ROLE_NAME} exists")
    except iam.exceptions.NoSuchEntityException:
        print(f"ERROR: Role {ROLE_NAME} does not exist. Run setup_ec2_iam.py first.")
        sys.exit(1)

    # Check if already attached
    attached = iam.list_attached_role_policies(RoleName=ROLE_NAME)
    for policy in attached["AttachedPolicies"]:
        if policy["PolicyArn"] == SSM_POLICY_ARN:
            print(f"SSM policy already attached to {ROLE_NAME}")
            return

    # Attach the policy
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=SSM_POLICY_ARN)
    print(f"Attached AmazonSSMManagedInstanceCore to {ROLE_NAME}")

    print(f"\nDone! Existing instances with {ROLE_NAME} will pick up SSM access")
    print(f"within a few minutes. Connect with:")
    print(f"  aws ssm start-session --target i-xxxxxxxxxxxx --region {REGION}")


def teardown() -> None:
    """Detach the SSM managed policy from the EC2 role."""
    import boto3

    iam = boto3.client("iam")

    try:
        iam.detach_role_policy(RoleName=ROLE_NAME, PolicyArn=SSM_POLICY_ARN)
        print(f"Detached AmazonSSMManagedInstanceCore from {ROLE_NAME}")
    except Exception as e:
        print(f"Could not detach policy: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach SSM Session Manager policy to hm2p-ec2-role",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="Print AWS CLI commands without executing")
    group.add_argument("--teardown", action="store_true",
                       help="Detach the SSM policy from the role")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    elif args.teardown:
        teardown()
    else:
        create()


if __name__ == "__main__":
    main()
