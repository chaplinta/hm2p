#!/usr/bin/env python3
"""Enable server access logging on hm2p S3 buckets.

Creates a dedicated logging bucket (hm2p-access-logs) and configures
hm2p-rawdata and hm2p-derivatives to write access logs there. The logging
bucket has public access blocked and a 90-day lifecycle expiration.

Run this from your local machine (not the devcontainer — IAM is blocked there).

Usage:
    uv run scripts/setup_s3_logging.py
    uv run scripts/setup_s3_logging.py --dry-run
    uv run scripts/setup_s3_logging.py --teardown
    uv run scripts/setup_s3_logging.py --teardown --delete-logging-bucket
"""

from __future__ import annotations

import argparse
import json
import sys

REGION = "ap-southeast-2"
LOGGING_BUCKET = "hm2p-access-logs"

# Source buckets and their log prefixes
BUCKETS = [
    ("hm2p-rawdata", "rawdata-logs/"),
    ("hm2p-derivatives", "derivatives-logs/"),
]

LIFECYCLE_RULE = {
    "Rules": [
        {
            "ID": "expire-logs-90-days",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Expiration": {"Days": 90},
        }
    ]
}


def dry_run() -> None:
    """Print what would change without making changes."""
    print(f"# Logging bucket: {LOGGING_BUCKET} (region: {REGION})")
    print()

    print("# 1. Create the logging bucket")
    print(f"aws s3api create-bucket \\")
    print(f"  --bucket {LOGGING_BUCKET} --region {REGION} \\")
    print(f"  --create-bucket-configuration LocationConstraint={REGION}")
    print()

    print("# 2. Block all public access on the logging bucket")
    print(f"aws s3api put-public-access-block \\")
    print(f"  --bucket {LOGGING_BUCKET} \\")
    print(f"  --public-access-block-configuration "
          f"BlockPublicAcls=true,IgnorePublicAcls=true,"
          f"BlockPublicPolicy=true,RestrictPublicBuckets=true")
    print()

    print("# 3. Set 90-day lifecycle on the logging bucket")
    print(f"aws s3api put-bucket-lifecycle-configuration \\")
    print(f"  --bucket {LOGGING_BUCKET} \\")
    print(f"  --lifecycle-configuration '{json.dumps(LIFECYCLE_RULE)}'")
    print()

    for bucket, prefix in BUCKETS:
        print(f"# 4. Enable logging on {bucket} -> {LOGGING_BUCKET}/{prefix}")
        logging_config = {
            "LoggingEnabled": {
                "TargetBucket": LOGGING_BUCKET,
                "TargetPrefix": prefix,
            }
        }
        print(f"aws s3api put-bucket-logging \\")
        print(f"  --bucket {bucket} \\")
        print(f"  --bucket-logging-status '{json.dumps(logging_config)}'")
        print()


def _ensure_logging_bucket(s3) -> None:
    """Create the logging bucket if it does not exist."""
    try:
        s3.head_bucket(Bucket=LOGGING_BUCKET)
        print(f"Logging bucket {LOGGING_BUCKET} already exists")
    except s3.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            print(f"Creating logging bucket: {LOGGING_BUCKET}")
            s3.create_bucket(
                Bucket=LOGGING_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
            print(f"  Created {LOGGING_BUCKET} in {REGION}")
        elif error_code == 403:
            print(f"ERROR: Access denied to {LOGGING_BUCKET}. "
                  f"Check your AWS credentials.")
            sys.exit(1)
        else:
            raise

    # Block all public access
    s3.put_public_access_block(
        Bucket=LOGGING_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"  Public access blocked on {LOGGING_BUCKET}")

    # Set lifecycle rule
    s3.put_bucket_lifecycle_configuration(
        Bucket=LOGGING_BUCKET,
        LifecycleConfiguration=LIFECYCLE_RULE,
    )
    print(f"  Lifecycle rule: delete logs after 90 days")


def create() -> None:
    """Enable S3 access logging on hm2p buckets."""
    import boto3

    s3 = boto3.client("s3", region_name=REGION)

    # Step 1: Ensure logging bucket exists and is configured
    _ensure_logging_bucket(s3)

    # Step 2: Enable logging on each source bucket
    for bucket, prefix in BUCKETS:
        try:
            s3.put_bucket_logging(
                Bucket=bucket,
                BucketLoggingStatus={
                    "LoggingEnabled": {
                        "TargetBucket": LOGGING_BUCKET,
                        "TargetPrefix": prefix,
                    }
                },
            )
            print(f"  Enabled logging: {bucket} -> {LOGGING_BUCKET}/{prefix}")
        except Exception as e:
            print(f"  ERROR enabling logging on {bucket}: {e}")

    print(f"\nDone! Access logs for both buckets will appear in s3://{LOGGING_BUCKET}/")


def teardown(delete_bucket: bool = False) -> None:
    """Disable logging and optionally delete the logging bucket."""
    import boto3

    s3 = boto3.client("s3", region_name=REGION)

    # Disable logging on source buckets
    for bucket, prefix in BUCKETS:
        try:
            s3.put_bucket_logging(
                Bucket=bucket,
                BucketLoggingStatus={},
            )
            print(f"Disabled logging on {bucket}")
        except Exception as e:
            print(f"Warning: could not disable logging on {bucket}: {e}")

    if delete_bucket:
        # Empty and delete the logging bucket
        try:
            s3.head_bucket(Bucket=LOGGING_BUCKET)
        except Exception:
            print(f"Logging bucket {LOGGING_BUCKET} does not exist, nothing to delete")
            return

        print(f"Emptying {LOGGING_BUCKET}...")
        s3_resource = boto3.resource("s3", region_name=REGION)
        bucket_obj = s3_resource.Bucket(LOGGING_BUCKET)
        bucket_obj.objects.all().delete()
        print(f"  All objects deleted from {LOGGING_BUCKET}")

        s3.delete_bucket(Bucket=LOGGING_BUCKET)
        print(f"  Deleted bucket {LOGGING_BUCKET}")
    else:
        print(f"\nLogging bucket {LOGGING_BUCKET} was kept. "
              f"Use --delete-logging-bucket to remove it.")

    print("\nTeardown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enable server access logging on hm2p S3 buckets",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without making changes",
    )
    group.add_argument(
        "--teardown", action="store_true",
        help="Disable logging on hm2p buckets",
    )
    parser.add_argument(
        "--delete-logging-bucket", action="store_true",
        help="With --teardown, also delete the logging bucket and its contents",
    )
    args = parser.parse_args()

    if args.delete_logging_bucket and not args.teardown:
        parser.error("--delete-logging-bucket requires --teardown")

    if args.dry_run:
        dry_run()
    elif args.teardown:
        teardown(delete_bucket=args.delete_logging_bucket)
    else:
        create()


if __name__ == "__main__":
    main()
