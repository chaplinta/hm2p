#!/usr/bin/env python3
"""Set up scheduled EC2 stop/start using EventBridge + Lambda.

Creates two Lambda functions that stop and start EC2 instances tagged with
Project=hm2p-dlc, Project=hm2p, or Name=hm2p-frontend. EventBridge rules
trigger them on a daily cron schedule.

Default schedule (AWST = UTC+8):
  - Stop at 22:00 AWST (14:00 UTC)
  - Start at 08:00 AWST (00:00 UTC)

Run this from your local machine (not the devcontainer — IAM is blocked there).

Usage:
    uv run scripts/setup_auto_shutdown.py                          # create everything
    uv run scripts/setup_auto_shutdown.py --dry-run                # print what would be created
    uv run scripts/setup_auto_shutdown.py --teardown               # delete everything
    uv run scripts/setup_auto_shutdown.py --stop-hour 23           # stop at 23:00 AWST
    uv run scripts/setup_auto_shutdown.py --start-hour 9           # start at 09:00 AWST
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile

ACCOUNT_ID = "390897005556"
REGION = "ap-southeast-2"

LAMBDA_ROLE_NAME = "hm2p-lambda-scheduler"
LAMBDA_ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/{LAMBDA_ROLE_NAME}"

STOP_FUNCTION_NAME = "hm2p-stop-instances"
START_FUNCTION_NAME = "hm2p-start-instances"

STOP_RULE_NAME = "hm2p-nightly-stop"
START_RULE_NAME = "hm2p-morning-start"

# Tags to match — instances with any of these tag combinations will be managed
MATCH_TAGS = [
    {"Key": "Project", "Values": ["hm2p-dlc", "hm2p"]},
    {"Key": "Name", "Values": ["hm2p-frontend"]},
]

TRUST_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
})

LAMBDA_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:StopInstances",
                "ec2:StartInstances",
            ],
            "Resource": "*",
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": "arn:aws:logs:*:*:*",
        },
    ],
}, indent=2)

LAMBDA_POLICY_NAME = "hm2p-lambda-scheduler-policy"
LAMBDA_POLICY_ARN = f"arn:aws:iam::{ACCOUNT_ID}:policy/{LAMBDA_POLICY_NAME}"

# ----- Lambda function code (embedded) -----

STOP_LAMBDA_CODE = '''\
import boto3

def handler(event, context):
    ec2 = boto3.client("ec2", region_name="ap-southeast-2")

    # Find running instances with matching tags
    filters = [
        {"Name": "instance-state-name", "Values": ["running"]},
    ]
    response = ec2.describe_instances(Filters=filters)

    instance_ids = []
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
            if (tags.get("Project") in ("hm2p-dlc", "hm2p")
                    or tags.get("Name") == "hm2p-frontend"):
                instance_ids.append(instance["InstanceId"])

    if instance_ids:
        ec2.stop_instances(InstanceIds=instance_ids)
        print(f"Stopped {len(instance_ids)} instances: {instance_ids}")
    else:
        print("No matching running instances to stop")

    return {"stopped": instance_ids}
'''

START_LAMBDA_CODE = '''\
import boto3

def handler(event, context):
    ec2 = boto3.client("ec2", region_name="ap-southeast-2")

    # Find stopped instances with matching tags
    filters = [
        {"Name": "instance-state-name", "Values": ["stopped"]},
    ]
    response = ec2.describe_instances(Filters=filters)

    instance_ids = []
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            tags = {t["Key"]: t["Value"] for t in instance.get("Tags", [])}
            if (tags.get("Project") in ("hm2p-dlc", "hm2p")
                    or tags.get("Name") == "hm2p-frontend"):
                instance_ids.append(instance["InstanceId"])

    if instance_ids:
        ec2.start_instances(InstanceIds=instance_ids)
        print(f"Started {len(instance_ids)} instances: {instance_ids}")
    else:
        print("No matching stopped instances to start")

    return {"started": instance_ids}
'''


def _zip_lambda(code: str) -> bytes:
    """Package Lambda code as an in-memory zip file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", code)
    return buf.getvalue()


def _awst_to_utc(hour: int) -> int:
    """Convert AWST hour (UTC+8) to UTC hour."""
    return (hour - 8) % 24


def dry_run(stop_hour: int, start_hour: int) -> None:
    """Print what would be created."""
    stop_utc = _awst_to_utc(stop_hour)
    start_utc = _awst_to_utc(start_hour)

    print("# Auto-shutdown setup for hm2p EC2 instances")
    print(f"# Stop:  {stop_hour:02d}:00 AWST ({stop_utc:02d}:00 UTC)")
    print(f"# Start: {start_hour:02d}:00 AWST ({start_utc:02d}:00 UTC)")
    print()

    print(f"# 1. Create IAM role for Lambda")
    print(f"aws iam create-role \\")
    print(f"  --role-name {LAMBDA_ROLE_NAME} \\")
    print(f"  --assume-role-policy-document '{TRUST_POLICY}'")
    print()

    print(f"# 2. Create and attach policy")
    print(f"aws iam create-policy \\")
    print(f"  --policy-name {LAMBDA_POLICY_NAME} \\")
    print(f"  --policy-document '<ec2 + logs policy>'")
    print(f"aws iam attach-role-policy \\")
    print(f"  --role-name {LAMBDA_ROLE_NAME} \\")
    print(f"  --policy-arn {LAMBDA_POLICY_ARN}")
    print()

    print(f"# 3. Create Lambda functions")
    print(f"aws lambda create-function \\")
    print(f"  --function-name {STOP_FUNCTION_NAME} \\")
    print(f"  --runtime python3.12 \\")
    print(f"  --handler lambda_function.handler \\")
    print(f"  --role {LAMBDA_ROLE_ARN} \\")
    print(f"  --zip-file fileb://stop.zip \\")
    print(f"  --region {REGION}")
    print()
    print(f"aws lambda create-function \\")
    print(f"  --function-name {START_FUNCTION_NAME} \\")
    print(f"  --runtime python3.12 \\")
    print(f"  --handler lambda_function.handler \\")
    print(f"  --role {LAMBDA_ROLE_ARN} \\")
    print(f"  --zip-file fileb://start.zip \\")
    print(f"  --region {REGION}")
    print()

    print(f"# 4. Create EventBridge rules")
    print(f"aws events put-rule \\")
    print(f"  --name {STOP_RULE_NAME} \\")
    print(f"  --schedule-expression 'cron(0 {stop_utc} * * ? *)' \\")
    print(f"  --region {REGION}")
    print()
    print(f"aws events put-rule \\")
    print(f"  --name {START_RULE_NAME} \\")
    print(f"  --schedule-expression 'cron(0 {start_utc} * * ? *)' \\")
    print(f"  --region {REGION}")
    print()

    print(f"# 5. Add Lambda targets to rules")
    print(f"aws events put-targets --rule {STOP_RULE_NAME} \\")
    print(f"  --targets Id=1,Arn=arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{STOP_FUNCTION_NAME}")
    print(f"aws events put-targets --rule {START_RULE_NAME} \\")
    print(f"  --targets Id=1,Arn=arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{START_FUNCTION_NAME}")
    print()

    print(f"# 6. Grant EventBridge permission to invoke Lambda")
    print(f"aws lambda add-permission \\")
    print(f"  --function-name {STOP_FUNCTION_NAME} \\")
    print(f"  --statement-id eventbridge-stop \\")
    print(f"  --action lambda:InvokeFunction \\")
    print(f"  --principal events.amazonaws.com \\")
    print(f"  --source-arn arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{STOP_RULE_NAME}")
    print(f"aws lambda add-permission \\")
    print(f"  --function-name {START_FUNCTION_NAME} \\")
    print(f"  --statement-id eventbridge-start \\")
    print(f"  --action lambda:InvokeFunction \\")
    print(f"  --principal events.amazonaws.com \\")
    print(f"  --source-arn arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{START_RULE_NAME}")


def create(stop_hour: int, start_hour: int) -> None:
    """Create Lambda functions, EventBridge rules, and IAM role."""
    import boto3

    iam = boto3.client("iam")
    lam = boto3.client("lambda", region_name=REGION)
    events = boto3.client("events", region_name=REGION)

    stop_utc = _awst_to_utc(stop_hour)
    start_utc = _awst_to_utc(start_hour)

    print(f"Schedule: stop {stop_hour:02d}:00 AWST ({stop_utc:02d}:00 UTC), "
          f"start {start_hour:02d}:00 AWST ({start_utc:02d}:00 UTC)")

    # 1. Create IAM role
    try:
        iam.get_role(RoleName=LAMBDA_ROLE_NAME)
        print(f"Role {LAMBDA_ROLE_NAME} already exists")
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(
            RoleName=LAMBDA_ROLE_NAME,
            AssumeRolePolicyDocument=TRUST_POLICY,
            Description="Lambda role for hm2p EC2 stop/start scheduler",
        )
        print(f"Created role: {LAMBDA_ROLE_NAME}")

    # 2. Create and attach policy
    try:
        iam.get_policy(PolicyArn=LAMBDA_POLICY_ARN)
        print(f"Policy {LAMBDA_POLICY_NAME} already exists")
    except iam.exceptions.NoSuchEntityException:
        iam.create_policy(
            PolicyName=LAMBDA_POLICY_NAME,
            PolicyDocument=LAMBDA_POLICY,
            Description="EC2 stop/start + CloudWatch Logs for hm2p scheduler Lambda",
        )
        print(f"Created policy: {LAMBDA_POLICY_NAME}")

    iam.attach_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyArn=LAMBDA_POLICY_ARN)
    print(f"Attached {LAMBDA_POLICY_NAME} to {LAMBDA_ROLE_NAME}")

    # Wait for role propagation (IAM is eventually consistent)
    print("Waiting 10s for IAM role propagation...")
    time.sleep(10)

    # 3. Create Lambda functions
    stop_zip = _zip_lambda(STOP_LAMBDA_CODE)
    start_zip = _zip_lambda(START_LAMBDA_CODE)

    for func_name, zip_bytes, desc in [
        (STOP_FUNCTION_NAME, stop_zip, "Stop hm2p EC2 instances nightly"),
        (START_FUNCTION_NAME, start_zip, "Start hm2p EC2 instances in the morning"),
    ]:
        try:
            lam.get_function(FunctionName=func_name)
            lam.update_function_code(
                FunctionName=func_name,
                ZipFile=zip_bytes,
            )
            print(f"Updated Lambda function: {func_name}")
        except lam.exceptions.ResourceNotFoundException:
            lam.create_function(
                FunctionName=func_name,
                Runtime="python3.12",
                Role=LAMBDA_ROLE_ARN,
                Handler="lambda_function.handler",
                Code={"ZipFile": zip_bytes},
                Description=desc,
                Timeout=60,
                MemorySize=128,
            )
            print(f"Created Lambda function: {func_name}")

    # 4. Create EventBridge rules
    stop_rule_arn = events.put_rule(
        Name=STOP_RULE_NAME,
        ScheduleExpression=f"cron(0 {stop_utc} * * ? *)",
        State="ENABLED",
        Description=f"Stop hm2p instances at {stop_hour:02d}:00 AWST",
    )["RuleArn"]
    print(f"Created/updated rule: {STOP_RULE_NAME}")

    start_rule_arn = events.put_rule(
        Name=START_RULE_NAME,
        ScheduleExpression=f"cron(0 {start_utc} * * ? *)",
        State="ENABLED",
        Description=f"Start hm2p instances at {start_hour:02d}:00 AWST",
    )["RuleArn"]
    print(f"Created/updated rule: {START_RULE_NAME}")

    # 5. Add Lambda targets
    stop_func_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{STOP_FUNCTION_NAME}"
    start_func_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{START_FUNCTION_NAME}"

    events.put_targets(
        Rule=STOP_RULE_NAME,
        Targets=[{"Id": "1", "Arn": stop_func_arn}],
    )
    events.put_targets(
        Rule=START_RULE_NAME,
        Targets=[{"Id": "1", "Arn": start_func_arn}],
    )
    print("Added Lambda targets to EventBridge rules")

    # 6. Grant EventBridge permission to invoke Lambda
    for func_name, stmt_id, rule_arn in [
        (STOP_FUNCTION_NAME, "eventbridge-stop", stop_rule_arn),
        (START_FUNCTION_NAME, "eventbridge-start", start_rule_arn),
    ]:
        try:
            lam.add_permission(
                FunctionName=func_name,
                StatementId=stmt_id,
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn,
            )
            print(f"Added EventBridge invoke permission for {func_name}")
        except lam.exceptions.ResourceConflictException:
            print(f"EventBridge permission already exists for {func_name}")

    print(f"\nDone! Instances tagged Project=hm2p-dlc, Project=hm2p, or")
    print(f"Name=hm2p-frontend will be stopped at {stop_hour:02d}:00 AWST and")
    print(f"started at {start_hour:02d}:00 AWST daily.")


def teardown() -> None:
    """Delete all resources created by this script."""
    import boto3

    iam = boto3.client("iam")
    lam = boto3.client("lambda", region_name=REGION)
    events = boto3.client("events", region_name=REGION)

    # Remove targets from rules
    for rule_name in [STOP_RULE_NAME, START_RULE_NAME]:
        try:
            events.remove_targets(Rule=rule_name, Ids=["1"])
            print(f"Removed targets from {rule_name}")
        except Exception:
            pass

    # Delete rules
    for rule_name in [STOP_RULE_NAME, START_RULE_NAME]:
        try:
            events.delete_rule(Name=rule_name)
            print(f"Deleted rule: {rule_name}")
        except Exception:
            pass

    # Delete Lambda functions
    for func_name in [STOP_FUNCTION_NAME, START_FUNCTION_NAME]:
        try:
            lam.delete_function(FunctionName=func_name)
            print(f"Deleted Lambda function: {func_name}")
        except Exception:
            pass

    # Detach policy and delete role
    try:
        iam.detach_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyArn=LAMBDA_POLICY_ARN)
        print(f"Detached policy from {LAMBDA_ROLE_NAME}")
    except Exception:
        pass

    try:
        iam.delete_role(RoleName=LAMBDA_ROLE_NAME)
        print(f"Deleted role: {LAMBDA_ROLE_NAME}")
    except Exception:
        pass

    try:
        iam.delete_policy(PolicyArn=LAMBDA_POLICY_ARN)
        print(f"Deleted policy: {LAMBDA_POLICY_NAME}")
    except Exception:
        pass

    print("\nTeardown complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up scheduled EC2 stop/start for hm2p instances",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true",
                       help="Print what would be created without executing")
    group.add_argument("--teardown", action="store_true",
                       help="Delete all resources created by this script")
    parser.add_argument("--stop-hour", type=int, default=22,
                        help="Hour to stop instances (AWST, 0-23, default: 22)")
    parser.add_argument("--start-hour", type=int, default=8,
                        help="Hour to start instances (AWST, 0-23, default: 8)")
    args = parser.parse_args()

    if not (0 <= args.stop_hour <= 23):
        print("ERROR: --stop-hour must be 0-23")
        sys.exit(1)
    if not (0 <= args.start_hour <= 23):
        print("ERROR: --start-hour must be 0-23")
        sys.exit(1)

    if args.dry_run:
        dry_run(args.stop_hour, args.start_hour)
    elif args.teardown:
        teardown()
    else:
        create(args.stop_hour, args.start_hour)


if __name__ == "__main__":
    main()
