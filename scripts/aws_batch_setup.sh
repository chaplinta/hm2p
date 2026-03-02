#!/usr/bin/env bash
# aws_batch_setup.sh — Create AWS Batch compute environments + job queues.
#
# Usage:
#   ./scripts/aws_batch_setup.sh [--region REGION] [--profile PROFILE]
#
# Creates:
#   - IAM service role for Batch (AWSBatchServiceRole)
#   - IAM instance role for EC2 (hm2p-batch-instance-role)
#   - CPU Spot compute environment (c5.4xlarge, 0-256 vCPUs)
#   - GPU Spot compute environment (g4dn.xlarge, 0-64 vCPUs)
#   - CPU job queue (hm2p-cpu-queue)
#   - GPU job queue (hm2p-gpu-queue)
#
# Prerequisites:
#   - AWS CLI configured with IAM + Batch admin permissions
#   - ECR repos created (run ecr_push.sh first)

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-ap-southeast-2}"
PROFILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)  REGION="$2";  shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

AWS_CMD=(aws)
if [[ -n "$PROFILE" ]]; then
    AWS_CMD+=(--profile "$PROFILE")
fi

ACCOUNT_ID=$("${AWS_CMD[@]}" sts get-caller-identity --query Account --output text --region "$REGION")

echo "Account:  $ACCOUNT_ID"
echo "Region:   $REGION"
echo ""

# ---------------------------------------------------------------------------
# 1. IAM — Batch service role
# ---------------------------------------------------------------------------
BATCH_SERVICE_ROLE="hm2p-batch-service-role"
echo "--- IAM: Batch service role ($BATCH_SERVICE_ROLE) ---"

if "${AWS_CMD[@]}" iam get-role --role-name "$BATCH_SERVICE_ROLE" 2>/dev/null; then
    echo "  Already exists, skipping."
else
    "${AWS_CMD[@]}" iam create-role \
        --role-name "$BATCH_SERVICE_ROLE" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "batch.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    "${AWS_CMD[@]}" iam attach-role-policy \
        --role-name "$BATCH_SERVICE_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole
    echo "  Created."
fi

# ---------------------------------------------------------------------------
# 2. IAM — EC2 instance role for Batch
# ---------------------------------------------------------------------------
INSTANCE_ROLE="hm2p-batch-instance-role"
INSTANCE_PROFILE="hm2p-batch-instance-profile"
echo ""
echo "--- IAM: Instance role ($INSTANCE_ROLE) ---"

if "${AWS_CMD[@]}" iam get-role --role-name "$INSTANCE_ROLE" 2>/dev/null; then
    echo "  Already exists, skipping."
else
    "${AWS_CMD[@]}" iam create-role \
        --role-name "$INSTANCE_ROLE" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    "${AWS_CMD[@]}" iam attach-role-policy \
        --role-name "$INSTANCE_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
    "${AWS_CMD[@]}" iam attach-role-policy \
        --role-name "$INSTANCE_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    "${AWS_CMD[@]}" iam attach-role-policy \
        --role-name "$INSTANCE_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

    # Create instance profile and add role
    "${AWS_CMD[@]}" iam create-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE" 2>/dev/null || true
    "${AWS_CMD[@]}" iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE" \
        --role-name "$INSTANCE_ROLE"
    echo "  Created."
fi

INSTANCE_PROFILE_ARN="arn:aws:iam::${ACCOUNT_ID}:instance-profile/${INSTANCE_PROFILE}"
SERVICE_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${BATCH_SERVICE_ROLE}"

# ---------------------------------------------------------------------------
# 3. Batch — CPU Spot compute environment
# ---------------------------------------------------------------------------
CPU_CE="hm2p-cpu-ce"
echo ""
echo "--- Batch: CPU compute environment ($CPU_CE) ---"

if "${AWS_CMD[@]}" batch describe-compute-environments \
    --compute-environments "$CPU_CE" --region "$REGION" \
    --query 'computeEnvironments[0].computeEnvironmentName' --output text 2>/dev/null | grep -q "$CPU_CE"; then
    echo "  Already exists, skipping."
else
    "${AWS_CMD[@]}" batch create-compute-environment \
        --compute-environment-name "$CPU_CE" \
        --type MANAGED \
        --state ENABLED \
        --service-role "$SERVICE_ROLE_ARN" \
        --compute-resources '{
            "type": "SPOT",
            "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
            "minvCpus": 0,
            "maxvCpus": 256,
            "desiredvCpus": 0,
            "instanceTypes": ["c5.4xlarge"],
            "instanceRole": "'"$INSTANCE_PROFILE_ARN"'",
            "spotIamFleetRole": "arn:aws:iam::'"$ACCOUNT_ID"':role/aws-ec2-spot-fleet-tagging-role",
            "subnets": [],
            "securityGroupIds": []
        }' \
        --region "$REGION"
    echo "  Created. NOTE: You may need to specify subnets and security groups."
fi

# ---------------------------------------------------------------------------
# 4. Batch — GPU Spot compute environment
# ---------------------------------------------------------------------------
GPU_CE="hm2p-gpu-ce"
echo ""
echo "--- Batch: GPU compute environment ($GPU_CE) ---"

if "${AWS_CMD[@]}" batch describe-compute-environments \
    --compute-environments "$GPU_CE" --region "$REGION" \
    --query 'computeEnvironments[0].computeEnvironmentName' --output text 2>/dev/null | grep -q "$GPU_CE"; then
    echo "  Already exists, skipping."
else
    "${AWS_CMD[@]}" batch create-compute-environment \
        --compute-environment-name "$GPU_CE" \
        --type MANAGED \
        --state ENABLED \
        --service-role "$SERVICE_ROLE_ARN" \
        --compute-resources '{
            "type": "SPOT",
            "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
            "minvCpus": 0,
            "maxvCpus": 64,
            "desiredvCpus": 0,
            "instanceTypes": ["g4dn.xlarge"],
            "instanceRole": "'"$INSTANCE_PROFILE_ARN"'",
            "spotIamFleetRole": "arn:aws:iam::'"$ACCOUNT_ID"':role/aws-ec2-spot-fleet-tagging-role",
            "subnets": [],
            "securityGroupIds": []
        }' \
        --region "$REGION"
    echo "  Created. NOTE: You may need to specify subnets and security groups."
fi

# ---------------------------------------------------------------------------
# 5. Batch — Job queues
# ---------------------------------------------------------------------------
echo ""
echo "--- Batch: Job queues ---"

for QUEUE_NAME in hm2p-cpu-queue hm2p-gpu-queue; do
    if [[ "$QUEUE_NAME" == "hm2p-cpu-queue" ]]; then
        CE="$CPU_CE"
    else
        CE="$GPU_CE"
    fi

    if "${AWS_CMD[@]}" batch describe-job-queues \
        --job-queues "$QUEUE_NAME" --region "$REGION" \
        --query 'jobQueues[0].jobQueueName' --output text 2>/dev/null | grep -q "$QUEUE_NAME"; then
        echo "  $QUEUE_NAME: already exists, skipping."
    else
        "${AWS_CMD[@]}" batch create-job-queue \
            --job-queue-name "$QUEUE_NAME" \
            --state ENABLED \
            --priority 1 \
            --compute-environment-order "order=1,computeEnvironment=$CE" \
            --region "$REGION"
        echo "  $QUEUE_NAME: created."
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "AWS Batch setup complete."
echo ""
echo "Queue ARNs (for workflow/profiles/aws-batch/config.yaml):"
for QUEUE_NAME in hm2p-cpu-queue hm2p-gpu-queue; do
    ARN=$("${AWS_CMD[@]}" batch describe-job-queues \
        --job-queues "$QUEUE_NAME" --region "$REGION" \
        --query 'jobQueues[0].jobQueueArn' --output text 2>/dev/null || echo "NOT FOUND")
    echo "  $QUEUE_NAME: $ARN"
done
echo "============================================================"
