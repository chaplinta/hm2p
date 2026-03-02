#!/usr/bin/env bash
# ecr_push.sh — Create ECR repositories, build Docker images, push to ECR.
#
# Usage:
#   ./scripts/ecr_push.sh [--region REGION] [--profile PROFILE]
#
# Prerequisites:
#   - AWS CLI configured with ECR push permissions
#   - Docker installed and running
#
# Outputs:
#   Prints the ECR prefix to set in config/pipeline.yaml → ecr_prefix

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-ap-southeast-2}"
PROFILE=""
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

# Get AWS account ID
ACCOUNT_ID=$("${AWS_CMD[@]}" sts get-caller-identity --query Account --output text --region "$REGION")
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "Account:  $ACCOUNT_ID"
echo "Region:   $REGION"
echo "ECR URI:  $ECR_URI"
echo ""

# --- Create ECR repositories (idempotent) -----------------------------------
for repo in hm2p-cpu hm2p-gpu; do
    echo "Creating ECR repo: $repo"
    "${AWS_CMD[@]}" ecr describe-repositories --repository-names "$repo" --region "$REGION" 2>/dev/null \
        || "${AWS_CMD[@]}" ecr create-repository --repository-name "$repo" --region "$REGION" \
             --image-scanning-configuration scanOnPush=true
done

# --- Docker login to ECR ----------------------------------------------------
echo ""
echo "Logging in to ECR..."
"${AWS_CMD[@]}" ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_URI"

# --- Build and push CPU image -----------------------------------------------
echo ""
echo "Building hm2p-cpu..."
docker build -f "$REPO_ROOT/docker/cpu.Dockerfile" -t hm2p-cpu "$REPO_ROOT"
docker tag hm2p-cpu:latest "$ECR_URI/hm2p-cpu:latest"
echo "Pushing hm2p-cpu..."
docker push "$ECR_URI/hm2p-cpu:latest"

# --- Build and push GPU image -----------------------------------------------
echo ""
echo "Building hm2p-gpu..."
docker build -f "$REPO_ROOT/docker/gpu.Dockerfile" -t hm2p-gpu "$REPO_ROOT"
docker tag hm2p-gpu:latest "$ECR_URI/hm2p-gpu:latest"
echo "Pushing hm2p-gpu..."
docker push "$ECR_URI/hm2p-gpu:latest"

# --- Print ECR prefix for pipeline.yaml -------------------------------------
echo ""
echo "============================================================"
echo "Done. Set this in config/pipeline.yaml:"
echo ""
echo "  ecr_prefix: \"${ECR_URI}/hm2p\""
echo ""
echo "============================================================"
