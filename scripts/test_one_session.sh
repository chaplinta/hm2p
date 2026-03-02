#!/usr/bin/env bash
# test_one_session.sh — Download one session from S3 and run the pipeline.
#
# Usage:
#   ./scripts/test_one_session.sh [--session EXP_ID] [--profile PROFILE] [--dry-run]
#
# Default session: 20221115_13_27_42_1118213 (primary_exp=1, exclude=0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${HM2P_TEST_SESSION:-20221115_13_27_42_1118213}"
PROFILE="${HM2P_AWS_PROFILE:-hm2p-agent}"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)  SESSION="$2";  shift 2 ;;
        --profile)  PROFILE="$2";  shift 2 ;;
        --dry-run)  DRY_RUN="--dry-run"; shift ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "============================================================"
echo "hm2p single-session test"
echo "Session:  $SESSION"
echo "Profile:  $PROFILE"
echo "Repo:     $REPO_ROOT"
echo "============================================================"

# --- Step 1: Download from S3 -----------------------------------------------
echo ""
echo "--- Step 1: Download session from S3 ---"
python "$REPO_ROOT/scripts/download_from_s3.py" \
    --session "$SESSION" \
    --profile "$PROFILE" \
    --data-root "$REPO_ROOT/data" \
    --yes \
    $DRY_RUN

if [[ -n "$DRY_RUN" ]]; then
    echo "Dry run — skipping pipeline execution."
    exit 0
fi

# --- Step 2: Snakemake dry-run -----------------------------------------------
echo ""
echo "--- Step 2: Snakemake dry-run ---"
cd "$REPO_ROOT/workflow"
snakemake -n --profile profiles/local \
    --config data_root="$REPO_ROOT/data" metadata_dir="$REPO_ROOT/metadata"

# --- Step 3: Run CPU stages (0, 3, 4, 5) ------------------------------------
echo ""
echo "--- Step 3: Run CPU stages ---"

# Derive sub/ses from session ID
DATE="${SESSION%%_*}"
REST="${SESSION#*_}"
HH="${REST%%_*}"; REST="${REST#*_}"
MM="${REST%%_*}"; REST="${REST#*_}"
SS="${REST%%_*}"; REST="${REST#*_}"
ANIMAL_ID="$REST"

SUB="sub-${ANIMAL_ID}"
SES="ses-${DATE}T${HH}${MM}${SS}"

DATA_ROOT="$REPO_ROOT/data"

# Stage 0: timestamps
snakemake --profile profiles/local --cores 1 \
    --config data_root="$DATA_ROOT" metadata_dir="$REPO_ROOT/metadata" \
    "${DATA_ROOT}/derivatives/timestamps/${SUB}/${SES}/timestamps.h5"

# Stage 3: kinematics (requires pose output — skip if no pose data)
if ls "${DATA_ROOT}/derivatives/pose/${SUB}/${SES}/"*.h5 2>/dev/null || \
   ls "${DATA_ROOT}/derivatives/pose/${SUB}/${SES}/"*.csv 2>/dev/null; then
    snakemake --profile profiles/local --cores 1 \
        --config data_root="$DATA_ROOT" metadata_dir="$REPO_ROOT/metadata" \
        "${DATA_ROOT}/derivatives/movement/${SUB}/${SES}/kinematics.h5"
else
    echo "  [skip] No pose output found — Stage 3 requires GPU Stage 2 first."
fi

# Stage 4: calcium (requires suite2p output — skip if not available)
if [[ -d "${DATA_ROOT}/derivatives/ca_extraction/${SUB}/${SES}/suite2p" ]]; then
    snakemake --profile profiles/local --cores 1 \
        --config data_root="$DATA_ROOT" metadata_dir="$REPO_ROOT/metadata" \
        "${DATA_ROOT}/derivatives/calcium/${SUB}/${SES}/ca.h5"
else
    echo "  [skip] No suite2p output found — Stage 4 requires GPU Stage 1 first."
fi

# Stage 5: sync (requires both kinematics + calcium)
KIN="${DATA_ROOT}/derivatives/movement/${SUB}/${SES}/kinematics.h5"
CA="${DATA_ROOT}/derivatives/calcium/${SUB}/${SES}/ca.h5"
if [[ -f "$KIN" && -f "$CA" ]]; then
    snakemake --profile profiles/local --cores 1 \
        --config data_root="$DATA_ROOT" metadata_dir="$REPO_ROOT/metadata" \
        "${DATA_ROOT}/derivatives/sync/${SUB}/${SES}/sync.h5"
else
    echo "  [skip] Missing kinematics.h5 or ca.h5 — Stage 5 skipped."
fi

# --- Step 4: Verify outputs -------------------------------------------------
echo ""
echo "--- Step 4: Verify outputs ---"
for f in \
    "${DATA_ROOT}/derivatives/timestamps/${SUB}/${SES}/timestamps.h5" \
    "${DATA_ROOT}/derivatives/movement/${SUB}/${SES}/kinematics.h5" \
    "${DATA_ROOT}/derivatives/calcium/${SUB}/${SES}/ca.h5" \
    "${DATA_ROOT}/derivatives/sync/${SUB}/${SES}/sync.h5"; do
    if [[ -f "$f" ]]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OK]   $f ($SIZE)"
    else
        echo "  [SKIP] $f (not produced — GPU stage prerequisite missing)"
    fi
done

echo ""
echo "============================================================"
echo "Single-session test complete."
echo "============================================================"
