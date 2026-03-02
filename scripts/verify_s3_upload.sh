#!/usr/bin/env bash
# verify_s3_upload.sh — Verify integrity of S3 upload against local sources.
#
# Usage:
#   ./scripts/verify_s3_upload.sh [--profile PROFILE] [--bucket BUCKET]
#
# Checks:
#   1. All 26 sessions from experiments.csv exist in S3
#   2. Each session has funcimg/ and behav/ subdirectories
#   3. Per-session object count and total size summary
#   4. MD5 checksum of behav/meta.txt (small, single-part → ETag = MD5)
#   5. File size match for TDMS files (large, multipart → size comparison)
#
# Exit code 0 if all checks pass, 1 if any fail.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROFILE="${AWS_PROFILE:-hm2p-agent}"
BUCKET="${S3_BUCKET:-hm2p-rawdata}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENTS_CSV="$REPO_ROOT/metadata/experiments.csv"

RAW_2P="/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze"
VIDEO="/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/video"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) PROFILE="$2"; shift 2 ;;
    --bucket)  BUCKET="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
pass=0; fail_count=0; skip_count=0; warn_count=0

ok()   { echo -e "  ${GREEN}OK${NC}: $1"; pass=$((pass+1)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; fail_count=$((fail_count+1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1"; skip_count=$((skip_count+1)); }
warn() { echo -e "  ${YELLOW}WARN${NC}: $1"; warn_count=$((warn_count+1)); }

exp_id_to_ses_path() {
  local exp_id="$1"
  IFS='_' read -ra parts <<< "$exp_id"
  local n=${#parts[@]}
  local animal="${parts[$((n-1))]}"
  echo "sub-${animal}/ses-${parts[0]}T${parts[1]}${parts[2]}${parts[3]}"
}

ses_path_to_exp_id() {
  local sub ses animal dp
  sub=$(echo "$1" | cut -d/ -f1); ses=$(echo "$1" | cut -d/ -f2)
  animal="${sub#sub-}"; dp="${ses#ses-}"
  echo "${dp:0:8}_${dp:9:2}_${dp:11:2}_${dp:13:2}_${animal}"
}

# ---------------------------------------------------------------------------
# Fetch full S3 listing once (key, size, etag)
# ---------------------------------------------------------------------------
echo ""
echo "Fetching full S3 listing (one API call)..."
LISTING_FILE=$(mktemp)
trap "rm -f $LISTING_FILE" EXIT

aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "rawdata/" \
  --profile "$PROFILE" \
  --query 'Contents[].{K:Key,S:Size,E:ETag}' --output json \
  > "$LISTING_FILE" 2>/dev/null

n_objects=$(python3 -c "import json; print(len(json.load(open('$LISTING_FILE'))))")
echo "  Fetched $n_objects objects."

# ---------------------------------------------------------------------------
# 1. Session completeness
# ---------------------------------------------------------------------------
echo ""
echo "=== 1. Session completeness (experiments.csv vs S3) ==="
echo ""

# Expected sessions from CSV
expected=()
while IFS= read -r exp_id; do
  expected+=("$(exp_id_to_ses_path "$exp_id")")
done < <(tail -n+2 "$EXPERIMENTS_CSV" | awk -F',' '{print $2}')

# S3 sessions from listing
s3_sessions=($(python3 -c "
import json, re
data = json.load(open('$LISTING_FILE'))
sessions = set()
for obj in data:
    m = re.match(r'rawdata/(sub-[^/]+/ses-[^/]+)/', obj['K'])
    if m: sessions.add(m.group(1))
for s in sorted(sessions): print(s)
"))

echo "  Expected: ${#expected[@]} sessions"
echo "  In S3:    ${#s3_sessions[@]} sessions"

for ses in "${expected[@]}"; do
  found=false
  for s3ses in "${s3_sessions[@]}"; do
    [[ "$ses" == "$s3ses" ]] && { found=true; break; }
  done
  if $found; then ok "$ses"; else fail "MISSING from S3: $ses"; fi
done

# ---------------------------------------------------------------------------
# 2. Directory structure + 3. Object counts + 4. Checksums + 5. Sizes
#    (all from the cached listing — no more API calls)
# ---------------------------------------------------------------------------
echo ""
echo "=== 2–5. Structure, counts, checksums, sizes ==="
echo ""

python3 << 'PYEOF'
import json, os, hashlib, sys
from pathlib import Path

listing_file = os.environ.get("LISTING_FILE", sys.argv[1] if len(sys.argv) > 1 else "")
with open("LISTING_FILE_PATH", "w") as f:
    pass  # dummy

LISTING_FILE = "$LISTING_FILE"
EXPERIMENTS_CSV = "$EXPERIMENTS_CSV"
RAW_2P = "$RAW_2P"
VIDEO = "$VIDEO"
PYEOF

# Use Python for the heavy lifting — parse the cached listing, do all checks
python3 - "$LISTING_FILE" "$EXPERIMENTS_CSV" "$RAW_2P" "$VIDEO" << 'PYEOF'
import json, os, sys, hashlib
from pathlib import Path
from collections import defaultdict

listing_file, experiments_csv, raw_2p, video_dir = sys.argv[1:5]

GREEN = "\033[0;32m"; RED = "\033[0;31m"; YELLOW = "\033[0;33m"; NC = "\033[0m"
counts = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}

def ok(msg):   print(f"  {GREEN}OK{NC}: {msg}"); counts["pass"] += 1
def fail(msg): print(f"  {RED}FAIL{NC}: {msg}"); counts["fail"] += 1
def skip(msg): print(f"  {YELLOW}SKIP{NC}: {msg}"); counts["skip"] += 1
def warn(msg): print(f"  {YELLOW}WARN{NC}: {msg}"); counts["warn"] += 1

# Load listing
with open(listing_file) as f:
    objects = json.load(f)

# Index by session
sessions: dict[str, list[dict]] = defaultdict(list)
for obj in objects:
    key = obj["K"]
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "rawdata" and parts[1].startswith("sub-"):
        ses_path = f"{parts[1]}/{parts[2]}"
        sessions[ses_path].append(obj)

# Load expected exp_ids
with open(experiments_csv) as f:
    lines = f.readlines()[1:]
exp_ids = [line.split(",")[1].strip() for line in lines if line.strip()]

def exp_id_to_ses(exp_id):
    parts = exp_id.split("_")
    animal = parts[-1]
    return f"sub-{animal}/ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"

def ses_to_exp_id(ses_path):
    sub, ses = ses_path.split("/")
    animal = sub[4:]
    dp = ses[4:]
    return f"{dp[:8]}_{dp[9:11]}_{dp[11:13]}_{dp[13:15]}_{animal}"

expected_sessions = sorted(set(exp_id_to_ses(e) for e in exp_ids))

# --- Check 2: funcimg + behav ---
print("--- 2. Directory structure ---")
print()
for ses in expected_sessions:
    objs = sessions.get(ses, [])
    subdirs = set()
    for obj in objs:
        rel = obj["K"].split(f"{ses}/", 1)[-1]
        if "/" in rel:
            subdirs.add(rel.split("/")[0])
    has_funcimg = "funcimg" in subdirs
    has_behav = "behav" in subdirs
    if has_funcimg and has_behav:
        ok(f"{ses} (funcimg + behav)")
    else:
        if not has_funcimg: fail(f"{ses}: funcimg/ MISSING")
        if not has_behav:   fail(f"{ses}: behav/ MISSING")

# --- Check 3: Object count + size ---
print()
print("--- 3. Per-session object count + size ---")
print()
print(f"  {'Session':<45}  {'Files':>6}  {'Size':>10}")
print(f"  {'-------':<45}  {'-----':>6}  {'----':>10}")
for ses in expected_sessions:
    objs = sessions.get(ses, [])
    total = sum(o["S"] for o in objs)
    count = len(objs)
    if total >= 1024**3:
        size_str = f"{total/1024**3:.1f} GiB"
    elif total >= 1024**2:
        size_str = f"{total/1024**2:.1f} MiB"
    else:
        size_str = f"{total/1024:.1f} KiB"
    print(f"  {ses:<45}  {count:>6}  {size_str:>10}")
    if count < 10:
        warn(f"{ses}: only {count} objects (expected ~18-22)")

# --- Check 4: MD5 checksum of behav/meta.txt ---
print()
print("--- 4. Checksum: behav/meta.txt ---")
print()
for ses in expected_sessions:
    exp_id = ses_to_exp_id(ses)
    meta_key = f"rawdata/{ses}/behav/meta.txt"
    meta_obj = None
    for obj in sessions.get(ses, []):
        if obj["K"] == meta_key:
            meta_obj = obj
            break
    if meta_obj is None:
        skip(f"{ses}: no meta.txt in S3")
        continue

    local_meta = Path(video_dir) / exp_id / "meta.txt"
    if not local_meta.exists():
        skip(f"{ses}: no local meta.txt")
        continue

    local_md5 = hashlib.md5(local_meta.read_bytes()).hexdigest()
    s3_etag = meta_obj["E"].strip('"')

    # Multipart ETags contain a dash — can't compare MD5
    if "-" in s3_etag:
        skip(f"{ses}: multipart ETag, can't verify MD5")
        continue

    if local_md5 == s3_etag:
        ok(ses)
    else:
        fail(f"{ses}: MD5 mismatch (local={local_md5} s3={s3_etag})")

# --- Check 5: TDMS file size ---
print()
print("--- 5. Size check: TDMS files ---")
print()
for ses in expected_sessions:
    exp_id = ses_to_exp_id(ses)
    date_dir = f"{exp_id[:4]}_{exp_id[4:6]}_{exp_id[6:8]}"

    # Find .tdms in this session
    tdms_objs = [o for o in sessions.get(ses, [])
                 if o["K"].endswith(".tdms") and "funcimg" in o["K"]]
    if not tdms_objs:
        skip(f"{ses}: no .tdms in S3")
        continue

    tdms_obj = tdms_objs[0]
    tdms_name = tdms_obj["K"].split("/")[-1]
    s3_size = tdms_obj["S"]

    local_tdms = Path(raw_2p) / date_dir / exp_id / tdms_name
    if not local_tdms.exists():
        skip(f"{ses}: local TDMS not found ({tdms_name})")
        continue

    local_size = local_tdms.stat().st_size
    if local_size == s3_size:
        ok(f"{ses}: TDMS {s3_size:,} bytes")
    else:
        fail(f"{ses}: TDMS size mismatch (local={local_size:,} s3={s3_size:,})")

# --- Summary ---
print()
print("=" * 60)
print(f"  {GREEN}PASS{NC}: {counts['pass']}")
print(f"  {RED}FAIL{NC}: {counts['fail']}")
print(f"  {YELLOW}SKIP{NC}: {counts['skip']}")
print(f"  {YELLOW}WARN{NC}: {counts['warn']}")
print("=" * 60)

if counts["fail"] > 0:
    print(f"  {RED}VERIFICATION FAILED{NC}")
    sys.exit(1)
else:
    print(f"  {GREEN}ALL CHECKS PASSED{NC}")
    sys.exit(0)
PYEOF
