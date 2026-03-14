"""AWS Cost Report — estimated S3 storage, transfer, and EC2 compute costs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    RAWDATA_BUCKET,
    REGION,
    PIPELINE_STAGES,
    get_s3_bucket_size,
    get_s3_prefix_sizes,
)

log = logging.getLogger("hm2p.frontend.cost")

st.title("AWS Cost Report")
st.caption("Estimated costs based on S3 storage and EC2 instance usage.")

# ── Pricing constants (ap-southeast-2, March 2026) ──────────────────────────
S3_STORAGE_PER_GB_MONTH = 0.025  # USD
S3_PUT_PER_1K = 0.005
S3_GET_PER_1K = 0.0004
S3_EGRESS_PER_GB = 0.09

# EC2 On-Demand pricing (ap-southeast-2)
EC2_PRICING = {
    "g4dn.xlarge": {"on_demand": 0.789, "spot_approx": 0.24, "gpu": "T4 16GB"},
    "g5.xlarge": {"on_demand": 1.19, "spot_approx": 0.36, "gpu": "A10G 24GB"},
    "g5.2xlarge": {"on_demand": 1.69, "spot_approx": 0.51, "gpu": "A10G 24GB"},
    "p3.2xlarge": {"on_demand": 4.234, "spot_approx": 1.27, "gpu": "V100 16GB"},
    "c5.xlarge": {"on_demand": 0.226, "spot_approx": 0.07, "gpu": "None (CPU)"},
}

# ── Compute all costs before display ─────────────────────────────────────────

# S3 costs
try:
    raw_size = get_s3_bucket_size(RAWDATA_BUCKET)
    raw_monthly = raw_size["total_gb"] * S3_STORAGE_PER_GB_MONTH
except Exception:
    raw_size = None
    raw_monthly = 0

stage_prefixes = [
    info["s3_prefix"]
    for info in PIPELINE_STAGES.values()
    if info.get("s3_prefix")
]
# Deduplicate (e.g. kinematics appears twice: kinematics + kpms)
stage_prefixes = list(dict.fromkeys(stage_prefixes))

try:
    prefix_sizes = get_s3_prefix_sizes(DERIVATIVES_BUCKET, stage_prefixes)
    total_deriv_gb = sum(v["total_gb"] for v in prefix_sizes.values())
    deriv_monthly = total_deriv_gb * S3_STORAGE_PER_GB_MONTH
except Exception:
    prefix_sizes = None
    total_deriv_gb = 0
    deriv_monthly = 0

s3_monthly = raw_monthly + deriv_monthly

# EC2 completed jobs
completed_jobs = [
    {
        "stage": "Stage 1 — Suite2p",
        "instance": "g4dn.xlarge",
        "hours": 52,
        "spot": True,
        "sessions": 26,
        "note": "Completed 26/26 sessions",
    },
    {
        "stage": "Stage 2 — DLC (sequential, 7 sessions)",
        "instance": "g4dn.xlarge",
        "hours": 19,
        "spot": False,
        "sessions": 7,
        "note": "7/26 sessions on T4 On-Demand (~2.7h each, Mar 7-8)",
    },
]

total_completed_cost = 0
for job in completed_jobs:
    pricing = EC2_PRICING.get(job["instance"], EC2_PRICING["g4dn.xlarge"])
    rate = pricing["spot_approx"] if job.get("spot") else pricing["on_demand"]
    n_inst = job.get("n_instances", 1)
    total_completed_cost += rate * job["hours"] * n_inst

# EC2 active / in-progress jobs
planned_jobs = []
try:
    import boto3
    import datetime

    ec2_client = boto3.client("ec2", region_name=REGION)
    resp = ec2_client.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": ["hm2p-dlc", "hm2p"]},
            {"Name": "instance-state-name", "Values": ["running", "stopped", "stopping"]},
        ]
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    for res in resp["Reservations"]:
        for inst in res["Instances"]:
            tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
            launch_time = inst.get("LaunchTime")
            if launch_time:
                hours_running = (now - launch_time).total_seconds() / 3600
                itype = inst["InstanceType"]
                state = inst["State"]["Name"]
                name = tags.get("Name", inst["InstanceId"])
                is_spot = inst.get("InstanceLifecycle") == "spot"

                planned_jobs.append({
                    "stage": f"Active — {name}",
                    "instance": itype,
                    "hours": round(hours_running, 1),
                    "spot": is_spot,
                    "n_instances": 1,
                    "sessions": 0,
                    "note": f"{state} since {str(launch_time)[:19]} UTC"
                        f" ({'Spot' if is_spot else 'On-Demand'})",
                })
except Exception as e:
    log.warning("Could not query EC2 instances: %s", e)

total_active_cost = 0
for job in planned_jobs:
    pricing = EC2_PRICING.get(job["instance"], EC2_PRICING["g4dn.xlarge"])
    rate = pricing["spot_approx"] if job.get("spot") else pricing["on_demand"]
    n_inst = job.get("n_instances", 1)
    total_active_cost += rate * job["hours"] * n_inst

total_ec2 = total_completed_cost + total_active_cost
total_all = total_ec2 + s3_monthly


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY — ordered by importance: summary first, details after
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Total Project Cost Summary ────────────────────────────────────────────

st.header("Total Project Cost Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("S3 Storage (monthly)", f"${s3_monthly:.2f}")
col2.metric("EC2 Completed", f"${total_completed_cost:.2f}")
col3.metric("EC2 Active", f"${total_active_cost:.2f}")
col4.metric("Total to Date", f"${total_all:.2f}")

st.caption(
    "S3 cost is per month. EC2 cost is cumulative (completed + active). "
    "Actual AWS bill may differ slightly due to EBS volumes, data transfer, "
    "and API request costs. Check AWS Cost Explorer for exact figures."
)


# ── 2. EC2 Compute Costs ─────────────────────────────────────────────────────

st.markdown("---")
st.header("EC2 Compute Costs")

st.markdown(
    "Estimated compute costs for completed and planned pipeline stages. "
    "Based on On-Demand and approximate Spot pricing in ap-southeast-2."
)

st.subheader("Completed Jobs")
for job in completed_jobs:
    pricing = EC2_PRICING.get(job["instance"], EC2_PRICING["g4dn.xlarge"])
    rate = pricing["spot_approx"] if job.get("spot") else pricing["on_demand"]
    n_inst = job.get("n_instances", 1)
    cost = rate * job["hours"] * n_inst

    spot_label = "Spot" if job.get("spot") else "On-Demand"
    st.markdown(
        f"**{job['stage']}** — `{job['instance']}` ({pricing['gpu']}) "
        f"× {n_inst} — {job['hours']}h {spot_label} — **${cost:.2f}**"
    )
    st.caption(f"  {job['note']}")

st.metric("Total Completed", f"${total_completed_cost:.2f}")

if planned_jobs:
    st.subheader("Active / In-Progress")
    for job in planned_jobs:
        pricing = EC2_PRICING.get(job["instance"], EC2_PRICING["g4dn.xlarge"])
        rate = pricing["spot_approx"] if job.get("spot") else pricing["on_demand"]
        n_inst = job.get("n_instances", 1)
        cost = rate * job["hours"] * n_inst
        spot_label = "Spot" if job.get("spot") else "On-Demand"

        st.markdown(
            f"**{job['stage']}** — `{job['instance']}` ({pricing['gpu']}) "
            f"× {n_inst} — {job['hours']}h {spot_label} — **${cost:.2f}**"
        )
        st.caption(f"  {job['note']}")

    st.metric("Active Instance Cost (so far)", f"${total_active_cost:.2f}")


# ── 3. S3 Storage ────────────────────────────────────────────────────────────

st.markdown("---")
st.header("S3 Storage")

if st.button("Scan S3 buckets", key="scan_s3"):
    st.cache_data.clear()

tab_raw, tab_deriv = st.tabs(["Raw Data", "Derivatives"])

with tab_raw:
    if raw_size is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Cost", f"${raw_monthly:.2f}")
        col2.metric("Total Size", f"{raw_size['total_gb']:.1f} GB")
        col3.metric("Objects", f"{raw_size['n_objects']:,}")
        st.caption(f"Bucket: `{RAWDATA_BUCKET}` ({REGION})")
    else:
        log.exception("Could not scan raw data bucket")
        st.warning("Could not scan raw data bucket. Check server logs for details.")

with tab_deriv:
    if prefix_sizes is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Cost", f"${deriv_monthly:.2f}")
        col2.metric("Total Size", f"{total_deriv_gb:.1f} GB")
        col3.metric("Objects", f"{sum(v['n_objects'] for v in prefix_sizes.values()):,}")

        # Per-stage breakdown table
        st.subheader("By Pipeline Stage")
        for _key, info in PIPELINE_STAGES.items():
            s3_prefix = info.get("s3_prefix")
            if not s3_prefix:
                continue
            pinfo = prefix_sizes.get(s3_prefix, {"total_gb": 0, "n_objects": 0})
            if pinfo["n_objects"] > 0:
                cost = pinfo["total_gb"] * S3_STORAGE_PER_GB_MONTH
                st.markdown(
                    f"**{info['label']}** — {pinfo['total_gb']:.2f} GB "
                    f"({pinfo['n_objects']} files) — ${cost:.3f}/month"
                )

        st.caption(f"Bucket: `{DERIVATIVES_BUCKET}` ({REGION})")
    else:
        log.exception("Could not scan derivatives bucket")
        st.warning("Could not scan derivatives bucket. Check server logs for details.")


# ── 4. Cost Calculator ───────────────────────────────────────────────────────

st.markdown("---")
st.header("Cost Calculator")

st.markdown("Estimate cost for a custom job.")

calc_col1, calc_col2, calc_col3 = st.columns(3)
with calc_col1:
    calc_instance = st.selectbox("Instance type", list(EC2_PRICING.keys()), key="calc_inst")
with calc_col2:
    calc_hours = st.number_input("Hours", min_value=1, max_value=500, value=10, key="calc_hours")
with calc_col3:
    calc_n = st.number_input("Instances", min_value=1, max_value=10, value=1, key="calc_n")

calc_spot = st.checkbox("Use Spot pricing", value=True, key="calc_spot")

pricing = EC2_PRICING[calc_instance]
rate = pricing["spot_approx"] if calc_spot else pricing["on_demand"]
calc_cost = rate * calc_hours * calc_n

col1, col2, col3 = st.columns(3)
col1.metric("Rate", f"${rate:.3f}/hr")
col2.metric("Total Hours", f"{calc_hours * calc_n}h")
col3.metric("Estimated Cost", f"${calc_cost:.2f}")


# ── 5. Instance Pricing Reference ────────────────────────────────────────────

st.markdown("---")
st.header("Instance Pricing Reference")
st.caption("ap-southeast-2 (Sydney), approximate as of March 2026")

import pandas as pd

pricing_df = pd.DataFrame([
    {
        "Instance": itype,
        "GPU": info["gpu"],
        "On-Demand ($/hr)": f"${info['on_demand']:.3f}",
        "Spot ($/hr, approx)": f"${info['spot_approx']:.3f}",
        "Spot Savings": f"{(1 - info['spot_approx'] / info['on_demand']) * 100:.0f}%",
        "24h On-Demand": f"${info['on_demand'] * 24:.2f}",
        "24h Spot": f"${info['spot_approx'] * 24:.2f}",
    }
    for itype, info in EC2_PRICING.items()
])
st.dataframe(pricing_df, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("AWS Cost Report | hm2p v2")
