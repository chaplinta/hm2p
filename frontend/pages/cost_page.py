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
    STAGE_PREFIXES,
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


# ── S3 Storage Costs ────────────────────────────────────────────────────────

st.header("S3 Storage")

if st.button("Scan S3 buckets", key="scan_s3"):
    st.cache_data.clear()

tab_raw, tab_deriv = st.tabs(["Raw Data", "Derivatives"])

with tab_raw:
    try:
        raw_size = get_s3_bucket_size(RAWDATA_BUCKET)
        monthly_cost = raw_size["total_gb"] * S3_STORAGE_PER_GB_MONTH

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Size", f"{raw_size['total_gb']:.1f} GB")
        col2.metric("Objects", f"{raw_size['n_objects']:,}")
        col3.metric("Monthly Storage", f"${monthly_cost:.2f}")

        st.caption(f"Bucket: `{RAWDATA_BUCKET}` ({REGION})")
    except Exception as e:
        st.warning(f"Could not scan raw data bucket: {e}")

with tab_deriv:
    try:
        # Per-stage breakdown
        stage_prefixes = list(STAGE_PREFIXES.keys())
        prefix_sizes = get_s3_prefix_sizes(DERIVATIVES_BUCKET, stage_prefixes)

        total_deriv_gb = sum(v["total_gb"] for v in prefix_sizes.values())
        total_deriv_cost = total_deriv_gb * S3_STORAGE_PER_GB_MONTH

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Size", f"{total_deriv_gb:.1f} GB")
        col2.metric("Objects", f"{sum(v['n_objects'] for v in prefix_sizes.values()):,}")
        col3.metric("Monthly Storage", f"${total_deriv_cost:.2f}")

        # Per-stage breakdown table
        st.subheader("By Pipeline Stage")
        for prefix, label in STAGE_PREFIXES.items():
            info = prefix_sizes.get(prefix, {"total_gb": 0, "n_objects": 0})
            if info["n_objects"] > 0:
                cost = info["total_gb"] * S3_STORAGE_PER_GB_MONTH
                st.markdown(
                    f"**{label}** — {info['total_gb']:.2f} GB "
                    f"({info['n_objects']} files) — ${cost:.3f}/month"
                )

        st.caption(f"Bucket: `{DERIVATIVES_BUCKET}` ({REGION})")
    except Exception as e:
        st.warning(f"Could not scan derivatives bucket: {e}")


# ── EC2 Compute Costs ───────────────────────────────────────────────────────

st.markdown("---")
st.header("EC2 Compute Costs")

st.markdown(
    "Estimated compute costs for completed and planned pipeline stages. "
    "Based on On-Demand and approximate Spot pricing in ap-southeast-2."
)

# Known completed jobs
completed_jobs = [
    {
        "stage": "Stage 1 — Suite2p",
        "instance": "g4dn.xlarge",
        "hours": 52,
        "spot": True,
        "sessions": 26,
        "note": "Completed 26/26 sessions",
    },
]

# Planned/in-progress jobs
planned_jobs = [
    {
        "stage": "Stage 2 — DLC (current: single instance)",
        "instance": "g4dn.xlarge",
        "hours": 78,
        "spot": False,
        "sessions": 26,
        "note": "~3h/session on T4",
    },
    {
        "stage": "Stage 2 — DLC (planned: 4x parallel A10G)",
        "instance": "g5.xlarge",
        "hours": 10,
        "n_instances": 4,
        "spot": True,
        "sessions": 26,
        "note": "4x parallel, ~1.5h/session on A10G",
    },
]

st.subheader("Completed Jobs")
total_completed_cost = 0
for job in completed_jobs:
    pricing = EC2_PRICING.get(job["instance"], EC2_PRICING["g4dn.xlarge"])
    rate = pricing["spot_approx"] if job.get("spot") else pricing["on_demand"]
    n_inst = job.get("n_instances", 1)
    cost = rate * job["hours"] * n_inst
    total_completed_cost += cost

    spot_label = "Spot" if job.get("spot") else "On-Demand"
    st.markdown(
        f"**{job['stage']}** — `{job['instance']}` ({pricing['gpu']}) "
        f"× {n_inst} — {job['hours']}h {spot_label} — **${cost:.2f}**"
    )
    st.caption(f"  {job['note']}")

st.metric("Total Completed", f"${total_completed_cost:.2f}")

st.subheader("Planned / In-Progress")
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


# ── EC2 Instance Pricing Reference ──────────────────────────────────────────

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


# ── Cost Calculator ─────────────────────────────────────────────────────────

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


# ── Total Project Cost Summary ──────────────────────────────────────────────

st.markdown("---")
st.header("Total Project Cost Summary")

try:
    raw_size = get_s3_bucket_size(RAWDATA_BUCKET)
    raw_monthly = raw_size["total_gb"] * S3_STORAGE_PER_GB_MONTH
except Exception:
    raw_monthly = 0

try:
    stage_prefixes = list(STAGE_PREFIXES.keys())
    prefix_sizes = get_s3_prefix_sizes(DERIVATIVES_BUCKET, stage_prefixes)
    deriv_monthly = sum(v["total_gb"] for v in prefix_sizes.values()) * S3_STORAGE_PER_GB_MONTH
except Exception:
    deriv_monthly = 0

s3_monthly = raw_monthly + deriv_monthly

col1, col2, col3 = st.columns(3)
col1.metric("S3 Storage (monthly)", f"${s3_monthly:.2f}")
col2.metric("EC2 Compute (to date)", f"${total_completed_cost:.2f}")
col3.metric("Total to Date", f"${total_completed_cost + s3_monthly:.2f}")

st.caption(
    "S3 cost is per month. EC2 cost is cumulative. "
    "Does not include data transfer egress or API request costs (typically < $0.10)."
)

st.markdown("---")
st.caption("AWS Cost Report | hm2p v2")
