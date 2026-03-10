"""Anatomy — Injection site localization and brain registration.

Displays injection site coordinates extracted from brainreg-registered
serial2p whole-brain volumes. Shows per-animal injection locations in
Allen CCFv3 atlas coordinates with interactive 3D visualization.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

st.header("Anatomy & Injection Sites")

st.markdown("""
Injection site locations extracted from serial2p whole-brain volumes
registered to the Allen Mouse Brain CCFv3 atlas using
[brainreg](https://brainglobe.info/documentation/brainreg/).
""")

# ── Imports ──────────────────────────────────────────────────────────────

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        RAWDATA_BUCKET,
        download_s3_bytes,
        load_animals,
        load_experiments,
        parse_session_id,
    )
except ImportError:
    st.error("Frontend data module not available.")
    st.stop()

# ── Load metadata ────────────────────────────────────────────────────────

animals = load_animals()
experiments = load_experiments()

if not animals:
    st.warning("No animals.csv data available.")
    st.stop()

animals_df = pd.DataFrame(animals)

# ── Brain volumes on S3 ─────────────────────────────────────────────────

st.subheader("Serial2p Brain Volumes")


@st.cache_data(ttl=3600)
def _list_brain_volumes() -> list[str]:
    """List brain volume files on S3."""
    import boto3

    try:
        s3 = boto3.client("s3", region_name="ap-southeast-2")
        resp = s3.list_objects_v2(
            Bucket=RAWDATA_BUCKET,
            Prefix="sourcedata/brains-sorted/",
            Delimiter="/",
        )
        files = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            if fname and not fname.startswith("."):
                files.append(fname)
        return sorted(files)
    except Exception as e:
        log.warning("Failed to list brain volumes: %s", e)
        return []


brain_files = _list_brain_volumes()

if brain_files:
    st.success(f"{len(brain_files)} brain volume files on S3")

    brain_animals = set()
    for f in brain_files:
        parts = f.split("_")
        if len(parts) >= 3:
            brain_animals.add(parts[2])

    green_files = [f for f in brain_files if "green" in f.lower()]
    red_files = [f for f in brain_files if "red" in f.lower()]
    blue_files = [f for f in brain_files if "blue" in f.lower()]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Animals", len(brain_animals))
    col2.metric("Green (signal)", len(green_files))
    col3.metric("Red", len(red_files))
    col4.metric("Blue", len(blue_files))

    with st.expander("All brain volume files"):
        st.dataframe(pd.DataFrame({"File": brain_files}), use_container_width=True)
else:
    st.info("No brain volumes found on S3 yet.")

# ── brainreg region volumes ──────────────────────────────────────────────

st.subheader("Brain Region Volumes (brainreg)")


@st.cache_data(ttl=3600)
def _load_brainreg_volumes() -> dict[str, pd.DataFrame]:
    """Load volumes.csv for all animals from S3 brains-reg."""
    import boto3

    result = {}
    try:
        s3 = boto3.client("s3", region_name="ap-southeast-2")
        resp = s3.list_objects_v2(
            Bucket=DERIVATIVES_BUCKET,
            Prefix="brains-reg/",
        )
        vol_keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith("volumes.csv") and obj["Size"] > 0
        ]

        for key in vol_keys:
            # Extract animal ID from path
            dir_name = key.split("/")[1]  # e.g. ds_TC_1114353_...
            parts = dir_name.split("_")
            animal_id = parts[2] if len(parts) >= 3 else dir_name

            obj = s3.get_object(Bucket=DERIVATIVES_BUCKET, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            if not df.empty:
                result[animal_id] = df
    except Exception as e:
        log.warning("Failed to load brainreg volumes: %s", e)
    return result


volumes_data = _load_brainreg_volumes()

if volumes_data:
    st.success(f"Region volumes available for {len(volumes_data)} animals")

    selected_animal = st.selectbox(
        "Select animal for region volumes",
        options=sorted(volumes_data.keys()),
    )

    if selected_animal and selected_animal in volumes_data:
        vol_df = volumes_data[selected_animal]

        # RSP regions
        rsp_mask = vol_df["structure_name"].str.contains(
            "Retrosplenial|RSP", case=False, na=False
        )
        rsp_df = vol_df[rsp_mask]

        if len(rsp_df) > 0:
            st.markdown("**Retrosplenial cortex (RSP) regions:**")
            st.dataframe(
                rsp_df.style.format({
                    "left_volume_mm3": "{:.4f}",
                    "right_volume_mm3": "{:.4f}",
                    "total_volume_mm3": "{:.4f}",
                }),
                use_container_width=True,
            )
            rsp_total = rsp_df["total_volume_mm3"].sum()
            st.metric("Total RSP volume", f"{rsp_total:.3f} mm³")

        # Top regions by volume
        with st.expander("Top 20 regions by volume"):
            top = vol_df.nlargest(20, "total_volume_mm3")
            st.dataframe(
                top.style.format({
                    "left_volume_mm3": "{:.4f}",
                    "right_volume_mm3": "{:.4f}",
                    "total_volume_mm3": "{:.4f}",
                }),
                use_container_width=True,
            )

        with st.expander(f"All regions ({len(vol_df)})"):
            st.dataframe(vol_df, use_container_width=True)
else:
    st.info("No brainreg volumes data available on S3.")

# ── Injection site coordinates ───────────────────────────────────────────

st.subheader("Injection Site Coordinates")

inj_cols = ["inj_ap", "inj_ml", "inj_dv"]
has_inj_data = all(col in animals_df.columns for col in inj_cols)

if has_inj_data:
    inj_df = animals_df[["animal_id", "celltype"] + inj_cols].copy()
    for col in inj_cols:
        inj_df[col] = pd.to_numeric(inj_df[col], errors="coerce")
    inj_df = inj_df.dropna(subset=inj_cols, how="all")

    if len(inj_df) > 0:
        st.dataframe(
            inj_df.style.format(
                {col: "{:.3f}" for col in inj_cols if col in inj_df.columns},
                na_rep="—",
            ),
            use_container_width=True,
        )

        # ── Interactive 3D Injection Site Visualization ──────────────────
        st.subheader("3D Injection Site Map")
        st.caption(
            "Interactive 3D view of injection sites in Allen CCFv3 atlas space. "
            "Blue = Penk+, Red = CamKII+. Drag to rotate, scroll to zoom."
        )

        fig_3d = go.Figure()

        # Add a transparent brain outline (simplified ellipsoid)
        # Allen mouse brain approximate dimensions: AP ~13.2mm, DV ~8mm, ML ~11.4mm
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        # Brain centroid approximately at AP=6.6, DV=4, ML=5.7
        brain_ap = 6.6 + 6.0 * np.outer(np.cos(u), np.sin(v))
        brain_dv = 4.0 + 3.5 * np.outer(np.sin(u), np.sin(v))
        brain_ml = 5.7 + 5.0 * np.outer(np.ones_like(u), np.cos(v))

        fig_3d.add_trace(go.Surface(
            x=brain_ap, y=brain_ml, z=brain_dv,
            opacity=0.08,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            showscale=False,
            name="Brain outline",
            hoverinfo="skip",
        ))

        # Add RSP region (approximate location)
        # RSP: AP ~5-8mm, DV ~0.5-2mm, ML ~4-7mm (bilateral)
        rsp_u = np.linspace(0, 2 * np.pi, 20)
        rsp_v = np.linspace(0, np.pi, 10)
        rsp_ap = 6.5 + 1.5 * np.outer(np.cos(rsp_u), np.sin(rsp_v))
        rsp_dv = 1.0 + 0.5 * np.outer(np.sin(rsp_u), np.sin(rsp_v))
        rsp_ml = 5.7 + 0.8 * np.outer(np.ones_like(rsp_u), np.cos(rsp_v))

        fig_3d.add_trace(go.Surface(
            x=rsp_ap, y=rsp_ml, z=rsp_dv,
            opacity=0.15,
            colorscale=[[0, "green"], [1, "green"]],
            showscale=False,
            name="RSP (approx)",
            hoverinfo="skip",
        ))

        # Add injection sites as 3D markers
        for celltype, color, symbol in [("penk", "blue", "circle"), ("nonpenk", "red", "square")]:
            ct_data = inj_df[inj_df["celltype"] == celltype]
            if len(ct_data) == 0:
                continue

            ap = pd.to_numeric(ct_data["inj_ap"], errors="coerce")
            ml = pd.to_numeric(ct_data["inj_ml"], errors="coerce")
            dv = pd.to_numeric(ct_data["inj_dv"], errors="coerce")
            labels = ct_data["animal_id"].astype(str)

            ct_label = "Penk+" if celltype == "penk" else "CamKII+"

            fig_3d.add_trace(go.Scatter3d(
                x=ap, y=ml, z=dv,
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=color,
                    opacity=0.9,
                    symbol=symbol,
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=9),
                name=ct_label,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "AP: %{x:.3f} mm<br>"
                    "ML: %{y:.3f} mm<br>"
                    "DV: %{z:.3f} mm<br>"
                    f"Type: {ct_label}<extra></extra>"
                ),
            ))

        # Add midline plane
        midline_ap = np.array([0, 13.2, 13.2, 0])
        midline_dv = np.array([0, 0, 8, 8])
        midline_ml = np.array([5.7, 5.7, 5.7, 5.7])

        fig_3d.update_layout(
            scene=dict(
                xaxis_title="AP (mm)",
                yaxis_title="ML (mm)",
                zaxis_title="DV (mm)",
                zaxis=dict(autorange="reversed"),  # DV: 0=dorsal at top
                aspectmode="data",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    up=dict(x=0, y=0, z=-1),
                ),
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(x=0.02, y=0.98),
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # ── 2D projections ───────────────────────────────────────────────
        with st.expander("2D projection views"):
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            for celltype, color, marker in [("penk", "blue", "o"), ("nonpenk", "red", "s")]:
                ct_data = inj_df[inj_df["celltype"] == celltype]
                if len(ct_data) == 0:
                    continue

                ap = pd.to_numeric(ct_data["inj_ap"], errors="coerce")
                ml = pd.to_numeric(ct_data["inj_ml"], errors="coerce")
                dv = pd.to_numeric(ct_data["inj_dv"], errors="coerce")
                ct_label = "Penk+" if celltype == "penk" else "CamKII+"

                axes[0].scatter(ml, ap, c=color, marker=marker, s=80, alpha=0.7, label=ct_label)
                axes[1].scatter(ap, dv, c=color, marker=marker, s=80, alpha=0.7, label=ct_label)
                axes[2].scatter(ml, dv, c=color, marker=marker, s=80, alpha=0.7, label=ct_label)

            axes[0].set_xlabel("ML (mm)")
            axes[0].set_ylabel("AP (mm)")
            axes[0].set_title("Top view (AP vs ML)")
            axes[0].legend(fontsize=8)

            axes[1].set_xlabel("AP (mm)")
            axes[1].set_ylabel("DV (mm)")
            axes[1].set_title("Side view (AP vs DV)")
            axes[1].invert_yaxis()

            axes[2].set_xlabel("ML (mm)")
            axes[2].set_ylabel("DV (mm)")
            axes[2].set_title("Coronal view (ML vs DV)")
            axes[2].invert_yaxis()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Per-animal detail ────────────────────────────────────────────
        with st.expander("Per-animal injection details"):
            for _, row in inj_df.iterrows():
                ct_label = "Penk+" if row.get("celltype") == "penk" else "CamKII+"
                st.markdown(
                    f"**{row['animal_id']}** ({ct_label}): "
                    f"AP={row.get('inj_ap', 0):.3f}, "
                    f"ML={row.get('inj_ml', 0):.3f}, "
                    f"DV={row.get('inj_dv', 0):.3f} mm"
                )
    else:
        st.info(
            "No injection coordinates in animals.csv yet. "
            "Run brainreg registration and injection extraction to populate."
        )
else:
    st.info(
        "animals.csv does not have injection coordinate columns (inj_ap, inj_ml, inj_dv). "
        "These will be added after brainreg registration."
    )

# ── Z-stacks on S3 ──────────────────────────────────────────────────────

st.subheader("Z-Stacks")

zstack_sessions = [e for e in experiments if e.get("zstack_id", "").strip()]
st.metric("Sessions with z-stacks", f"{len(zstack_sessions)} / {len(experiments)}")

if zstack_sessions:
    zstack_df = pd.DataFrame([
        {
            "Session": e["exp_id"],
            "Z-stack ID": e["zstack_id"],
            "Animal": e["exp_id"].split("_")[-1],
        }
        for e in zstack_sessions
    ])
    st.dataframe(zstack_df, use_container_width=True)

# ── Methods ──────────────────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
    **Brain registration** uses brainreg (BrainGlobe) to register serial2p
    autofluorescence volumes to the Allen Mouse Brain Common Coordinate
    Framework v3 (CCFv3). Registration involves reorientation, affine
    alignment, and freeform deformation.

    **Injection site segmentation** uses brainglobe-segmentation (napari
    plugin) with Otsu thresholding on the signal channel. Centroid, volume,
    and brain region overlap are computed automatically.

    **3D visualization** uses Plotly for interactive rendering. Brain outline
    and RSP region are approximate ellipsoids for spatial reference. Exact
    atlas meshes can be rendered with brainrender (VTK-based, not available
    in Streamlit).

    **Coordinate system** (BrainGlobe Allen 25 um atlas):
    - X: anterior-posterior
    - Y: dorsal-ventral
    - Z: left-right (midline at 5.7 mm)

    Left hemisphere coordinates are mirrored to right hemisphere.

    **References:**
    - Tyson et al. 2022. "Accurate determination of marker location within
      whole-brain microscopy images." *Scientific Reports*.
      doi:10.1038/s41598-021-04676-9
    - Wang et al. 2020. "The Allen Mouse Brain Common Coordinate Framework."
      *Cell*. doi:10.1016/j.cell.2020.04.007
    - Claudi et al. 2021. "Visualizing anatomically registered data with
      brainrender." *eLife*. doi:10.7554/eLife.65751
    """)
