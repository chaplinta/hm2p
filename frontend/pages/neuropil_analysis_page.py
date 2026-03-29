"""Neuropil Analysis — local network input from Fneu.

The neuropil signal reflects aggregate axonal/dendritic activity surrounding
each ROI (Kerr et al. 2005). This page analyses how the neuropil signal
relates to movement, light condition, and cell type.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from frontend.data import (
    DERIVATIVES_BUCKET,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)

log = logging.getLogger("hm2p.frontend.neuropil_analysis")


def _page() -> None:
    st.title("Neuropil Analysis")
    st.caption(
        "The neuropil signal (Fneu) represents local network input — aggregate "
        "axonal and dendritic fluorescence surrounding each ROI (Kerr et al. 2005). "
        "Correlations with behaviour reveal brain-state modulation independent of "
        "single-cell tuning."
    )

    experiments = load_experiments()
    animals = load_animals()
    animal_map = {a["animal_id"]: a for a in animals}

    exp_ids = [e["exp_id"] for e in experiments]
    selected = st.selectbox(
        "Session", exp_ids,
        format_func=lambda x: f"{x} ({animal_map.get(x.split('_')[-1], {}).get('celltype', '?')})",
        key="npil_session",
    )
    if not selected:
        st.stop()

    sub, ses = parse_session_id(selected)
    celltype = animal_map.get(selected.split("_")[-1], {}).get("celltype", "?")

    @st.cache_data(ttl=300)
    def _load(sub: str, ses: str) -> dict | None:
        import h5py

        # Load Fneu and F from Suite2p
        prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0"
        fneu_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/Fneu.npy")
        f_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/F.npy")
        iscell_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/iscell.npy")

        if fneu_bytes is None or f_bytes is None:
            return None

        Fneu = np.load(io.BytesIO(fneu_bytes))
        F = np.load(io.BytesIO(f_bytes))
        iscell = np.load(io.BytesIO(iscell_bytes)) if iscell_bytes else None
        cell_mask = iscell[:, 0].astype(bool) if iscell is not None else np.ones(F.shape[0], dtype=bool)

        result = {"Fneu": Fneu, "F": F, "cell_mask": cell_mask}

        # Load ca.h5 for dF/F and fps
        ca_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
        if ca_bytes:
            with h5py.File(io.BytesIO(ca_bytes), "r") as f:
                result["dff"] = f["dff"][:]
                result["fps"] = float(f.attrs.get("fps_imaging", 9.8))
                if "roi_types" in f:
                    result["roi_types"] = f["roi_types"][:]

        # Load sync.h5 for behaviour
        sync_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
        if sync_bytes:
            with h5py.File(io.BytesIO(sync_bytes), "r") as f:
                for key in ("speed_cm_s", "ahv_deg_s", "light_on", "active", "bad_behav"):
                    if key in f:
                        result[key] = f[key][:]

        return result

    with st.spinner("Loading neuropil data..."):
        data = _load(sub, ses)

    if data is None:
        st.warning("No Suite2p data (Fneu.npy) found for this session.")
        st.stop()

    from hm2p.calcium.neuropil_analysis import (
        compute_mean_neuropil,
        compute_neuropil_ratio,
        neuropil_behaviour_correlation,
        neuropil_soma_correlation,
    )

    Fneu = data["Fneu"]
    F = data["F"]
    cell_mask = data["cell_mask"]
    fps = data.get("fps", 9.8)
    n_rois_all = Fneu.shape[0]
    n_frames = Fneu.shape[1]
    time_s = np.arange(n_frames) / fps
    n_accepted = int(cell_mask.sum())

    st.markdown(f"**{sub}/{ses}** — {celltype} — {n_rois_all} total ROIs, {n_accepted} accepted")

    mean_fneu = compute_mean_neuropil(Fneu, cell_mask)
    neuropil_ratios = compute_neuropil_ratio(F, Fneu)

    has_behaviour = "speed_cm_s" in data
    has_dff = "dff" in data

    # ── Tabs ──
    tabs = st.tabs(["Overview", "Behaviour", "Celltype", "Decorrelation"])

    # ── Tab 1: Overview ──
    with tabs[0]:
        st.subheader("Mean Neuropil Signal")

        ds = max(1, n_frames // 3000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_s[::ds], y=mean_fneu[::ds],
            mode="lines", line=dict(color="steelblue", width=1),
            name="Mean Fneu",
        ))

        # Add light shading if available
        if "light_on" in data:
            light = data["light_on"][:n_frames].astype(bool)
            dark_starts, dark_ends = [], []
            in_dark = not light[0]
            if in_dark:
                dark_starts.append(time_s[0])
            for i in range(1, min(len(light), n_frames)):
                if not light[i] and light[i - 1]:
                    dark_starts.append(time_s[i])
                elif light[i] and not light[i - 1]:
                    dark_ends.append(time_s[i])
            if len(dark_starts) > len(dark_ends):
                dark_ends.append(time_s[-1])
            for ds_t, de_t in zip(dark_starts, dark_ends):
                fig.add_vrect(x0=ds_t, x1=de_t, fillcolor="rgba(50,50,50,0.12)",
                              layer="below", line_width=0)

        fig.update_layout(
            height=250, xaxis_title="Time (s)", yaxis_title="Mean Fneu",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Neuropil ratio distribution
        st.subheader("Neuropil-to-Soma Ratio")
        valid_ratios = neuropil_ratios[np.isfinite(neuropil_ratios)]
        fig_ratio = px.histogram(x=valid_ratios, nbins=40, title="Fneu/F ratio per ROI")
        fig_ratio.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=40),
                                xaxis_title="Fneu / F", yaxis_title="ROI count")
        st.plotly_chart(fig_ratio, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Median ratio", f"{np.nanmedian(valid_ratios):.3f}")
        c2.metric("Mean Fneu", f"{np.nanmean(mean_fneu):.1f}")
        c3.metric("Std Fneu", f"{np.nanstd(mean_fneu):.1f}")

    # ── Tab 2: Behaviour correlations ──
    with tabs[1]:
        st.subheader("Neuropil × Behaviour")
        if not has_behaviour:
            st.warning("No behavioural data (sync.h5) for this session.")
        else:
            speed = data["speed_cm_s"][:n_frames]
            ahv = data.get("ahv_deg_s")
            if ahv is not None:
                ahv = ahv[:n_frames]
            light_on = data.get("light_on")
            if light_on is not None:
                light_on = light_on[:n_frames]
            bad = data.get("bad_behav")
            active_mask = ~bad[:n_frames].astype(bool) if bad is not None else np.ones(n_frames, dtype=bool)

            # Resample Fneu to imaging rate if needed
            fneu_signal = mean_fneu
            if len(speed) != n_frames:
                n = min(len(speed), n_frames)
                fneu_signal = mean_fneu[:n]
                speed = speed[:n]
                if ahv is not None:
                    ahv = ahv[:n]
                if light_on is not None:
                    light_on = light_on[:n]
                active_mask = active_mask[:n]

            corr = neuropil_behaviour_correlation(fneu_signal, speed, ahv, light_on, active_mask)

            cols = st.columns(4)
            if "speed_corr" in corr:
                cols[0].metric("Speed ρ", f"{corr['speed_corr']:.3f}")
            if "ahv_corr" in corr:
                cols[1].metric("|AHV| ρ", f"{corr['ahv_corr']:.3f}")
            if "light_mod_index" in corr:
                cols[2].metric("Light mod", f"{corr['light_mod_index']:.3f}")
            if "movement_mod_index" in corr:
                cols[3].metric("Movement mod", f"{corr['movement_mod_index']:.3f}")

            # Condition means
            if "mean_fneu_light" in corr:
                st.markdown(
                    f"**Light:** {corr['mean_fneu_light']:.1f} | "
                    f"**Dark:** {corr['mean_fneu_dark']:.1f} | "
                    f"p = {corr.get('light_dark_p', 'N/A')}"
                )

            if "mean_fneu_moving" in corr:
                st.markdown(
                    f"**Moving:** {corr['mean_fneu_moving']:.1f} | "
                    f"**Stationary:** {corr['mean_fneu_stationary']:.1f}"
                )

            # Scatter: Fneu vs speed
            v = active_mask & np.isfinite(fneu_signal) & np.isfinite(speed)
            ds2 = max(1, v.sum() // 2000)
            idx = np.where(v)[0][::ds2]
            fig_sc = px.scatter(
                x=speed[idx], y=fneu_signal[idx], opacity=0.3,
                title="Neuropil vs Speed",
                labels={"x": "Speed (cm/s)", "y": "Mean Fneu"},
            )
            fig_sc.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── Tab 3: Celltype comparison ──
    with tabs[2]:
        st.subheader("Cross-Session Celltype Comparison")
        st.caption("Compare neuropil properties between Penk+ and Penk⁻CamKII+ across all sessions.")

        @st.cache_data(ttl=600)
        def _load_all_neuropil():
            import h5py as _h5py
            rows = []
            for exp in experiments:
                eid = exp["exp_id"]
                parts = eid.split("_")
                aid = parts[-1]
                s, ss = parse_session_id(eid)
                ct = animal_map.get(aid, {}).get("celltype", "?")

                prefix = f"ca_extraction/{s}/{ss}/suite2p/plane0"
                fneu_b = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/Fneu.npy")
                f_b = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/F.npy")
                if fneu_b is None or f_b is None:
                    continue

                _Fneu = np.load(io.BytesIO(fneu_b))
                _F = np.load(io.BytesIO(f_b))
                _ratios = compute_neuropil_ratio(_F, _Fneu)

                # Load behaviour
                sync_b = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{s}/{ss}/sync.h5")
                _corr = {}
                if sync_b:
                    with _h5py.File(io.BytesIO(sync_b), "r") as sf:
                        _spd = sf["speed_cm_s"][:] if "speed_cm_s" in sf else None
                        _lt = sf["light_on"][:] if "light_on" in sf else None
                        _bad = sf["bad_behav"][:] if "bad_behav" in sf else None

                    if _spd is not None:
                        n = min(_Fneu.shape[1], len(_spd))
                        _mfneu = np.nanmean(_Fneu[:, :n], axis=0)
                        _amask = ~_bad[:n].astype(bool) if _bad is not None else np.ones(n, dtype=bool)
                        _corr = neuropil_behaviour_correlation(_mfneu, _spd[:n], light_on=_lt[:n] if _lt is not None else None, active_mask=_amask)

                rows.append({
                    "session": eid,
                    "celltype": ct,
                    "median_ratio": float(np.nanmedian(_ratios)),
                    "mean_fneu": float(np.nanmean(_Fneu)),
                    "n_rois": _Fneu.shape[0],
                    **{k: v for k, v in _corr.items() if isinstance(v, (int, float))},
                })
            return pd.DataFrame(rows)

        with st.spinner("Loading neuropil data across sessions..."):
            all_npil = _load_all_neuropil()

        if all_npil.empty:
            st.warning("No data.")
        else:
            penk = all_npil[all_npil["celltype"] == "penk"]
            nonpenk = all_npil[all_npil["celltype"] == "nonpenk"]

            metrics = [
                ("speed_corr", "Speed ρ"),
                ("light_mod_index", "Light mod"),
                ("movement_mod_index", "Movement mod"),
                ("median_ratio", "Fneu/F ratio"),
            ]

            cols = st.columns(len(metrics))
            for i, (key, label) in enumerate(metrics):
                pv = penk[key].dropna()
                nv = nonpenk[key].dropna()
                with cols[i]:
                    if len(pv) >= 2 and len(nv) >= 2:
                        _, p = mannwhitneyu(pv, nv, alternative="two-sided")
                        st.metric(label, f"P:{pv.median():.3f} N:{nv.median():.3f}")
                        st.caption(f"p={p:.3f}" + (" *" if p < 0.05 else ""))

            fig_ct = px.strip(
                all_npil, x="celltype", y="speed_corr", color="celltype",
                color_discrete_map={"penk": "#1f77b4", "nonpenk": "#ff7f0e"},
                title="Neuropil–Speed Correlation by Cell Type",
            )
            fig_ct.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_ct, use_container_width=True)

    # ── Tab 4: Decorrelation ──
    with tabs[3]:
        st.subheader("Neuropil–Soma Decorrelation")
        st.caption(
            "Spearman correlation between each ROI's neuropil (Fneu) and its somatic dF/F. "
            "High = soma dominated by shared network input. Low = independent cell activity."
        )

        if not has_dff:
            st.warning("No dF/F data available.")
        else:
            dff = data["dff"]
            # Match Fneu to dff ROIs (dff now includes all ROIs)
            n_match = min(Fneu.shape[0], dff.shape[0])
            corrs = neuropil_soma_correlation(dff[:n_match], Fneu[:n_match])

            fig_dec = px.histogram(
                x=corrs[np.isfinite(corrs)], nbins=40,
                title="Neuropil–Soma Correlation Distribution",
                labels={"x": "Spearman ρ", "y": "ROI count"},
            )
            fig_dec.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_dec, use_container_width=True)

            c1, c2 = st.columns(2)
            valid_c = corrs[np.isfinite(corrs)]
            c1.metric("Median ρ", f"{np.median(valid_c):.3f}")
            c2.metric("% with ρ > 0.5", f"{(valid_c > 0.5).mean() * 100:.1f}%")

    with st.expander("Methods & References"):
        st.markdown(
            "**Neuropil signal:** Suite2p Fneu.npy — fluorescence from the annular "
            "region surrounding each ROI, primarily axonal and dendritic processes.\n\n"
            "**Kerr et al. 2005.** \"Imaging input and output of neocortical networks in vivo.\" "
            "PNAS 102(39):14063-14068. — Neuropil calcium signal = 'optical encephalogram' "
            "reflecting local input activity.\n\n"
            "**Dipoppa et al. 2018.** \"Vision and locomotion shape the interactions between "
            "neuron types in mouse visual cortex.\" Neuron 98(3):602-615. — Locomotion modulates "
            "baseline activity and effective synaptic connectivity."
        )


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        _page()
except ImportError:
    _page()
