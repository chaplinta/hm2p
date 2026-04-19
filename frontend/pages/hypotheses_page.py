"""Hypothesis Test Report — comprehensive results across all signal types.

Shows a single table of all measures (dF/F0, deconv, events, CASCADE spikes,
neuropil) with Penk+ vs Penk-CamKII+ comparisons, light vs dark, and
movement effects. All tests are non-parametric.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import mannwhitneyu, wilcoxon

from frontend.data import (
    DERIVATIVES_BUCKET,
    check_stale_data_warning,
    download_s3_bytes,
    load_animals,
    load_experiments,
    parse_session_id,
)


def _page() -> None:
    st.title("Hypothesis Tests")
    check_stale_data_warning(stages=["sync"])
    st.caption(
        "Non-parametric tests across all signal types. "
        "Session-level summaries compared with Mann-Whitney U (between celltypes) "
        "and Wilcoxon signed-rank (within-session paired comparisons)."
    )

    @st.cache_data(ttl=600, show_spinner="Running hypothesis tests across all sessions...")
    def _compute_all_tests():
        import h5py

        experiments = load_experiments()
        animals_list = load_animals()
        animal_map = {a["animal_id"]: a.get("celltype", "?") for a in animals_list}

        # Collect per-session, per-signal metrics
        rows = []

        for exp in experiments:
            eid = exp["exp_id"]
            if exp.get("exclude", "0") == "1":
                continue
            parts = eid.split("_")
            aid = parts[-1]
            sub, ses = parse_session_id(eid)
            ct = animal_map.get(aid, "?")

            # Load ca.h5
            ca_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")
            if ca_bytes is None:
                continue

            try:
                with h5py.File(io.BytesIO(ca_bytes), "r") as f:
                    dff = f["dff"][:]
                    fps = float(f.attrs.get("fps_imaging", 9.8))
                    roi_types = f["roi_types"][:] if "roi_types" in f else np.zeros(dff.shape[0], dtype=np.uint8)
                    signals = {"dF/F0": dff}
                    for key, label in [("deconv_norm", "Deconv"), ("event_masks", "Events (V&H)"),
                                       ("event_masks_sd", "Events (SD)"), ("spikes", "CASCADE spikes")]:
                        if key in f:
                            signals[label] = f[key][:]
            except Exception:
                continue

            # Load sync for behaviour
            sync_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"sync/{sub}/{ses}/sync.h5")
            speed = light_on = bad = None
            if sync_bytes:
                try:
                    with h5py.File(io.BytesIO(sync_bytes), "r") as sf:
                        speed = sf["speed_cm_s"][:] if "speed_cm_s" in sf else None
                        light_on = sf["light_on"][:].astype(bool) if "light_on" in sf else None
                        bad = sf["bad_behav"][:].astype(bool) if "bad_behav" in sf else None
                except Exception:
                    pass

            # Load Fneu for neuropil
            prefix = f"ca_extraction/{sub}/{ses}/suite2p/plane0"
            fneu_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/Fneu.npy")
            if fneu_bytes is not None:
                Fneu = np.load(io.BytesIO(fneu_bytes))
                # Use iscell mask for neuropil mean
                iscell_bytes = download_s3_bytes(DERIVATIVES_BUCKET, f"{prefix}/iscell.npy")
                if iscell_bytes is not None:
                    cell_mask = np.load(io.BytesIO(iscell_bytes))[:, 0].astype(bool)
                    signals["Neuropil (Fneu)"] = np.nanmean(Fneu[cell_mask], axis=0, keepdims=True)
                else:
                    signals["Neuropil (Fneu)"] = np.nanmean(Fneu, axis=0, keepdims=True)

            n_frames = dff.shape[1]
            soma_mask = roi_types == 0

            for sig_name, sig_arr in signals.items():
                sig_arr = np.nan_to_num(sig_arr.astype(np.float32))

                # For soma-specific signals, filter to soma
                if sig_name != "Neuropil (Fneu)" and sig_arr.shape[0] > 1:
                    if soma_mask.sum() > 0 and sig_arr.shape[0] == len(soma_mask):
                        sig_arr = sig_arr[soma_mask]

                n_rois = sig_arr.shape[0]
                n = min(sig_arr.shape[1], n_frames)
                if n_rois == 0:
                    continue

                mean_signal = np.nanmean(sig_arr[:, :n])

                # Condition means
                row = {
                    "session": eid,
                    "animal_id": aid,
                    "celltype": ct,
                    "signal": sig_name,
                    "n_rois": n_rois,
                    "mean_all": mean_signal,
                }

                if speed is not None and light_on is not None and bad is not None:
                    ns = min(n, len(speed), len(light_on), len(bad))
                    valid = ~bad[:ns]
                    moving = valid & (speed[:ns] >= 2.5)
                    stationary = valid & (speed[:ns] < 2.5)
                    light = valid & light_on[:ns]
                    dark = valid & ~light_on[:ns]

                    for cond_name, cond_mask in [("light", light), ("dark", dark),
                                                  ("moving", moving), ("stationary", stationary)]:
                        if cond_mask.sum() > 0:
                            row[f"mean_{cond_name}"] = float(np.nanmean(sig_arr[:, :ns][:, cond_mask]))

                rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df, pd.DataFrame()

        # Run tests
        test_rows = []
        signal_names = df["signal"].unique()

        for sig in signal_names:
            sig_df = df[df["signal"] == sig]
            penk = sig_df[sig_df["celltype"] == "penk"]
            nonpenk = sig_df[sig_df["celltype"] == "nonpenk"]

            # 1. Penk+ vs Penk-CamKII+ overall activity
            pv = penk["mean_all"].dropna()
            nv = nonpenk["mean_all"].dropna()
            if len(pv) >= 2 and len(nv) >= 2:
                U, p = mannwhitneyu(pv, nv, alternative="two-sided")
                test_rows.append({
                    "Signal": sig,
                    "Comparison": "Penk+ vs Penk⁻CamKII+ (overall)",
                    "Test": "Mann-Whitney U",
                    "Penk+ median": f"{pv.median():.4f}",
                    "NonPenk median": f"{nv.median():.4f}",
                    "Statistic": f"{U:.0f}",
                    "p-value": p,
                    "n_penk": len(pv),
                    "n_nonpenk": len(nv),
                })

            # 2. Light vs Dark (within-session, paired)
            valid_ld = sig_df[["mean_light", "mean_dark"]].dropna()
            if len(valid_ld) >= 3:
                stat, p = wilcoxon(valid_ld["mean_light"], valid_ld["mean_dark"])
                test_rows.append({
                    "Signal": sig,
                    "Comparison": "Light vs Dark (all sessions)",
                    "Test": "Wilcoxon signed-rank",
                    "Penk+ median": f"{valid_ld['mean_light'].median():.4f}",
                    "NonPenk median": f"{valid_ld['mean_dark'].median():.4f}",
                    "Statistic": f"{stat:.0f}",
                    "p-value": p,
                    "n_penk": len(valid_ld),
                    "n_nonpenk": len(valid_ld),
                })

            # 3. Moving vs Stationary (within-session, paired)
            valid_ms = sig_df[["mean_moving", "mean_stationary"]].dropna()
            if len(valid_ms) >= 3:
                stat, p = wilcoxon(valid_ms["mean_moving"], valid_ms["mean_stationary"])
                test_rows.append({
                    "Signal": sig,
                    "Comparison": "Moving vs Stationary (all sessions)",
                    "Test": "Wilcoxon signed-rank",
                    "Penk+ median": f"{valid_ms['mean_moving'].median():.4f}",
                    "NonPenk median": f"{valid_ms['mean_stationary'].median():.4f}",
                    "Statistic": f"{stat:.0f}",
                    "p-value": p,
                    "n_penk": len(valid_ms),
                    "n_nonpenk": len(valid_ms),
                })

            # 4. Light modulation index by celltype
            for cond_pair, label in [(("mean_light", "mean_dark"), "Light mod"),
                                      (("mean_moving", "mean_stationary"), "Movement mod")]:
                valid = sig_df[[cond_pair[0], cond_pair[1]]].dropna()
                if len(valid) < 3:
                    continue
                denom = valid[cond_pair[0]] + valid[cond_pair[1]]
                mod_idx = (valid[cond_pair[0]] - valid[cond_pair[1]]) / denom.replace(0, np.nan)
                sig_df_mod = sig_df.loc[mod_idx.index].copy()
                sig_df_mod["mod_idx"] = mod_idx
                p_mod = sig_df_mod[sig_df_mod["celltype"] == "penk"]["mod_idx"].dropna()
                n_mod = sig_df_mod[sig_df_mod["celltype"] == "nonpenk"]["mod_idx"].dropna()
                if len(p_mod) >= 2 and len(n_mod) >= 2:
                    U, p = mannwhitneyu(p_mod, n_mod, alternative="two-sided")
                    test_rows.append({
                        "Signal": sig,
                        "Comparison": f"{label} index: Penk+ vs Penk⁻CamKII+",
                        "Test": "Mann-Whitney U",
                        "Penk+ median": f"{p_mod.median():.4f}",
                        "NonPenk median": f"{n_mod.median():.4f}",
                        "Statistic": f"{U:.0f}",
                        "p-value": p,
                        "n_penk": len(p_mod),
                        "n_nonpenk": len(n_mod),
                    })

        return df, pd.DataFrame(test_rows)

    raw_df, tests_df = _compute_all_tests()

    if tests_df.empty:
        st.warning("No data available for hypothesis testing.")
        st.stop()

    # Summary
    n_tests = len(tests_df)
    n_sig = (tests_df["p-value"] < 0.05).sum()
    n_signals = tests_df["Signal"].nunique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Signal types", n_signals)
    c2.metric("Tests", n_tests)
    c3.metric("Significant (p < 0.05)", n_sig)

    # Format p-values
    def _fmt_p(p):
        if p < 0.001:
            return "< 0.001 ***"
        if p < 0.01:
            return f"{p:.3f} **"
        if p < 0.05:
            return f"{p:.3f} *"
        return f"{p:.3f}"

    tests_df["p"] = tests_df["p-value"].apply(_fmt_p)

    # Display table
    st.subheader("Results")
    display_cols = ["Signal", "Comparison", "Test", "Penk+ median", "NonPenk median", "p", "n_penk", "n_nonpenk"]

    # Colour significant rows
    st.dataframe(
        tests_df[display_cols].sort_values(["Signal", "p"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "p": st.column_config.TextColumn("p-value"),
            "Penk+ median": st.column_config.TextColumn("Group 1 median"),
            "NonPenk median": st.column_config.TextColumn("Group 2 median"),
        },
    )

    # Significant results highlighted
    sig_tests = tests_df[tests_df["p-value"] < 0.05]
    if len(sig_tests) > 0:
        st.subheader(f"Significant Results ({len(sig_tests)})")
        for _, row in sig_tests.iterrows():
            st.markdown(
                f"**{row['Signal']}** — {row['Comparison']}: "
                f"p = {_fmt_p(row['p-value'])}"
            )

    with st.expander("Methods"):
        st.markdown(
            "**Between-group (Penk+ vs Penk⁻CamKII+):** Session-level means compared "
            "with Mann-Whitney U test. Each session contributes one data point.\n\n"
            "**Within-session (light vs dark, moving vs stationary):** Paired Wilcoxon "
            "signed-rank test on session-level condition means.\n\n"
            "**Modulation indices:** (condition1 - condition2) / (condition1 + condition2), "
            "compared between celltypes with Mann-Whitney U.\n\n"
            "**Signal types:** dF/F0 (raw calcium), Deconv (Suite2p normalized), "
            "Events V&H (Voigts & Harnett 2020), Events SD (Zong et al. 2022), "
            "CASCADE spikes (Rupprecht et al. 2021), Neuropil (mean Fneu from Suite2p).\n\n"
            "All tests are non-parametric. No multiple comparisons correction is applied "
            "to this exploratory table — interpret accordingly."
        )


_page()
