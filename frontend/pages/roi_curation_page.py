"""ROI Curation — manual review of ambiguous soma/dend/artefact ROIs.

Surfaces ROIs whose soma classifier probabilities are ambiguous (default:
``0.3 < p_soma < 0.7``) and lets the curator confirm or override the
model's argmax label.  Labels are appended to ``metadata/roi_curation.csv``
in append-only fashion (re-labelling adds a new row; latest timestamp wins
on read) and can be applied to the local ``ca.h5`` for the session via a
button.

The CSV is the same one consumed by ``scripts/train_soma_classifier.py``,
so curation work accumulates a growing labelled training set over time.

Frontend rules followed (CLAUDE.md):

* Real S3-backed data only — no synthetic fallback. The page shows a clear
  message if no calcium data is available.
* All controls in the page body (no sidebar filters).
* Methods & References expander documents the methodology and cites the
  classifier framework.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from frontend.data import (
    DERIVATIVES_BUCKET,
    METADATA_DIR,
    download_s3_bytes,
    load_animals,
    load_ca_one,
    load_experiments,
    load_suite2p_spatial_one,
    parse_session_id,
)

from hm2p.extraction.curation import (
    append_curation_row,
    apply_curation_to_ca_h5,
    labels_for_session,
    load_latest_labels,
)
from hm2p.extraction.soma_classifier import CLASS_NAMES

log = logging.getLogger("hm2p.frontend.roi_curation")

CURATION_CSV = METADATA_DIR / "roi_curation.csv"

ROI_TYPE_NAMES = {0: "Soma", 1: "Dendrite", 2: "Non-cell"}
ROI_TYPE_COLORS = {0: "turquoise", 1: "darkorchid", 2: "gray"}

# ROI-mask overlay colour per class label — vivid on a grayscale background.
LABEL_COLORS = {
    "soma": "rgba(0,229,255,0.45)",  # cyan
    "dend": "rgba(255,64,255,0.45)",  # magenta
    "artefact": "rgba(255,140,0,0.45)",  # orange
}


# ── Page header ───────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
try:
    _CURATION_CSV_DISPLAY = str(CURATION_CSV.relative_to(_REPO_ROOT))
except ValueError:
    _CURATION_CSV_DISPLAY = str(CURATION_CSV)

# Use the full browser width for this page (the anatomy image benefits from it).
st.markdown(
    "<style>.block-container{max-width:100% !important;"
    "padding-left:1.5rem;padding-right:1.5rem;}</style>",
    unsafe_allow_html=True,
)

st.title("ROI Curation")
st.caption(
    "Review ROIs that the soma classifier is uncertain about and assign a "
    "confirmed soma / dendrite / artefact label. Labels are written to "
    f"`{_CURATION_CSV_DISPLAY}` "
    "and feed both the runtime label resolver and offline classifier training."
)


# ── Cached loaders ────────────────────────────────────────────────────────


@st.cache_data(ttl=1800, show_spinner=False)
def _session_list() -> list[dict]:
    """Lightweight per-session metadata (exp_id, sub, ses, celltype).

    Built from metadata CSVs only — no ca.h5 downloads — so the page renders
    the session selector immediately. Each session's calcium data is loaded
    lazily once selected.
    """
    animals = {a["animal_id"]: a for a in load_animals()}
    out = []
    for exp in load_experiments():
        exp_id = exp["exp_id"]
        sub, ses = parse_session_id(exp_id)
        animal_id = exp_id.split("_")[-1]
        out.append(
            {
                "exp_id": exp_id,
                "sub": sub,
                "ses": ses,
                "animal_id": animal_id,
                "celltype": animals.get(animal_id, {}).get("celltype", "unknown"),
            }
        )
    return out


@st.cache_data(ttl=300)
def _download_session_ca_h5(sub: str, ses: str) -> bytes | None:
    """Fetch the raw ca.h5 bytes for a session (for local apply-curation)."""
    return download_s3_bytes(DERIVATIVES_BUCKET, f"calcium/{sub}/{ses}/ca.h5")


sessions_meta = _session_list()

if not sessions_meta:
    st.warning("No sessions found in metadata. Check that `experiments.csv` is present.")
    st.stop()


# Curator is fixed for this single-operator project (logged with each label).
curator = os.environ.get("HM2P_CURATOR") or os.environ.get("USER") or "tristan"


# ── Session selector (page body, not sidebar) ─────────────────────────────

session_options = [s["exp_id"] for s in sessions_meta]
sel_label = st.selectbox(
    "Session",
    options=session_options,
    key="rc_session",
)
ses_idx = session_options.index(sel_label)
ses_meta = sessions_meta[ses_idx]

# Load only the selected session's calcium data (dff, roi_qc, event masks).
with st.spinner(f"Loading calcium data for {ses_meta['exp_id']}..."):
    ca = load_ca_one(ses_meta["exp_id"])
if ca is None:
    st.warning(
        f"Could not load `ca.h5` for {ses_meta['exp_id']}. Run Stage 4 to "
        "generate it, or check S3 connectivity."
    )
    st.stop()
ses = {**ses_meta, **ca}

celltype_label = "Penk+" if ses["celltype"] == "penk" else "Penk⁻CamKII+"
st.markdown(
    f"<small style='color:gray'>{ses['exp_id']} &mdash; "
    f"{celltype_label}, {ses['n_rois']} ROIs, {ses['n_frames']} frames</small>",
    unsafe_allow_html=True,
)


# ── Filter controls (page body) ───────────────────────────────────────────

roi_qc_ses = ses.get("roi_qc")
n_rois_ses = ses["n_rois"]

if roi_qc_ses is None or "p_soma" not in roi_qc_ses:
    st.warning(
        "No soma classifier probabilities available for this session. Re-run "
        "Stage 4 with the soma classifier enabled to populate "
        "`roi_qc/{p_soma, p_dend, p_artefact}` in `ca.h5`."
    )
    st.stop()

p_soma = np.asarray(roi_qc_ses.get("p_soma", np.full(n_rois_ses, np.nan, dtype=np.float32)))
p_dend = np.asarray(roi_qc_ses.get("p_dend", np.full(n_rois_ses, np.nan, dtype=np.float32)))
p_art = np.asarray(roi_qc_ses.get("p_artefact", np.full(n_rois_ses, np.nan, dtype=np.float32)))


fc1, fc2, fc3 = st.columns([2, 2, 1])
with fc1:
    p_lo, p_hi = st.slider(
        "p_soma range (ambiguous)",
        min_value=0.0,
        max_value=1.0,
        value=(0.3, 0.7),
        step=0.05,
        key="rc_psoma_range",
        help=(
            "Default is the 0.3–0.7 ambiguous band. Widen to review more "
            "ROIs; narrow to 0.4–0.6 for the hardest cases."
        ),
    )
with fc2:
    show_mode = st.radio(
        "Filter mode",
        [
            "Ambiguous (p_soma in range)",
            "Soma-flagged (argmax)",
            "Artefact-flagged (argmax)",
            "Dend-flagged (argmax)",
            "All",
        ],
        index=0,
        key="rc_mode",
        horizontal=False,
    )
with fc3:
    hide_curated = st.checkbox(
        "Hide already-curated",
        value=True,
        key="rc_hide_curated",
        help="Hide ROIs that already have a row in roi_curation.csv.",
    )


# Existing curation labels for this session.
existing = labels_for_session(CURATION_CSV, ses["exp_id"])


# Build candidate ROI list.
finite = np.isfinite(p_soma) & np.isfinite(p_dend) & np.isfinite(p_art)
argmax = np.argmax(np.stack([p_soma, p_dend, p_art], axis=1), axis=1)
arg_labels = [CLASS_NAMES[i] if finite[j] else "soma" for j, i in enumerate(argmax)]

if show_mode == "Ambiguous (p_soma in range)":
    candidate = np.where(finite & (p_soma >= p_lo) & (p_soma <= p_hi))[0]
elif show_mode == "Soma-flagged (argmax)":
    candidate = np.where(np.array([al == "soma" for al in arg_labels]))[0]
elif show_mode == "Artefact-flagged (argmax)":
    candidate = np.where(np.array([al == "artefact" for al in arg_labels]))[0]
elif show_mode == "Dend-flagged (argmax)":
    candidate = np.where(np.array([al == "dend" for al in arg_labels]))[0]
else:  # "All"
    candidate = np.arange(n_rois_ses, dtype=np.int64)

if hide_curated:
    candidate = np.array([i for i in candidate if int(i) not in existing], dtype=np.int64)

# Sort ambiguous mode by distance from 0.5 — most uncertain first.
if show_mode == "Ambiguous (p_soma in range)" and len(candidate) > 0:
    order = np.argsort(np.abs(p_soma[candidate] - 0.5))
    candidate = candidate[order]

n_candidate = int(len(candidate))


# ── Progress bar ──────────────────────────────────────────────────────────

# Total count of labels in the CSV across all sessions.
all_labels = load_latest_labels(CURATION_CSV)
n_all_labels = int(len(all_labels))
n_session_labels = int(len(existing))

m1, m2, m3 = st.columns(3)
m1.metric("Candidates in this session", n_candidate)
m2.metric("Labels saved (this session)", n_session_labels)
m3.metric("Labels saved (all sessions)", n_all_labels)

if n_candidate == 0:
    st.success(
        "No ROIs match the current filter. Either the ambiguous band is empty "
        "or all candidates have been curated."
    )
    st.stop()


@st.fragment
def _roi_review() -> None:
    """Render the ROI review (selector, image, trace, label) as a
    fragment so Prev/Next navigation reruns only this section and does
    not scroll the page back to the top."""
    # ── ROI selector (Prev / Next buttons + ← / → keyboard shortcuts) ─────────

    if "rc_pos" not in st.session_state:
        st.session_state.rc_pos = 0
    # Keep the position in range if the candidate list shrank (e.g. filter change).
    st.session_state.rc_pos = int(min(max(st.session_state.rc_pos, 0), n_candidate - 1))

    nav_prev, nav_num, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        st.write("")
        if st.button("◀ Prev", key="rc_prev", use_container_width=True):
            st.session_state.rc_pos = max(0, st.session_state.rc_pos - 1)
    with nav_next:
        st.write("")
        if st.button("Next ▶", key="rc_next", use_container_width=True):
            st.session_state.rc_pos = min(n_candidate - 1, st.session_state.rc_pos + 1)
    with nav_num:
        # The widget writes to st.session_state.rc_pos via its key; we read that
        # (not the return value) so the Prev/Next buttons stay authoritative.
        st.number_input(
            f"ROI to review (0–{n_candidate - 1}, {n_candidate} candidates)",
            min_value=0,
            max_value=n_candidate - 1,
            step=1,
            key="rc_pos",
            help="Use ← / → arrow keys or the Prev/Next buttons to step through candidates.",
        )
    roi_idx = int(candidate[int(st.session_state.rc_pos)])

    # Bind ← / → to the Prev/Next buttons. Handled in the CAPTURE phase with
    # stopPropagation so other widgets (e.g. the vertical filter-mode radio,
    # which also cycles on ← / →) never receive the key. ↑ / ↓ are left alone,
    # so radios can still be navigated vertically, and real text-entry fields
    # keep the arrows for cursor movement.
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const handler = function(e) {
          if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
          const el = e.target || {};
          const tag = (el.tagName || '').toLowerCase();
          const type = (el.type || '').toLowerCase();
          const textInput = tag === 'textarea' || el.isContentEditable ||
            (tag === 'input' &&
             ['text', 'search', 'email', 'url', 'tel', 'password', ''].includes(type));
          if (textInput) return;
          const want = e.key === 'ArrowRight' ? 'next' : 'prev';
          const btn = Array.from(doc.querySelectorAll('button'))
            .find(b => (b.textContent || '').toLowerCase().includes(want));
          if (btn) { e.preventDefault(); e.stopPropagation(); btn.click(); }
        };
        // Re-register each render: a fragment rerun tears down this iframe and
        // the browser drops the listener it created, so remove the stale one
        // and add a fresh handler bound to the current (live) iframe.
        if (doc._rcKeyNavFn) doc.removeEventListener('keydown', doc._rcKeyNavFn, true);
        doc._rcKeyNavFn = handler;
        doc.addEventListener('keydown', handler, true);
        </script>
        """,
        height=0,
    )

    # ── ROI summary metrics ───────────────────────────────────────────────────

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROI", roi_idx)
    c2.metric("p_soma", f"{p_soma[roi_idx]:.3f}" if np.isfinite(p_soma[roi_idx]) else "—")
    c3.metric("p_dend", f"{p_dend[roi_idx]:.3f}" if np.isfinite(p_dend[roi_idx]) else "—")
    c4.metric("p_artefact", f"{p_art[roi_idx]:.3f}" if np.isfinite(p_art[roi_idx]) else "—")
    c5.metric("Argmax", arg_labels[roi_idx])

    # QC sub-metrics — re-use the per-ROI QC arrays the ROI viewer already uses.
    qc_keys = ("snr_event", "decay_tau_s", "fneu_dff_corr", "bleach_slope", "active_fraction")
    qc_vals = {
        k: roi_qc_ses.get(k, np.full(n_rois_ses, np.nan, dtype=np.float32)) for k in qc_keys
    }
    qc_cols = st.columns(len(qc_keys))
    for col, key in zip(qc_cols, qc_keys, strict=True):
        val = float(qc_vals[key][roi_idx]) if roi_idx < len(qc_vals[key]) else np.nan
        col.metric(key, f"{val:.3f}" if np.isfinite(val) else "—")

    # ── Spatial view ──────────────────────────────────────────────────────────

    spatial = load_suite2p_spatial_one(ses["exp_id"]) or {}
    mean_img = spatial.get("mean_img")
    max_img = spatial.get("max_img")
    shape_features = spatial.get("shape_features", [])
    # Background image selector (mean vs max projection) and ROI-overlay toggle.
    # Controls live in the page body (never the sidebar) per project rules.
    bg_options: list[str] = []
    if max_img is not None:
        bg_options.append("Max projection")
    if mean_img is not None:
        bg_options.append("Mean image")

    ctrl_bg, ctrl_roi, ctrl_contrast = st.columns([2, 1, 1])
    with ctrl_bg:
        bg_choice = st.radio(
            "Background image",
            options=bg_options or ["(none)"],
            horizontal=True,
            key="rc_bg_choice",
            disabled=not bg_options,
        )
    with ctrl_roi:
        show_roi = st.toggle("Show ROI overlay", value=True, key="rc_show_roi")
    with ctrl_contrast:
        auto_contrast = st.toggle(
            "Auto-contrast",
            value=True,
            key="rc_auto_contrast",
            help=(
                "Base the display range on the 1st–99.5th intensity percentile "
                "instead of the raw min/max, so faint somata and dendrites are "
                "visible. Applies to both the mean and max images."
            ),
        )

    sl_contrast, sl_bright = st.columns(2)
    with sl_contrast:
        contrast = st.slider(
            "Contrast",
            min_value=0.5,
            max_value=6.0,
            value=1.0,
            step=0.1,
            key="rc_contrast",
            help="Higher narrows the display range around its midpoint (more contrast).",
        )
    with sl_bright:
        brightness = st.slider(
            "Brightness",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="rc_brightness",
            help="Shifts the display range: positive brightens, negative darkens.",
        )

    # Radio options only include images that exist, so the selected one is present.
    bg_img = mean_img if bg_choice == "Mean image" else max_img

    # Image (left) and the "Label this ROI" widget (right) sit side by side.
    col_img, col_label = st.columns([3, 1], gap="large")

    if bg_img is not None:
        # Pick a base display range (robust percentiles or raw min/max), then apply
        # the contrast (narrow around the midpoint) and brightness (shift) sliders.
        # Only the display colour scale changes here, never the underlying data.
        zmin = zmax = None
        finite_vals = np.asarray(bg_img)[np.isfinite(bg_img)]
        if finite_vals.size:
            if auto_contrast:
                lo, hi = (float(v) for v in np.percentile(finite_vals, [1.0, 99.5]))
            else:
                lo, hi = float(finite_vals.min()), float(finite_vals.max())
            if hi > lo:
                mid = (lo + hi) / 2.0
                span = hi - lo
                half = (span / 2.0) / max(contrast, 1e-6)
                shift = brightness * span
                zmin = mid - half - shift
                zmax = mid + half - shift
        fig = go.Figure(
            data=go.Heatmap(z=bg_img, colorscale="gray", showscale=False, zmin=zmin, zmax=zmax)
        )
        img_h, img_w = bg_img.shape[:2]
        if show_roi and roi_idx < len(shape_features) and shape_features[roi_idx] is not None:
            sf = shape_features[roi_idx]
            ypix = np.asarray(sf.get("ypix", []), dtype=int)
            xpix = np.asarray(sf.get("xpix", []), dtype=int)
            if xpix.size > 0:
                # Shade the ROI footprint as a translucent region (NaN elsewhere
                # renders transparent), coloured by the ROI's label — the
                # curated label if one exists, else the classifier's argmax.
                roi_label = existing.get(roi_idx)
                if roi_label not in LABEL_COLORS:
                    roi_label = arg_labels[roi_idx]
                mask_color = LABEL_COLORS.get(roi_label, "rgba(255,215,0,0.45)")
                overlay = np.full((img_h, img_w), np.nan)
                inb = (ypix >= 0) & (ypix < img_h) & (xpix >= 0) & (xpix < img_w)
                overlay[ypix[inb], xpix[inb]] = 1.0
                fig.add_trace(
                    go.Heatmap(
                        z=overlay,
                        showscale=False,
                        colorscale=[[0, mask_color], [1, mask_color]],
                        hoverinfo="skip",
                        name=f"ROI {roi_idx} ({roi_label})",
                    )
                )
        fig.update_layout(
            height=700,
            title=f"ROI {roi_idx} — {bg_choice}",
            xaxis=dict(range=[0, img_w], constrain="domain"),
            yaxis=dict(range=[img_h, 0], scaleanchor="x", constrain="domain"),
            margin=dict(t=30, b=5, l=5, r=5),
            showlegend=False,
        )
        col_img.plotly_chart(fig, use_container_width=True, key="rc_img")
        col_img.markdown(
            "<small>Mask colour = label: "
            "<span style='color:#00E5FF'>■ soma</span> &nbsp; "
            "<span style='color:#FF40FF'>■ dendrite</span> &nbsp; "
            "<span style='color:#FF8C00'>■ artefact</span> "
            "(curated label if set, else classifier argmax)</small>",
            unsafe_allow_html=True,
        )
    else:
        col_img.info("No spatial data for this session.")

    # dF/F trace (matches roi_viewer plotting convention), full width below image.
    dff = ses["dff"][roi_idx]
    frame_times = ses.get("frame_times")
    t = (frame_times - frame_times[0]) if frame_times is not None else np.arange(len(dff)) / 9.6
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=dff,
            mode="lines",
            line=dict(color="royalblue", width=1),
            name="dF/F₀",
        )
    )
    event_masks = ses.get("event_masks")
    if event_masks is not None and roi_idx < event_masks.shape[0]:
        events = event_masks[roi_idx].astype(bool)
        if events.any():
            # Draw events as segments over the trace: overlay the dF/F only during
            # event frames (NaN elsewhere) with connectgaps off, so each
            # contiguous event renders as a red segment rather than isolated dots.
            y_event = np.where(events, dff, np.nan)
            fig.add_trace(
                go.Scattergl(
                    x=t,
                    y=y_event,
                    mode="lines",
                    line=dict(color="red", width=2.5),
                    connectgaps=False,
                    name="Events",
                )
            )
    # Classifier event_rate threshold: median + 2·MAD (robust noise) of this
    # trace — the level above which the classifier counts threshold crossings.
    _med = float(np.median(dff))
    _thr = _med + 2.0 * 1.4826 * float(np.median(np.abs(dff - _med)))
    fig.add_hline(
        y=_thr,
        line=dict(color="green", width=1, dash="dot"),
        annotation_text="event_rate threshold (median + 2·MAD)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="green"),
    )
    fig.update_layout(
        height=280,
        margin=dict(t=20, b=35, l=55, r=15),
        xaxis_title="Time (s)",
        yaxis_title="dF/F₀",
        showlegend=True,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True, key="rc_trace")
    st.caption(
        "Events (red segments) are calcium transients detected with the "
        "Voigts & Harnett (2020) method: a per-cell noise model is estimated from "
        "the dF/F distribution (percentile-based Gaussian), and runs of frames whose "
        "signal exceeds a CDF probability threshold under that model are flagged as "
        "events. Voigts & Harnett 2020, *Neuron* 105(2):237–245, "
        "doi:10.1016/j.neuron.2019.10.016. The green dotted line is the "
        "classifier's `event_rate` threshold (median + 2·MAD) — a separate, "
        "simpler crossing count, not the V&H detection."
    )

    # ── Label widget ──────────────────────────────────────────────────────────

    # Default selection: existing curation if any, else model argmax.
    existing_label = existing.get(roi_idx)
    if existing_label in CLASS_NAMES:
        default_choice = existing_label
    else:
        default_choice = arg_labels[roi_idx] if arg_labels[roi_idx] in CLASS_NAMES else "soma"

    choices = list(CLASS_NAMES) + ["skip"]
    choice_idx = choices.index(default_choice) if default_choice in choices else 0

    # "Label this ROI" renders in the right-hand column, next to the image.
    col_label.markdown("### Label this ROI")
    chosen = col_label.radio(
        "Decision",
        choices,
        index=choice_idx,
        key=f"rc_choice_{ses['exp_id']}_{roi_idx}",
        help=(
            "Default is the model argmax (or your previous label if you've "
            "already reviewed this ROI). Click Save to commit; choose 'skip' "
            "to leave the current label unchanged."
        ),
    )

    if existing_label:
        # Look up the latest curator/timestamp for this ROI from the full table.
        sub_rows = all_labels[
            (all_labels["session_id"] == ses["exp_id"]) & (all_labels["roi_index"] == roi_idx)
        ]
        prev_curator = sub_rows.iloc[0]["curator"] if len(sub_rows) else "unknown"
        col_label.caption(
            f"Previously labelled as **{existing_label}** by {prev_curator}. "
            "Saving a new label appends a new row; the most recent timestamp wins on read."
        )

    save_clicked = col_label.button(
        "Save label",
        type="primary",
        disabled=(chosen == "skip"),
        use_container_width=True,
    )

    # Session-level action, full width below the image/trace.
    apply_clicked = st.button(
        "Apply curation to ca.h5 (this session)",
        help=(
            "Downloads ca.h5 from S3, writes roi_qc/curated_label using the "
            "latest labels in metadata/roi_curation.csv, and offers a download. "
            "Does NOT push back to S3 — that is a separate, deliberate step."
        ),
        use_container_width=True,
    )

    if save_clicked:
        try:
            append_curation_row(
                CURATION_CSV,
                session_id=ses["exp_id"],
                roi_index=roi_idx,
                label=chosen,
                curator=curator.strip(),
            )
            st.success(f"Saved: {ses['exp_id']} ROI {roi_idx} → {chosen}.")
            # st.rerun re-reads the CSV (it's loaded fresh each render — no
            # @st.cache_data on load_latest_labels) so the metric updates.
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save label: {exc}")

    if apply_clicked:
        raw = _download_session_ca_h5(ses["sub"], ses["ses"])
        if raw is None:
            st.error("Could not download ca.h5 from S3 — apply skipped.")
        else:
            try:
                with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = Path(tmp.name)
                n_written = apply_curation_to_ca_h5(CURATION_CSV, ses["exp_id"], tmp_path)
                with open(tmp_path, "rb") as fh:
                    payload = fh.read()
                st.success(
                    f"Applied {n_written} curated labels to a local copy of ca.h5. "
                    "Download below; uploading to S3 is a separate step."
                )
                st.download_button(
                    "Download curated ca.h5",
                    data=payload,
                    file_name=f"{ses['sub']}_{ses['ses']}_ca_curated.h5",
                    mime="application/x-hdf5",
                    key="rc_download",
                )
            except Exception as exc:
                st.error(f"Failed to apply curation: {exc}")


_roi_review()


# ── Curation log preview ──────────────────────────────────────────────────

with st.expander("Curation log (this session)", expanded=False):
    if n_session_labels == 0:
        st.caption("No labels saved for this session yet.")
    else:
        sub = all_labels[all_labels["session_id"] == ses["exp_id"]]
        st.dataframe(
            sub[["roi_index", "label", "curator", "timestamp"]]
            .sort_values("roi_index")
            .reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

with st.expander("All sessions — label counts", expanded=False):
    if n_all_labels == 0:
        st.caption("No labels in `metadata/roi_curation.csv` yet.")
    else:
        counts = (
            all_labels.groupby(["session_id", "label"]).size().unstack(fill_value=0).reset_index()
        )
        st.dataframe(counts, use_container_width=True, hide_index=True)


# ── Methods & References ──────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
**Soma classifier framework** (`hm2p.extraction.soma_classifier`): produces
calibrated per-ROI probabilities `p_soma`, `p_dend`, `p_artefact` for each
Suite2p ROI. The current default is a provisional rule-based scorer whose
argmax matches the legacy shape-only thresholds. Once enough manual labels
have accumulated in `metadata/roi_curation.csv`, a logistic-regression
classifier can be trained via `scripts/train_soma_classifier.py` and dropped
in at `sourcedata/trackers/suite2p/soma_classifier.pkl`. See
`docs/soma-classifier.md` for the full workflow.

**Curation policy:**

* Append-only writes — re-labelling a ROI never deletes the old row; the
  most recent timestamp wins on read.
* `roi_qc/curated_label` in `ca.h5` is preferred over the model argmax by
  the runtime resolver `hm2p.extraction.curation.effective_roi_label`.
* Apply-to-ca.h5 produces a local file; pushing back to S3 is a separate,
  deliberate operation.

**References:**

- Pachitariu et al. 2017. *Suite2p: beyond 10,000 neurons with standard
  two-photon microscopy.* bioRxiv. doi:10.1101/061507.
  https://github.com/MouseLand/suite2p
- Pedregosa et al. 2011. *Scikit-learn: Machine Learning in Python.*
  *Journal of Machine Learning Research* 12:2825–2830.
  https://scikit-learn.org
- Voigts & Harnett 2020. *Somatic and dendritic encoding of spatial variables
  in retrosplenial cortex differs during 2D navigation.* *Neuron*
  105(2):237–245. doi:10.1016/j.neuron.2019.10.016
""")
