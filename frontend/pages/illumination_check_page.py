"""Illumination Check — test whether the overhead camera responds to 450 nm room lights.

The Basler acA1300-200um camera is fitted with an IR-pass filter and should not
detect visible light.  The experiment alternates 1-minute lights-on / lights-off
epochs.  This page samples mean pixel intensity from the overhead video and checks
whether intensity changes correlate with the light-on/off schedule.

If a systematic on/off difference is observed it indicates the IR filter is
passing some 450 nm light, which would contaminate calcium imaging
synchronisation checks or any analysis that relies on the camera being
light-insensitive.

Statistical test: Wilcoxon signed-rank test on per-session (mean_on − mean_off)
pairs (non-parametric, as required by project policy).

    Wilcoxon F. 1945. "Individual comparisons by ranking methods."
    Biometrics Bulletin 1(6):80-83. doi:10.2307/3001968
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger("hm2p.frontend.illumination_check")

st.title("Illumination Check")
st.caption(
    "Tests whether the overhead camera image brightness changes with 450 nm room "
    "light on/off epochs.  A large systematic difference indicates visible-light "
    "leakage through the IR-pass filter."
)

# ── Imports ───────────────────────────────────────────────────────────────

try:
    from frontend.data import (
        DERIVATIVES_BUCKET,
        RAWDATA_BUCKET,
        download_s3_bytes,
        get_s3_client,
        list_s3_session_files,
        load_experiments,
        parse_session_id,
    )
except ImportError as _err:
    st.error(f"Frontend data module not available: {_err}")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────

_CAMERA_FPS = 100          # assumed; not stored in metadata
_SAMPLE_EVERY_N = 100      # sample 1 frame per second at 100 fps
_CACHE_KEY_PREFIX = "_illum_check_"

# ── Helpers ───────────────────────────────────────────────────────────────


def _cache_key(sub: str, ses: str) -> str:
    return f"{_CACHE_KEY_PREFIX}{sub}_{ses}"


def _load_timestamps(sub: str, ses: str) -> dict | None:
    """Load light on/off times from sync.h5.

    Reconstructs light_on_times and light_off_times from the boolean
    ``light_on`` array and ``frame_times`` in sync.h5.

    Returns dict with keys:
        light_on_times  : np.ndarray (seconds from session start)
        light_off_times : np.ndarray (seconds from session start)
        frame_times     : np.ndarray — imaging frame timestamps (seconds)
    Returns None if the file is unavailable.
    """
    import h5py

    key = f"sync/{sub}/{ses}/sync.h5"
    data = download_s3_bytes(DERIVATIVES_BUCKET, key)
    if data is None:
        return None
    try:
        with h5py.File(io.BytesIO(data), "r") as f:
            if "light_on" not in f or "frame_times" not in f:
                return None
            light_on = f["light_on"][()].astype(bool)
            frame_times = f["frame_times"][()].astype(float)

        # Reconstruct on/off transition times from boolean array
        transitions = np.diff(light_on.astype(np.int8))
        on_idx = np.where(transitions == 1)[0] + 1
        off_idx = np.where(transitions == -1)[0] + 1

        # Handle edge cases: if light starts on, add t=0 as first on time
        light_on_times = frame_times[on_idx] if len(on_idx) > 0 else np.array([], dtype=float)
        light_off_times = frame_times[off_idx] if len(off_idx) > 0 else np.array([], dtype=float)

        if light_on[0]:
            light_on_times = np.r_[frame_times[0], light_on_times]

        return {
            "light_on_times": light_on_times,
            "light_off_times": light_off_times,
            "frame_times": frame_times,
        }
    except Exception as exc:
        log.warning("Failed to read sync.h5 for %s/%s: %s", sub, ses, exc)
        return None


def _find_video_key(sub: str, ses: str) -> str | None:
    """Find the overhead (non-side) MP4 key for a session on S3.

    Returns the first .mp4 key that does NOT contain 'side' in the filename.
    """
    prefix = f"rawdata/{sub}/{ses}/behav/"
    files = list_s3_session_files(RAWDATA_BUCKET, prefix)
    for f in files:
        key = f["key"]
        filename = key.rsplit("/", 1)[-1].lower()
        if filename.endswith(".mp4") and "side" not in filename:
            return key
    return None


def _get_video_size(bucket: str, key: str) -> int:
    """Return size in bytes of an S3 object."""
    s3 = get_s3_client()
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        return head["ContentLength"]
    except Exception:
        return 0


def _sample_video_intensity(
    bucket: str,
    key: str,
    sample_every: int,
    progress_bar: st.delta_generator.DeltaGenerator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Download a video to a temp file and sample mean pixel intensity.

    Downloads the full video (necessary for random frame access via OpenCV).
    Returns (frame_indices, mean_intensities) or None on failure.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    key:
        S3 object key of the MP4 file.
    sample_every:
        Sample one frame every this many frames (e.g. 100 = 1 fps at 100 fps).
    progress_bar:
        Streamlit progress bar widget to update during download.
    """
    try:
        import cv2
    except ImportError:
        st.error(
            "OpenCV (cv2) is required for video sampling. "
            "Install with: uv pip install opencv-python-headless"
        )
        return None

    s3 = get_s3_client()

    # Stream download with progress tracking
    size_bytes = _get_video_size(bucket, key)
    log.info("Downloading video s3://%s/%s (%.1f MB)", bucket, key, size_bytes / 1e6)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        downloaded = 0
        with open(tmp_path, "wb") as fout:
            resp = s3.get_object(Bucket=bucket, Key=key)
            for chunk in resp["Body"].iter_chunks(chunk_size=8 * 1024 * 1024):
                fout.write(chunk)
                downloaded += len(chunk)
                if size_bytes > 0:
                    frac = min(downloaded / size_bytes, 1.0)
                    mb_done = downloaded / 1e6
                    mb_total = size_bytes / 1e6
                    progress_bar.progress(
                        frac * 0.8,
                        text=f"Downloading… {mb_done:.0f} / {mb_total:.0f} MB",
                    )

        progress_bar.progress(0.8, text="Sampling frames…")

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            log.error("OpenCV could not open %s", tmp_path)
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            log.warning("No frames reported by OpenCV for %s", key)
            # Fall back to sequential read
            total_frames = None

        frame_indices = []
        intensities = []

        if total_frames is not None:
            sample_indices = list(range(0, total_frames, sample_every))
        else:
            # Read sequentially and pick every Nth frame
            sample_indices = None

        if sample_indices is not None:
            n = len(sample_indices)
            for i, idx in enumerate(sample_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                intensities.append(float(np.mean(gray)))
                frame_indices.append(idx)
                if i % max(1, n // 20) == 0:
                    frac = 0.8 + 0.18 * (i / n)
                    progress_bar.progress(frac, text=f"Sampling frames… {i}/{n}")
        else:
            # Sequential fallback
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                    intensities.append(float(np.mean(gray)))
                    frame_indices.append(frame_idx)
                frame_idx += 1

        cap.release()
        progress_bar.progress(1.0, text="Done.")

        if not intensities:
            return None
        return np.array(frame_indices, dtype=np.int64), np.array(intensities, dtype=np.float64)

    except Exception as exc:
        log.error("Error sampling video %s/%s: %s", bucket, key, exc)
        return None
    finally:
        import contextlib
        with contextlib.suppress(Exception):
            tmp_path.unlink(missing_ok=True)


def _frame_index_to_time(
    frame_indices: np.ndarray,
    frame_times: np.ndarray | None,
    fps: float = _CAMERA_FPS,
) -> np.ndarray:
    """Convert frame indices to seconds.

    Uses frame_times array if available, otherwise divides by fps.
    """
    if frame_times is not None and len(frame_times) > 0:
        # Interpolate: some sampled indices may exceed frame_times length
        max_idx = len(frame_times) - 1
        clipped = np.clip(frame_indices, 0, max_idx)
        return frame_times[clipped]
    return frame_indices.astype(float) / fps


def _classify_frames_by_light(
    times: np.ndarray,
    light_on_times: np.ndarray,
    light_off_times: np.ndarray,
) -> np.ndarray:
    """Return boolean array: True if frame time falls within a light-on epoch.

    Each epoch is [light_on_times[i], light_off_times[i]).
    Frames before the first on-time or after the last off-time are classified
    as light-off.
    """
    n_epochs = min(len(light_on_times), len(light_off_times))
    is_on = np.zeros(len(times), dtype=bool)
    for i in range(n_epochs):
        mask = (times >= light_on_times[i]) & (times < light_off_times[i])
        is_on |= mask
    return is_on


# ── Session selector ──────────────────────────────────────────────────────

experiments = load_experiments()
if not experiments:
    st.warning("No experiments found.")
    st.stop()

col_sel, col_run = st.columns([3, 1])

with col_sel:
    exp_ids = [e["exp_id"] for e in experiments]
    selected_id = st.selectbox("Session", exp_ids, key="illum_session_select")

with col_run:
    st.write("")  # vertical spacer to align with selectbox
    run_button = st.button("Run analysis", key="illum_run_btn", type="primary")

if selected_id is None:
    st.stop()

sub, ses = parse_session_id(selected_id)

# ── Per-session analysis ──────────────────────────────────────────────────

st.header("Per-session analysis")
_sample_interval_s = _SAMPLE_EVERY_N / _CAMERA_FPS
st.caption(
    f"Session: **{selected_id}** ({sub} / {ses}).  "
    f"Video sampled every {_SAMPLE_EVERY_N} frames ({_sample_interval_s:.1f} s per sample)."
)

cache_key = _cache_key(sub, ses)

if run_button and cache_key in st.session_state:
    # Clear any previous result for this session so a fresh run is performed
    del st.session_state[cache_key]

# Check for cached result
cached = st.session_state.get(cache_key)

if cached is None and not run_button:
    st.info(
        "Select a session and press **Run analysis** to download and sample the video.  "
        "Results are cached in the browser session — pressing Run again clears and re-runs."
    )

elif cached is None and run_button:
    # --- Run the analysis ---
    ts_data = _load_timestamps(sub, ses)
    if ts_data is None:
        st.error(
            f"sync.h5 not found on S3 for {sub}/{ses}.  "
            "Run pipeline stages 0-5 first."
        )
        st.stop()

    light_on_times: np.ndarray = ts_data["light_on_times"]
    light_off_times: np.ndarray = ts_data["light_off_times"]
    frame_times: np.ndarray | None = ts_data.get("frame_times")

    if len(light_on_times) == 0 or len(light_off_times) == 0:
        st.warning(
            "No light timing data found in sync.h5 for this session.  "
            "light_on_times and/or light_off_times arrays are empty."
        )

    video_key = _find_video_key(sub, ses)
    if video_key is None:
        st.error(
            f"No overhead video found on S3 at rawdata/{sub}/{ses}/behav/*.mp4.  "
            "Check that the session has been ingested."
        )
        st.stop()

    size_mb = _get_video_size(RAWDATA_BUCKET, video_key) / 1e6
    st.write(f"Video: `{video_key.rsplit('/', 1)[-1]}` ({size_mb:.0f} MB)")
    progress = st.progress(0.0, text="Starting download…")

    result = _sample_video_intensity(RAWDATA_BUCKET, video_key, _SAMPLE_EVERY_N, progress)
    if result is None:
        st.error("Video sampling failed. Check the logs for details.")
        st.stop()

    frame_indices, intensities = result
    times = _frame_index_to_time(frame_indices, frame_times)
    is_on = _classify_frames_by_light(times, light_on_times, light_off_times)

    # Store in session_state
    st.session_state[cache_key] = {
        "frame_indices": frame_indices,
        "intensities": intensities,
        "times": times,
        "is_on": is_on,
        "light_on_times": light_on_times,
        "light_off_times": light_off_times,
        "video_key": video_key,
    }
    cached = st.session_state[cache_key]

# ── Display per-session results ───────────────────────────────────────────

if cached is not None:
    intensities: np.ndarray = cached["intensities"]
    times: np.ndarray = cached["times"]
    is_on: np.ndarray = cached["is_on"]
    light_on_times: np.ndarray = cached["light_on_times"]
    light_off_times: np.ndarray = cached["light_off_times"]

    # -- Metrics --
    on_vals = intensities[is_on]
    off_vals = intensities[~is_on]

    m_on = float(np.mean(on_vals)) if len(on_vals) > 0 else float("nan")
    m_off = float(np.mean(off_vals)) if len(off_vals) > 0 else float("nan")
    diff = m_on - m_off

    # Cohen's d analogue: difference / pooled SD (descriptive, not a test)
    sd_on = float(np.std(on_vals, ddof=1)) if len(on_vals) > 1 else float("nan")
    sd_off = float(np.std(off_vals, ddof=1)) if len(off_vals) > 1 else float("nan")
    if not (np.isnan(sd_on) or np.isnan(sd_off)) and (sd_on + sd_off) > 0:
        pooled_sd = np.sqrt((sd_on**2 + sd_off**2) / 2)
        cohens_d = diff / pooled_sd
    else:
        cohens_d = float("nan")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Mean intensity — lights on", f"{m_on:.2f}")
    mc2.metric("Mean intensity — lights off", f"{m_off:.2f}")
    mc3.metric("Difference (on − off)", f"{diff:+.2f}")
    _d_str = f"{abs(cohens_d):.3f}" if not np.isnan(cohens_d) else "n/a"
    _d_help = (
        "Standardised effect size (|difference| / pooled SD). "
        "Descriptive only — statistical test is in the cross-session section."
    )
    mc4.metric("|Cohen's d|", _d_str, help=_d_help)

    # -- Time-series plot --
    fig = go.Figure()

    # Shade light-on epochs
    n_epochs = min(len(light_on_times), len(light_off_times))
    t_max = float(times[-1]) if len(times) > 0 else 1.0
    for i in range(n_epochs):
        t0 = float(light_on_times[i])
        t1 = float(light_off_times[i])
        fig.add_vrect(
            x0=t0, x1=min(t1, t_max),
            fillcolor="rgba(255, 220, 50, 0.25)",
            layer="below",
            line_width=0,
            annotation_text="lights on" if i == 0 else "",
            annotation_position="top left",
        )

    # Mean intensity trace
    fig.add_trace(go.Scatter(
        x=times.tolist(),
        y=intensities.tolist(),
        mode="lines",
        line=dict(color="steelblue", width=1.2),
        name="Mean pixel intensity",
    ))

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Mean pixel intensity (8-bit)",
        title=f"Mean pixel intensity over time — {selected_id}",
        height=380,
        margin=dict(t=50, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Yellow shading = lights-on epochs.  "
        "If the IR filter is effective, intensity should not change with light state."
    )

    # -- Light-on vs light-off distribution --
    if len(on_vals) > 0 and len(off_vals) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=on_vals.tolist(), name="Lights on",
            marker_color="rgba(255, 180, 0, 0.6)",
            nbinsx=40,
        ))
        fig2.add_trace(go.Histogram(
            x=off_vals.tolist(), name="Lights off",
            marker_color="rgba(60, 100, 180, 0.6)",
            nbinsx=40,
        ))
        fig2.update_layout(
            barmode="overlay",
            xaxis_title="Mean pixel intensity (8-bit)",
            yaxis_title="Count",
            title="Distribution of sampled frame intensities by light state",
            height=300,
            margin=dict(t=50, b=50),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Cross-session summary ─────────────────────────────────────────────────

st.header("Cross-session summary")
st.markdown(
    "Shows mean on/off intensity and their difference for every session where "
    "results have been cached in this browser session.  "
    "Run the per-session analysis for each session to populate this table."
)

# Collect all cached results
summary_rows = []
for exp in experiments:
    eid = exp["exp_id"]
    s, ss = parse_session_id(eid)
    ck = _cache_key(s, ss)
    c = st.session_state.get(ck)
    if c is None:
        continue
    i_arr = c["intensities"]
    io_arr = c["is_on"]
    on_v = i_arr[io_arr]
    off_v = i_arr[~io_arr]
    m_on_c = float(np.mean(on_v)) if len(on_v) > 0 else float("nan")
    m_off_c = float(np.mean(off_v)) if len(off_v) > 0 else float("nan")
    diff_c = m_on_c - m_off_c
    summary_rows.append({
        "session": eid,
        "mean_on": round(m_on_c, 3),
        "mean_off": round(m_off_c, 3),
        "diff_on_minus_off": round(diff_c, 3),
        "n_samples_on": int(len(on_v)),
        "n_samples_off": int(len(off_v)),
    })

if not summary_rows:
    st.info(
        "No cached results yet.  Run the per-session analysis for one or more sessions "
        "to populate the cross-session summary."
    )
else:
    df_summary = pd.DataFrame(summary_rows)

    # -- Summary table --
    st.dataframe(
        df_summary.rename(columns={
            "session": "Session",
            "mean_on": "Mean intensity (on)",
            "mean_off": "Mean intensity (off)",
            "diff_on_minus_off": "Difference (on − off)",
            "n_samples_on": "N samples (on)",
            "n_samples_off": "N samples (off)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # -- Bar chart of differences --
    fig3 = go.Figure()
    colours = [
        "rgba(200, 50, 50, 0.7)" if d > 0 else "rgba(50, 100, 200, 0.7)"
        for d in df_summary["diff_on_minus_off"]
    ]
    fig3.add_trace(go.Bar(
        x=df_summary["session"].tolist(),
        y=df_summary["diff_on_minus_off"].tolist(),
        marker_color=colours,
        name="on − off",
    ))
    fig3.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig3.update_layout(
        xaxis_title="Session",
        yaxis_title="Mean intensity difference (on − off)",
        title="Per-session light-on minus light-off intensity difference",
        xaxis_tickangle=-45,
        height=420,
        margin=dict(t=50, b=120),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -- Wilcoxon signed-rank test --
    diffs = df_summary["diff_on_minus_off"].dropna().values

    if len(diffs) < 4:
        st.info(
            f"Only {len(diffs)} session(s) cached — need at least 4 for a meaningful "
            "Wilcoxon signed-rank test.  Run more sessions to enable the statistical test."
        )
    else:
        from scipy import stats as _scipy_stats

        # Wilcoxon signed-rank test: H0 = population median difference is zero.
        # Tests whether the on−off differences are systematically non-zero.
        # Reference: Wilcoxon F. 1945. doi:10.2307/3001968
        stat, pval = _scipy_stats.wilcoxon(diffs, alternative="two-sided")

        st.subheader("Wilcoxon signed-rank test")
        st.markdown(
            "Tests whether the per-session (lights-on minus lights-off) intensity "
            "difference has a median significantly different from zero.  "
            "H\u2080: median difference = 0.  Non-parametric; no normality assumption.\n\n"
            "> Wilcoxon F. 1945. \"Individual comparisons by ranking methods.\" "
            "*Biometrics Bulletin* 1(6):80–83. doi:10.2307/3001968"
        )

        med_diff = float(np.median(diffs))
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Wilcoxon statistic", f"{stat:.1f}")
        wc2.metric("p-value", f"{pval:.4f}")
        wc3.metric("Median difference", f"{med_diff:+.3f}")

        alpha = 0.05
        if pval < alpha:
            st.warning(
                f"p = {pval:.4f} (< {alpha}).  The median on−off intensity difference "
                f"({med_diff:+.3f} grey levels) is significantly different from zero.  "
                "This suggests the IR filter may be passing visible 450 nm light."
            )
        else:
            st.success(
                f"p = {pval:.4f} (>= {alpha}).  No significant difference between "
                "lights-on and lights-off mean pixel intensity.  "
                "The IR filter appears to be blocking visible light effectively."
            )

# ── Methods & References ──────────────────────────────────────────────────

with st.expander("Methods & References"):
    st.markdown("""
**Illumination check method**

Mean pixel intensity is sampled from the overhead video (Basler acA1300-200um,
~100 fps) every 100 frames (approximately 1 sample per second).  Light epoch
boundaries are reconstructed from the `light_on` boolean array in `sync.h5` (datasets `light_on` and
`light_off_times`, seconds from session start).  Each sampled frame is
classified as lights-on or lights-off by checking whether its timestamp falls
within a [on, off) interval.

Camera frame timestamps, when available, are taken from `frame_times_camera`
in `sync.h5`; otherwise frame index / 100 fps is used.

The statistical test is a two-sided Wilcoxon signed-rank test applied to the
per-session (mean lights-on minus mean lights-off) intensity differences.

> Wilcoxon F. 1945. "Individual comparisons by ranking methods."
> *Biometrics Bulletin* 1(6):80–83. doi:10.2307/3001968

**Equipment**

Camera: Basler acA1300-200um (1296 × 966, monochrome near-infrared).
Filter: IR-pass (passes wavelengths > ~700 nm; blocks visible including 450 nm room lights).
Room lights: ~450 nm blue LEDs, 1 min on / 1 min off alternating throughout each session.
""")
