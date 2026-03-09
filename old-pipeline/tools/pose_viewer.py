"""
Interactive DLC Pose Estimation Viewer
=======================================
Loads experiments from metadata/experiments.csv, auto-discovers video and DLC
tracking files, then opens an interactive viewer with:
  - Video playback and frame scrubbing
  - Body part overlays (coloured dots + red rings on errors)
  - Automatic error detection (three signals: likelihood, diverge, distance)
  - Diagnostic plots: ear-to-ear distance and ear velocity over time
  - Error timeline showing where errors cluster
  - Jump-to-error navigation

CLI usage:
    python tools/pose_viewer.py                        # primary, non-excluded exps only
    python tools/pose_viewer.py --all                  # all experiments
    python tools/pose_viewer.py --thresh-likelihood 0.5 --outlier-percentile 99.9

Notebook usage:
    %matplotlib widget   # or %matplotlib tk
    from tools.pose_viewer import launch_viewer
    launch_viewer()
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import imageio

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paths.config import M2PConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_PARTS = ["ear-left", "ear-right", "back-upper", "back-middle", "back-tail"]

BODY_PART_COLORS = {
    "ear-left":    "#00FF00",
    "ear-right":   "#0088FF",
    "back-upper":  "#FF8800",
    "back-middle": "#FF00FF",
    "back-tail":   "#FFFF00",
}

PLAY_SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_SPEED_IDX = 3  # 1.0x

# Half-window of frames shown in the diagnostic plots around the current frame
DIAG_WINDOW = 600

# Minimum gap (frames) between error clusters for Prev/Next Err navigation
CLUSTER_GAP = 15


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------

def _count_ear_errors(dlc_path, thresh_likelihood=0.1, outlier_percentile=99.95):
    """
    Load a DLC h5 file and return (n_error_frames, n_frames) for the ears.
    Returns (-1, -1) if the file cannot be read.
    """
    try:
        df = pd.read_hdf(dlc_path)
        scorer = df.columns.get_level_values(0)[0]
        n_frames = len(df)
        n_bp = len(BODY_PARTS)
        xs = np.empty((n_bp, n_frames))
        ys = np.empty((n_bp, n_frames))
        liks = np.empty((n_bp, n_frames))
        for i, bp in enumerate(BODY_PARTS):
            xs[i] = df[scorer][bp]["x"].values
            ys[i] = df[scorer][bp]["y"].values
            liks[i] = df[scorer][bp]["likelihood"].values
        error_mask, *_ = detect_errors(xs, ys, liks,
                                       thresh_likelihood=thresh_likelihood,
                                       outlier_percentile=outlier_percentile)
        return int(np.any(error_mask, axis=0).sum()), n_frames
    except Exception:
        return -1, -1


def load_experiments(cfg, include_all=False,
                     thresh_likelihood=0.1, outlier_percentile=99.95):
    """
    Read experiments.csv, find those that have both a video and a DLC file,
    and pre-compute ear error counts for each.

    Returns a list of dicts with keys:
        exp_id, video_path, dlc_path, primary_exp, exclude, n_errors, n_frames
    """
    df = pd.read_csv(cfg.meta_exps_file)

    if not include_all:
        df = df[(df["primary_exp"] == 1) & (df["exclude"] == 0)]

    results = []
    for _, row in df.iterrows():
        exp_id = str(row["exp_id"])
        video_dir = cfg.video_path / exp_id

        if not video_dir.exists():
            continue

        cropped_videos = list(video_dir.glob("*-cropped.mp4"))
        if not cropped_videos:
            continue

        overhead = [v for v in cropped_videos if "overhead" in v.name.lower()]
        video_path = overhead[0] if overhead else cropped_videos[0]

        video_stem = video_path.stem
        dlc_filename = video_stem + cfg.dlc_iter_name + ".h5"
        dlc_path = cfg.dlc_tracked_path / dlc_filename

        if not dlc_path.exists():
            continue

        results.append({
            "exp_id": exp_id,
            "video_path": video_path,
            "dlc_path": dlc_path,
            "primary_exp": row.get("primary_exp", 0),
            "exclude": row.get("exclude", 0),
            "n_errors": None,
            "n_frames": None,
        })

    print(f"Scanning {len(results)} experiment(s) for ear errors...")
    for exp in results:
        n_err, n_fr = _count_ear_errors(
            exp["dlc_path"],
            thresh_likelihood=thresh_likelihood,
            outlier_percentile=outlier_percentile,
        )
        exp["n_errors"] = n_err
        exp["n_frames"] = n_fr

    return results


def select_experiment(experiments, include_all=False):
    """Print experiment list with error counts and prompt user for selection."""
    label = "all" if include_all else "primary, non-excluded"
    print(f"\nAvailable experiments ({label}, with video + DLC files found):\n")
    print(f"  {'#':>3}  {'Experiment ID':<30}  {'Errors':>7}  {'Rate':>6}")
    print(f"  {'─'*3}  {'─'*30}  {'─'*7}  {'─'*6}")

    for i, exp in enumerate(experiments, 1):
        n_err = exp.get("n_errors", None)
        n_fr  = exp.get("n_frames", None)
        if n_err is None or n_err < 0:
            err_str  = "     ?"
            rate_str = "     ?"
        else:
            rate = 100.0 * n_err / n_fr if n_fr > 0 else 0.0
            err_str  = f"{n_err:>7d}"
            rate_str = f"{rate:>5.1f}%"
        flags = []
        if exp.get("exclude"):
            flags.append("EXCL")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        print(f"  {i:>3d}. {exp['exp_id']:<30}  {err_str}  {rate_str}{flag_str}")

    if not experiments:
        print("  (no experiments found)")
        return None

    print()
    while True:
        try:
            raw = input(f"Select experiment [1-{len(experiments)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(experiments):
                return experiments[idx]
            print(f"  Please enter a number between 1 and {len(experiments)}.")
        except (ValueError, EOFError):
            print("  Invalid input.")
        except KeyboardInterrupt:
            print()
            return None


# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------

def detect_errors(xs, ys, likelihoods,
                  thresh_likelihood=0.1,
                  outlier_percentile=99.95):
    """
    Ear-focused pose estimation error detector.

    Only the ear body parts are checked. Three signals:

    Signal 1 — Low likelihood  : DLC confidence < thresh_likelihood (default 0.1).
    Signal 2 — Ear distance    : inter-ear distance above outlier_percentile.
    Signal 3 — Ear divergence  : one ear moves ≥30× faster than the other AND
                                 the faster ear exceeded 8 px/frame. This avoids
                                 false positives when both ears are nearly still.

    Velocity is computed but NOT used as a standalone error signal — fast
    simultaneous movement of both ears is legitimate (mouse running).

    Parameters
    ----------
    xs, ys            : (n_bodyparts, n_frames) pixel arrays
    likelihoods       : (n_bodyparts, n_frames) DLC confidence, 0–1
    thresh_likelihood : flag when ear likelihood < this value (default 0.1)
    outlier_percentile: percentile cutoff for distance signal (default 99.95)

    Returns
    -------
    error_mask   : bool (n_bodyparts, n_frames)
    lik_mask     : bool (n_bodyparts, n_frames)  — signal 1
    vel_mask     : bool (n_bodyparts, n_frames)  — all-False (kept for compat)
    geo_mask     : bool (n_bodyparts, n_frames)  — signal 2
    div_mask     : bool (n_bodyparts, n_frames)  — signal 3
    ear_dist     : float (n_frames,)             — inter-ear distance
    vel_L_full   : float (n_frames,)             — ear-left velocity (px/frame)
    vel_R_full   : float (n_frames,)             — ear-right velocity (px/frame)
    dist_thresh  : float                         — threshold used for ear_dist
    vel_L_thresh : float                         — 99th-pct velocity (display only)
    vel_R_thresh : float                         — 99th-pct velocity (display only)
    """
    n_bp, n_frames = xs.shape
    lik_mask = np.zeros((n_bp, n_frames), dtype=bool)
    vel_mask = np.zeros((n_bp, n_frames), dtype=bool)   # not used as error signal
    geo_mask = np.zeros((n_bp, n_frames), dtype=bool)
    div_mask = np.zeros((n_bp, n_frames), dtype=bool)

    EAR_L = BODY_PARTS.index("ear-left")
    EAR_R = BODY_PARTS.index("ear-right")

    # -- Signal 1: low DLC likelihood --
    lik_mask[EAR_L] = likelihoods[EAR_L] < thresh_likelihood
    lik_mask[EAR_R] = likelihoods[EAR_R] < thresh_likelihood

    # -- Velocities (used for diverge + diagnostic plot, not flagged directly) --
    vel_L = np.sqrt(np.diff(xs[EAR_L]) ** 2 + np.diff(ys[EAR_L]) ** 2)
    vel_R = np.sqrt(np.diff(xs[EAR_R]) ** 2 + np.diff(ys[EAR_R]) ** 2)
    # Reference thresholds for the diagnostic plot (not used for flagging)
    vel_L_thresh = np.percentile(vel_L, 99.0)
    vel_R_thresh = np.percentile(vel_R, 99.0)

    # -- Signal 2: ear-to-ear distance too large --
    ear_dist = np.sqrt(
        (xs[EAR_L] - xs[EAR_R]) ** 2 + (ys[EAR_L] - ys[EAR_R]) ** 2
    )
    dist_thresh = np.percentile(ear_dist, outlier_percentile)
    dist_flag = ear_dist > dist_thresh
    geo_mask[EAR_L] |= dist_flag
    geo_mask[EAR_R] |= dist_flag

    # -- Signal 3: ear divergence (one ear jumps, the other stays put) --
    # Only flag when the faster ear moved at least 8 px (noise filter).
    diverge_thresh = 30.0
    min_vel_px = 8.0
    ratio_L = vel_L / np.maximum(vel_R, 0.5)
    ratio_R = vel_R / np.maximum(vel_L, 0.5)
    div_mask[EAR_L] |= np.concatenate(
        [[False], (vel_L > min_vel_px) & (ratio_L > diverge_thresh)]
    )
    div_mask[EAR_R] |= np.concatenate(
        [[False], (vel_R > min_vel_px) & (ratio_R > diverge_thresh)]
    )

    error_mask = lik_mask | geo_mask | div_mask

    # Pad velocity arrays to length n_frames (frame 0 has no prior frame)
    vel_L_full = np.concatenate([[0.0], vel_L])
    vel_R_full = np.concatenate([[0.0], vel_R])

    return (error_mask, lik_mask, vel_mask, geo_mask, div_mask,
            ear_dist, vel_L_full, vel_R_full,
            dist_thresh, vel_L_thresh, vel_R_thresh)


# ---------------------------------------------------------------------------
# Viewer class
# ---------------------------------------------------------------------------

class PoseViewer:
    """
    Interactive matplotlib viewer for DLC pose estimation overlaid on video.

    Parameters
    ----------
    video_path        : str or Path
    dlc_h5_path       : str or Path
    scorer            : str, optional  — auto-detected from HDF5 if None
    thresh_likelihood : float  — DLC confidence below which a detection is
                        flagged (default 0.5)
    outlier_percentile: float  — velocity/geometry above this percentile are
                        flagged. 99.9 → top 0.1%; 99.0 → top 1% (default 99.9)
    """

    def __init__(self, video_path, dlc_h5_path, scorer=None,
                 thresh_likelihood=0.5, outlier_percentile=99.9):
        self.video_path = Path(video_path)
        self.dlc_h5_path = Path(dlc_h5_path)
        self.scorer = scorer
        self.thresh_likelihood = thresh_likelihood
        self.outlier_percentile = outlier_percentile

        self.current_frame = 0
        self._playing = False
        self._speed_idx = DEFAULT_SPEED_IDX
        self._play_speed = PLAY_SPEEDS[self._speed_idx]
        self._timer = None
        self._last_drawn_frame = -1
        self._advance_in_progress = False  # re-entrant guard for timer callbacks

        print(f"Loading DLC data: {self.dlc_h5_path.name} ...")
        self._load_dlc_data()
        self._detect_errors_all()

        print(f"Opening video: {self.video_path.name} ...")
        self._open_video_reader()

        print(f"Detected {int(self.any_error.sum())} error frames "
              f"out of {self.n_frames} total "
              f"({100 * self.any_error.mean():.1f}%)")

        self._build_figure()
        self._go_to_frame(0)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_dlc_data(self):
        df = pd.read_hdf(self.dlc_h5_path)

        if self.scorer is None:
            self.scorer = df.columns.get_level_values(0)[0]

        self.n_frames = len(df)
        n_bp = len(BODY_PARTS)

        self.xs = np.empty((n_bp, self.n_frames))
        self.ys = np.empty((n_bp, self.n_frames))
        self.likelihoods = np.empty((n_bp, self.n_frames))

        for i, bp in enumerate(BODY_PARTS):
            self.xs[i] = df[self.scorer][bp]["x"].values
            self.ys[i] = df[self.scorer][bp]["y"].values
            self.likelihoods[i] = df[self.scorer][bp]["likelihood"].values

    def _detect_errors_all(self):
        (self.error_mask, self.lik_mask, self.vel_mask, self.geo_mask, self.div_mask,
         self.ear_dist, self.vel_L_full, self.vel_R_full,
         self.dist_thresh, self.vel_L_thresh, self.vel_R_thresh) = detect_errors(
            self.xs, self.ys, self.likelihoods,
            thresh_likelihood=self.thresh_likelihood,
            outlier_percentile=self.outlier_percentile,
        )
        self.any_error = np.any(self.error_mask, axis=0)
        self.error_frame_indices = np.where(self.any_error)[0]

        # Precompute cluster start frames: first frame of each contiguous run
        # (runs separated by more than CLUSTER_GAP frames are distinct clusters)
        if len(self.error_frame_indices) > 0:
            diffs = np.diff(self.error_frame_indices)
            new_cluster = np.concatenate([[True], diffs > CLUSTER_GAP])
            self.cluster_starts = self.error_frame_indices[new_cluster]
        else:
            self.cluster_starts = np.array([], dtype=int)

        EAR_L = BODY_PARTS.index("ear-left")
        EAR_R = BODY_PARTS.index("ear-right")
        self._EAR_L = EAR_L
        self._EAR_R = EAR_R

        n_lik = int(np.any(self.lik_mask, axis=0).sum())
        n_geo = int(np.any(self.geo_mask, axis=0).sum())
        n_div = int(np.any(self.div_mask, axis=0).sum())
        print(f"  Likelihood: {n_lik}  |  Dist: {n_geo}  |  Diverge: {n_div}  frames")

    def _open_video_reader(self):
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video file not found (may not be synced from Dropbox):\n"
                f"  {self.video_path}"
            )
        try:
            self.video_reader = imageio.get_reader(str(self.video_path))
        except OSError as e:
            raise OSError(
                f"Could not open video — ffmpeg failed to read the file.\n"
                f"  {self.video_path}\n"
                f"Try making sure the file is fully downloaded (not Dropbox online-only).\n"
                f"Original error: {e}"
            ) from e
        meta = self.video_reader.get_meta_data()
        self.fps = meta["fps"]
        video_n_frames = self.video_reader.count_frames()
        self.n_frames = min(self.n_frames, video_n_frames)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor("#1a1a1a")

        # 6-row × 7-col layout:
        #   Row 0 (video):        ax_video (cols 0-5) | ax_info (col 6)
        #   Row 1 (diag dist):    ax_diag_dist (all 7 cols)
        #   Row 2 (diag vel):     ax_diag_vel  (all 7 cols)
        #   Row 3 (timeline):     ax_timeline  (all 7 cols)
        #   Row 4 (slider):       ax_slider    (all 7 cols)
        #   Row 5 (buttons):      7 button axes
        #     PrevErr | PrevFr | Play | NextFr | NextErr | Speed- | Speed+
        gs = gridspec.GridSpec(
            6, 7,
            figure=self.fig,
            height_ratios=[4.5, 1.4, 1.4, 0.8, 0.65, 0.65],
            hspace=0.18,
            wspace=0.25,
            left=0.05, right=0.98,
            top=0.97, bottom=0.03,
        )

        self.ax_video      = self.fig.add_subplot(gs[0, :6])
        self.ax_info       = self.fig.add_subplot(gs[0, 6])
        self.ax_diag_dist  = self.fig.add_subplot(gs[1, :])
        self.ax_diag_vel   = self.fig.add_subplot(gs[2, :])
        self.ax_timeline   = self.fig.add_subplot(gs[3, :])
        self.ax_slider     = self.fig.add_subplot(gs[4, :])
        self.ax_btn_prev   = self.fig.add_subplot(gs[5, 0])
        self.ax_btn_frprev = self.fig.add_subplot(gs[5, 1])
        self.ax_btn_play   = self.fig.add_subplot(gs[5, 2])
        self.ax_btn_frnext = self.fig.add_subplot(gs[5, 3])
        self.ax_btn_next   = self.fig.add_subplot(gs[5, 4])
        self.ax_btn_spdn   = self.fig.add_subplot(gs[5, 5])
        self.ax_btn_spup   = self.fig.add_subplot(gs[5, 6])

        self.ax_video.set_facecolor("#111111")
        self.ax_info.set_facecolor("#111111")
        self.ax_video.axis("off")
        self.ax_info.axis("off")

        self._setup_video_ax()
        self._setup_diag_axes()
        self._setup_timeline_ax()
        self._setup_widgets()

    def _setup_video_ax(self):
        placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        self.im = self.ax_video.imshow(
            placeholder, aspect="equal", interpolation="nearest"
        )
        self._video_extent_set = False

        # Only show ear body parts in the video overlay
        EAR_PARTS = ["ear-left", "ear-right"]
        self.dot_artists = {}
        self.ring_artists = {}
        for bp in EAR_PARTS:
            color = BODY_PART_COLORS[bp]
            dot, = self.ax_video.plot(
                [], [], "o",
                color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.6,
                label=bp, zorder=10,
            )
            ring, = self.ax_video.plot(
                [], [], "o",
                color="red", markersize=15,
                markerfacecolor="none",
                markeredgecolor="red", markeredgewidth=1.5,
                zorder=9,
            )
            self.dot_artists[bp] = dot
            self.ring_artists[bp] = ring

        self.ax_video.legend(
            loc="upper right", fontsize=7,
            facecolor="#2a2a2a", labelcolor="white",
            framealpha=0.8,
        )
        self.title_text = self.ax_video.set_title(
            "Frame 0 / 0", color="white", fontsize=10, pad=4,
        )

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    def _shade_errors(self, ax, error_flags):
        """Shade error frames with a semi-transparent red background band."""
        if not np.any(error_flags):
            return
        frames = np.arange(len(error_flags))
        ax.fill_between(
            frames, 0, 1,
            where=error_flags,
            transform=ax.get_xaxis_transform(),
            color="#ff2222", alpha=0.18, linewidth=0,
        )

    def _setup_diag_axes(self):
        """Draw static content of the diagnostic plots (full session data)."""
        frames = np.arange(self.n_frames)
        err_any = self.any_error[:self.n_frames]

        col_L = BODY_PART_COLORS["ear-left"]   # green
        col_R = BODY_PART_COLORS["ear-right"]  # blue

        def _style_diag(ax, title, ylabel):
            ax.set_facecolor("#111111")
            for spine in ax.spines.values():
                spine.set_color("#333333")
            ax.tick_params(colors="#888888", labelsize=6)
            ax.set_ylabel(ylabel, fontsize=6, color="#888888")
            ax.set_title(title, fontsize=7, color="#888888", pad=2)
            ax.set_xlim(0, self.n_frames)

        # ---- Ear-to-ear distance ----------------------------------------
        ax = self.ax_diag_dist
        _style_diag(ax, "Ear-to-ear distance", "dist (px)")

        self._shade_errors(ax, err_any)
        ax.plot(frames, self.ear_dist[:self.n_frames],
                color="#aaaaaa", lw=0.6, alpha=0.85)
        ax.axhline(self.dist_thresh, color="#ff6666", lw=1.2, ls="--",
                   label=f"thresh {self.dist_thresh:.0f}px")
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#2a2a2a", labelcolor="white", framealpha=0.7)

        # Cap y-axis just above threshold so signal is readable
        y_max = max(self.dist_thresh * 1.5,
                    np.percentile(self.ear_dist[:self.n_frames], 99.5) * 1.1)
        ax.set_ylim(0, y_max)

        self.diag_dist_cursor = ax.axvline(x=0, color="#ff4444", lw=1.5, zorder=10)
        # Highlight current frame
        self.diag_dist_pt, = ax.plot([], [], "o", color="#ff4444",
                                     markersize=5, zorder=11)

        # ---- Ear velocity -----------------------------------------------
        ax = self.ax_diag_vel
        _style_diag(ax, "Ear velocity", "vel (px/fr)")

        self._shade_errors(ax, err_any)
        ax.plot(frames, self.vel_L_full[:self.n_frames],
                color=col_L, lw=0.6, alpha=0.85, label="ear-left")
        ax.plot(frames, self.vel_R_full[:self.n_frames],
                color=col_R, lw=0.6, alpha=0.85, label="ear-right")
        ax.axhline(self.vel_L_thresh, color=col_L, lw=1.2, ls="--",
                   alpha=0.75, label=f"L thr {self.vel_L_thresh:.0f}px")
        ax.axhline(self.vel_R_thresh, color=col_R, lw=1.2, ls="--",
                   alpha=0.75, label=f"R thr {self.vel_R_thresh:.0f}px")
        ax.legend(fontsize=6, loc="upper right",
                  facecolor="#2a2a2a", labelcolor="white", framealpha=0.7)

        # Cap y so threshold lines are visible
        y_max_vel = max(self.vel_L_thresh, self.vel_R_thresh) * 3.5
        ax.set_ylim(0, y_max_vel)

        self.diag_vel_cursor = ax.axvline(x=0, color="#ff4444", lw=1.5, zorder=10)
        self.diag_vel_pt_L, = ax.plot([], [], "o", color=col_L,
                                      markersize=5, zorder=11)
        self.diag_vel_pt_R, = ax.plot([], [], "o", color=col_R,
                                      markersize=5, zorder=11)

    def _update_diag_cursors(self, frame_idx):
        """Move cursor lines and highlight current-frame values in the diag plots."""
        f = frame_idx

        # Cursor lines
        self.diag_dist_cursor.set_xdata([f, f])
        self.diag_vel_cursor.set_xdata([f, f])

        # Current-frame dot markers
        self.diag_dist_pt.set_data([f], [self.ear_dist[f]])
        self.diag_vel_pt_L.set_data([f], [self.vel_L_full[f]])
        self.diag_vel_pt_R.set_data([f], [self.vel_R_full[f]])

        # Scroll window: keep current frame centred in a ±DIAG_WINDOW view
        half = min(DIAG_WINDOW, max(self.n_frames // 2, 1))
        lo = max(0, f - half)
        hi = lo + 2 * half
        if hi > self.n_frames:
            hi = self.n_frames
            lo = max(0, hi - 2 * half)
        self.ax_diag_dist.set_xlim(lo, hi)
        self.ax_diag_vel.set_xlim(lo, hi)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _setup_timeline_ax(self):
        EAR_PARTS = ["ear-left", "ear-right"]
        ax = self.ax_timeline
        ax.set_facecolor("#111111")
        ax.set_xlim(0, self.n_frames)
        ax.set_ylim(-0.6, len(EAR_PARTS) - 0.4)
        ax.set_yticks(range(len(EAR_PARTS)))
        ax.set_yticklabels(EAR_PARTS, fontsize=6, color="#cccccc")
        ax.set_xlabel("Frame", fontsize=6, color="#888888")
        ax.tick_params(axis="x", colors="#888888", labelsize=6)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.set_title("Error timeline", fontsize=7, color="#888888", pad=2)

        for row, bp in enumerate(EAR_PARTS):
            i = BODY_PARTS.index(bp)
            error_frames = np.where(self.error_mask[i])[0]
            if len(error_frames) > 0:
                ax.scatter(
                    error_frames,
                    np.full(len(error_frames), row, dtype=float),
                    s=3, c=BODY_PART_COLORS[bp],
                    marker="|", linewidths=0.5, zorder=5,
                )

        self.timeline_line = ax.axvline(x=0, color="red", lw=1.5, zorder=10)

    # ------------------------------------------------------------------
    # Widgets and info panel
    # ------------------------------------------------------------------

    def _setup_widgets(self):
        self.slider = Slider(
            ax=self.ax_slider,
            label="Frame",
            valmin=0,
            valmax=max(self.n_frames - 1, 1),
            valinit=0,
            valstep=1,
            color="#445577",
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.ax_slider.set_facecolor("#222233")
        self.slider.on_changed(self._on_slider_changed)

        def make_btn(ax, label, face="#2a2a3a", hover="#3a3a4a"):
            btn = Button(ax, label, color=face, hovercolor=hover)
            btn.label.set_color("white")
            btn.label.set_fontsize(8)
            return btn

        self.btn_prev   = make_btn(self.ax_btn_prev,   "<< Err",  "#2a2a3a")
        self.btn_frprev = make_btn(self.ax_btn_frprev, "< Frame", "#1a2a3a")
        self.btn_play   = make_btn(self.ax_btn_play,   "Play",    "#2a3a2a")
        self.btn_frnext = make_btn(self.ax_btn_frnext, "Frame >", "#1a2a3a")
        self.btn_next   = make_btn(self.ax_btn_next,   "Err >>",  "#2a2a3a")
        self.btn_spdn   = make_btn(self.ax_btn_spdn,   "Speed -", "#3a2a2a")
        self.btn_spup   = make_btn(self.ax_btn_spup,   "Speed +", "#3a2a2a")

        self.btn_prev.on_clicked(self._on_prev_error)
        self.btn_frprev.on_clicked(lambda e: self._go_to_frame(self.current_frame - 1))
        self.btn_play.on_clicked(self._on_play_pause)
        self.btn_frnext.on_clicked(lambda e: self._go_to_frame(self.current_frame + 1))
        self.btn_next.on_clicked(self._on_next_error)
        self.btn_spdn.on_clicked(self._on_speed_down)
        self.btn_spup.on_clicked(self._on_speed_up)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        # Pre-create a persistent text artist — in-place updates are fast
        self.info_text_artist = self.ax_info.text(
            0.05, 0.98, "",
            transform=self.ax_info.transAxes,
            fontsize=7, va="top", color="#cccccc",
            fontfamily="monospace",
        )

        self._update_info_panel()

    def _update_info_panel(self):
        f = self.current_frame
        t = f / self.fps if hasattr(self, "fps") else 0.0
        EL = self._EAR_L
        ER = self._EAR_R

        has_error = bool(self.any_error[f]) if f < len(self.any_error) else False

        lines = [
            f"Frame: {f}",
            f"Time:  {t:.2f}s",
            f"Speed: {self._play_speed:.2f}x",
            "",
        ]

        if has_error:
            lines.append("!! ERROR FRAME !!")
        else:
            lines.append("Frame OK")
        lines.append("")

        # Per-ear signal details for the current frame
        for i, short in ((EL, "L"), (ER, "R")):
            lik_val = self.likelihoods[i, f]
            vel_val = self.vel_L_full[f] if i == EL else self.vel_R_full[f]
            dist_val = self.ear_dist[f]

            sig_L = "L!" if self.lik_mask[i, f] else "L "
            sig_D = "D!" if self.div_mask[i, f] else "D "
            sig_G = "G!" if self.geo_mask[i, f] else "G "

            lines.append(f"Ear-{short} [{sig_L}{sig_D}{sig_G}]")
            lines.append(f" lik={lik_val:.3f}"
                         + (" <!!" if self.lik_mask[i, f] else ""))
            lines.append(f" vel={vel_val:.1f}px"
                         + (" !!div" if self.div_mask[i, f] else ""))
            lines.append(f" dist={dist_val:.1f}px"
                         + (f" thr={self.dist_thresh:.0f}" if self.geo_mask[i, f] else ""))
            lines.append("")

        # Session-wide summary
        n_err_total = len(self.error_frame_indices)
        n_L = int(np.any(self.lik_mask, axis=0).sum())
        n_D = int(np.any(self.div_mask, axis=0).sum())
        n_G = int(np.any(self.geo_mask, axis=0).sum())
        lines += [
            "── Session ──",
            f"errs: {n_err_total}/{self.n_frames}",
            f"rate: {100*self.any_error.mean():.1f}%",
            "",
            f"L:{n_L} D:{n_D} G:{n_G}",
            "",
            "── Keys ──",
            "←/→ step  ⇧±10",
            "Space play",
            "E/Q errors",
            "+/- speed",
        ]

        self.info_text_artist.set_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _go_to_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        self.current_frame = frame_idx

        # Load video frame (sequential read is faster)
        if frame_idx == self._last_drawn_frame + 1:
            img = self.video_reader.get_next_data()
        else:
            self.video_reader.set_image_index(frame_idx)
            img = self.video_reader.get_next_data()
        self._last_drawn_frame = frame_idx

        self.im.set_data(img)
        if not self._video_extent_set:
            h, w = img.shape[:2]
            self.im.set_extent([0, w, h, 0])
            self.ax_video.set_xlim(0, w)
            self.ax_video.set_ylim(h, 0)
            self._video_extent_set = True

        # Pose overlays (ears only)
        n_errors = 0
        for bp in ("ear-left", "ear-right"):
            i = BODY_PARTS.index(bp)
            x = self.xs[i, frame_idx]
            y = self.ys[i, frame_idx]
            self.dot_artists[bp].set_data([x], [y])
            if self.error_mask[i, frame_idx]:
                self.ring_artists[bp].set_data([x], [y])
                n_errors += 1
            else:
                self.ring_artists[bp].set_data([], [])

        # Timeline cursor
        self.timeline_line.set_xdata([frame_idx, frame_idx])

        # Title
        t_sec = frame_idx / self.fps
        err_tag = f"  [!{n_errors} error(s)]" if n_errors else ""
        self.title_text.set_text(
            f"Frame {frame_idx} / {self.n_frames - 1}  |  "
            f"t = {t_sec:.2f}s  |  fps={self.fps:.0f}{err_tag}"
        )
        self.title_text.set_color("#ff6666" if n_errors else "white")

        # Update slider, info panel, and diagnostic cursors (skip during playback)
        if not self._playing:
            self.slider.eventson = False
            self.slider.set_val(frame_idx)
            self.slider.eventson = True
            self._update_info_panel()
            self._update_diag_cursors(frame_idx)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _on_play_pause(self, event=None):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self._playing = True
        self.btn_play.label.set_text("Pause")
        # Hide the expensive diagnostic plots while playing; they're static anyway
        self.ax_diag_dist.set_visible(False)
        self.ax_diag_vel.set_visible(False)
        interval_ms = max(1, int(1000 / (self.fps * self._play_speed)))
        self._timer = self.fig.canvas.new_timer(interval=interval_ms)
        self._timer.add_callback(self._advance_frame)
        self._timer.start()

    def _stop_playback(self):
        self._playing = False
        self.btn_play.label.set_text("Play")
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        # Restore diagnostic plots and sync all UI elements
        self.ax_diag_dist.set_visible(True)
        self.ax_diag_vel.set_visible(True)
        self.slider.eventson = False
        self.slider.set_val(self.current_frame)
        self.slider.eventson = True
        self._update_info_panel()
        self._update_diag_cursors(self.current_frame)
        self.fig.canvas.draw_idle()

    def _advance_frame(self):
        # Guard against stacked timer callbacks when rendering falls behind
        if self._advance_in_progress:
            return
        self._advance_in_progress = True
        try:
            next_frame = self.current_frame + 1
            if next_frame >= self.n_frames:
                self._stop_playback()
                return
            self._go_to_frame(next_frame)
        finally:
            self._advance_in_progress = False

    def _restart_timer_if_playing(self):
        if self._playing:
            self._stop_playback()
            self._start_playback()

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------

    def _on_slider_changed(self, val):
        if not self._playing:
            self._go_to_frame(int(val))

    def _on_prev_error(self, event=None):
        if len(self.cluster_starts) == 0:
            return
        # Jump to the start of the cluster that begins strictly before the
        # current cluster (i.e., cluster_start < current_frame - CLUSTER_GAP)
        candidates = self.cluster_starts[
            self.cluster_starts < self.current_frame - CLUSTER_GAP
        ]
        if len(candidates):
            self._go_to_frame(candidates[-1])

    def _on_next_error(self, event=None):
        if len(self.cluster_starts) == 0:
            return
        # Jump to the start of the next cluster beyond the current position
        candidates = self.cluster_starts[
            self.cluster_starts > self.current_frame + CLUSTER_GAP
        ]
        if len(candidates):
            self._go_to_frame(candidates[0])

    def _on_speed_down(self, event=None):
        self._speed_idx = max(0, self._speed_idx - 1)
        self._play_speed = PLAY_SPEEDS[self._speed_idx]
        self._restart_timer_if_playing()
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def _on_speed_up(self, event=None):
        self._speed_idx = min(len(PLAY_SPEEDS) - 1, self._speed_idx + 1)
        self._play_speed = PLAY_SPEEDS[self._speed_idx]
        self._restart_timer_if_playing()
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def _on_key_press(self, event):
        key = event.key
        if key == "right":
            self._go_to_frame(self.current_frame + 1)
        elif key == "left":
            self._go_to_frame(self.current_frame - 1)
        elif key == "shift+right":
            self._go_to_frame(self.current_frame + 10)
        elif key == "shift+left":
            self._go_to_frame(self.current_frame - 10)
        elif key == " ":
            self._on_play_pause()
        elif key == "e":
            self._on_next_error()
        elif key == "q":
            self._on_prev_error()
        elif key in ("+", "="):
            self._on_speed_up()
        elif key == "-":
            self._on_speed_down()

    def _on_close(self, event=None):
        self._stop_playback()
        if hasattr(self, "video_reader") and self.video_reader is not None:
            self.video_reader.close()
            self.video_reader = None

    def close(self):
        self._on_close()
        plt.close(self.fig)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def launch_viewer(include_all=False, thresh_likelihood=0.1, outlier_percentile=99.95):
    """
    Convenience function for notebook use.

    Call ``%matplotlib widget`` (JupyterLab) or ``%matplotlib tk`` before this.
    Returns the PoseViewer instance (keep a reference to prevent GC).
    """
    cfg = M2PConfig()
    experiments = load_experiments(cfg, include_all=include_all,
                                   thresh_likelihood=thresh_likelihood,
                                   outlier_percentile=outlier_percentile)
    exp = select_experiment(experiments, include_all=include_all)
    if exp is None:
        return None

    viewer = PoseViewer(
        video_path=exp["video_path"],
        dlc_h5_path=exp["dlc_path"],
        thresh_likelihood=thresh_likelihood,
        outlier_percentile=outlier_percentile,
    )
    plt.show()
    return viewer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive DLC pose estimation viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all", dest="include_all", action="store_true",
        help="Include all experiments (not just primary, non-excluded)",
    )
    parser.add_argument(
        "--thresh-likelihood", type=float, default=0.1,
        help="Likelihood below which a detection is flagged (0–1)",
    )
    parser.add_argument(
        "--outlier-percentile", type=float, default=99.95,
        help="Distance percentile threshold (99.95 = top 0.05%%)",
    )
    args = parser.parse_args()

    try:
        current = matplotlib.get_backend()
        if current.lower() in ("agg", ""):
            matplotlib.use("TkAgg")
    except Exception:
        pass

    cfg = M2PConfig()
    experiments = load_experiments(cfg, include_all=args.include_all,
                                   thresh_likelihood=args.thresh_likelihood,
                                   outlier_percentile=args.outlier_percentile)
    exp = select_experiment(experiments, include_all=args.include_all)

    if exp is None:
        sys.exit(0)

    print(f"\nLoading: {exp['exp_id']}")
    print(f"  Video: {exp['video_path']}")
    print(f"  DLC:   {exp['dlc_path']}\n")

    with PoseViewer(
        video_path=exp["video_path"],
        dlc_h5_path=exp["dlc_path"],
        thresh_likelihood=args.thresh_likelihood,
        outlier_percentile=args.outlier_percentile,
    ):
        plt.show()


if __name__ == "__main__":
    main()
