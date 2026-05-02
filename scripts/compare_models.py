#!/usr/bin/env python3
"""Compare a baseline DLC model to a candidate via the SA fine-tune gate.

Loads two sets of per-session DLC predictions (pose .h5 files on S3) plus
the project's local labelled-data ground truth, computes per-frame
Euclidean errors per keypoint, and runs the v2 plan §4.6 paired
non-parametric promotion gate via :mod:`hm2p.pose.finetune`.

Method: Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
behavioral analysis." *Nature Communications* 15:5165.
doi:10.1038/s41467-024-48792-2.
Code: https://github.com/DeepLabCut/DeepLabCut.

Usage::

    uv run python scripts/compare_models.py \\
        --mode predict \\
        --baseline-h5-prefix s3://hm2p-derivatives/pose-archive/<id>/ \\
        --candidate-h5-prefix s3://hm2p-derivatives/pose/ \\
        --labels-dir sourcedata/trackers/dlc/hm2p-retrain-tristan-2026-03-20/labeled-data/ \\
        --baseline-id dlc-20260430-hrnetw32-snap110 \\
        --candidate-id dlc-20260501-hrnetw32-snap60 \\
        --output verdict.json \\
        [--upload-s3]

Exit codes:

- 0 — overall_pass=True (gate accepts the candidate as champion)
- 2 — overall_pass=False (one or more gate predicates failed)
- 3 — comparison could not be performed (no overlapping sessions, missing
  labels). ``verdict.json`` is still written with ``meta.error`` populated.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Allow `import hm2p.pose.finetune` when invoked directly from a clone.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hm2p.pose.finetune import (  # noqa: E402
    HM2P_BODYPARTS,
    GateConfig,
    Verdict,
    bonferroni_alpha,
    evaluate_promotion_gate,
    hd_from_ear_vector,
    per_frame_euclidean_error,
    verdict_to_json,
)

log = logging.getLogger("compare_models")

# Constants -----------------------------------------------------------------

DERIVATIVES_BUCKET = "hm2p-derivatives"
VERDICT_S3_KEY = "dlc-retrain/models/_compare_verdict.json"
RAW_FPS = 100.0  # overhead camera (matches compute_bodypart_rmse.py)
DLC_FPS = 30.0  # downsample target before DLC inference

#: Default Bonferroni-corrected α for the 8-keypoint family.
DEFAULT_ALPHA = bonferroni_alpha(0.05, len(HM2P_BODYPARTS))


# ---------------------------------------------------------------------------
# Frame index mapping
# ---------------------------------------------------------------------------


def map_raw_to_dlc_frame(raw_frame_index: int) -> int:
    """Map a raw 100-fps frame index to the corresponding DLC 30-fps index.

    The pipeline subsamples the overhead video to 30 fps before DLC
    inference (``run_dlc_retrain.py:infer``). Ground-truth labels are
    indexed against the raw video; DLC predictions against the subsampled
    one. Mirrors ``compute_bodypart_rmse.py``.
    """
    return int(round(raw_frame_index * DLC_FPS / RAW_FPS))


# ---------------------------------------------------------------------------
# GT label loading
# ---------------------------------------------------------------------------


def list_gt_session_dirs(labels_dir: Path) -> list[Path]:
    """Return labelled-data subdirectories (one per labelled session/clip)."""
    if not labels_dir.exists():
        return []
    return sorted(d for d in labels_dir.iterdir() if d.is_dir())


def load_gt_keypoints(
    h5_path: Path,
    keypoint_names: list[str],
) -> tuple[np.ndarray, list[int]] | None:
    """Load ground-truth keypoint coordinates from a CollectedData_*.h5.

    Returns ``(coords, raw_frame_indices)`` where ``coords`` has shape
    ``(n_frames, n_keypoints, 2)`` (NaN where the GT is missing) and
    ``raw_frame_indices`` is the corresponding list of integer frame
    indices extracted from the index column ``frame_<N>.png``.

    Returns ``None`` when the file is empty, unreadable, or has no
    parseable rows.
    """
    import pandas as pd

    try:
        gt = pd.read_hdf(h5_path)
    except Exception:
        log.warning("Could not load GT %s", h5_path)
        return None
    if gt is None or len(gt) == 0:
        return None
    scorer = gt.columns.get_level_values(0)[0]
    bps_available = set(gt.columns.get_level_values(1).unique().tolist())

    coords = np.full((len(gt), len(keypoint_names), 2), np.nan, dtype=np.float64)
    raw_indices: list[int] = []
    for i in range(len(gt)):
        idx = gt.index[i]
        frame_file = idx[-1] if isinstance(idx, tuple) else str(idx).split("/")[-1]
        m = re.match(r"frame_(\d+)\.png", str(frame_file))
        if not m:
            raw_indices.append(-1)
            continue
        raw_indices.append(int(m.group(1)))
        for k, bp in enumerate(keypoint_names):
            if bp not in bps_available:
                # Legacy alias: head_midpoint <-> implant_base_rear.
                if bp == "head_midpoint" and "implant_base_rear" in bps_available:
                    bp_lookup = "implant_base_rear"
                else:
                    continue
            else:
                bp_lookup = bp
            try:
                gx = float(gt.iloc[i][(scorer, bp_lookup, "x")])
                gy = float(gt.iloc[i][(scorer, bp_lookup, "y")])
            except (KeyError, ValueError):
                continue
            coords[i, k, 0] = gx
            coords[i, k, 1] = gy
    if not raw_indices:
        return None
    return coords, raw_indices


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def load_predictions_from_h5(
    h5_path: Path,
    keypoint_names: list[str],
    raw_frame_indices: list[int],
) -> np.ndarray:
    """Load DLC predictions for a list of raw frame indices.

    Parameters
    ----------
    h5_path
        Local path to the DLC prediction ``.h5``.
    keypoint_names
        Project bodypart names. ``head_midpoint`` falls back to the legacy
        ``implant_base_rear`` alias.
    raw_frame_indices
        Raw frame indices (100 fps); each is mapped through
        :func:`map_raw_to_dlc_frame` before reading.

    Returns
    -------
    np.ndarray
        Shape ``(len(raw_frame_indices), n_keypoints, 2)``, NaN where the
        prediction is missing or out-of-range.
    """
    import pandas as pd

    pred = pd.read_hdf(h5_path)
    scorer = pred.columns.get_level_values(0)[0]
    bps_available = set(pred.columns.get_level_values(1).unique().tolist())

    out = np.full((len(raw_frame_indices), len(keypoint_names), 2), np.nan)
    for i, raw_fi in enumerate(raw_frame_indices):
        if raw_fi < 0:
            continue
        dlc_fi = map_raw_to_dlc_frame(raw_fi)
        if dlc_fi >= len(pred):
            continue
        for k, bp in enumerate(keypoint_names):
            bp_lookup = bp
            if bp not in bps_available:
                if bp == "head_midpoint" and "implant_base_rear" in bps_available:
                    bp_lookup = "implant_base_rear"
                else:
                    continue
            try:
                px = float(pred.iloc[dlc_fi][(scorer, bp_lookup, "x")])
                py = float(pred.iloc[dlc_fi][(scorer, bp_lookup, "y")])
            except (KeyError, ValueError):
                continue
            out[i, k, 0] = px
            out[i, k, 1] = py
    return out


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/prefix/`` -> ``(bucket, prefix)``.

    Always returns ``prefix`` with a trailing slash if the user supplied
    one (the listing API treats it that way). Raises on non-S3 URIs.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri!r}")
    rest = uri[len("s3://") :]
    parts = rest.split("/", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def fetch_prediction_h5(
    s3_client: Any,
    bucket: str,
    prefix: str,
    sub: str,
    ses: str,
    *,
    select_fn: Any = None,
) -> Path | None:
    """Locate and download the best prediction ``.h5`` for a session.

    Returns ``None`` when no .h5 is found under ``{prefix}/{sub}/{ses}/``.
    The downloaded file lives in a fresh temp file; the caller owns
    cleanup.
    """
    if select_fn is None:
        from hm2p.pose.select import select_best_dlc_h5_s3

        select_fn = select_best_dlc_h5_s3
    full_prefix = f"{prefix.rstrip('/')}/{sub}/{ses}/"
    key = select_fn(s3_client, bucket, full_prefix)
    if key is None:
        return None
    fd = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)  # noqa: SIM115
    fd.close()
    s3_client.download_file(bucket, key, fd.name)
    return Path(fd.name)


# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------


def clip_dir_to_sub_ses(
    clip_dir_name: str,
    retrain_frames_dir: Path | None = None,
) -> tuple[str, str] | None:
    """Map a labelled-data clip directory to ``(sub, ses)``.

    Mirrors the helper in ``compute_bodypart_rmse.py`` but is folded
    in-line so this script has only one-direction dependency on shared
    modules. Returns ``None`` when the clip cannot be matched.
    """
    parts = clip_dir_name.split("_")
    if len(parts) < 5:
        return None
    date = parts[0]
    try:
        clip_time = int(parts[1] + parts[2] + parts[3])
    except ValueError:
        return None
    animal = parts[4].split("-")[0]
    if retrain_frames_dir is None:
        retrain_frames_dir = _REPO_ROOT / "metadata" / "retrain_frames"
    if not retrain_frames_dir.exists():
        return None
    candidates: list[tuple[int, str, str]] = []
    for f in retrain_frames_dir.glob("*.json"):
        fp = f.stem.split("_")
        if len(fp) < 2:
            continue
        f_animal = fp[0].replace("sub-", "")
        f_ses_full = fp[1].replace("ses-", "")
        f_date = f_ses_full[:8]
        if f_animal != animal or f_date != date:
            continue
        try:
            f_time = int(f_ses_full[9:])
        except ValueError:
            continue
        candidates.append((abs(f_time - clip_time), fp[0], fp[1]))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


# ---------------------------------------------------------------------------
# Comparison core
# ---------------------------------------------------------------------------


def collect_paired_errors(
    labels_dir: Path,
    s3_client: Any,
    baseline_bucket: str,
    baseline_prefix: str,
    candidate_bucket: str,
    candidate_prefix: str,
    *,
    keypoint_names: list[str] | None = None,
    select_fn: Any = None,
    label_filename: str = "CollectedData_tristan.h5",
    retrain_frames_dir: Path | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]
]:
    """Build paired per-frame error arrays across labelled sessions.

    Returns
    -------
    e_baseline, e_candidate
        Arrays of shape ``(n_frames_total, n_keypoints)``.
    hd_baseline_rad, hd_candidate_rad, hd_gt_rad
        Per-frame ear-vector head-direction (radians) for the matched
        frames, or ``None`` when the ear keypoints are not in
        ``keypoint_names``.
    skipped_sessions
        Short descriptors of sessions that were skipped (no GT, no
        matching prediction in either prefix, etc.).
    """
    if keypoint_names is None:
        keypoint_names = list(HM2P_BODYPARTS)
    e_b_chunks: list[np.ndarray] = []
    e_c_chunks: list[np.ndarray] = []
    hd_b_chunks: list[np.ndarray] = []
    hd_c_chunks: list[np.ndarray] = []
    hd_g_chunks: list[np.ndarray] = []
    skipped: list[str] = []

    has_ear_pair = "left_ear" in keypoint_names and "right_ear" in keypoint_names
    if has_ear_pair:
        le_idx = keypoint_names.index("left_ear")
        re_idx = keypoint_names.index("right_ear")
    else:
        le_idx = re_idx = -1

    for clip_dir in list_gt_session_dirs(labels_dir):
        gt_h5 = clip_dir / label_filename
        if not gt_h5.exists():
            continue
        loaded = load_gt_keypoints(gt_h5, keypoint_names)
        if loaded is None:
            skipped.append(f"{clip_dir.name}:no_gt")
            continue
        gt_xy, raw_idx = loaded
        sub_ses = clip_dir_to_sub_ses(clip_dir.name, retrain_frames_dir)
        if sub_ses is None:
            skipped.append(f"{clip_dir.name}:no_match")
            continue
        sub, ses = sub_ses

        b_h5 = fetch_prediction_h5(
            s3_client,
            baseline_bucket,
            baseline_prefix,
            sub,
            ses,
            select_fn=select_fn,
        )
        if b_h5 is None:
            skipped.append(f"{sub}/{ses}:no_baseline_prediction")
            continue
        c_h5 = fetch_prediction_h5(
            s3_client,
            candidate_bucket,
            candidate_prefix,
            sub,
            ses,
            select_fn=select_fn,
        )
        if c_h5 is None:
            skipped.append(f"{sub}/{ses}:no_candidate_prediction")
            b_h5.unlink(missing_ok=True)
            continue
        try:
            pred_b = load_predictions_from_h5(b_h5, keypoint_names, raw_idx)
            pred_c = load_predictions_from_h5(c_h5, keypoint_names, raw_idx)
        finally:
            b_h5.unlink(missing_ok=True)
            c_h5.unlink(missing_ok=True)

        e_b = per_frame_euclidean_error(pred_b, gt_xy)
        e_c = per_frame_euclidean_error(pred_c, gt_xy)
        e_b_chunks.append(e_b)
        e_c_chunks.append(e_c)

        if has_ear_pair:
            hd_b_chunks.append(hd_from_ear_vector(pred_b[:, le_idx, :], pred_b[:, re_idx, :]))
            hd_c_chunks.append(hd_from_ear_vector(pred_c[:, le_idx, :], pred_c[:, re_idx, :]))
            hd_g_chunks.append(hd_from_ear_vector(gt_xy[:, le_idx, :], gt_xy[:, re_idx, :]))

    if not e_b_chunks:
        empty = np.empty((0, len(keypoint_names)))
        return empty, empty, None, None, None, skipped

    e_baseline = np.concatenate(e_b_chunks, axis=0)
    e_candidate = np.concatenate(e_c_chunks, axis=0)
    if has_ear_pair and hd_b_chunks:
        hd_baseline = np.concatenate(hd_b_chunks, axis=0)
        hd_candidate = np.concatenate(hd_c_chunks, axis=0)
        hd_gt = np.concatenate(hd_g_chunks, axis=0)
    else:
        hd_baseline = hd_candidate = hd_gt = None
    return e_baseline, e_candidate, hd_baseline, hd_candidate, hd_gt, skipped


# ---------------------------------------------------------------------------
# rmse-json triage mode
# ---------------------------------------------------------------------------


def build_descriptive_verdict_from_rmse(
    baseline_json: dict,
    candidate_json: dict,
    *,
    baseline_id: str,
    candidate_id: str,
    keypoint_names: list[str] | None = None,
    gate: GateConfig | None = None,
) -> Verdict:
    """Triage-mode verdict from two ``_bodypart_rmse.json`` summaries.

    The pre-aggregated JSONs lack per-frame pairs, so paired Wilcoxon
    cannot run; ``p_value_wilcoxon`` is NaN throughout. Use this mode for
    quick visual inspection only — it is non-authoritative.

    Returns a :class:`Verdict` whose per-keypoint stats reflect the
    median/PCK numbers in the two JSONs and whose ``meta`` carries an
    ``error="rmse-json mode is descriptive only"`` annotation.
    """
    if gate is None:
        gate = GateConfig()
    if keypoint_names is None:
        keypoint_names = list(HM2P_BODYPARTS)
    n_frames = len(keypoint_names) * 0  # synthesise a 0-frame e_arr later
    n_kp = len(keypoint_names)
    e_b = np.empty((n_frames, n_kp))
    e_c = np.empty((n_frames, n_kp))
    v = evaluate_promotion_gate(
        e_b,
        e_c,
        keypoint_names,
        None,
        None,
        None,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        gate=gate,
        meta={
            "mode": "rmse-json",
            "error": "rmse-json mode is descriptive only; no paired Wilcoxon",
            "baseline_summary": baseline_json.get("bodyparts", {}),
            "candidate_summary": candidate_json.get("bodyparts", {}),
        },
    )
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--mode",
        choices=["predict", "rmse-json"],
        default="predict",
        help="`predict` reads pairs of prediction .h5 files (full gate); "
        "`rmse-json` reads two pre-aggregated _bodypart_rmse.json "
        "files (descriptive only, non-authoritative).",
    )
    p.add_argument(
        "--baseline-id",
        required=True,
        help="Champion-id of the baseline (for verdict provenance).",
    )
    p.add_argument(
        "--candidate-id",
        required=True,
        help="Champion-id of the candidate (for verdict provenance).",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        required=False,
        help="Local labelled-data directory (CollectedData_*.h5 per "
        "session). Required in --mode predict.",
    )
    p.add_argument("--baseline-h5-prefix", default="s3://hm2p-derivatives/pose-archive/")
    p.add_argument("--candidate-h5-prefix", default="s3://hm2p-derivatives/pose/")
    p.add_argument(
        "--baseline-rmse-json",
        type=Path,
        required=False,
        help="Required in --mode rmse-json. Path to local copy of "
        "the baseline's _bodypart_rmse.json.",
    )
    p.add_argument("--candidate-rmse-json", type=Path, required=False)
    p.add_argument("--output", type=Path, default=Path("./verdict.json"))
    p.add_argument(
        "--upload-s3",
        action="store_true",
        help="Also upload the verdict to s3://hm2p-derivatives/" + VERDICT_S3_KEY,
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Bonferroni-corrected per-keypoint α (default {DEFAULT_ALPHA:g}).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the bootstrap (default 42).")
    p.add_argument(
        "--label-filename",
        default="CollectedData_tristan.h5",
        help="Per-session GT filename inside each clip dir.",
    )
    p.add_argument("--region", default="ap-southeast-2")
    return p


def _make_s3_client(region: str) -> Any:
    """Lazy boto3 import + client construction (kept out of test path)."""
    import boto3

    return boto3.client("s3", region_name=region)


def _build_gate_from_args(alpha: float) -> GateConfig:
    """Build a :class:`GateConfig` honouring the CLI ``--alpha`` override."""
    return GateConfig(alpha=alpha)


def _write_verdict(
    v: Verdict,
    out_path: Path,
    *,
    upload_s3: bool,
    s3_client: Any,
) -> None:
    """Write verdict JSON to disk and (optionally) S3."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(verdict_to_json(v))
    log.info("Wrote verdict to %s", out_path)
    if upload_s3:
        s3_client.put_object(
            Bucket=DERIVATIVES_BUCKET,
            Key=VERDICT_S3_KEY,
            Body=out_path.read_bytes(),
            ContentType="application/json",
        )
        log.info("Uploaded verdict to s3://%s/%s", DERIVATIVES_BUCKET, VERDICT_S3_KEY)


def _summarise_to_stdout(v: Verdict) -> None:
    """Print a one-page summary of the verdict to stdout."""
    print(f"\n=== SA fine-tune verdict ({v.candidate_id} vs {v.baseline_id}) ===")
    print(f"frames compared: {v.n_frames_compared}")
    print(f"overall_pass:    {v.overall_pass}")
    if v.fail_reasons:
        print("fail_reasons:")
        for r in v.fail_reasons:
            print(f"  - {r}")
    print("\nper-keypoint:")
    print(f"  {'keypoint':<16}  {'med_b':>6}  {'med_c':>6}  {'%Δ':>7}  {'p':>9}  {'r':>6}")
    for kp in v.keypoints:
        pct = (
            f"{kp.pct_change_median * 100:+6.1f}%"
            if not np.isnan(kp.pct_change_median)
            else "    nan"
        )
        p = f"{kp.p_value_wilcoxon:.2e}" if not np.isnan(kp.p_value_wilcoxon) else "      nan"
        print(
            f"  {kp.keypoint:<16}  {kp.median_baseline_px:>6.1f}  "
            f"{kp.median_candidate_px:>6.1f}  {pct}  {p}  "
            f"{kp.rank_biserial_r:>+6.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write a verdict.

    Parameters
    ----------
    argv
        Optional argv override (used by tests). When ``None``, uses
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 (pass), 2 (fail), 3 (no comparison possible).
    """
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    args = _build_parser().parse_args(argv)
    gate = _build_gate_from_args(args.alpha)
    rng = np.random.default_rng(args.seed)

    if args.mode == "rmse-json":
        if args.baseline_rmse_json is None or args.candidate_rmse_json is None:
            print(
                "ERROR: --mode rmse-json requires --baseline-rmse-json + --candidate-rmse-json",
                file=sys.stderr,
            )
            return 1
        b_json = json.loads(args.baseline_rmse_json.read_text())
        c_json = json.loads(args.candidate_rmse_json.read_text())
        v = build_descriptive_verdict_from_rmse(
            b_json,
            c_json,
            baseline_id=args.baseline_id,
            candidate_id=args.candidate_id,
            gate=gate,
        )
        s3 = _make_s3_client(args.region) if args.upload_s3 else None
        _write_verdict(v, args.output, upload_s3=args.upload_s3, s3_client=s3)
        _summarise_to_stdout(v)
        # rmse-json mode is non-authoritative; always exit 0 (the verdict
        # carries meta.error explaining the limitation).
        return 0

    # --- mode predict ---------------------------------------------------
    if args.labels_dir is None:
        print("ERROR: --mode predict requires --labels-dir", file=sys.stderr)
        return 1

    s3 = _make_s3_client(args.region)
    b_bucket, b_prefix = parse_s3_uri(args.baseline_h5_prefix)
    c_bucket, c_prefix = parse_s3_uri(args.candidate_h5_prefix)

    e_b, e_c, hd_b, hd_c, hd_g, skipped = collect_paired_errors(
        args.labels_dir,
        s3,
        b_bucket,
        b_prefix,
        c_bucket,
        c_prefix,
        label_filename=args.label_filename,
    )

    meta: dict[str, Any] = {
        "mode": "predict",
        "labels_dir": str(args.labels_dir),
        "baseline_h5_prefix": args.baseline_h5_prefix,
        "candidate_h5_prefix": args.candidate_h5_prefix,
        "rng_seed": args.seed,
        "skipped_sessions": skipped,
    }

    if e_b.shape[0] == 0:
        # Build an empty-but-valid verdict so downstream tooling has
        # something to read.
        meta["error"] = "no overlapping (gt, baseline_pred, candidate_pred) frames"
        empty = np.empty((0, len(HM2P_BODYPARTS)))
        v = evaluate_promotion_gate(
            empty,
            empty,
            list(HM2P_BODYPARTS),
            None,
            None,
            None,
            baseline_id=args.baseline_id,
            candidate_id=args.candidate_id,
            gate=gate,
            rng=rng,
            meta=meta,
        )
        _write_verdict(v, args.output, upload_s3=args.upload_s3, s3_client=s3)
        _summarise_to_stdout(v)
        return 3

    v = evaluate_promotion_gate(
        e_b,
        e_c,
        list(HM2P_BODYPARTS),
        hd_b,
        hd_c,
        hd_g,
        baseline_id=args.baseline_id,
        candidate_id=args.candidate_id,
        gate=gate,
        rng=rng,
        meta=meta,
    )
    _write_verdict(v, args.output, upload_s3=args.upload_s3, s3_client=s3)
    _summarise_to_stdout(v)
    return 0 if v.overall_pass else 2


if __name__ == "__main__":
    sys.exit(main())
