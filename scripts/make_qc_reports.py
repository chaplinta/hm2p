#!/usr/bin/env python3
"""Collect per-session data for the five standalone data-quality reports.

For every session in ``metadata/experiments.csv`` (all 26, regardless of the
``exclude`` / ``primary_exp`` flags, which are shown in the reports instead)
this downloads the Stage 2b-4 outputs from S3, summarises them with
:mod:`hm2p.qc`, and writes one JSON per report to ``results/qc/data/``:

=============  ===========================================================
tracking       champion DLC pose .h5 (``pose/{sub}/{ses}/``)
movement       ``kinematics/{sub}/{ses}/kinematics.h5``
syllables      ``kinematics/{sub}/{ses}/syllables.npz`` + provenance, model
rois           ``ca_extraction/{sub}/{ses}/suite2p/plane0/*.npy`` + ca.h5
spikes         ``calcium/{sub}/{ses}/ca.h5``
=============  ===========================================================

Per-session summaries are cached in ``results/qc/cache/{report}/`` so an
interrupted run resumes; ``--refresh`` ignores the cache. Nothing is written
to S3. ``--classifier-reference`` also computes leave-one-session-out
predictions on the manually labelled legacy sessions in ``/data/s2p``
(read-only) for the classifier report.

Usage
-----
    python scripts/make_qc_reports.py                       # all reports
    python scripts/make_qc_reports.py --reports tracking movement
    python scripts/make_qc_reports.py --sessions 20220804_13_52_02_1117646
    python scripts/make_qc_reports.py --reports rois --classifier-reference
    python scripts/build_qc_reports_html.py
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hm2p.qc import movement as qmov  # noqa: E402
from hm2p.qc import rois as qroi  # noqa: E402
from hm2p.qc import spikes as qspk  # noqa: E402
from hm2p.qc import syllables as qsyl  # noqa: E402
from hm2p.qc import tracking as qtrk  # noqa: E402
from hm2p.qc.common import hist, span_fps  # noqa: E402

log = logging.getLogger("make_qc_reports")

BUCKET = "hm2p-derivatives"
OUT_DIR = REPO / "results" / "qc"
REPORTS = ("tracking", "movement", "syllables", "rois", "spikes")
CA_KEYS_SPIKES = (
    "dff",
    "F_corr",
    "F0_rolling",
    "F0_percentile",
    "F_raw",
    "event_masks",
    "event_masks_sd",
    "spikes",
    "roi_types",
    "frame_times",
)
KIN_SKIP_PREFIX = "bp_"

_S3: Any = None


# ---------------------------------------------------------------------------
# S3 access (read-only, with retries: the container connection is flaky)
# ---------------------------------------------------------------------------


def s3() -> Any:
    global _S3
    if _S3 is None:
        import boto3

        profile = os.environ.get("HM2P_AWS_PROFILE", "hm2p-agent")
        _S3 = boto3.Session(profile_name=profile).client("s3")
    return _S3


def get_bytes(key: str, retries: int = 4) -> bytes | None:
    """Object body, or None if it does not exist / every attempt failed."""
    for attempt in range(retries):
        try:
            return s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except Exception as exc:  # noqa: BLE001
            # non-ClientError exceptions (connection errors) carry response=None
            code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404"):
                return None
            log.debug("get %s failed (attempt %d): %s", key, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    log.warning("download failed after %d attempts: %s", retries, key)
    return None


def list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3().list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]


def get_json(key: str) -> dict | None:
    b = get_bytes(key)
    return None if b is None else json.loads(b)


def read_h5(key: str, wanted: Callable[[str], bool] | None = None) -> tuple[dict, dict] | None:
    """Datasets (all, or those whose path passes *wanted*) and root attrs of an S3 HDF5."""
    b = get_bytes(key)
    return None if b is None else h5_from_bytes(b, wanted)


def h5_from_bytes(b: bytes, wanted: Callable[[str], bool] | None = None) -> tuple[dict, dict]:
    """Datasets (all, or those whose path passes *wanted*) and root attrs of an in-memory HDF5."""
    data: dict[str, np.ndarray] = {}
    with h5py.File(io.BytesIO(b), "r") as f:
        attrs = dict(f.attrs)

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and (wanted is None or wanted(name)):
                data[name] = obj[()]

        f.visititems(visit)
    return data, attrs


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


def sessions_table() -> pd.DataFrame:
    exps = pd.read_csv(REPO / "metadata" / "experiments.csv", dtype={"exp_id": str})
    animals = pd.read_csv(REPO / "metadata" / "animals.csv", dtype={"animal_id": str})
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    p = exps["exp_id"].str.split("_")
    exps["sub"] = "sub-" + exps["animal_id"]
    exps["ses"] = "ses-" + p.str[0] + "T" + p.str[1] + p.str[2] + p.str[3]
    return exps.merge(animals[["animal_id", "celltype"]], on="animal_id", how="left")


def mm_per_px(sub: str, ses: str) -> float | None:
    import configparser

    path = REPO / "metadata" / "video_meta" / f"{sub}_{ses}_meta.txt"
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    try:
        return float(cfg["scale"]["mm_per_pix"])
    except (KeyError, ValueError):
        return None


def session_meta(row: pd.Series) -> dict:
    return {
        "exp_id": row["exp_id"],
        "sub": row["sub"],
        "ses": row["ses"],
        "animal": row["animal_id"],
        "celltype": None if pd.isna(row.get("celltype")) else row["celltype"],
        "exclude": bool(row.get("exclude", 0) == 1),
        "primary": bool(row.get("primary_exp", 0) == 1),
        "notes": None if pd.isna(row.get("Notes")) else str(row["Notes"]),
    }


# ---------------------------------------------------------------------------
# Per-report collectors (each returns a summary dict for one session)
# ---------------------------------------------------------------------------


class Ctx:
    """Per-session download cache shared by the collectors."""

    def __init__(self, sub: str, ses: str, champion: dict | None) -> None:
        self.sub, self.ses, self.champion = sub, ses, champion
        self._kin: tuple[dict, dict] | None | bool = False
        self._ca_bytes: bytes | None | bool = False

    def kinematics(self) -> tuple[dict, dict] | None:
        if self._kin is False:
            self._kin = read_h5(
                f"kinematics/{self.sub}/{self.ses}/kinematics.h5",
                lambda n: not n.startswith(KIN_SKIP_PREFIX),
            )
        return self._kin  # type: ignore[return-value]

    def ca(self, wanted: Callable[[str], bool]) -> tuple[dict, dict] | None:
        """Selected datasets of ca.h5; the file is downloaded once per session."""
        if self._ca_bytes is False:
            self._ca_bytes = get_bytes(f"calcium/{self.sub}/{self.ses}/ca.h5")
        if self._ca_bytes is None:
            return None
        return h5_from_bytes(self._ca_bytes, wanted)  # type: ignore[arg-type]

    def ca_labels(self) -> tuple[dict, dict] | None:
        """roi_types, frame_times and roi_qc from ca.h5 (small datasets only)."""
        return self.ca(
            lambda n: n in ("roi_types", "frame_times", "iscell") or n.startswith("roi_qc/")
        )


def collect_tracking(c: Ctx) -> dict:
    from hm2p.kinematics.compute import load_pose_dataset
    from hm2p.pose.select import select_champion_h5

    prefix = f"pose/{c.sub}/{c.ses}/"
    keys = list_keys(prefix)
    champ_id = (c.champion or {}).get("champion_id")
    if champ_id:
        key = select_champion_h5(keys, champ_id)
    else:  # no manifest: newest-looking non-filtered .h5
        cands = sorted(
            k for k in keys if k.endswith(".h5") and "_filtered" not in k and "_single" not in k
        )
        if not cands:
            raise FileNotFoundError(f"no pose .h5 under {prefix}")
        key = cands[-1]
    kin = c.kinematics()
    meta = get_json(prefix + "dlc_meta.json") or {}
    fps = float(meta.get("tracking_fps", 30.0))
    scale = mm_per_px(c.sub, c.ses)
    fps_source, scale_source = (
        "dlc_meta.json" if "tracking_fps" in meta else "default 30",
        "video_meta",
    )
    body = get_bytes(key)
    if body is None:
        raise FileNotFoundError(key)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        tmp.write(body)
        tmp.flush()
        ds = load_pose_dataset(Path(tmp.name), "dlc")
    kp = qtrk.keypoints_from_dataset(ds)
    n_pose = len(next(iter(kp.values()))["x"])
    light = None
    if kin is not None:
        k, attrs = kin
        # kinematics.h5 is computed from this pose file, frame for frame: use its
        # frame rate and video scale so tracking QC matches stage 3
        if "frame_times" in k and len(k["frame_times"]) == n_pose:
            fps, fps_source = span_fps(k["frame_times"]), "kinematics frame_times"
        if attrs.get("scale_mm_per_px"):
            scale, scale_source = float(attrs["scale_mm_per_px"]), "kinematics.h5"
        light = k.get("light_on")
    s = qtrk.summarise_tracking(kp, fps=fps, mm_per_px=scale, light_on=light)
    s["fps_source"], s["scale_source"] = fps_source, scale_source
    s["pose_file"] = key.split("/")[-1]
    s["champion_id"] = champ_id
    s["light_from_kinematics"] = light is not None and len(light) == s["n_frames"]
    return s


def collect_movement(c: Ctx) -> dict:
    kin = c.kinematics()
    if kin is None:
        raise FileNotFoundError("kinematics.h5")
    s = qmov.summarise_movement(*kin)
    cid = s["attrs"].get("dlc_champion_id")
    want = (c.champion or {}).get("champion_id")
    s["champion_current"] = None if want is None else bool(cid == want)
    return s


def collect_syllables(c: Ctx) -> dict:
    b = get_bytes(f"kinematics/{c.sub}/{c.ses}/syllables.npz")
    if b is None:
        raise FileNotFoundError("syllables.npz")
    z = np.load(io.BytesIO(b))
    sid = z["syllable_id"] if "syllable_id" in z else z["syllable_ids"]
    kin = c.kinematics()
    k = kin[0] if kin is not None else {}
    fps = 30.0
    if "frame_times" in k and len(k["frame_times"]) == len(sid):
        fps = span_fps(k["frame_times"])
    s = qsyl.summarise_syllables(
        sid,
        fps=fps,
        speed_cm_s=k.get("speed_cm_s"),
        ahv_deg_s=k.get("ahv_deg_s"),
        light_on=k.get("light_on"),
        bad_behav=k.get("bad_behav"),
    )
    s["aligned_to_kinematics"] = "frame_times" in k and len(k["frame_times"]) == len(sid)
    s["provenance"] = get_json(f"kinematics/{c.sub}/{c.ses}/syllables.provenance.json")
    if "syllable_prob" in z:
        p = np.asarray(z["syllable_prob"], dtype=np.float64)
        s["max_prob_hist"] = hist(p.max(axis=1), 0.0, 1.0, 40) if p.ndim == 2 else None
    return s


_MODEL: Any = None


def collect_rois(c: Ctx) -> dict:
    """Classifier QC for one session.

    Stored labels and probabilities come from ``roi_class.npy`` /
    ``roi_class_prob.npy`` when present, else from ca.h5 (``roi_types`` and
    ``roi_qc/p_*``; the EC2 stage 1 run classified inline and kept them only
    there). The classifier is then re-applied to features recomputed from the
    Suite2p outputs twice: at the frame rate the pipeline used
    (``ops["fs"]``), to check the stored labels can be reproduced, and at the
    imaging frame rate from ca.h5 (the rate of the training data), whose
    label changes measure the effect of the frame rate given to
    ``classify_session``.
    """
    from hm2p.extraction.roi_classify import load_model
    from hm2p.extraction.soma_features import FEATURE_COLUMNS, extract_soma_features

    global _MODEL
    base = f"ca_extraction/{c.sub}/{c.ses}/suite2p/plane0/"

    def npy(name: str, pickle: bool = False) -> Any:
        b = get_bytes(base + name)
        return None if b is None else np.load(io.BytesIO(b), allow_pickle=pickle)

    stat = npy("stat.npy", True)
    if stat is None:
        raise FileNotFoundError("stat.npy")
    stat = list(stat)
    ca = c.ca_labels()
    ca_d, ca_attrs = ca if ca is not None else ({}, {})
    roi_qc = {k.split("/", 1)[1]: v for k, v in ca_d.items() if k.startswith("roi_qc/")}
    labels, probs = npy("roi_class.npy"), npy("roi_class_prob.npy")
    ca_types = ca_d.get("roi_types")
    if labels is not None and probs is not None:
        source = "roi_class.npy"
    elif ca_types is not None and all(f"p_{k}" in roi_qc for k in qroi.LABELS):
        source = "ca.h5 (roi_types, roi_qc/p_*)"
        labels = np.array([qroi.CA_TO_CLASS.get(int(v), -1) for v in ca_types])
        probs = np.column_stack([roi_qc[f"p_{k}"] for k in qroi.LABELS])
        ca_types = None  # same source, so the consistency check would be trivial
    else:
        raise FileNotFoundError(
            "no classifier output (roi_class.npy, or ca.h5 roi_types + roi_qc/p_*)"
        )
    ops = npy("ops.npy", True)
    ops = ops.item() if ops is not None else {}
    F, Fneu = npy("F.npy"), npy("Fneu.npy")
    pipeline_fps = float(ops.get("fs", np.nan))
    imaging_fps = float(ca_attrs.get("fps_imaging", 0) or 0)
    if imaging_fps <= 0 and "frame_times" in ca_d:
        imaging_fps = span_fps(ca_d["frame_times"])
    feats = recomputed = None
    repro: dict = {
        "pipeline_fps": pipeline_fps,
        "imaging_fps": imaging_fps,
        "label_source": source,
    }
    if F is not None and Fneu is not None:
        if _MODEL is None:
            _MODEL = load_model()
        model, medians, _meta = _MODEL
        fill = pd.Series(medians, index=list(FEATURE_COLUMNS))

        def predict(fps: float) -> tuple[pd.DataFrame, np.ndarray]:
            f = extract_soma_features(stat, F.astype(np.float32), Fneu.astype(np.float32), fps=fps)
            return f, model.predict_proba(f.fillna(fill).values)

        if np.isfinite(pipeline_fps) and pipeline_fps > 0:
            _f, p_pipe = predict(pipeline_fps)
            repro["pipeline_max_abs_diff"] = (
                float(np.abs(p_pipe - probs).max()) if len(probs) else 0.0
            )
            repro["pipeline_label_changes"] = int((p_pipe.argmax(axis=1) != labels).sum())
        if imaging_fps > 0:
            feats, recomputed = predict(imaging_fps)
    s = qroi.summarise_rois(
        labels,
        probs,
        stat=stat,
        features=feats,
        mean_img=ops.get("meanImg"),
        ca_roi_types=ca_types,
        recomputed_probs=recomputed,
        roi_qc=roi_qc,
    )
    s["repro"] = repro
    if recomputed is not None:
        s["imaging_fps_counts"] = {
            k: int((recomputed.argmax(axis=1) == i).sum()) for i, k in enumerate(qroi.LABELS)
        }
    return s


def collect_spikes(c: Ctx) -> dict:
    got = c.ca(lambda n: n in CA_KEYS_SPIKES or n.startswith("roi_qc/"))
    if got is None:
        raise FileNotFoundError("ca.h5")
    ca, attrs = got
    light = None
    kin = c.kinematics()
    if kin is not None and "frame_times" in ca and "light_on" in kin[0]:
        kt, kl = kin[0]["frame_times"], kin[0]["light_on"].astype(float)
        idx = np.clip(np.searchsorted(kt, ca["frame_times"]), 0, kt.size - 1)
        light = kl[idx]
    return qspk.summarise_spikes(ca, attrs, light_on=light)


COLLECTORS: dict[str, tuple[Callable[[Ctx], dict], Callable[[dict], dict]]] = {
    "tracking": (collect_tracking, qtrk.overview_row),
    "movement": (collect_movement, qmov.overview_row),
    "syllables": (collect_syllables, qsyl.overview_row),
    "rois": (collect_rois, qroi.overview_row),
    "spikes": (collect_spikes, qspk.overview_row),
}


# ---------------------------------------------------------------------------
# Project-level extras
# ---------------------------------------------------------------------------


def syllable_model_info() -> dict:
    out = {"summary": get_json("kinematics/kpms_summary.json")}
    for name in ("pca_variance.json", "fit_info.json", "convergence.json"):
        out[name.removesuffix(".json")] = get_json(f"kinematics/kpms_model/{name}")
    return out


def classifier_info(with_reference: bool) -> dict:
    meta_path = REPO / "sourcedata" / "trackers" / "suite2p" / "roi_classifier_xgb.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out: dict = {"model": meta}
    ref_path = OUT_DIR / "cache" / "rois" / "_reference.json"
    if with_reference:
        from hm2p.extraction.roi_training_data import load_all_sessions

        X, y, groups = load_all_sessions()
        ref = qroi.cv_reference(X, y, groups, meta.get("best_params", {}), n_jobs=-1)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(json.dumps(ref))
    if ref_path.exists():
        out["reference"] = json.loads(ref_path.read_text())
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(reports: list[str], only: set[str] | None, refresh: bool, with_reference: bool) -> None:
    table = sessions_table()
    if only:
        table = table[table["exp_id"].isin(only)]
    champion = get_json("dlc-champion.json")
    log.info("champion: %s", (champion or {}).get("champion_id"))
    results: dict[str, list[dict]] = {r: [] for r in reports}
    for _, row in table.iterrows():
        meta = session_meta(row)
        ctx = Ctx(meta["sub"], meta["ses"], champion)
        for rep in reports:
            cache = OUT_DIR / "cache" / rep / f"{meta['exp_id']}.json"
            entry: dict = dict(meta)
            cached = json.loads(cache.read_text()) if cache.exists() and not refresh else None
            if cached is not None and cached.get("error") is None:
                entry.update(cached)
            else:  # no cache, --refresh, or a cached failure (retried: S3 errors are often transient)
                fn, ov = COLLECTORS[rep]
                t0 = time.time()
                try:
                    s = fn(ctx)
                    entry_data = {"summary": s, "overview": ov(s), "error": None}
                    log.info("%s %s ok (%.0f s)", rep, meta["exp_id"], time.time() - t0)
                except Exception as exc:  # noqa: BLE001 - recorded in the report
                    log.warning("%s %s failed: %s", rep, meta["exp_id"], exc)
                    log.debug(traceback.format_exc())
                    entry_data = {
                        "summary": None,
                        "overview": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(_clean(entry_data)))
                entry.update(entry_data)
            results[rep].append(entry)
    for rep in reports:
        doc: dict = {
            "report": rep,
            "generated": date.today().isoformat(),
            "champion": champion,
            "sessions": results[rep],
        }
        if rep == "syllables":
            doc["model"] = syllable_model_info()
        if rep == "rois":
            doc["classifier"] = classifier_info(with_reference)
        path = OUT_DIR / "data" / f"{rep}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_clean(doc), separators=(",", ":")))
        n_ok = sum(1 for e in results[rep] if e.get("error") is None)
        log.info(
            "wrote %s (%d/%d sessions ok, %.1f MB)",
            path,
            n_ok,
            len(results[rep]),
            path.stat().st_size / 1e6,
        )


def _clean(o: Any) -> Any:
    """Make numpy scalars and non-finite floats JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, bytes):
        return o.decode(errors="replace")
    return o


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - S3 entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", nargs="+", choices=REPORTS, default=list(REPORTS))
    ap.add_argument("--sessions", nargs="+", help="exp_id values (default: all)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached per-session summaries")
    ap.add_argument(
        "--classifier-reference", action="store_true", help="LOSO CV on /data/s2p manual labels"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    run(a.reports, set(a.sessions) if a.sessions else None, a.refresh, a.classifier_reference)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
