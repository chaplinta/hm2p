#!/usr/bin/env python3
"""Penk+ vs Penk⁻CamKII+ programme runner (hypotheses H2–H10).

One subcommand per hypothesis in docs/plan-penk-vs-nonpenk.md. Each
subcommand loads the per-session ``sync.h5`` files from S3 (with retries),
computes per-cell or per-session measures with the pure functions in
``hm2p.analysis``, and reports every between-group comparison at four levels
(naive cell-level, animal-level Mann-Whitney, animal-level cluster
permutation, supplementary LMM) plus a leave-one-animal-out direction check
and the equipment-matched subset. Outputs go to
``results/celltype_programme/<hypothesis>/`` (gitignored).

Nothing here is executed automatically. See ``--dry-run`` and ``--sessions``.

Usage
-----
    python scripts/run_celltype_programme.py h2 --dry-run
    python scripts/run_celltype_programme.py h2 --sessions 2      # smoke run
    python scripts/run_celltype_programme.py h6 --features results/celltype_programme/h2/cells.csv
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.analysis.mixed_stats import (  # noqa: E402
    equipment_matched_subset,
    fdr_correct,
    lmm_celltype_test,
    loao_between_group,
    run_between_group_test,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("celltype_programme")

BUCKET = "hm2p-derivatives"
REPO = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO / "results" / "celltype_programme"
SOMA = 0  # roi_types code for soma
SYNC_KEYS = [
    "dff",
    "hd_deg",
    "ahv_deg_s",
    "speed_cm_s",
    "light_on",
    "active",
    "bad_behav",
    "x_mm",
    "y_mm",
    "x_maze",
    "y_maze",
    "roi_types",
    "syllable_id",
    "event_masks",
    "spikes",
    "frame_times",
]

_S3 = None


# ---------------------------------------------------------------------------
# S3 + metadata
# ---------------------------------------------------------------------------


def _s3():  # pragma: no cover - network
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.Session(profile_name="hm2p-agent").client("s3")
    return _S3


def _download_h5(key: str, retries: int = 4):  # pragma: no cover - network
    """Download an HDF5 file from S3 into memory, retrying on transient errors."""
    for attempt in range(retries):
        try:
            obj = _s3().get_object(Bucket=BUCKET, Key=key)
            return h5py.File(io.BytesIO(obj["Body"].read()), "r")
        except Exception as exc:  # noqa: BLE001
            log.debug("download failed %s (attempt %d): %s", key, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    log.warning("download FAILED after %d attempts: %s", retries, key)
    return None


def sync_key(exp_id: str, animal_id: str) -> str:
    parts = exp_id.split("_")
    return f"sync/sub-{animal_id}/ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}/sync.h5"


def load_metadata(base: Path | None = None) -> pd.DataFrame:
    """Join experiments.csv with animals.csv into one row per session.

    Columns: ``exp_id, animal_id, celltype, fibre, lens, virus_id, gcamp,
    exclude, primary_exp``. Excluded sessions are kept (analysis-time
    filtering is explicit via ``--include-excluded``).
    """
    base = base or (REPO / "metadata")
    animals = pd.read_csv(base / "animals.csv", dtype={"animal_id": str})
    exps = pd.read_csv(base / "experiments.csv")
    exps["animal_id"] = exps["exp_id"].astype(str).str.split("_").str[-1]
    cols = ["animal_id", "celltype", "virus_id", "gcamp"]
    merged = exps.merge(animals[cols], on="animal_id", how="left", validate="many_to_one")
    for c in ("exclude", "primary_exp"):
        merged[c] = pd.to_numeric(merged.get(c, 0), errors="coerce").fillna(0).astype(int)
    keep = ["exp_id", "animal_id", "celltype", "fibre", "lens", "virus_id", "gcamp"]
    keep += ["exclude", "primary_exp"]
    return merged[keep]


def select_sessions(
    meta: pd.DataFrame,
    limit: int | None = None,
    include_excluded: bool = False,
    primary_only: bool = False,
) -> pd.DataFrame:
    """Apply the standard session filters and an optional smoke-run limit."""
    sel = meta.copy()
    if not include_excluded:
        sel = sel[sel["exclude"] == 0]
    if primary_only:
        sel = sel[sel["primary_exp"] == 1]
    sel = sel.dropna(subset=["celltype"])
    if limit is not None:
        # keep both cell types represented in a smoke run
        parts = [g.head(max(1, limit // 2)) for _, g in sel.groupby("celltype")]
        sel = pd.concat(parts).head(limit) if parts else sel.head(limit)
    return sel.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Session arrays
# ---------------------------------------------------------------------------


def read_session_arrays(f: h5py.File, soma_only: bool = True) -> dict[str, Any] | None:
    """Read the sync.h5 datasets used by the programme into a dict.

    Returns None when a required dataset is absent or the sync failed.
    Row selection restricts ``dff``/``event_masks``/``spikes`` to soma ROIs
    when *soma_only* is set and ``roi_types`` is present.
    """
    required = ["dff", "hd_deg", "speed_cm_s", "light_on", "active", "bad_behav"]
    if any(k not in f for k in required):
        return None
    status = f.attrs.get("sync_status", "OK")
    if isinstance(status, bytes):
        status = status.decode()
    if str(status).startswith("FAILED"):
        return None
    out: dict[str, Any] = {}
    for k in SYNC_KEYS:
        if k in f:
            out[k] = f[k][:]
    n = out["dff"].shape[1]
    for k in ("hd_deg", "ahv_deg_s", "speed_cm_s", "light_on", "active", "bad_behav"):
        if k in out:
            out[k] = out[k][:n]
    for k in ("light_on", "active", "bad_behav"):
        out[k] = out[k].astype(bool)
    for k in ("dff", "hd_deg", "ahv_deg_s", "speed_cm_s", "x_mm", "y_mm", "spikes"):
        if k in out:
            out[k] = out[k].astype(np.float64)
    if "ahv_deg_s" not in out:
        from hm2p.analysis.ahv import compute_ahv

        out["ahv_deg_s"] = compute_ahv(out["hd_deg"], fps=float(f.attrs.get("fps_imaging", 9.6)))
    out["fps"] = float(f.attrs.get("fps_imaging", 9.6))
    out["roi_idx"] = np.arange(out["dff"].shape[0])
    if soma_only and "roi_types" in out:
        keep = out["roi_types"] == SOMA
        for k in ("dff", "event_masks", "spikes", "roi_types", "roi_idx"):
            if k in out:
                out[k] = out[k][keep]
    if out["dff"].shape[0] == 0:
        return None
    finite = np.isfinite(out["hd_deg"]) & np.isfinite(out["speed_cm_s"])
    finite &= np.isfinite(out["ahv_deg_s"])
    out["mask"] = out["active"] & ~out["bad_behav"] & finite
    return out


def attach_session_meta(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Add the per-session metadata columns to a per-cell (or per-session) table."""
    out = df.copy()
    for c in ("exp_id", "animal_id", "celltype", "fibre", "lens", "virus_id", "gcamp"):
        out[c] = row[c]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def naive_cell_level(df: pd.DataFrame, metric: str, group_col: str = "celltype") -> dict:
    """Descriptive cell-level Mann-Whitney (reported, never used for inference)."""
    from scipy import stats

    groups = sorted(df[group_col].dropna().unique())
    if len(groups) != 2:
        return {"naive_p_value": np.nan, "naive_n": int(len(df))}
    a = df.loc[df[group_col] == groups[0], metric].dropna().to_numpy()
    b = df.loc[df[group_col] == groups[1], metric].dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return {"naive_p_value": np.nan, "naive_n": int(len(a) + len(b))}
    p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return {"naive_p_value": p, "naive_n": int(len(a) + len(b))}


def between_group_report(
    df: pd.DataFrame,
    metrics: list[str],
    family: str,
    n_perms: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Four-level between-group report for a family of metrics.

    Per metric: naive cell-level p (descriptive), animal-level MWU and
    cluster permutation (``run_between_group_test``), LMM with ICC,
    leave-one-animal-out direction stability, and the animal-level MWU on the
    equipment-matched subset when one exists. FDR is applied to the cluster
    permutation p-values within the family.
    """
    rows: list[dict] = []
    matched = None
    if {"fibre", "lens"} <= set(df.columns):
        matched = equipment_matched_subset(df)
    for m in metrics:
        sub = df.dropna(subset=[m, "celltype", "animal_id"])
        if sub["celltype"].nunique() != 2 or sub["animal_id"].nunique() < 3:
            rows.append({"metric": m, "family": family, "skipped": "insufficient groups"})
            continue
        res = run_between_group_test(sub, m, n_perms=n_perms, seed=seed)
        res.update(naive_cell_level(sub, m))
        lmm = lmm_celltype_test(sub, m)
        res.update({f"lmm_{k}": v for k, v in lmm.items() if k not in ("metric", "levels")})
        loao = loao_between_group(sub, m, n_perms=min(n_perms, 2000), seed=seed)
        res["loao_direction_stable"] = loao["direction_stable"]
        res["loao_max_p"] = loao["max_p_value"]
        if matched is not None and not matched.empty:
            msub = matched.dropna(subset=[m])
            if msub["celltype"].nunique() == 2 and msub["animal_id"].nunique() >= 3:
                mres = run_between_group_test(msub, m, n_perms=min(n_perms, 2000), seed=seed)
                res["matched_summary_p"] = mres["summary_p_value"]
                full_diff = res["summary_penk_mean"] - res["summary_nonpenk_mean"]
                matched_diff = mres["summary_penk_mean"] - mres["summary_nonpenk_mean"]
                res["matched_direction_agrees"] = bool(np.sign(full_diff) == np.sign(matched_diff))
        res["family"] = family
        rows.append(res)
    valid = [r for r in rows if "perm_p_value" in r]
    if valid:
        corrected = fdr_correct(valid)
        by_metric = {r["metric"]: r for r in corrected}
        rows = [by_metric.get(r["metric"], r) for r in rows]
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, name: str, obj: Any) -> Path:
    """Write a DataFrame (csv) or dict (json) under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        path = out_dir / f"{name}.csv"
        obj.to_csv(path, index=False)
    else:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2, default=_json_default))
    log.info("wrote %s", path)
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def iter_sessions(args: argparse.Namespace):  # pragma: no cover - network
    """Yield (meta_row, arrays) for each selected session."""
    meta = select_sessions(
        load_metadata(),
        limit=args.sessions,
        include_excluded=args.include_excluded,
        primary_only=args.primary_only,
    )
    for _, row in meta.iterrows():
        f = _download_h5(sync_key(row["exp_id"], row["animal_id"]))
        if f is None:
            continue
        with f:
            arrays = read_session_arrays(f, soma_only=not args.all_rois)
        if arrays is None:
            log.warning("skipping %s (missing datasets or failed sync)", row["exp_id"])
            continue
        yield row, arrays


# ---------------------------------------------------------------------------
# Hypothesis subcommands (H2–H10) — bodies in HYPOTHESES below
# ---------------------------------------------------------------------------

HYPOTHESES: dict[str, dict[str, Any]] = {
    "h2": {"title": "Calcium activity signature (cell feature table)", "families": []},
    "h3": {"title": "Angular head velocity coding", "families": []},
    "h4": {"title": "State dependence: transient vs sustained", "families": []},
    "h5": {"title": "Light-dark transition dynamics", "families": []},
    "h6": {"title": "Heterogeneity and omnibus separability", "families": []},
    "h7": {"title": "Population HD code: decoding and topology", "families": []},
    "h8": {"title": "Encoding models (Poisson GLM)", "families": []},
    "h9": {"title": "Network coupling", "families": []},
    "h10": {"title": "Behaviour-coupled navigational coding", "families": []},
}


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("hypothesis", choices=sorted(HYPOTHESES), help="which hypothesis to run")
    ap.add_argument("--sessions", type=int, default=None, help="smoke-run limit on sessions")
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-excluded", action="store_true")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--all-rois", action="store_true", help="include dendrite/artefact ROIs")
    ap.add_argument("--signal", default="dff", choices=["dff", "event_masks", "spikes"])
    ap.add_argument("--features", type=Path, default=None, help="h6: existing cell table")
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--no-gbm", dest="gbm", action="store_false", help="h6: skip GBM classifier")
    ap.add_argument(
        "--no-glm-selection",
        dest="glm_selection",
        action="store_false",
        help="h8: skip forward selection (partial deviances only)",
    )
    ap.add_argument("--n-cells-matched", type=int, default=None, help="h7: matched cell count")
    ap.add_argument("--dry-run", action="store_true", help="validate metadata, write nothing")
    return ap


def dry_run(args: argparse.Namespace) -> dict:
    """Validate inputs and describe what a run would do without touching S3."""
    meta = select_sessions(
        load_metadata(), args.sessions, args.include_excluded, args.primary_only
    )
    summary = {
        "hypothesis": args.hypothesis,
        "title": HYPOTHESES[args.hypothesis]["title"],
        "n_sessions": int(len(meta)),
        "n_animals": int(meta["animal_id"].nunique()),
        "sessions_by_celltype": meta["celltype"].value_counts().to_dict(),
        "animals_by_celltype": meta.groupby("celltype")["animal_id"].nunique().to_dict(),
        "out_dir": str(args.out / args.hypothesis),
        "n_perms": args.n_perms,
    }
    log.info("dry run: %s", json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network entry point
    args = _build_arg_parser().parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    from run_celltype_hypotheses import RUNNERS  # noqa: PLC0415

    runner = RUNNERS[args.hypothesis]
    runner(args, iter_sessions(args), args.out / args.hypothesis)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
