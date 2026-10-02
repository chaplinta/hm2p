#!/usr/bin/env python3
"""Query public Allen Institute datasets for Penk expression in mouse retrosplenial cortex.

Runs on a short-lived EC2 instance (``scripts/launch_allen_penk_ec2.py``); the
devcontainer cannot reach the Allen hosts or the us-west-2 ABC Atlas bucket.
Three independent parts, each wrapped so a failure in one does not stop the
others. Every invocation updates ``manifest.json`` in the output directory
(status, outputs with size and row count, errors) and, when ``--s3-uri`` is
given, uploads the outputs, the manifest and the log after each part.

Part A -- whole-mouse-brain 10x scRNA-seq (WMB-10Xv2/v3/Multi), cells dissected
from RSP regions of interest; per-cell expression of a fixed gene list and
per-subclass/supertype/cluster Penk summaries.
    Yao Z, et al. 2023. "A high-resolution transcriptomic and spatial atlas of
    cell types in the whole mouse brain." Nature 624:317-332.
    doi:10.1038/s41586-023-06812-z

Part B -- MERFISH spatial atlases registered to the Allen CCFv3, cells in
RSPagl/RSPd/RSPv with layer substructure; Penk-positive fraction by
subdivision x layer x subclass.
    Yao Z, et al. 2023 (above), dataset MERFISH-C57BL6J-638850.
    Zhang M, et al. 2023. "Molecularly defined and spatially resolved cell
    atlas of the whole mouse brain." Nature 624:343-354.
    doi:10.1038/s41586-023-06808-9 (datasets Zhuang-ABCA-1..4)

Part C -- Patch-seq / intrinsic electrophysiology (best effort, URLs logged):
    Scala F, et al. 2021. "Phenotypic variation of transcriptomic cell types in
    mouse motor cortex." Nature 598:144-150. doi:10.1038/s41586-020-2907-3
    https://github.com/berenslab/mini-atlas
    Gouwens NW, et al. 2020. "Integrated morphoelectric and transcriptomic
    classification of cortical GABAergic cells." Cell 183:935-953.
    doi:10.1016/j.cell.2020.09.057
    Allen Cell Types Database (Cre-line labelled cells, including
    Penk-IRES2-Cre-neo): Gouwens NW, et al. 2019. "Classification of
    electrophysiological and morphological neuron types in the mouse visual
    cortex." Nature Neuroscience 22:1182-1195. doi:10.1038/s41593-019-0417-0

Data access: Allen Brain Cell Atlas, ``abc_atlas_access``
(https://github.com/AllenInstitute/abc_atlas_access), public bucket
``s3://allen-brain-cell-atlas`` (us-west-2).

Statistics are non-parametric (Mann-Whitney U, Cliff's delta,
Benjamini-Hochberg FDR).

Usage (on the instance)
-----------------------
    python allen_penk_query.py --parts A B C --outdir /opt/hm2p/allen_out \\
        --cache-dir /opt/hm2p/abc_cache --s3-uri s3://hm2p-derivatives/allen/penk
    python allen_penk_query.py --write-done /tmp/done.json --parts A B C --outdir ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("allen_penk")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENES: tuple[str, ...] = (
    "Penk",
    "Cxcl14",
    "Calb1",
    "Slc17a7",
    "Camk2a",
    "Gad1",
    "Gad2",
    "Vip",
    "Sst",
    "Pvalb",
    "Lamp5",
    "Rorb",
    "Cux2",
    "Otof",
    "Rbp4",
    "Fezf2",
    "Oprd1",
    "Oprm1",
    "Oprk1",
    "Pdyn",
    "Tac1",
    "Cck",
    "Nnat",
    "Tshz2",
    "Rxfp1",
    "Scnn1a",
    "Kcnc1",
    "Kcnc2",
    "Kcna1",
    "Kcna4",
    "Hcn1",
    "Hcn2",
    "Cacna1h",
)
PENK = "Penk"
ALL_PARTS: tuple[str, ...] = ("A", "B", "C")
MANIFEST_NAME = "manifest.json"
LOG_NAME = "allen_penk_query.log"

# Part A (WMB 10x)
TENX_META_DIR = "WMB-10X"
TAXONOMY_DIR = "WMB-taxonomy"
TAXONOMY_PIVOT = "cluster_to_cluster_annotation_membership_pivoted"
TAXONOMY_LEVELS: tuple[str, ...] = ("class", "subclass", "supertype", "cluster")
RSP_TOKEN = "RSP"
# Subclass names for upper-layer intratelencephalic glutamatergic neurons,
# e.g. "007 L2/3 IT CTX Glut", "L2/3 IT RSP Glut".
L23_IT_PATTERN = re.compile(r"L2/3 IT", re.IGNORECASE)
GLUT_PATTERN = re.compile(r"Glut", re.IGNORECASE)
SUMMARY_LEVELS: tuple[tuple[str, ...], ...] = (
    ("subclass",),
    ("supertype",),
    ("cluster",),
    ("library_method", "subclass"),
)
ROW_CHUNK = 50_000

# Part B (MERFISH)
MERFISH_DATASETS: tuple[str, ...] = (
    "MERFISH-C57BL6J-638850",
    "Zhuang-ABCA-1",
    "Zhuang-ABCA-2",
    "Zhuang-ABCA-3",
    "Zhuang-ABCA-4",
)
CCF_DIR = "Allen-CCF-2020"
CCF_PARCELLATION = "parcellation_to_parcellation_term_membership_acronym"
RSP_STRUCTURES: tuple[str, ...] = ("RSPagl", "RSPd", "RSPv")
LAYER_PATTERN = re.compile(r"(1|2/3|4|5|6a|6b)$")

# Part C (patch-seq / ephys)
MINI_ATLAS_REPO = "https://github.com/berenslab/mini-atlas.git"
MINI_ATLAS_RAW = "https://raw.githubusercontent.com/berenslab/mini-atlas/{branch}/data/{name}"
MINI_ATLAS_FILES: tuple[str, ...] = (
    "m1_patchseq_meta_data.csv",
    "m1_patchseq_ephys_features.csv",
    "m1_patchseq_exon_counts.csv.gz",
    "m1_patchseq_intron_counts.csv.gz",
)
# Pages / directory listings scraped for Allen patch-seq download links.
ALLEN_PATCHSEQ_DISCOVERY_URLS: tuple[str, ...] = (
    "https://portal.brain-map.org/explore/classes/multimodal-characterization/"
    "multimodal-characterization-mouse-visual-cortex",
    "https://portal.brain-map.org/explore/classes/multimodal-characterization",
    "https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/"
    "processed/analysis/20200611/",
    "https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/"
    "processed/analysis/",
    "https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/",
)
ALLEN_LINK_PATTERN = re.compile(r"(meta|ephys|feature|count|cpm|patch)", re.IGNORECASE)
ALLEN_MAX_DOWNLOAD_BYTES = 2_000_000_000
CTDB_URL = (
    "https://api.brain-map.org/api/v2/data/query.json?criteria="
    "model::ApiCellTypesSpecimenDetail,"
    "rma::criteria,[donor__species$eq'Mus musculus'],"
    "rma::options[num_rows$eqall]"
)
IT_GROUPS: tuple[str, ...] = ("L2/3 IT", "L4/5 IT", "L5 IT")
TOP_QUANTILE = 0.75
MIN_GROUP_N = 3

# Canonical intrinsic-property names -> regex over lower-cased column names.
# Order matters: the first column matching a pattern is used.
EPHYS_PATTERNS: dict[str, str] = {
    "input_resistance": r"input.?resist|^ef__ri$|\bri\b",
    "rheobase": r"rheobase|threshold_i_long_square",
    "ap_width": r"ap.?(half.?)?width|half.?width|width",
    "upstroke_downstroke_ratio": r"upstroke.?downstroke",
    "adaptation_index": r"adapt",
    "max_firing": r"max.*(firing|number of ap|rate|spike)|avg_firing_rate",
    "fi_slope": r"f.?i.?(curve)?.?slope",
    "sag": r"\bsag",
    "resting_vm": r"resting|vrest|v_rest",
    "latency": r"latency",
    "tau": r"time.?constant|^ef__tau$|\btau\b",
    "ap_threshold": r"ap.?threshold|threshold_v",
    "ahp": r"ahp|afterhyperpol|fast_trough",
    "ap_amplitude": r"ap.?amplitude|peak_v",
}


# ---------------------------------------------------------------------------
# Manifest / part bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class OutputRecord:
    """One output file written by a part."""

    file: str
    size_bytes: int
    n_rows: int | None
    s3_uri: str | None = None


@dataclass
class PartResult:
    """Outcome of one part: status, timing, outputs, notes, and any error."""

    part: str
    status: str = "pending"
    started: str | None = None
    finished: str | None = None
    duration_s: float | None = None
    outputs: list[OutputRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None

    def note(self, msg: str) -> None:
        """Record a note in the manifest and the log."""
        log.info("[%s] %s", self.part, msg)
        self.notes.append(msg)


@dataclass
class Context:
    """Run configuration shared by all parts."""

    outdir: Path
    cache_dir: Path
    s3_uri: str | None = None
    genes: tuple[str, ...] = GENES
    merfish_datasets: tuple[str, ...] = MERFISH_DATASETS
    keep_downloads: bool = False
    max_files: int | None = None


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_part(name: str, fn: Callable[[Context, PartResult], None], ctx: Context) -> PartResult:
    """Run one part, catching any exception so later parts still run.

    Parameters
    ----------
    name : str
        Part label ("A", "B", "C").
    fn : callable
        ``fn(ctx, result)``; appends outputs/notes to ``result``.
    ctx : Context
        Shared configuration.

    Returns
    -------
    PartResult
        ``status`` is "ok" or "failed"; outputs written before a failure are kept.
    """
    result = PartResult(part=name, started=_now())
    t0 = time.time()
    log.info("=== part %s start ===", name)
    try:
        fn(ctx, result)
        result.status = "ok"
    except Exception as exc:  # noqa: BLE001 - parts are isolated by design
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback = traceback.format_exc()
        log.error("part %s failed: %s\n%s", name, result.error, result.traceback)
    result.finished = _now()
    result.duration_s = round(time.time() - t0, 1)
    log.info("=== part %s %s (%.0f s) ===", name, result.status, result.duration_s)
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    """Load an existing manifest, or return an empty one."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("manifest %s unreadable; starting a new one", path)
    return {"created": _now(), "parts": {}}


def update_manifest(manifest: dict[str, Any], result: PartResult) -> dict[str, Any]:
    """Insert/replace the entry for ``result.part`` and stamp the update time."""
    manifest = dict(manifest)
    parts = dict(manifest.get("parts", {}))
    parts[result.part] = asdict(result)
    manifest["parts"] = parts
    manifest["updated"] = _now()
    manifest["genes"] = list(GENES)
    return manifest


def done_status(manifest: dict[str, Any], parts: Sequence[str]) -> dict[str, Any]:
    """Summarise a manifest into the completion marker.

    Returns ``status`` "ok" when every requested part succeeded, "partial" when
    some did, "failed" when none did.
    """
    entries = manifest.get("parts", {})
    per_part = {p: entries.get(p, {}).get("status", "missing") for p in parts}
    n_ok = sum(s == "ok" for s in per_part.values())
    if n_ok == len(parts) and parts:
        status = "ok"
    elif n_ok > 0:
        status = "partial"
    else:
        status = "failed"
    return {"status": status, "parts": per_part, "finished": _now()}


def write_table(df: pd.DataFrame, path: Path, result: PartResult) -> OutputRecord:
    """Write ``df`` as CSV (gzip if the name ends in .gz) and record it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.5g")
    rec = OutputRecord(file=path.name, size_bytes=path.stat().st_size, n_rows=int(len(df)))
    result.outputs.append(rec)
    log.info("[%s] wrote %s (%d rows, %d bytes)", result.part, path, len(df), rec.size_bytes)
    return rec


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` into (bucket, prefix without trailing slash)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, prefix.strip("/")


def s3_key(prefix: str, name: str) -> str:
    """Join an S3 prefix and a file name."""
    return f"{prefix}/{name}" if prefix else name


def upload_files(paths: Iterable[Path], s3_uri: str | None) -> dict[str, str]:  # pragma: no cover
    """Upload files to ``s3_uri``; failures are logged, never raised."""
    if not s3_uri:
        return {}
    import boto3

    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    done: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        key = s3_key(prefix, p.name)
        try:
            s3.upload_file(str(p), bucket, key)
            done[p.name] = f"s3://{bucket}/{key}"
        except Exception as exc:  # noqa: BLE001
            log.warning("upload of %s failed: %s", p, exc)
    return done


# ---------------------------------------------------------------------------
# Pure transforms (unit-tested with synthetic data)
# ---------------------------------------------------------------------------


def select_genes_present(available: Iterable[str], requested: Sequence[str]) -> tuple[list, list]:
    """Split ``requested`` genes into those present in ``available`` and those missing.

    Order of ``requested`` is preserved.
    """
    avail = set(available)
    present = [g for g in requested if g in avail]
    missing = [g for g in requested if g not in avail]
    return present, missing


def resolve_gene_columns(
    var: pd.DataFrame,
    genes: Sequence[str],
    id_to_symbol: dict[str, str] | None = None,
) -> tuple[list[int], list[str], list[str]]:
    """Find column positions of ``genes`` in an AnnData ``var`` table.

    Symbols are taken from ``var['gene_symbol']`` when present, else by mapping
    the var index through ``id_to_symbol`` (e.g. Ensembl id -> symbol), else
    the var index itself is treated as symbols.

    Returns
    -------
    positions, present, missing
        Integer column positions (in ``genes`` order) for present genes.
    """
    if "gene_symbol" in var.columns:
        symbols = var["gene_symbol"].astype(str).to_numpy()
    elif id_to_symbol:
        symbols = np.array([id_to_symbol.get(str(i), str(i)) for i in var.index])
    else:
        symbols = var.index.astype(str).to_numpy()
    first: dict[str, int] = {}
    for i, s in enumerate(symbols):
        first.setdefault(s, i)
    present = [g for g in genes if g in first]
    missing = [g for g in genes if g not in first]
    return [first[g] for g in present], present, missing


def rsp_roi_mask(roi: pd.Series, token: str = RSP_TOKEN) -> pd.Series:
    """Boolean mask of dissection ROI acronyms containing ``token`` (case-insensitive).

    Matches combined ROIs such as "RSP" or "ACA-RSP"; NaN is False.
    """
    return roi.astype("string").str.contains(token, case=False, regex=False).fillna(False)


def annotate_with_taxonomy(cells: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    """Join WMB taxonomy names (class/subclass/supertype/cluster) onto cells by cluster_alias.

    ``pivot`` may carry ``cluster_alias`` as a column or as its index. Aliases
    are compared as strings to avoid int/str dtype mismatches.
    """
    piv = pivot.reset_index() if "cluster_alias" not in pivot.columns else pivot.copy()
    if "cluster_alias" not in piv.columns:
        raise KeyError("taxonomy table has no cluster_alias")
    keep = ["cluster_alias"] + [c for c in (*TAXONOMY_LEVELS, "neurotransmitter") if c in piv]
    piv = piv[keep].copy()
    piv["cluster_alias"] = piv["cluster_alias"].astype(str)
    out = cells.copy()
    out["cluster_alias"] = out["cluster_alias"].astype(str)
    drop = [c for c in keep if c != "cluster_alias" and c in out.columns]
    return out.drop(columns=drop).merge(piv, on="cluster_alias", how="left")


def is_l23_it_like(subclass: object) -> bool:
    """True for glutamatergic L2/3 IT subclass names (e.g. "007 L2/3 IT CTX Glut")."""
    if not isinstance(subclass, str):
        return False
    return bool(L23_IT_PATTERN.search(subclass)) and bool(GLUT_PATTERN.search(subclass))


def tenx_directory_for_label(feature_matrix_label: str) -> str:
    """ABC data directory for a 10x feature-matrix label.

    "WMB-10Xv3-Isocortex-1" -> "WMB-10Xv3"; "WMB-10XMulti" -> "WMB-10XMulti".
    """
    m = re.match(r"^(WMB-10X(?:v2|v3|Multi))", feature_matrix_label)
    if not m:
        raise ValueError(f"unrecognised 10x feature_matrix_label {feature_matrix_label!r}")
    return m.group(1)


def pick_data_file(files: Iterable[str], label: str, kind: str = "log2") -> str | None:
    """Pick the expression file for ``label`` of type ``kind`` from a listing.

    Prefers an exact ``"{label}/{kind}"`` (with or without .h5ad), then any
    name containing both ``label`` and ``kind``.
    """
    files = list(files)
    exact = {f"{label}/{kind}", f"{label}/{kind}.h5ad", f"{label}-{kind}.h5ad"}
    for f in files:
        if f in exact:
            return f
    for f in files:
        if label in f and kind in f:
            return f
    return None


def parse_layer(substructure: object) -> str | None:
    """Cortical layer from a CCF substructure acronym ("RSPd2/3" -> "L2/3")."""
    if not isinstance(substructure, str):
        return None
    m = LAYER_PATTERN.search(substructure)
    return f"L{m.group(1)}" if m else None


def rsp_parcellation_mask(
    df: pd.DataFrame,
    structure_col: str = "parcellation_structure",
    substructure_col: str = "parcellation_substructure",
) -> pd.Series:
    """Cells whose CCF structure is RSPagl/RSPd/RSPv (or substructure starts with one)."""
    mask = pd.Series(False, index=df.index)
    if structure_col in df.columns:
        mask |= df[structure_col].astype("string").isin(RSP_STRUCTURES).fillna(False)
    if substructure_col in df.columns:
        sub = df[substructure_col].astype("string")
        for s in RSP_STRUCTURES:
            mask |= sub.str.startswith(s).fillna(False)
    return mask.astype(bool)


def rsp_structure_from_substructure(substructure: object) -> str | None:
    """RSP subdivision from a substructure acronym ("RSPagl5" -> "RSPagl")."""
    if not isinstance(substructure, str):
        return None
    for s in sorted(RSP_STRUCTURES, key=len, reverse=True):
        if substructure.startswith(s):
            return s
    return None


def _group_key(values: object) -> str:
    if isinstance(values, tuple):
        return " | ".join(str(v) for v in values)
    return str(values)


def penk_group_summary(
    df: pd.DataFrame,
    levels: Sequence[tuple[str, ...]] = SUMMARY_LEVELS,
    genes: Sequence[str] = GENES,
    gene: str = PENK,
) -> pd.DataFrame:
    """Per-group Penk summary with Penk+ vs Penk- means of every gene.

    Penk+ is defined as log2 expression > 0. For each grouping in ``levels``
    returns one row per group: n cells, n and fraction Penk+, mean log2 Penk,
    the modal subclass, whether that subclass is glutamatergic L2/3 IT-like,
    and ``{g}_mean_penk_pos`` / ``{g}_mean_penk_neg`` for each gene. A pooled
    row (level ``"l23_it_like_pooled"``) covers all L2/3 IT-like cells.
    """
    if gene not in df.columns:
        raise KeyError(f"{gene} not in expression columns")
    data = df.reset_index(drop=True)
    pos = (data[gene] > 0).to_numpy()
    genes_here = [g for g in genes if g in data.columns]

    def row_for(level: str, group: str, idx: np.ndarray) -> dict[str, Any]:
        sub = data.iloc[idx]
        p = pos[idx]
        subclass = None
        if "subclass" in sub.columns and sub["subclass"].notna().any():
            subclass = sub["subclass"].mode(dropna=True).iloc[0]
        row: dict[str, Any] = {
            "level": level,
            "group": group,
            "modal_subclass": subclass,
            "l23_it_like": is_l23_it_like(subclass),
            "n_cells": int(len(sub)),
            "n_penk_pos": int(p.sum()),
            "frac_penk_pos": float(p.mean()) if len(sub) else np.nan,
            "mean_log2_penk": float(sub[gene].mean()) if len(sub) else np.nan,
        }
        for g in genes_here:
            vals = sub[g].to_numpy(dtype=float)
            row[f"{g}_mean_penk_pos"] = float(vals[p].mean()) if p.any() else np.nan
            row[f"{g}_mean_penk_neg"] = float(vals[~p].mean()) if (~p).any() else np.nan
        return row

    rows: list[dict[str, Any]] = []
    for cols in levels:
        if not all(c in data.columns for c in cols):
            continue
        key = list(cols) if len(cols) > 1 else cols[0]
        for name, idx in data.groupby(key, sort=True, dropna=True).indices.items():
            rows.append(row_for("x".join(cols), _group_key(name), np.asarray(idx)))
    if "subclass" in data.columns:
        l23 = np.flatnonzero(data["subclass"].map(is_l23_it_like).to_numpy(dtype=bool))
        if len(l23):
            rows.append(row_for("l23_it_like_pooled", "L2/3 IT-like (pooled)", l23))
    return pd.DataFrame(rows)


def positive_fraction_summary(
    df: pd.DataFrame,
    keys: Sequence[str],
    gene: str = PENK,
    all_label: str = "ALL",
    marginal_key: str | None = "subclass",
) -> pd.DataFrame:
    """Counts and fraction of ``gene`` > 0 cells grouped by ``keys``.

    When ``marginal_key`` is one of ``keys``, rows pooled over it are added
    with that column set to ``all_label``.
    """
    data = df.copy()
    data["_pos"] = data[gene] > 0

    def agg(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        g = frame.groupby(by, dropna=False, sort=True)
        out = g.agg(
            n_cells=("_pos", "size"),
            n_penk_pos=("_pos", "sum"),
            mean_log2_penk=(gene, "mean"),
        ).reset_index()
        out["frac_penk_pos"] = out["n_penk_pos"] / out["n_cells"]
        return out

    parts = [agg(data, list(keys))]
    if marginal_key and marginal_key in keys:
        rest = [k for k in keys if k != marginal_key]
        m = agg(data, rest)
        m[marginal_key] = all_label
        parts.append(m[parts[0].columns])
    out = pd.concat(parts, ignore_index=True)
    out["n_penk_pos"] = out["n_penk_pos"].astype(int)
    return out


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta, P(X > Y) - P(X < Y), ignoring NaNs.

    Cliff N. 1993. "Dominance statistics: Ordinal analyses to answer ordinal
    questions." Psychological Bulletin 114:494-509. doi:10.1037/0033-2909.114.3.494
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    xa = xa[~np.isnan(xa)]
    ya = np.sort(ya[~np.isnan(ya)])
    if len(xa) == 0 or len(ya) == 0:
        return float("nan")
    n_less = np.searchsorted(ya, xa, side="left")  # y < x
    n_greater = len(ya) - np.searchsorted(ya, xa, side="right")  # y > x
    return float((n_less.sum() - n_greater.sum()) / (len(xa) * len(ya)))


def bh_fdr(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaNs are passed through.

    Benjamini Y, Hochberg Y. 1995. "Controlling the false discovery rate."
    J R Stat Soc B 57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
    """
    pa = np.asarray(p, dtype=float)
    out = np.full_like(pa, np.nan)
    ok = ~np.isnan(pa)
    m = int(ok.sum())
    if m == 0:
        return out
    vals = pa[ok]
    order = np.argsort(vals)
    ranked = vals[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """Counts-per-million with cells as rows and genes as columns."""
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0) * 1e6


def orient_cells_by_genes(counts: pd.DataFrame, marker: str = PENK) -> pd.DataFrame:
    """Return a count table with cells as rows, genes as columns.

    The orientation is detected from where ``marker`` appears (index or columns).
    """
    if marker in counts.columns:
        return counts
    if marker in counts.index:
        return counts.T
    raise KeyError(f"{marker} found in neither rows nor columns of the count table")


def map_ephys_columns(
    columns: Iterable[str], patterns: dict[str, str] | None = None
) -> dict[str, str]:
    """Map canonical intrinsic-property names to the first matching source column.

    Each source column is used at most once (patterns are applied in order).
    """
    patterns = patterns or EPHYS_PATTERNS
    cols = list(columns)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for canon, pat in patterns.items():
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if c in used:
                continue
            if rx.search(str(c).strip().lower()):
                mapping[canon] = c
                used.add(c)
                break
    return mapping


def ttype_group(ttype: object) -> str | None:
    """Coarse IT group from a transcriptomic type / family name.

    "L2/3 IT_1" -> "L2/3 IT"; "L4/5 IT_2" -> "L4/5 IT"; "L5 IT_1" -> "L5 IT";
    "L6 IT Car3" -> "L6 IT"; others -> None.
    """
    if not isinstance(ttype, str):
        return None
    for grp in ("L2/3 IT", "L4/5 IT", "L5 IT", "L6 IT"):
        if grp.lower() in ttype.lower():
            return grp
    return None


def penk_split(values: pd.Series, method: str = "nonzero", q: float = TOP_QUANTILE) -> pd.Series:
    """Label cells Penk+ (True) / Penk- (False).

    ``method="nonzero"``: Penk > 0. ``method="topq"``: Penk at or above the
    ``q`` quantile *and* > 0 (so an all-zero group has no Penk+ cells).
    """
    v = pd.to_numeric(values, errors="coerce")
    if method == "nonzero":
        lab = v > 0
    elif method == "topq":
        thr = v.quantile(q) if v.notna().any() else np.inf
        lab = (v >= thr) & (v > 0)
    else:
        raise ValueError(f"unknown method {method!r}")
    return lab.where(v.notna())


def compare_penk_groups(
    df: pd.DataFrame,
    features: Sequence[str],
    split_cols: Sequence[str],
    dataset_col: str = "dataset",
    group_col: str = "ttype_group",
    min_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """Penk+ vs Penk- comparison of each feature per dataset x group x split.

    Mann-Whitney U (two-sided) and Cliff's delta (positive = Penk+ larger);
    BH-FDR within each dataset x split. Groups with fewer than ``min_n`` cells
    on either side get NaN statistics. A pooled ``"ALL"`` group is included.
    """
    from scipy.stats import mannwhitneyu

    rows: list[dict[str, Any]] = []
    for split in split_cols:
        for ds, dsub in df.groupby(dataset_col, sort=True):
            groups = [("ALL", dsub)] + [(g, s) for g, s in dsub.groupby(group_col, sort=True)]
            for grp, sub in groups:
                lab = sub[split]
                for feat in features:
                    if feat not in sub.columns:
                        continue
                    vals = pd.to_numeric(sub[feat], errors="coerce")
                    x = vals[lab == True].dropna().to_numpy()  # noqa: E712 - NA-aware
                    y = vals[lab == False].dropna().to_numpy()  # noqa: E712
                    if len(x) + len(y) == 0:
                        continue
                    p = np.nan
                    if len(x) >= min_n and len(y) >= min_n:
                        p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                    rows.append(
                        {
                            "dataset": ds,
                            "group": grp,
                            "split": split,
                            "feature": feat,
                            "n_penk_pos": len(x),
                            "n_penk_neg": len(y),
                            "median_penk_pos": float(np.median(x)) if len(x) else np.nan,
                            "median_penk_neg": float(np.median(y)) if len(y) else np.nan,
                            "mannwhitney_p": p,
                            "cliffs_delta": cliffs_delta(x, y) if len(x) and len(y) else np.nan,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh"] = np.nan
    for _, idx in out.groupby(["dataset", "split"]).groups.items():
        out.loc[idx, "q_bh"] = bh_fdr(out.loc[idx, "mannwhitney_p"].to_numpy())
    return out


def extract_links(html: str, base_url: str, pattern: re.Pattern[str] | None = None) -> list[str]:
    """Absolute href targets in ``html`` pointing to data files (csv/zip/tar/...).

    Optionally filtered by ``pattern`` on the link text. Order preserved, de-duplicated.
    """
    from urllib.parse import urljoin

    exts = (".csv", ".csv.gz", ".zip", ".tar", ".tar.gz", ".tgz", ".xlsx", ".feather", ".h5ad")
    out: list[str] = []
    for href in re.findall(r"""href\s*=\s*["']([^"']+)["']""", html, flags=re.IGNORECASE):
        url = urljoin(base_url, href)
        low = url.lower().split("?")[0]
        if not low.endswith(exts):
            continue
        if pattern is not None and not pattern.search(url):
            continue
        if url not in out:
            out.append(url)
    return out


def ctdb_to_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Allen Cell Types DB specimen records -> table with Penk Cre-line label.

    Keeps spiny (putative excitatory) cells. ``penk_cre`` is True when the
    transgenic line name contains "Penk".
    """
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    if "tag__dendrite_type" in df.columns:
        df = df[df["tag__dendrite_type"].astype(str).str.lower() == "spiny"].copy()
    line = df.get("line_name", pd.Series("", index=df.index)).astype(str)
    df["penk_cre"] = line.str.contains("Penk", case=False)
    layer = df.get("structure__layer", pd.Series("", index=df.index)).astype(str)
    df["ttype_group"] = "layer " + layer
    df["dataset"] = "allen_ctdb_cre"
    id_col = "specimen__id" if "specimen__id" in df.columns else df.columns[0]
    df["cell_id"] = df[id_col].astype(str)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ABC atlas helpers (network; run on the instance only)
# ---------------------------------------------------------------------------


def open_abc_cache(cache_dir: Path) -> Any:  # pragma: no cover - network
    """Open the ABC Atlas S3 cache and load the latest manifest."""
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    cache_dir.mkdir(parents=True, exist_ok=True)
    abc = AbcProjectCache.from_s3_cache(cache_dir)
    with contextlib.suppress(Exception):
        abc.load_latest_manifest()
    with contextlib.suppress(Exception):
        log.info("ABC manifest: %s", abc.current_manifest)
    return abc


def log_inventory(
    abc: Any, directories: Iterable[str] | None = None
) -> dict[str, Any]:  # pragma: no cover
    """Log (and return) directories plus metadata/data files of the selected ones."""
    inv: dict[str, Any] = {}
    try:
        dirs = abc.list_directories  # property in current releases; method in some older ones
        dirs = list(dirs() if callable(dirs) else dirs)
    except Exception as exc:  # noqa: BLE001
        log.warning("list_directories failed: %s", exc)
        dirs = []
    log.info("ABC directories (%d): %s", len(dirs), dirs)
    inv["directories"] = dirs
    for d in directories or []:
        entry: dict[str, Any] = {}
        for kind, fn in (("metadata", "list_metadata_files"), ("data", "list_data_files")):
            try:
                entry[kind] = list(getattr(abc, fn)(d))
            except Exception as exc:  # noqa: BLE001
                entry[kind] = f"error: {exc}"
            log.info("  %s %s files: %s", d, kind, entry[kind])
        inv[d] = entry
    return inv


def get_metadata(
    abc: Any, directory: str, names: Sequence[str], **kw: Any
) -> pd.DataFrame:  # pragma: no cover
    """Load the first metadata file in ``names`` that exists in ``directory``."""
    errors = []
    for name in names:
        try:
            df = abc.get_metadata_dataframe(directory=directory, file_name=name, **kw)
            log.info(
                "loaded %s/%s: %d rows, columns %s", directory, name, len(df), list(df.columns)
            )
            return df
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            log.info("metadata %s/%s unavailable: %s", directory, name, exc)
    raise FileNotFoundError(f"none of {names} in {directory}: {errors}")


def read_expression_subset(
    h5ad_path: Path,
    cell_labels: Sequence[str],
    genes: Sequence[str],
    id_to_symbol: dict[str, str] | None = None,
    chunk: int = ROW_CHUNK,
) -> tuple[pd.DataFrame, list[str]]:  # pragma: no cover - needs anndata + real files
    """Read ``genes`` for ``cell_labels`` from an h5ad opened in backed mode.

    Rows are read in sorted chunks to bound memory. Returns a cells x genes
    DataFrame indexed by cell label and the list of missing genes.
    """
    import anndata
    import scipy.sparse as sp

    ad = anndata.read_h5ad(h5ad_path, backed="r")
    try:
        gpos, present, missing = resolve_gene_columns(ad.var, genes, id_to_symbol)
        log.info(
            "%s: %d x %d; genes present %d, missing %s",
            h5ad_path.name,
            ad.n_obs,
            ad.n_vars,
            len(present),
            missing,
        )
        rows = ad.obs_names.get_indexer(pd.Index([str(c) for c in cell_labels]))
        rows = np.unique(rows[rows >= 0])
        blocks = []
        for s in range(0, len(rows), chunk):
            r = rows[s : s + chunk]
            try:
                X = ad[r, gpos].X
            except Exception as exc:  # noqa: BLE001 - fall back to raw row read
                log.info("view slicing failed (%s); reading rows then columns", exc)
                X = ad.X[r]
                X = X[:, gpos]
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            blocks.append(
                pd.DataFrame(X.astype(np.float32), index=ad.obs_names[r], columns=present)
            )
            log.info("  rows %d-%d of %d", s, s + len(r), len(rows))
        out = pd.concat(blocks) if blocks else pd.DataFrame(columns=present, dtype=np.float32)
    finally:
        with contextlib.suppress(Exception):
            ad.file.close()
    return out, missing


def _drop_download(path: Path, keep: bool) -> None:  # pragma: no cover
    if keep:
        return
    try:
        path.unlink()
        log.info("deleted cached %s", path)
    except OSError as exc:
        log.info("could not delete %s: %s", path, exc)


def _gene_symbol_map(gene_meta: pd.DataFrame | None) -> dict[str, str] | None:
    """gene_identifier -> gene_symbol from an ABC gene table (None if unavailable)."""
    if gene_meta is None:
        return None
    g = gene_meta.reset_index()
    if "gene_identifier" in g.columns and "gene_symbol" in g.columns:
        return dict(
            zip(g["gene_identifier"].astype(str), g["gene_symbol"].astype(str), strict=False)
        )
    return None


# ---------------------------------------------------------------------------
# Part A -- WMB 10x scRNA-seq
# ---------------------------------------------------------------------------


def part_a(ctx: Context, result: PartResult) -> None:  # pragma: no cover - network
    """RSP cells from WMB-10X: per-cell gene table and Penk summaries."""
    abc = open_abc_cache(ctx.cache_dir)
    log_inventory(abc, [TENX_META_DIR, TAXONOMY_DIR, "WMB-10Xv2", "WMB-10Xv3", "WMB-10XMulti"])

    cells = get_metadata(abc, TENX_META_DIR, ["cell_metadata"], dtype={"cell_label": str})
    if "cell_label" not in cells.columns:
        cells = cells.reset_index()
    roi_col = "region_of_interest_acronym"
    if roi_col not in cells.columns:
        raise KeyError(f"{roi_col} not in cell_metadata columns {list(cells.columns)}")
    rois = sorted(cells[roi_col].dropna().astype(str).unique())
    log.info("all ROI acronyms (%d): %s", len(rois), rois)
    mask = rsp_roi_mask(cells[roi_col])
    rsp_rois = sorted(cells.loc[mask, roi_col].astype(str).unique())
    result.note(f"ROI values containing RSP: {rsp_rois}")
    rsp = cells.loc[mask].copy()
    result.note(f"{len(rsp)} RSP cells of {len(cells)} 10x cells")
    del cells
    if rsp.empty:
        raise RuntimeError("no 10x cells with an RSP region of interest")

    pivot = get_metadata(abc, TAXONOMY_DIR, [TAXONOMY_PIVOT], keep_default_na=False)
    rsp = annotate_with_taxonomy(rsp, pivot)
    result.note(f"cells without taxonomy annotation: {int(rsp['subclass'].isna().sum())}")

    gene_meta = None
    with contextlib.suppress(Exception):
        gene_meta = get_metadata(abc, TENX_META_DIR, ["gene"])
    id_to_symbol = _gene_symbol_map(gene_meta)
    if gene_meta is not None and "gene_symbol" in gene_meta.columns:
        _, missing = select_genes_present(gene_meta["gene_symbol"].astype(str), ctx.genes)
        result.note(f"genes missing from 10x gene table: {missing}")

    fm_col = "feature_matrix_label"
    labels = rsp[fm_col].value_counts()
    result.note(f"RSP cells per expression file: {labels.to_dict()}")
    expr_blocks = []
    for i, (label, n) in enumerate(labels.items()):
        if ctx.max_files is not None and i >= ctx.max_files:
            result.note(f"max_files={ctx.max_files}: skipping remaining files")
            break
        directory = (
            str(rsp.loc[rsp[fm_col] == label, "dataset_label"].iloc[0])
            if "dataset_label" in rsp.columns
            else tenx_directory_for_label(str(label))
        )
        files = list(abc.list_data_files(directory))
        fname = pick_data_file(files, str(label), "log2")
        if fname is None:
            result.note(f"no log2 file for {label} in {directory}: {files}")
            continue
        log.info("file %s/%s (%d RSP cells): downloading", directory, fname, n)
        path = Path(abc.get_data_path(directory=directory, file_name=fname))
        cell_ids = rsp.loc[rsp[fm_col] == label, "cell_label"].astype(str)
        expr, missing = read_expression_subset(path, cell_ids, ctx.genes, id_to_symbol)
        if missing:
            result.note(f"{label}: genes missing {missing}")
        expr_blocks.append(expr)
        _drop_download(path, ctx.keep_downloads)

    if not expr_blocks:
        raise RuntimeError("no expression read for RSP cells")
    expr = pd.concat(expr_blocks)
    expr.index.name = "cell_label"
    out = rsp.rename(columns={roi_col: "roi"})
    keep = [
        "cell_label",
        "library_method",
        "roi",
        "feature_matrix_label",
        "dataset_label",
        "donor_label",
        "donor_sex",
        "anatomical_division_label",
        "cluster_alias",
        *TAXONOMY_LEVELS,
        "neurotransmitter",
    ]
    out = out[[c for c in keep if c in out.columns]]
    out = out.merge(expr.reset_index(), on="cell_label", how="inner")
    result.note(f"{len(out)} RSP cells with expression")
    write_table(out, ctx.outdir / "rsp_10x_cells.csv.gz", result)
    write_table(
        penk_group_summary(out, genes=ctx.genes),
        ctx.outdir / "rsp_10x_penk_by_cluster.csv",
        result,
    )


# ---------------------------------------------------------------------------
# Part B -- MERFISH
# ---------------------------------------------------------------------------


def _merfish_parcellation(
    abc: Any, dataset: str, cells: pd.DataFrame
) -> pd.DataFrame:  # pragma: no cover
    """Attach parcellation_{division,structure,substructure} to MERFISH cells."""
    have = {"parcellation_structure", "parcellation_substructure"}
    if have <= set(cells.columns):
        return cells
    ccf_dir = f"{dataset}-CCF"
    try:
        ann = get_metadata(
            abc, ccf_dir, ["cell_metadata_with_parcellation_annotation"], dtype={"cell_label": str}
        )
        if "cell_label" not in ann.columns:
            ann = ann.reset_index()
        cols = ["cell_label"] + [c for c in ann.columns if c.startswith("parcellation_")]
        cols += [c for c in ("x_ccf", "y_ccf", "z_ccf") if c in ann.columns]
        return cells.merge(ann[cols], on="cell_label", how="left", suffixes=("", "_ccf"))
    except FileNotFoundError:
        pass
    coords = get_metadata(
        abc, ccf_dir, ["ccf_coordinates", "reconstructed_coordinates"], dtype={"cell_label": str}
    )
    if "cell_label" not in coords.columns:
        coords = coords.reset_index()
    coords = coords.rename(columns={c: f"{c}_ccf" for c in ("x", "y", "z") if c in coords.columns})
    parc = get_metadata(abc, CCF_DIR, [CCF_PARCELLATION])
    if "parcellation_index" not in parc.columns:
        parc = parc.reset_index()
    parc = parc.rename(
        columns={c: f"parcellation_{c}" for c in parc.columns if c != "parcellation_index"}
    )
    keep = ["cell_label", "parcellation_index"] + [c for c in coords.columns if c.endswith("_ccf")]
    merged = coords[keep].merge(parc, on="parcellation_index", how="left")
    return cells.merge(merged, on="cell_label", how="left")


def part_b(ctx: Context, result: PartResult) -> None:  # pragma: no cover - network
    """RSP cells from MERFISH atlases with layer substructure and Penk summary."""
    abc = open_abc_cache(ctx.cache_dir)
    dirs = [d for ds in ctx.merfish_datasets for d in (ds, f"{ds}-CCF")] + [CCF_DIR, TAXONOMY_DIR]
    log_inventory(abc, dirs)
    pivot = get_metadata(abc, TAXONOMY_DIR, [TAXONOMY_PIVOT], keep_default_na=False)

    tables = []
    for i, ds in enumerate(ctx.merfish_datasets):
        if ctx.max_files is not None and i >= ctx.max_files:
            break
        try:
            cells = get_metadata(abc, ds, ["cell_metadata"], dtype={"cell_label": str})
            if "cell_label" not in cells.columns:
                cells = cells.reset_index()
            cells = _merfish_parcellation(abc, ds, cells)
            mask = rsp_parcellation_mask(cells)
            subs = sorted(
                cells.loc[mask, "parcellation_substructure"].dropna().astype(str).unique()
            )
            result.note(f"{ds}: {int(mask.sum())} RSP cells; substructures {subs}")
            rsp = cells.loc[mask].copy()
            del cells
            if rsp.empty:
                continue
            if "cluster_alias" in rsp.columns:
                rsp = annotate_with_taxonomy(rsp, pivot)
            gene_meta = None
            with contextlib.suppress(Exception):
                gene_meta = get_metadata(abc, ds, ["gene"])
            if gene_meta is not None and "gene_symbol" in gene_meta.columns:
                present, missing = select_genes_present(
                    gene_meta["gene_symbol"].astype(str), ctx.genes
                )
                result.note(f"{ds}: panel genes present {present}; missing {missing}")
            files = list(abc.list_data_files(ds))
            fm = str(rsp["feature_matrix_label"].iloc[0]) if "feature_matrix_label" in rsp else ds
            fname = pick_data_file(files, fm, "log2") or pick_data_file(files, "", "log2")
            if fname is None:
                result.note(f"{ds}: no log2 expression file in {files}")
                continue
            path = Path(abc.get_data_path(directory=ds, file_name=fname))
            expr, missing = read_expression_subset(
                path, rsp["cell_label"], ctx.genes, _gene_symbol_map(gene_meta)
            )
            _drop_download(path, ctx.keep_downloads)
            rsp["dataset"] = ds
            rsp["layer"] = rsp["parcellation_substructure"].map(parse_layer)
            rsp["rsp_subdivision"] = rsp["parcellation_substructure"].map(
                rsp_structure_from_substructure
            )
            keep = [
                "cell_label",
                "dataset",
                "brain_section_label",
                "x",
                "y",
                "z",
                "x_ccf",
                "y_ccf",
                "z_ccf",
                "x_reconstructed",
                "y_reconstructed",
                "z_reconstructed",
                "parcellation_division",
                "parcellation_structure",
                "parcellation_substructure",
                "rsp_subdivision",
                "layer",
                "cluster_alias",
                *TAXONOMY_LEVELS,
            ]
            rsp = rsp[[c for c in keep if c in rsp.columns]]
            expr.index.name = "cell_label"
            tables.append(rsp.merge(expr.reset_index(), on="cell_label", how="inner"))
        except Exception as exc:  # noqa: BLE001 - one dataset failing must not stop others
            result.note(f"{ds}: failed {type(exc).__name__}: {exc}")
            log.debug(traceback.format_exc())

    if not tables:
        raise RuntimeError("no MERFISH dataset produced RSP cells")
    out = pd.concat(tables, ignore_index=True)
    write_table(out, ctx.outdir / "rsp_merfish_cells.csv.gz", result)
    if PENK in out.columns:
        with_penk = out[out[PENK].notna()]
        keys = ["dataset", "rsp_subdivision", "layer", "subclass"]
        summary = positive_fraction_summary(with_penk.fillna({"layer": "NA"}), keys)
        write_table(summary, ctx.outdir / "rsp_merfish_penk_summary.csv", result)
    else:
        result.note("Penk not in any MERFISH panel read; no summary written")


# ---------------------------------------------------------------------------
# Part C -- Patch-seq / intrinsic electrophysiology
# ---------------------------------------------------------------------------


class UrlLog:
    """Records every URL tried with its outcome."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, url: str, status: str, detail: str = "", size: int | None = None) -> None:
        """Append one URL attempt and log it."""
        log.info("URL %s -> %s %s", url, status, detail)
        self.rows.append({"url": url, "status": status, "detail": detail, "size_bytes": size})

    def frame(self) -> pd.DataFrame:
        """URL attempts as a table."""
        return pd.DataFrame(self.rows, columns=["url", "status", "detail", "size_bytes"])


def http_get(
    url: str,
    dest: Path | None,
    urls: UrlLog,
    timeout: int = 120,
    max_bytes: int = ALLEN_MAX_DOWNLOAD_BYTES,
) -> bytes | Path | None:  # pragma: no cover
    """GET ``url`` to ``dest`` (or return bytes when ``dest`` is None); log the outcome."""
    req = urllib.request.Request(url, headers={"User-Agent": "hm2p-allen-penk/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            size = resp.headers.get("Content-Length")
            size_i = int(size) if size and size.isdigit() else None
            if size_i is not None and size_i > max_bytes:
                urls.add(url, f"{resp.status} skipped", f"too large ({size_i} bytes)", size_i)
                return None
            if dest is None:
                data = resp.read()
                urls.add(url, str(resp.status), "", len(data))
                return data
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            urls.add(url, str(resp.status), f"saved {dest.name}", dest.stat().st_size)
            return dest
    except urllib.error.HTTPError as exc:
        urls.add(url, str(exc.code), str(exc.reason))
    except Exception as exc:  # noqa: BLE001
        urls.add(url, "error", f"{type(exc).__name__}: {exc}")
    return None


def _read_table_any(path: Path) -> pd.DataFrame:  # pragma: no cover
    """Read a CSV/TSV with delimiter sniffing."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if head.startswith(b"version https://git-lfs"):
        raise ValueError(f"{path.name} is a git-lfs pointer, not data")
    sep = "\t" if b"\t" in head.split(b"\n", 1)[0] else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _find(files: Iterable[Path], *tokens: str) -> Path | None:
    """First path whose lower-cased name contains all ``tokens``."""
    for f in sorted(files):
        name = f.name.lower()
        if all(t in name for t in tokens):
            return f
    return None


def _id_column(df: pd.DataFrame) -> str:
    """Best guess at the cell-identifier column of a patch-seq table."""
    for c in df.columns:
        if str(c).strip().lower() in {
            "cell",
            "cell id",
            "cell_id",
            "cell_specimen_id",
            "specimen_id",
            "sample",
            "unnamed: 0",
        }:
            return c
    return df.columns[0]


def _mini_atlas(
    ctx: Context, result: PartResult, urls: UrlLog
) -> pd.DataFrame | None:  # pragma: no cover
    """Scala et al. 2021 MOp patch-seq: metadata + ephys + Penk CPM."""
    root = ctx.cache_dir / "patchseq" / "mini-atlas"
    data_dir = root / "data"
    if not data_dir.exists():
        proc = subprocess.run(
            ["git", "clone", "-q", "--depth", "1", MINI_ATLAS_REPO, str(root)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        urls.add(
            MINI_ATLAS_REPO, "git clone rc=" + str(proc.returncode), proc.stderr.strip()[:300]
        )
    files = list(data_dir.glob("*")) if data_dir.exists() else []
    log.info("mini-atlas data files: %s", [f.name for f in files])
    for name in MINI_ATLAS_FILES:
        f = data_dir / name
        if f.exists() and f.stat().st_size > 1024:
            continue
        for branch in ("master", "main"):
            if http_get(MINI_ATLAS_RAW.format(branch=branch, name=name), f, urls):
                break
    files = list(data_dir.glob("*")) if data_dir.exists() else []
    meta_f = _find(files, "meta")
    ephys_f = _find(files, "ephys")
    exon_f = _find(files, "exon", "count")
    intron_f = _find(files, "intron", "count")
    if not (meta_f and ephys_f and exon_f):
        result.note(f"mini-atlas: missing files (meta={meta_f}, ephys={ephys_f}, exon={exon_f})")
        return None
    meta = _read_table_any(meta_f)
    ephys = _read_table_any(ephys_f)
    result.note(f"mini-atlas meta columns: {list(meta.columns)}")
    result.note(f"mini-atlas ephys columns: {list(ephys.columns)}")
    counts = _read_table_any(exon_f)
    counts = counts.set_index(counts.columns[0])
    counts = orient_cells_by_genes(counts)
    if intron_f is not None:
        try:
            intr = _read_table_any(intron_f)
            intr = orient_cells_by_genes(intr.set_index(intr.columns[0]))
            counts = counts.add(intr.reindex_like(counts).fillna(0), fill_value=0)
            result.note("mini-atlas: CPM from exon + intron counts")
        except Exception as exc:  # noqa: BLE001
            result.note(f"mini-atlas: intron counts unused ({exc})")
    counts = counts.apply(pd.to_numeric, errors="coerce").fillna(0)
    penk_cpm = cpm(counts)[PENK]
    penk_counts = counts[PENK]

    mid, eid = _id_column(meta), _id_column(ephys)
    result.note(f"mini-atlas id columns: meta={mid!r}, ephys={eid!r}")
    ttype_col = next(
        (c for c in meta.columns if re.search(r"rna type|t-type|ttype", str(c), re.I)), None
    )
    layer_col = next((c for c in meta.columns if re.search(r"layer", str(c), re.I)), None)
    mapping = map_ephys_columns([c for c in ephys.columns if c != eid])
    result.note(f"mini-atlas ephys column mapping: {mapping}")
    df = pd.DataFrame({"cell_id": meta[mid].astype(str)})
    df["dataset"] = "scala2021_mop"
    df["ttype"] = meta[ttype_col].astype(str) if ttype_col else None
    df["layer"] = meta[layer_col].astype(str) if layer_col else None
    df["penk_counts"] = df["cell_id"].map(penk_counts)
    df["penk_cpm"] = df["cell_id"].map(penk_cpm)
    eph = ephys.rename(columns={v: k for k, v in mapping.items()})
    eph["cell_id"] = ephys[eid].astype(str)
    df = df.merge(eph[["cell_id", *mapping.keys()]], on="cell_id", how="left")
    df["ttype_group"] = df["ttype"].map(ttype_group)
    return df


def _allen_patchseq_discovery(
    ctx: Context, result: PartResult, urls: UrlLog
) -> pd.DataFrame:  # pragma: no cover
    """Scrape known Allen pages/listings for patch-seq data links; download small CSVs.

    Returns a table of discovered links. Parsing of these files into the
    common schema is deferred until their structure is known (second pass).
    """
    found: list[dict[str, Any]] = []
    dest_dir = ctx.cache_dir / "patchseq" / "allen"
    for page in ALLEN_PATCHSEQ_DISCOVERY_URLS:
        raw = http_get(page, None, urls, timeout=60)
        if not isinstance(raw, bytes):
            continue
        links = extract_links(raw.decode("utf-8", "replace"), page, ALLEN_LINK_PATTERN)
        result.note(f"{page}: {len(links)} candidate data links")
        for link in links:
            row: dict[str, Any] = {"source_page": page, "url": link, "downloaded": False}
            if link.lower().split("?")[0].endswith((".csv", ".csv.gz")):
                dest = dest_dir / Path(link.split("?")[0]).name
                got = http_get(link, dest, urls, max_bytes=200_000_000)
                if isinstance(got, Path):
                    row["downloaded"] = True
                    with contextlib.suppress(Exception):
                        head = _read_table_any(got).head(0)
                        row["columns"] = "|".join(map(str, head.columns))[:2000]
            found.append(row)
    return pd.DataFrame(found)


def _allen_ctdb(result: PartResult, urls: UrlLog) -> pd.DataFrame | None:  # pragma: no cover
    """Allen Cell Types Database specimen ephys features with Cre-line labels."""
    raw = http_get(CTDB_URL, None, urls, timeout=300)
    if not isinstance(raw, bytes):
        return None
    payload = json.loads(raw)
    records = payload.get("msg", [])
    if not isinstance(records, list) or not records:
        result.note(f"CTDB: no records ({str(payload)[:300]})")
        return None
    df = ctdb_to_table(records)
    mapping = map_ephys_columns([c for c in df.columns if str(c).startswith("ef__")])
    penk_lines = sorted(df.loc[df["penk_cre"], "line_name"].astype(str).unique())
    result.note(f"CTDB: {len(df)} spiny cells; Penk lines {penk_lines}")
    result.note(f"CTDB ephys column mapping: {mapping}")
    out = df.rename(columns={v: k for k, v in mapping.items()})
    keep = [
        "cell_id",
        "dataset",
        "line_name",
        "structure__acronym",
        "structure__layer",
        "ttype_group",
        "penk_cre",
        *mapping.keys(),
    ]
    return out[[c for c in keep if c in out.columns]]


def part_c(ctx: Context, result: PartResult) -> None:  # pragma: no cover - network
    """Patch-seq / intrinsic ephys split by Penk (best effort; every URL logged)."""
    urls = UrlLog()
    frames: list[pd.DataFrame] = []
    splits: list[str] = []
    try:
        try:
            mini = _mini_atlas(ctx, result, urls)
        except Exception as exc:  # noqa: BLE001
            result.note(f"mini-atlas failed: {type(exc).__name__}: {exc}")
            mini = None
        if mini is not None:
            mini = mini[mini["ttype_group"].isin(IT_GROUPS)].copy()
            mini["penk_pos_nonzero"] = penk_split(mini["penk_cpm"], "nonzero")
            mini["penk_pos_topq"] = pd.concat(
                [penk_split(s["penk_cpm"], "topq") for _, s in mini.groupby("ttype_group")]
            ).reindex(mini.index)
            result.note(f"mini-atlas IT cells: {mini['ttype_group'].value_counts().to_dict()}")
            frames.append(mini)
            splits += ["penk_pos_nonzero", "penk_pos_topq"]
        try:
            disc = _allen_patchseq_discovery(ctx, result, urls)
            if not disc.empty:
                write_table(disc, ctx.outdir / "patchseq_allen_discovered_links.csv", result)
        except Exception as exc:  # noqa: BLE001
            result.note(f"Allen patch-seq discovery failed: {type(exc).__name__}: {exc}")
        try:
            ctdb = _allen_ctdb(result, urls)
        except Exception as exc:  # noqa: BLE001
            result.note(f"CTDB failed: {type(exc).__name__}: {exc}")
            ctdb = None
        if ctdb is not None and not ctdb.empty:
            frames.append(ctdb)
            splits.append("penk_cre")
    finally:
        write_table(urls.frame(), ctx.outdir / "patchseq_url_log.csv", result)

    if not frames:
        raise RuntimeError("no patch-seq / ephys source could be read")
    cells = pd.concat(frames, ignore_index=True, sort=False)
    write_table(cells, ctx.outdir / "patchseq_penk_ephys.csv", result)
    features = [k for k in EPHYS_PATTERNS if k in cells.columns]
    summary = compare_penk_groups(cells, features, split_cols=sorted(set(splits)))
    write_table(summary, ctx.outdir / "patchseq_penk_summary.csv", result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PART_FUNCS: dict[str, Callable[[Context, PartResult], None]] = {
    "A": part_a,
    "B": part_b,
    "C": part_c,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", default=list(ALL_PARTS), choices=ALL_PARTS)
    ap.add_argument("--outdir", type=Path, default=Path("allen_penk_out"))
    ap.add_argument("--cache-dir", type=Path, default=Path("abc_cache"))
    ap.add_argument("--s3-uri", default=None, help="e.g. s3://hm2p-derivatives/allen/penk")
    ap.add_argument("--merfish-datasets", nargs="+", default=list(MERFISH_DATASETS))
    ap.add_argument("--keep-downloads", action="store_true", help="keep cached h5ad files")
    ap.add_argument("--max-files", type=int, default=None, help="debug: limit files per part")
    ap.add_argument(
        "--write-done",
        type=Path,
        default=None,
        help="only summarise manifest.json into this completion marker",
    )
    return ap


def _setup_logging(outdir: Path) -> Path:  # pragma: no cover
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / LOG_NAME
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(log_path, mode="a")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return log_path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    args = _build_arg_parser().parse_args(argv)
    manifest_path = args.outdir / MANIFEST_NAME
    if args.write_done is not None:
        done = done_status(load_manifest(manifest_path), args.parts)
        args.write_done.write_text(json.dumps(done, indent=2))
        print(json.dumps(done))
        return 0
    log_path = _setup_logging(args.outdir)
    ctx = Context(
        outdir=args.outdir,
        cache_dir=args.cache_dir,
        s3_uri=args.s3_uri,
        merfish_datasets=tuple(args.merfish_datasets),
        keep_downloads=args.keep_downloads,
        max_files=args.max_files,
    )
    for part in args.parts:
        result = run_part(part, PART_FUNCS[part], ctx)
        uploaded = upload_files([args.outdir / o.file for o in result.outputs], ctx.s3_uri)
        for o in result.outputs:
            o.s3_uri = uploaded.get(o.file)
        manifest = update_manifest(load_manifest(manifest_path), result)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        upload_files([manifest_path, log_path], ctx.s3_uri)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
