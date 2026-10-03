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

Continuum vs discrete analyses (all parts)
------------------------------------------
Asks whether Penk marks a discrete subset of L2/3 IT neurons or varies along a
continuum. Per part, alongside the outputs above:

- A: Penk zero fraction and fixed-bin histograms (RSP L2/3 IT and the L2/3 IT
  subclass in each isocortex dissection ROI); Silverman's test (k=1) on
  expressing cells; share of Penk rank variance explained by cluster and
  supertype (Kruskal-Wallis H -> epsilon-squared) with per-taxon medians and
  expressing fractions; Spearman correlation of Penk with every gene expressed
  in >= 5 % of RSP L2/3 IT cells (BH-FDR; also partial on genes detected per
  cell), with listed L2/L3 depth markers reported explicitly; Penk by RSP
  subregion and by isocortex ROI.
- B: for MERFISH RSP L2/3 IT cells, Spearman of Penk with layer (ordinal),
  CCF coordinates, distance from the midline and any depth column present;
  Penk by RSP subdivision; Silverman's test; a subsampled spatial map table.
- C: Spearman of log2(Penk CPM + 1) with each ephys feature (BH-FDR) and
  quantile shift functions (Penk > 0 vs Penk = 0) for IT patch-seq cells.

Caveat: log2(CPM + 1) values of low-count genes are close to a lattice (one,
two, three UMIs at a given library size). Silverman's test can return small p
values from that lattice alone, so per-chemistry rows and the histograms
should be read alongside the p values.

These use ``hm2p.analysis.continuum`` (numpy only), imported from the
repository's ``src`` directory (the launcher sets ``PYTHONPATH``).

    Silverman BW. 1981. "Using kernel density estimates to investigate
    multimodality." J R Stat Soc B 43:97-99.
    doi:10.1111/j.2517-6161.1981.tb01155.x
    Rousselet GA, Pernet CR, Wilcox RR. 2017. "Beyond differences in means:
    robust graphical methods to compare two groups in neuroscience." Eur J
    Neurosci 46:1738-1748. doi:10.1111/ejn.13610
    Spearman C. 1904. "The proof and measurement of association between two
    things." Am J Psychol 15:72-101. doi:10.2307/1412159
    Kruskal WH, Wallis WA. 1952. "Use of ranks in one-criterion variance
    analysis." J Am Stat Assoc 47:583-621. doi:10.1080/01621459.1952.10483441
    Tomczak M, Tomczak E. 2014. "The need to report effect size estimates
    revisited. An overview of some recommended measures of effect size."
    Trends Sport Sci 21:19-25. (epsilon-squared for Kruskal-Wallis)

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

# Shared continuum tools (numpy only). On the instance the repository's src/
# directory is on PYTHONPATH; the import failure is kept and reported when a
# continuum analysis is requested, so the other outputs are still written.
try:
    from hm2p.analysis import continuum as _continuum_module

    _CONTINUUM_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover - depends on the environment
    _continuum_module = None
    _CONTINUUM_IMPORT_ERROR = _exc

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

# Continuum vs discrete analyses
# Genes reported as candidate upper-layer (L2 vs L3) depth markers; their
# Spearman rho with Penk is reported whether or not they pass the 5 % filter.
DEPTH_MARKERS: tuple[str, ...] = (
    "Cux2",
    "Otof",
    "Cdh13",
    "Rorb",
    "Lamp5",
    "Stard8",
    "Calb1",
    "Ccbe1",
    "Cxcl14",
)
SEED = 0
HIST_EDGES: np.ndarray = np.round(np.arange(0.0, 16.0 + 0.25, 0.25), 2)  # log2 expression
SILVERMAN_MAX_N = 3000
SILVERMAN_N_BOOT = 200
CORR_MIN_FRAC = 0.05  # genes expressed in >= 5 % of cells enter the correlation family
CORR_TOP_N = 50
CORR_MAX_CELLS = 10_000  # RSP L2/3 IT cells read with all genes (seeded subsample)
CORR_MIN_CELLS = 50  # minimum cells per chemistry stratum for gene correlations
AREA_MAX_CELLS = 5_000  # L2/3 IT cells per non-RSP isocortex ROI (seeded subsample)
MAP_MAX_CELLS = 20_000
MIN_TAXON_N = 5  # taxa with fewer cells are left out of Kruskal-Wallis
SHIFT_MIN_N = 5
SHIFT_N_BOOT = 1000
ISOCORTEX_COL = "anatomical_division_label"
LAYER_ORDINAL: dict[str, int] = {"L1": 1, "L2/3": 2, "L4": 3, "L5": 4, "L6a": 5, "L6b": 6}
# CCFv3 (10 um, 1140 voxels left-right): midline at 5.7 mm.
CCF_MIDLINE_Z_MM = 5.7
# Assumed axis meaning of coordinate columns (CCFv3 convention for *_ccf).
COORD_AXES: dict[str, str] = {
    "x_ccf": "CCF anterior-posterior",
    "y_ccf": "CCF dorsal-ventral",
    "z_ccf": "CCF left-right",
    "ccf_ml_from_midline": "CCF distance from midline",
    "layer_ordinal": "CCF layer (L1=1 ... L6b=6)",
    "x_reconstructed": "reconstructed x (section plane)",
    "y_reconstructed": "reconstructed y (section plane)",
    "z_reconstructed": "reconstructed z (section order)",
}
DEPTH_COLUMN_PATTERN = re.compile(r"depth|pia", re.IGNORECASE)


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


def write_json(obj: Any, path: Path, result: PartResult) -> OutputRecord:
    """Write ``obj`` as indented JSON (non-finite floats -> null) and record it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2, default=str))
    rec = OutputRecord(file=path.name, size_bytes=path.stat().st_size, n_rows=None)
    result.outputs.append(rec)
    log.info("[%s] wrote %s (%d bytes)", result.part, path, rec.size_bytes)
    return rec


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None and numpy scalars with Python ones."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


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


def var_symbols(var: pd.DataFrame, id_to_symbol: dict[str, str] | None = None) -> np.ndarray:
    """Gene symbols of an AnnData ``var`` table, one per column.

    From ``var['gene_symbol']`` when present, else the var index mapped through
    ``id_to_symbol``, else the var index itself.
    """
    if "gene_symbol" in var.columns:
        return var["gene_symbol"].astype(str).to_numpy()
    if id_to_symbol:
        return np.array([id_to_symbol.get(str(i), str(i)) for i in var.index])
    return var.index.astype(str).to_numpy()


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
    symbols = var_symbols(var, id_to_symbol)
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


def alias_keys(alias: pd.Series) -> pd.Series:
    """cluster_alias values as strings, with integral floats written as integers (1.0 -> "1").

    A missing value in an integer column makes pandas read it as float; this
    keeps such aliases matching the taxonomy table.
    """
    num = pd.to_numeric(alias, errors="coerce")
    integral = num.notna() & (num == num.round())
    out = alias.astype(str)
    out[integral] = num[integral].astype("int64").astype(str)
    return out


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
    piv["cluster_alias"] = alias_keys(piv["cluster_alias"])
    out = cells.copy()
    out["cluster_alias"] = alias_keys(out["cluster_alias"])
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
# Continuum vs discrete helpers (pure; unit-tested with synthetic data)
# ---------------------------------------------------------------------------


def continuum_module() -> Any:
    """Return ``hm2p.analysis.continuum`` or raise an ImportError that says how to fix it."""
    if _continuum_module is None:
        raise ImportError(
            f"hm2p.analysis.continuum is not importable ({_CONTINUUM_IMPORT_ERROR}); "
            "put the repository's src/ directory on PYTHONPATH "
            "(the EC2 launcher exports PYTHONPATH=/opt/hm2p/repo/src)"
        )
    return _continuum_module


def run_substeps(result: PartResult, steps: Sequence[tuple[str, Callable[[], None]]]) -> None:
    """Run independent analysis steps; a failing step is noted and the rest still run.

    Raises
    ------
    RuntimeError
        After all steps ran, if any failed (so the part is marked failed while
        keeping every output written).
    """
    failed: list[str] = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - steps are isolated by design
            failed.append(name)
            result.note(f"{name} failed: {type(exc).__name__}: {exc}")
            log.debug(traceback.format_exc())
    if failed:
        raise RuntimeError(f"continuum steps failed: {failed}")


def subsample_rows(
    df: pd.DataFrame, max_n: int, seed: int = SEED, by: str | Sequence[str] | None = None
) -> pd.DataFrame:
    """Seeded subsample to at most ``max_n`` rows (per group of ``by`` when given).

    Row order of the input is preserved.
    """
    if by is None:
        return df if len(df) <= max_n else df.sample(n=max_n, random_state=seed).sort_index()
    keys = [by] if isinstance(by, str) else list(by)
    parts = [
        g if len(g) <= max_n else g.sample(n=max_n, random_state=seed)
        for _, g in df.groupby(keys, sort=True, dropna=False)
    ]
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts).sort_index()


def subclass_by_alias(pivot: pd.DataFrame) -> dict[str, str]:
    """cluster_alias (as str) -> subclass name from the WMB taxonomy pivot table."""
    piv = pivot.reset_index() if "cluster_alias" not in pivot.columns else pivot
    if "cluster_alias" not in piv.columns or "subclass" not in piv.columns:
        raise KeyError("taxonomy table needs cluster_alias and subclass")
    return dict(zip(alias_keys(piv["cluster_alias"]), piv["subclass"].astype(str), strict=False))


def l23_it_alias_mask(cluster_alias: pd.Series, alias_to_subclass: dict[str, str]) -> pd.Series:
    """Cells whose cluster maps to a glutamatergic L2/3 IT subclass."""
    sub = alias_keys(cluster_alias).map(alias_to_subclass)
    return sub.map(is_l23_it_like).fillna(False).astype(bool)


def isocortex_mask(df: pd.DataFrame, col: str = ISOCORTEX_COL) -> pd.Series:
    """Cells whose anatomical division contains "Isocortex"; all True if ``col`` is absent."""
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return (
        df[col].astype("string").str.contains("Isocortex", case=False).fillna(False).astype(bool)
    )


def rsp_subregion_from_roi(roi: object) -> str | None:
    """RSP subregion named in a dissection ROI ("RSPv" -> "RSPv", "RSP" -> "RSP"); else None."""
    if not isinstance(roi, str) or RSP_TOKEN.lower() not in roi.lower():
        return None
    for s in sorted(RSP_STRUCTURES, key=len, reverse=True):
        if s.lower() in roi.lower():
            return s
    return RSP_TOKEN


def fixed_histogram(
    values: Sequence[float] | np.ndarray, edges: np.ndarray = HIST_EDGES
) -> pd.DataFrame:
    """Counts in fixed bins; values outside the edges go to the first/last bin, NaN dropped."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    e = np.asarray(edges, dtype=float)
    counts, _ = np.histogram(np.clip(v, e[0], e[-1]), bins=e)
    total = counts.sum()
    return pd.DataFrame(
        {
            "bin_lo": e[:-1],
            "bin_hi": e[1:],
            "count": counts.astype(int),
            "frac": counts / total if total else np.full(counts.shape, np.nan),
        }
    )


def expression_distribution(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Zero fraction and quantiles of one gene's log expression (zero = not detected)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    e = v[v > 0]
    n = int(v.size)

    def q(a: np.ndarray, p: float) -> float:
        return float(np.quantile(a, p)) if a.size else float("nan")

    return {
        "n": n,
        "n_zero": int(n - e.size),
        "n_expressing": int(e.size),
        "frac_zero": float((n - e.size) / n) if n else float("nan"),
        "frac_expressing": float(e.size / n) if n else float("nan"),
        "mean_all": float(v.mean()) if n else float("nan"),
        "median_all": q(v, 0.5),
        "median_expressing": q(e, 0.5),
        "q10_expressing": q(e, 0.1),
        "q25_expressing": q(e, 0.25),
        "q75_expressing": q(e, 0.75),
        "q90_expressing": q(e, 0.9),
    }


def silverman_expressing(
    values: Sequence[float] | np.ndarray,
    max_n: int = SILVERMAN_MAX_N,
    n_boot: int = SILVERMAN_N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Silverman's test (H0: at most one mode) on expressing cells (value > 0).

    Cells are subsampled (seeded) to ``max_n`` before the test.

    Silverman BW. 1981. "Using kernel density estimates to investigate
    multimodality." J R Stat Soc B 43:97-99. doi:10.1111/j.2517-6161.1981.tb01155.x
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    n_expr = int(v.size)
    if v.size > max_n:
        v = np.random.default_rng(seed).choice(v, size=max_n, replace=False)
    res = continuum_module().silverman_mode_test(v, k=1, n_boot=n_boot, seed=seed)
    return {
        "n_silverman": int(res["n"]),
        "silverman_h_crit": res["h_crit"],
        "silverman_p_unimodal": res["p"],
        "silverman_subsampled": bool(n_expr > max_n),
    }


def _groups(
    df: pd.DataFrame, keys: Sequence[str], pool_col: str | None = None, all_label: str = "ALL"
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """(key values, rows) for each group of ``keys``; [({}, df)] when ``keys`` is empty.

    When ``pool_col`` is one of ``keys``, groups pooled over it (with that key
    set to ``all_label``) are added after the per-group entries.
    """
    keys = list(keys)
    if not keys:
        return [({}, df)]
    out: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for name, sub in df.groupby(keys, sort=True, dropna=False):
        name_t = name if isinstance(name, tuple) else (name,)
        out.append((dict(zip(keys, name_t, strict=True)), sub))
    if pool_col is not None and pool_col in keys:
        rest = [k for k in keys if k != pool_col]
        for kv, sub in _groups(df, rest):
            out.append(({**kv, pool_col: all_label}, sub))
    return out


def penk_distribution_tables(
    df: pd.DataFrame,
    groupings: Sequence[Sequence[str]],
    scope: str,
    gene: str = PENK,
    edges: np.ndarray = HIST_EDGES,
    silverman: bool = True,
    max_n: int = SILVERMAN_MAX_N,
    n_boot: int = SILVERMAN_N_BOOT,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group expression distribution, Silverman test and fixed-bin histograms.

    Parameters
    ----------
    df : DataFrame
        One row per cell with ``gene`` (log expression) and grouping columns.
    groupings : sequence of key lists
        ``[]`` gives one pooled group; groupings whose columns are missing are skipped.
    scope : str
        Label written to every row (e.g. "RSP L2/3 IT").

    Returns
    -------
    summary, histogram
        ``summary``: one row per group (:func:`expression_distribution` plus
        Silverman columns). ``histogram``: long table with ``population`` "all"
        (every cell) or "expressing" (value > 0).
    """
    sum_rows: list[dict[str, Any]] = []
    hist_parts: list[pd.DataFrame] = []
    for keys in groupings:
        if not all(k in df.columns for k in keys):
            continue
        grouping = "x".join(keys) if keys else "pooled"
        for kv, sub in _groups(df, keys):
            label = _group_key(tuple(kv.values())) if kv else "ALL"
            vals = pd.to_numeric(sub[gene], errors="coerce").to_numpy(dtype=float)
            row: dict[str, Any] = {"scope": scope, "grouping": grouping, "group": label}
            row.update(expression_distribution(vals))
            if silverman:
                row.update(silverman_expressing(vals, max_n=max_n, n_boot=n_boot, seed=seed))
            sum_rows.append(row)
            for pop, v in (("all", vals), ("expressing", vals[vals > 0])):
                h = fixed_histogram(v, edges)
                h.insert(0, "population", pop)
                h.insert(0, "group", label)
                h.insert(0, "grouping", grouping)
                h.insert(0, "scope", scope)
                hist_parts.append(h)
    hist = pd.concat(hist_parts, ignore_index=True) if hist_parts else pd.DataFrame()
    return pd.DataFrame(sum_rows), hist


def group_expression_table(
    df: pd.DataFrame, keys: Sequence[str], gene: str = PENK, min_n: int = 1
) -> pd.DataFrame:
    """Per-group n, expressing fraction, medians, and share of all expressing cells.

    ``share_of_expressing`` is the group's expressing cells over all expressing
    cells in groups kept (n >= ``min_n``); a discrete marker concentrates this
    share in few groups.
    """
    rows: list[dict[str, Any]] = []
    for kv, sub in _groups(df, keys):
        vals = pd.to_numeric(sub[gene], errors="coerce").to_numpy(dtype=float)
        d = expression_distribution(vals)
        if d["n"] < min_n:
            continue
        rows.append({**kv, **d})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    tot = out["n_expressing"].sum()
    out["share_of_expressing"] = out["n_expressing"] / tot if tot else np.nan
    return out


def kruskal_epsilon_squared(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    min_n: int = MIN_TAXON_N,
) -> dict[str, Any]:
    """Kruskal-Wallis H (tie-corrected) and epsilon-squared = H / (n - 1).

    Epsilon-squared is the between-group share of the rank sum of squares
    (eta-squared computed on ranks), in [0, 1]. Groups with fewer than
    ``min_n`` finite values are dropped first. The p value is the chi-squared
    approximation with k - 1 degrees of freedom.

    References
    ----------
    Kruskal WH, Wallis WA. 1952. "Use of ranks in one-criterion variance
    analysis." J Am Stat Assoc 47:583-621. doi:10.1080/01621459.1952.10483441
    Tomczak M, Tomczak E. 2014. "The need to report effect size estimates
    revisited. An overview of some recommended measures of effect size."
    Trends Sport Sci 21:19-25.
    """
    from scipy.stats import chi2, rankdata

    v = np.asarray(values, dtype=float)
    g = pd.Series(np.asarray(groups, dtype=object))
    ok = np.isfinite(v) & g.notna().to_numpy()
    v, g = v[ok], g[ok].astype(str).to_numpy()
    counts = pd.Series(g).value_counts()
    kept = counts[counts >= min_n].index
    m = np.isin(g, np.asarray(kept, dtype=object))
    v, g = v[m], g[m]
    n, k = int(v.size), int(len(kept))
    res: dict[str, Any] = {
        "n": n,
        "k": k,
        "n_groups_dropped": int(len(counts) - k),
        "H": float("nan"),
        "df": k - 1,
        "p": float("nan"),
        "epsilon_sq": float("nan"),
    }
    if k < 2 or n < 3:
        return res
    r = rankdata(v)
    codes, _ = pd.factorize(g)
    rank_sums = np.bincount(codes, weights=r)
    sizes = np.bincount(codes)
    h = 12.0 / (n * (n + 1)) * np.sum(rank_sums**2 / sizes) - 3.0 * (n + 1)
    _, ties = np.unique(v, return_counts=True)
    corr = 1.0 - np.sum(ties.astype(float) ** 3 - ties) / (float(n) ** 3 - n)
    if corr <= 0:
        return res
    h /= corr
    res.update(H=float(h), p=float(chi2.sf(h, k - 1)), epsilon_sq=float(h / (n - 1)))
    return res


def taxonomy_variance_table(
    df: pd.DataFrame,
    factors: Sequence[str],
    scope: str,
    gene: str = PENK,
    strata_col: str | None = None,
    min_n: int = MIN_TAXON_N,
) -> pd.DataFrame:
    """Kruskal-Wallis epsilon-squared of ``gene`` by each factor, pooled and per stratum.

    Also reports how concentrated expressing cells are: the share of all
    expressing cells in the three groups holding most of them, and the range
    of per-group expressing fractions. Factors missing from ``df`` are skipped.
    """
    strata: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    if strata_col is not None and strata_col in df.columns:
        strata += [(str(s), sub) for s, sub in df.groupby(strata_col, sort=True)]
    rows: list[dict[str, Any]] = []
    for factor in factors:
        if factor not in df.columns:
            continue
        for stratum, sub in strata:
            vals = pd.to_numeric(sub[gene], errors="coerce").to_numpy(dtype=float)
            kw = kruskal_epsilon_squared(vals, sub[factor].to_numpy(), min_n=min_n)
            tab = group_expression_table(sub, [factor], gene, min_n=min_n)
            conc: dict[str, float] = {
                "top3_share_of_expressing": float("nan"),
                "max_group_frac_expressing": float("nan"),
                "min_group_frac_expressing": float("nan"),
            }
            if not tab.empty:
                conc = {
                    "top3_share_of_expressing": float(
                        tab["share_of_expressing"].nlargest(3).sum()
                    ),
                    "max_group_frac_expressing": float(tab["frac_expressing"].max()),
                    "min_group_frac_expressing": float(tab["frac_expressing"].min()),
                }
            rows.append(
                {
                    "scope": scope,
                    "factor": factor,
                    "stratum_col": strata_col if stratum != "ALL" else None,
                    "stratum": stratum,
                    **kw,
                    **conc,
                }
            )
    return pd.DataFrame(rows)


def _spearman_p(rho: np.ndarray, n: int, n_cov: int = 0) -> np.ndarray:
    """Two-sided p for Spearman rho via the t approximation with n - 2 - n_cov df."""
    from scipy.stats import t as t_dist

    r = np.asarray(rho, dtype=float)
    dof = n - 2 - n_cov
    if dof <= 0:
        return np.full(r.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(dof / (1.0 - r**2))
    p = 2.0 * t_dist.sf(np.abs(t), dof)
    return np.where(np.isfinite(r), p, np.nan)


def spearman_against(target: Sequence[float] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Spearman rho of ``target`` with each column of ``matrix`` (finite values only).

    Ties get average ranks; a constant column (or target) gives NaN.

    Spearman C. 1904. "The proof and measurement of association between two
    things." Am J Psychol 15:72-101. doi:10.2307/1412159
    """
    from scipy.stats import rankdata

    t = np.asarray(target, dtype=float)
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    if m.shape[0] != t.size:
        raise ValueError(f"target has {t.size} values, matrix has {m.shape[0]} rows")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(m))):
        raise ValueError("spearman_against needs finite values")
    rt = rankdata(t)
    rt = rt - rt.mean()
    rm = rankdata(m, axis=0)
    rm = rm - rm.mean(axis=0)
    num = rt @ rm
    den = np.sqrt((rt @ rt) * np.sum(rm * rm, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.clip(rho, -1.0, 1.0)


def spearman_rho_p(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, int]:
    """Spearman rho, two-sided p and n over pairs where both values are finite."""
    xa = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    ya = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    n = int(ok.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    rho = float(spearman_against(xa[ok], ya[ok])[0])
    return rho, float(_spearman_p(np.array([rho]), n)[0]), n


def partial_rho(r_xy: np.ndarray, r_xz: float, r_yz: np.ndarray) -> np.ndarray:
    """First-order partial correlation of x and y given z, from pairwise coefficients."""
    r_xy = np.asarray(r_xy, dtype=float)
    r_yz = np.asarray(r_yz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
        out = (r_xy - r_xz * r_yz) / den
    return np.where(den > 0, np.clip(out, -1.0, 1.0), np.nan)


def gene_correlations(
    matrix: Any,
    genes: Sequence[str],
    target: str = PENK,
    min_frac: float = CORR_MIN_FRAC,
    always_include: Sequence[str] = DEPTH_MARKERS,
    covariate: Sequence[float] | np.ndarray | None = None,
    chunk: int = 2000,
) -> pd.DataFrame:
    """Spearman rho of ``target`` with every other gene of a cells x genes matrix.

    Parameters
    ----------
    matrix : ndarray or scipy.sparse matrix
        Cells x genes, log expression (0 = not detected).
    genes : sequence of str
        Column names; the first occurrence of a duplicated name is used.
    min_frac : float
        Genes detected in at least this fraction of cells form the tested
        family (BH-FDR within it).
    always_include : sequence of str
        Genes also reported when below ``min_frac`` (``passes_min_frac`` False,
        no q value).
    covariate : array, optional
        Per-cell nuisance variable (e.g. genes detected); adds the partial
        Spearman rho of target and gene given the covariate.

    Returns
    -------
    DataFrame
        gene, frac_expressing, passes_min_frac, n_cells, rho, p, q_bh (and
        rho_partial, p_partial, q_bh_partial), sorted by rho descending.
    """
    import scipy.sparse as sp

    genes = [str(g) for g in genes]
    first: dict[str, int] = {}
    for i, g in enumerate(genes):
        first.setdefault(g, i)
    if target not in first:
        raise KeyError(f"{target} not among the genes")
    x = sp.csc_matrix(matrix) if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    n = int(x.shape[0])
    if sp.issparse(x):
        frac = np.asarray((x > 0).sum(axis=0)).ravel() / n
    else:
        frac = (x > 0).sum(axis=0) / n

    def dense_cols(idx: np.ndarray) -> np.ndarray:
        block: Any = x[:, idx]
        return block.toarray() if sp.issparse(block) else np.asarray(block)

    t = dense_cols(np.array([first[target]]))[:, 0].astype(float)
    extra = set(always_include)
    idx = np.array(
        [i for g, i in first.items() if g != target and (frac[i] >= min_frac or g in extra)],
        dtype=int,
    )
    rho = np.empty(idx.size)
    rho_gz = np.empty(idx.size)
    cov = None if covariate is None else np.asarray(covariate, dtype=float)
    r_tz = float(spearman_against(t, cov)[0]) if cov is not None else float("nan")
    for s in range(0, idx.size, chunk):
        block = dense_cols(idx[s : s + chunk]).astype(float)
        rho[s : s + chunk] = spearman_against(t, block)
        if cov is not None:
            rho_gz[s : s + chunk] = spearman_against(cov, block)
    passes = frac[idx] >= min_frac
    out = pd.DataFrame(
        {
            "gene": [genes[i] for i in idx],
            "frac_expressing": frac[idx],
            "passes_min_frac": passes,
            "n_cells": n,
            "rho": rho,
            "p": _spearman_p(rho, n),
        }
    )
    out["q_bh"] = np.nan
    out.loc[passes, "q_bh"] = bh_fdr(out.loc[passes, "p"].to_numpy())
    if cov is not None:
        rp = partial_rho(rho, r_tz, rho_gz)
        out["rho_partial"] = rp
        out["p_partial"] = _spearman_p(rp, n, n_cov=1)
        out["q_bh_partial"] = np.nan
        out.loc[passes, "q_bh_partial"] = bh_fdr(out.loc[passes, "p_partial"].to_numpy())
    return out.sort_values("rho", ascending=False, na_position="last").reset_index(drop=True)


def top_correlated_genes(
    corr: pd.DataFrame, n: int = CORR_TOP_N, col: str = "rho"
) -> pd.DataFrame:
    """Top ``n`` positive and ``n`` negative genes by ``col`` among the tested family."""
    fam = corr[corr["passes_min_frac"].astype(bool) & corr[col].notna()]
    pos = fam[fam[col] > 0].sort_values(col, ascending=False).head(n).copy()
    neg = fam[fam[col] < 0].sort_values(col, ascending=True).head(n).copy()
    pos.insert(0, "rank", np.arange(1, len(pos) + 1))
    neg.insert(0, "rank", np.arange(1, len(neg) + 1))
    pos.insert(0, "direction", "positive")
    neg.insert(0, "direction", "negative")
    return pd.concat([pos, neg], ignore_index=True)


def marker_correlations(
    corr: pd.DataFrame, markers: Sequence[str] = DEPTH_MARKERS
) -> pd.DataFrame:
    """One row per marker: ``in_data`` flag plus its correlation columns (NaN if absent)."""
    by_gene = corr.drop_duplicates("gene").set_index("gene")
    out = by_gene.reindex(list(markers))
    out.insert(0, "in_data", out.index.isin(by_gene.index))
    out.index.name = "gene"
    return out.reset_index()


def spearman_table(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
    group_cols: Sequence[str],
    pool_col: str | None = None,
    min_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """Spearman rho of ``x_col`` with each of ``y_cols`` per group, BH-FDR within group.

    Pairs with a non-finite value are dropped per variable; variables with
    fewer than ``min_n`` pairs get NaN statistics. With ``pool_col`` (one of
    ``group_cols``) rows pooled over it are added with that column "ALL".
    """
    rows: list[dict[str, Any]] = []
    present = [y for y in y_cols if y in df.columns and y != x_col]
    for fam, (kv, sub) in enumerate(_groups(df, group_cols, pool_col)):
        for y in present:
            rho, p, n = spearman_rho_p(sub[x_col], sub[y])
            if n < min_n:
                rho, p = float("nan"), float("nan")
            rows.append({**kv, "variable": y, "n": n, "rho": rho, "p": p, "_fam": fam})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_bh"] = np.nan
    for _, idx in out.groupby("_fam").groups.items():
        out.loc[idx, "q_bh"] = bh_fdr(out.loc[idx, "p"].to_numpy())
    return out.drop(columns="_fam")


def shift_table(
    df: pd.DataFrame,
    features: Sequence[str],
    split_col: str,
    group_cols: Sequence[str],
    pool_col: str | None = None,
    min_n: int = SHIFT_MIN_N,
    n_boot: int = SHIFT_N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Quantile shift function (split True minus split False) for each feature and group.

    Cell-level bootstrap band (no animal/donor grouping). Groups with fewer
    than ``min_n`` cells on either side are skipped.

    Rousselet GA, Pernet CR, Wilcox RR. 2017. "Beyond differences in means:
    robust graphical methods to compare two groups in neuroscience." Eur J
    Neurosci 46:1738-1748. doi:10.1111/ejn.13610
    """
    cont = continuum_module()
    rows: list[dict[str, Any]] = []
    for kv, sub in _groups(df, group_cols, pool_col):
        lab = sub[split_col]
        for feat in features:
            if feat not in sub.columns:
                continue
            vals = pd.to_numeric(sub[feat], errors="coerce")
            a = vals[lab == True].dropna().to_numpy(dtype=float)  # noqa: E712 - NA-aware
            b = vals[lab == False].dropna().to_numpy(dtype=float)  # noqa: E712
            if len(a) < min_n or len(b) < min_n:
                continue
            res = cont.shift_function(a, b, n_boot=n_boot, seed=seed)
            for q, d, lo, hi in zip(
                res["quantiles"], res["diff"], res["lo"], res["hi"], strict=True
            ):
                rows.append(
                    {
                        **kv,
                        "feature": feat,
                        "n_pos": len(a),
                        "n_neg": len(b),
                        "quantile": q,
                        "diff": d,
                        "lo": lo,
                        "hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def distance_from_midline(z: Sequence[float] | np.ndarray) -> np.ndarray:
    """|z - midline| for CCF left-right coordinates in mm (or um if values exceed 100)."""
    za = np.asarray(z, dtype=float)
    finite = za[np.isfinite(za)]
    midline = CCF_MIDLINE_Z_MM
    if finite.size and np.nanmax(np.abs(finite)) > 100:
        midline *= 1000.0
    return np.abs(za - midline)


def find_depth_columns(columns: Iterable[str]) -> list[str]:
    """Columns whose names mention depth or pia (e.g. normalised cortical depth)."""
    return [str(c) for c in columns if DEPTH_COLUMN_PATTERN.search(str(c))]


def add_spatial_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``layer_ordinal`` (from ``layer``) and ``ccf_ml_from_midline`` (from ``z_ccf``)."""
    out = df.copy()
    if "layer" in out.columns:
        out["layer_ordinal"] = out["layer"].map(LAYER_ORDINAL).astype(float)
    if "z_ccf" in out.columns:
        out["ccf_ml_from_midline"] = distance_from_midline(
            pd.to_numeric(out["z_ccf"], errors="coerce").to_numpy()
        )
    return out


def spatial_variables(columns: Iterable[str]) -> list[str]:
    """Spatial proxy columns present: depth columns, layer ordinal, coordinates."""
    cols = list(columns)
    found = find_depth_columns(cols)
    found += [c for c in COORD_AXES if c in cols and c not in found]
    return found


def align_sparse_blocks(
    blocks: Sequence[tuple[Any, Sequence[str], Sequence[str]]],
) -> tuple[Any, list[str], list[str]]:
    """Stack (matrix, cell labels, gene symbols) blocks on their shared genes.

    Genes are kept in the first block's order; a duplicated symbol uses its
    first column. Returns a CSR matrix, the cell labels and the gene symbols.
    """
    import scipy.sparse as sp

    if not blocks:
        raise ValueError("no blocks to align")

    def first_pos(symbols: Sequence[str]) -> dict[str, int]:
        pos: dict[str, int] = {}
        for i, s in enumerate(symbols):
            pos.setdefault(str(s), i)
        return pos

    maps = [first_pos(b[2]) for b in blocks]
    common = [g for g in maps[0] if all(g in m for m in maps[1:])]
    mats: list[Any] = []
    cells: list[str] = []
    for (mat, labels, _), m in zip(blocks, maps, strict=True):
        cols = np.array([m[g] for g in common], dtype=int)
        mats.append(sp.csr_matrix(mat)[:, cols])
        cells.extend(str(c) for c in labels)
    return sp.vstack(mats, format="csr"), cells, common


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


def read_expression_all_genes(
    h5ad_path: Path,
    cell_labels: Sequence[str],
    id_to_symbol: dict[str, str] | None = None,
    chunk: int = ROW_CHUNK,
) -> tuple[Any, list[str], list[str]]:  # pragma: no cover - needs anndata + real files
    """Read every gene for ``cell_labels`` from a backed h5ad as a CSR matrix.

    Returns (matrix, cell labels in matrix row order, gene symbols).
    """
    import anndata
    import scipy.sparse as sp

    ad = anndata.read_h5ad(h5ad_path, backed="r")
    try:
        symbols = [str(s) for s in var_symbols(ad.var, id_to_symbol)]
        rows = ad.obs_names.get_indexer(pd.Index([str(c) for c in cell_labels]))
        rows = np.unique(rows[rows >= 0])
        blocks = []
        for s in range(0, len(rows), chunk):
            r = rows[s : s + chunk]
            try:
                x = ad[r].X
            except Exception as exc:  # noqa: BLE001 - fall back to raw row read
                log.info("view slicing failed (%s); reading raw rows", exc)
                x = ad.X[r]
            blocks.append(sp.csr_matrix(x, dtype=np.float32))
        mat = sp.vstack(blocks, format="csr") if blocks else sp.csr_matrix((0, len(symbols)))
        labels = [str(c) for c in ad.obs_names[rows]]
        log.info("%s: all genes for %d cells (%d nnz)", h5ad_path.name, mat.shape[0], mat.nnz)
    finally:
        with contextlib.suppress(Exception):
            ad.file.close()
    return mat, labels, symbols


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


TENX_KEEP: tuple[str, ...] = (
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
)


def part_a(ctx: Context, result: PartResult) -> None:  # pragma: no cover - network
    """RSP cells from WMB-10X: per-cell gene table, Penk summaries, continuum analyses.

    Reads, from each expression file holding RSP cells: ``ctx.genes`` plus
    :data:`DEPTH_MARKERS` for RSP cells and for L2/3 IT cells of other
    isocortex ROIs in the same file (at most :data:`AREA_MAX_CELLS` per ROI),
    and every gene for RSP L2/3 IT cells (at most :data:`CORR_MAX_CELLS`).
    """
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
    result.note(f"{int(mask.sum())} RSP cells of {len(cells)} 10x cells")
    if not mask.any():
        raise RuntimeError("no 10x cells with an RSP region of interest")

    pivot = get_metadata(abc, TAXONOMY_DIR, [TAXONOMY_PIVOT], keep_default_na=False)
    # L2/3 IT cells of other isocortex ROIs (cross-area reference for the continuum analyses)
    area_mask = pd.Series(False, index=cells.index)
    try:
        if ISOCORTEX_COL not in cells.columns:
            result.note(f"{ISOCORTEX_COL} absent: L2/3 IT reference not restricted to isocortex")
        area_mask = (
            l23_it_alias_mask(cells["cluster_alias"], subclass_by_alias(pivot))
            & isocortex_mask(cells)
            & ~mask
        )
    except Exception as exc:  # noqa: BLE001 - the RSP outputs do not depend on this
        result.note(f"cross-area L2/3 IT selection failed: {type(exc).__name__}: {exc}")
    rsp = cells.loc[mask].copy()
    area = cells.loc[area_mask].copy()
    del cells
    rsp = annotate_with_taxonomy(rsp, pivot)
    result.note(f"cells without taxonomy annotation: {int(rsp['subclass'].isna().sum())}")

    gene_meta = None
    with contextlib.suppress(Exception):
        gene_meta = get_metadata(abc, TENX_META_DIR, ["gene"])
    id_to_symbol = _gene_symbol_map(gene_meta)
    read_genes = tuple(dict.fromkeys((*ctx.genes, *DEPTH_MARKERS)))
    if gene_meta is not None and "gene_symbol" in gene_meta.columns:
        _, missing = select_genes_present(gene_meta["gene_symbol"].astype(str), read_genes)
        result.note(f"genes missing from 10x gene table: {missing}")

    fm_col = "feature_matrix_label"
    labels = rsp[fm_col].value_counts()
    result.note(f"RSP cells per expression file: {labels.to_dict()}")
    if ctx.max_files is not None:
        labels = labels.iloc[: ctx.max_files]
        result.note(f"max_files={ctx.max_files}: reading {list(labels.index)}")

    # Cross-area cells: only from files read for RSP; seeded cap per ROI.
    n_area_all = area[roi_col].value_counts().to_dict() if len(area) else {}
    area = area[area[fm_col].isin(labels.index)] if fm_col in area.columns else area.iloc[0:0]
    area = subsample_rows(area, AREA_MAX_CELLS, SEED, by=roi_col)
    if len(area):
        area = annotate_with_taxonomy(area, pivot)
    result.note(
        f"L2/3 IT isocortex cells outside RSP: all files {n_area_all}; "
        f"read (files with RSP cells, <= {AREA_MAX_CELLS}/ROI) "
        f"{area[roi_col].value_counts().to_dict() if len(area) else {}}"
    )
    rsp_l23 = rsp[rsp["subclass"].map(is_l23_it_like).fillna(False).astype(bool)]
    corr_ids = set(subsample_rows(rsp_l23, CORR_MAX_CELLS, SEED)["cell_label"].astype(str))
    result.note(f"RSP L2/3 IT cells: {len(rsp_l23)}; read with all genes: {len(corr_ids)}")

    expr_blocks = []
    full_blocks: list[tuple[Any, list[str], list[str]]] = []
    for label, n in labels.items():
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
        cell_ids = pd.concat(
            [
                rsp.loc[rsp[fm_col] == label, "cell_label"],
                area.loc[area[fm_col] == label, "cell_label"]
                if len(area)
                else pd.Series([], dtype=str),
            ]
        ).astype(str)
        expr, missing = read_expression_subset(path, cell_ids, read_genes, id_to_symbol)
        if missing:
            result.note(f"{label}: genes missing {missing}")
        expr_blocks.append(expr)
        file_corr = [c for c in rsp.loc[rsp[fm_col] == label, "cell_label"].astype(str)
                     if c in corr_ids]  # fmt: skip
        if file_corr:
            try:
                full_blocks.append(read_expression_all_genes(path, file_corr, id_to_symbol))
            except Exception as exc:  # noqa: BLE001 - the gene-correlation step reports it
                result.note(f"{label}: all-gene read failed: {type(exc).__name__}: {exc}")
        _drop_download(path, ctx.keep_downloads)

    if not expr_blocks:
        raise RuntimeError("no expression read for RSP cells")
    expr = pd.concat(expr_blocks)
    expr = expr[~expr.index.duplicated(keep="first")]
    expr.index.name = "cell_label"
    out = rsp.rename(columns={roi_col: "roi"})
    out = out[[c for c in TENX_KEEP if c in out.columns]]
    out = out.merge(expr.reset_index(), on="cell_label", how="inner")
    result.note(f"{len(out)} RSP cells with expression")
    write_table(out, ctx.outdir / "rsp_10x_cells.csv.gz", result)
    write_table(
        penk_group_summary(out, genes=ctx.genes),
        ctx.outdir / "rsp_10x_penk_by_cluster.csv",
        result,
    )
    area_out = area.rename(columns={roi_col: "roi"})
    area_out = area_out[[c for c in TENX_KEEP if c in area_out.columns]]
    if len(area_out):
        area_out = area_out.merge(expr.reset_index(), on="cell_label", how="inner")
    _part_a_continuum(ctx, result, out, area_out, full_blocks)


def _part_a_continuum(
    ctx: Context,
    result: PartResult,
    rsp: pd.DataFrame,
    area: pd.DataFrame,
    full_blocks: list[tuple[Any, list[str], list[str]]],
) -> None:  # orchestration; tested end to end with synthetic frames
    """Continuum vs discrete analyses for 10x L2/3 IT cells (see module docstring)."""
    continuum_module()  # fail early, with the PYTHONPATH hint, if the module is missing
    if PENK not in rsp.columns:
        raise KeyError("Penk not read for 10x cells")
    is_l23 = rsp["subclass"].map(is_l23_it_like).fillna(False).astype(bool)
    rsp_l23 = rsp[is_l23].copy()
    rsp_l23["rsp_subregion"] = rsp_l23["roi"].map(rsp_subregion_from_roi)
    frames = [rsp_l23]
    if len(area) and PENK in area.columns:
        frames.append(area[area["subclass"].map(is_l23_it_like).fillna(False).astype(bool)])
    ctx_l23 = pd.concat(frames, ignore_index=True, sort=False)
    ctx_l23["area"] = ctx_l23["roi"]
    scope_rsp, scope_ctx = "RSP L2/3 IT", "isocortex L2/3 IT"
    result.note(f"continuum: {len(rsp_l23)} RSP and {len(ctx_l23)} isocortex L2/3 IT cells")
    out = ctx.outdir

    def distribution() -> None:
        s1, h1 = penk_distribution_tables(
            rsp_l23, [[], ["library_method"], ["rsp_subregion"]], scope_rsp
        )
        s2, h2 = penk_distribution_tables(
            ctx_l23, [["area"], ["library_method", "area"]], scope_ctx
        )
        write_table(
            pd.concat([s1, s2], ignore_index=True), out / "l23it_10x_penk_distribution.csv", result
        )
        write_table(
            pd.concat([h1, h2], ignore_index=True), out / "l23it_10x_penk_histogram.csv", result
        )

    def taxonomy() -> None:
        parts = []
        for scope, frame in ((scope_rsp, rsp_l23), (scope_ctx, ctx_l23)):
            for level in ("supertype", "cluster"):
                if level not in frame.columns:
                    continue
                t = group_expression_table(frame, [level])
                t.insert(0, "level", level)
                t.insert(0, "scope", scope)
                parts.append(t.rename(columns={level: "taxon"}))
        write_table(
            pd.concat(parts, ignore_index=True), out / "l23it_10x_penk_by_taxon.csv", result
        )
        kw = pd.concat(
            [
                taxonomy_variance_table(
                    rsp_l23, ["supertype", "cluster", "rsp_subregion"], scope_rsp,
                    strata_col="library_method",
                ),
                taxonomy_variance_table(
                    ctx_l23, ["supertype", "cluster", "area"], scope_ctx,
                    strata_col="library_method",
                ),
            ],
            ignore_index=True,
        )  # fmt: skip
        write_table(kw, out / "l23it_10x_penk_kruskal.csv", result)

    def region() -> None:
        parts = []
        for scope, frame, keys in (
            (scope_rsp, rsp_l23, ["rsp_subregion"]),
            (scope_rsp, rsp_l23, ["library_method", "rsp_subregion"]),
            (scope_ctx, ctx_l23, ["area"]),
            (scope_ctx, ctx_l23, ["library_method", "area"]),
        ):
            t = group_expression_table(frame, keys)
            t.insert(0, "grouping", "x".join(keys))
            t.insert(0, "scope", scope)
            parts.append(t)
        write_table(
            pd.concat(parts, ignore_index=True), out / "l23it_10x_penk_by_region.csv", result
        )
        write_table(ctx_l23, out / "l23it_10x_cells.csv.gz", result)

    def gene_corr() -> None:
        if not full_blocks:
            raise RuntimeError("no all-gene expression read for RSP L2/3 IT cells")
        mat, labels, symbols = align_sparse_blocks(full_blocks)
        method = rsp.set_index("cell_label")["library_method"] if "library_method" in rsp else None
        strata: list[tuple[str, np.ndarray]] = [("ALL", np.arange(mat.shape[0]))]
        if method is not None:
            lm = pd.Series(labels).map(method.astype(str)).to_numpy()
            for m in sorted({x for x in lm if isinstance(x, str)}):
                rows = np.flatnonzero(lm == m)
                if rows.size >= CORR_MIN_CELLS:
                    strata.append((m, rows))
        tabs = []
        for name, rows in strata:
            sub = mat[rows]
            n_genes = np.asarray((sub > 0).sum(axis=1)).ravel()
            t = gene_correlations(sub, symbols, covariate=n_genes)
            t.insert(0, "library_method", name)
            tabs.append(t)
        corr = pd.concat(tabs, ignore_index=True)
        write_table(corr, out / "rsp_l23it_10x_penk_gene_corr.csv.gz", result)
        top = pd.concat([top_correlated_genes(t) for t in tabs], ignore_index=True)
        write_table(top, out / "rsp_l23it_10x_penk_gene_corr_top.csv", result)
        marks = pd.concat(
            [marker_correlations(t.drop(columns="library_method")).assign(library_method=t["library_method"].iloc[0])
             for t in tabs],
            ignore_index=True,
        )  # fmt: skip
        write_table(marks, out / "rsp_l23it_10x_penk_depth_markers.csv", result)

    write_json(_continuum_params(), out / "continuum_params_A.json", result)
    run_substeps(
        result,
        [
            ("A distribution", distribution),
            ("A taxonomy", taxonomy),
            ("A region", region),
            ("A gene correlations", gene_corr),
        ],
    )


def _continuum_params() -> dict[str, Any]:
    """Parameters of the continuum analyses, written next to the outputs."""
    return {
        "seed": SEED,
        "hist_edges_log2": HIST_EDGES.tolist(),
        "silverman": {"k": 1, "max_n": SILVERMAN_MAX_N, "n_boot": SILVERMAN_N_BOOT},
        "gene_corr": {
            "min_frac_expressing": CORR_MIN_FRAC,
            "top_n": CORR_TOP_N,
            "max_cells": CORR_MAX_CELLS,
            "min_cells_per_chemistry": CORR_MIN_CELLS,
            "partial_covariate": "genes detected per cell",
            "depth_markers": list(DEPTH_MARKERS),
        },
        "area_max_cells_per_roi": AREA_MAX_CELLS,
        "kruskal_min_taxon_n": MIN_TAXON_N,
        "map_max_cells": MAP_MAX_CELLS,
        "shift": {"min_n": SHIFT_MIN_N, "n_boot": SHIFT_N_BOOT},
        "coord_axes_assumed": COORD_AXES,
        "ccf_midline_z_mm": CCF_MIDLINE_Z_MM,
    }


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
        extra = ("x_reconstructed", "y_reconstructed", "z_reconstructed")
        cols += [
            c
            for c in (*extra, *find_depth_columns(ann.columns))
            if c in ann.columns and c not in cells.columns and c not in cols
        ]
        log.info("%s: CCF annotation columns %s; merged %s", dataset, list(ann.columns), cols)
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
                *find_depth_columns(rsp.columns),
            ]
            rsp = rsp[list(dict.fromkeys(c for c in keep if c in rsp.columns))]
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
        _part_b_continuum(ctx, result, with_penk)
    else:
        result.note("Penk not in any MERFISH panel read; no summary or continuum outputs written")


def merfish_l23_it(cells: pd.DataFrame) -> pd.DataFrame:
    """RSP MERFISH cells of an L2/3 IT subclass with a Penk value, plus spatial proxies.

    Cells are selected by transcriptomic subclass (not by CCF layer), so the
    CCF layer label of these cells can serve as an ordinal depth proxy.
    """
    if "subclass" not in cells.columns or PENK not in cells.columns:
        raise KeyError("MERFISH table needs subclass and Penk columns")
    keep = cells["subclass"].map(is_l23_it_like).fillna(False).astype(bool) & cells[PENK].notna()
    return add_spatial_proxies(cells[keep])


def _part_b_continuum(
    ctx: Context, result: PartResult, cells: pd.DataFrame
) -> None:  # orchestration; tested end to end with synthetic frames
    """Continuum vs discrete analyses for MERFISH RSP L2/3 IT cells."""
    continuum_module()
    l23 = merfish_l23_it(cells)
    out = ctx.outdir
    space = spatial_variables(l23.columns)
    result.note(
        f"continuum: {len(l23)} MERFISH RSP L2/3 IT cells; spatial variables {space}; "
        f"depth columns found {find_depth_columns(cells.columns)}"
    )
    if l23.empty:
        raise RuntimeError("no MERFISH RSP L2/3 IT cells with Penk")

    def distribution() -> None:
        s, h = penk_distribution_tables(
            l23, [["dataset"], ["dataset", "rsp_subdivision"]], "RSP L2/3 IT (MERFISH)"
        )
        write_table(s, out / "rsp_merfish_l23it_penk_distribution.csv", result)
        write_table(h, out / "rsp_merfish_l23it_penk_histogram.csv", result)

    def spatial() -> None:
        tab = pd.concat(
            [
                spearman_table(l23, PENK, space, ["dataset"]),
                spearman_table(l23, PENK, space, ["dataset", "rsp_subdivision"]),
            ],
            ignore_index=True,
            sort=False,
        )
        if not tab.empty:
            tab["axis_assumed"] = tab["variable"].map(COORD_AXES)
        write_table(tab, out / "rsp_merfish_l23it_penk_spatial_corr.csv", result)

    def subdivision() -> None:
        parts = []
        for keys in (["dataset", "rsp_subdivision"], ["dataset", "rsp_subdivision", "layer"]):
            if all(k in l23.columns for k in keys):
                t = group_expression_table(l23.fillna({"layer": "NA"}), keys)
                t.insert(0, "grouping", "x".join(keys))
                parts.append(t)
        write_table(
            pd.concat(parts, ignore_index=True, sort=False),
            out / "rsp_merfish_l23it_penk_by_subdivision.csv",
            result,
        )
        kw = [
            taxonomy_variance_table(
                sub, ["rsp_subdivision", "layer", "supertype", "cluster"], str(ds)
            )
            for ds, sub in l23.groupby("dataset", sort=True)
        ]
        write_table(
            pd.concat(kw, ignore_index=True).rename(columns={"scope": "dataset"}),
            out / "rsp_merfish_l23it_penk_kruskal.csv",
            result,
        )

    def spatial_map() -> None:
        cols = [
            "cell_label", "dataset", "brain_section_label", "x", "y", "z",
            "x_ccf", "y_ccf", "z_ccf", "ccf_ml_from_midline",
            "x_reconstructed", "y_reconstructed", "z_reconstructed",
            "rsp_subdivision", "layer", "supertype", "cluster", PENK,
        ]  # fmt: skip
        m = subsample_rows(l23[[c for c in cols if c in l23.columns]], MAP_MAX_CELLS, SEED)
        write_table(m, out / "rsp_merfish_l23it_penk_map.csv.gz", result)

    write_json(_continuum_params(), out / "continuum_params_B.json", result)
    run_substeps(
        result,
        [
            ("B distribution", distribution),
            ("B spatial correlation", spatial),
            ("B subdivision", subdivision),
            ("B map", spatial_map),
        ],
    )


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
    _part_c_continuum(ctx, result, cells, features)


def patchseq_penk_continuous(cells: pd.DataFrame) -> pd.DataFrame:
    """Patch-seq cells with a Penk CPM value, adding ``log2_penk_cpm`` = log2(CPM + 1)."""
    if "penk_cpm" not in cells.columns:
        return cells.iloc[0:0].assign(log2_penk_cpm=pd.Series(dtype=float))
    v = pd.to_numeric(cells["penk_cpm"], errors="coerce")
    out = cells[v.notna()].copy()
    out["log2_penk_cpm"] = np.log2(v[v.notna()] + 1.0)
    return out


def _part_c_continuum(
    ctx: Context, result: PartResult, cells: pd.DataFrame, features: Sequence[str]
) -> None:  # orchestration; tested end to end with synthetic frames
    """Penk as a continuous variable against intrinsic properties (patch-seq IT cells)."""
    continuum_module()
    pc = patchseq_penk_continuous(cells)
    out = ctx.outdir
    if pc.empty:
        result.note("continuum: no patch-seq cells with Penk expression; skipped")
        return
    result.note(
        "continuum: patch-seq cells with Penk CPM per group "
        f"{pc.groupby(['dataset', 'ttype_group']).size().to_dict()}"
    )
    keys = ["dataset", "ttype_group"]

    def spearman() -> None:
        tab = spearman_table(pc, "log2_penk_cpm", features, keys, pool_col="ttype_group")
        write_table(tab, out / "patchseq_penk_ephys_spearman.csv", result)

    def shift() -> None:
        tab = shift_table(pc, features, "penk_pos_nonzero", keys, pool_col="ttype_group")
        write_table(tab, out / "patchseq_penk_ephys_shift.csv", result)

    def distribution() -> None:
        s, h = penk_distribution_tables(
            pc, [keys], "patch-seq IT", gene="log2_penk_cpm", edges=np.arange(0.0, 20.5, 0.5)
        )
        write_table(s, out / "patchseq_penk_distribution.csv", result)
        write_table(h, out / "patchseq_penk_histogram.csv", result)

    write_json(_continuum_params(), out / "continuum_params_C.json", result)
    run_substeps(
        result,
        [("C spearman", spearman), ("C shift", shift), ("C distribution", distribution)],
    )


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
