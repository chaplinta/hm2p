"""Manual ROI curation — append-only label store and runtime label resolver.

This module provides the I/O layer for the manual ROI curation workflow that
sits on top of the soma classifier framework
(:mod:`hm2p.extraction.soma_classifier`).

Three pieces fit together:

1. :func:`append_curation_row` writes a single ``(session_id, roi_index,
   label, curator, timestamp)`` row to the on-disk label CSV in append-only
   mode.  Re-labelling the same ROI never overwrites the previous row; the
   reader resolves duplicates by taking the most recent timestamp.

2. :func:`load_latest_labels` reads the CSV and returns one row per
   ``(session_id, roi_index)`` pair, picking the highest-timestamp row.
   Validates the schema with a clear error message on malformed input.

3. :func:`apply_curation_to_ca_h5` reads the latest labels for a single
   session and writes a ``roi_qc/curated_label`` array (length ``n_rois``,
   string dtype, empty for un-curated) into the local ``ca.h5`` file.  This
   does *not* push to S3 — that is a separate operation.

The runtime side is handled by :func:`effective_roi_label`, which prefers
the curated label when present and falls back to the argmax of
``p_soma`` / ``p_dend`` / ``p_artefact`` otherwise.

CSV schema
----------
The CSV uses the same ``session_id, roi_index, label`` core columns that
:mod:`scripts.train_soma_classifier` expects, plus two provenance columns
(``curator``, ``timestamp``) that are ignored by the training script.  This
means the same file feeds both the runtime label resolver and offline
classifier training.

References
----------
Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
two-photon microscopy." bioRxiv. doi:10.1101/061507.
https://github.com/MouseLand/suite2p

Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
*Journal of Machine Learning Research* 12:2825–2830.
https://scikit-learn.org
"""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from hm2p.extraction.soma_classifier import CLASS_NAMES

log = logging.getLogger(__name__)


CURATION_CSV_COLUMNS: tuple[str, ...] = (
    "session_id",
    "roi_index",
    "label",
    "curator",
    "timestamp",
)
"""Canonical column order for ``metadata/roi_curation.csv``.

The first three columns match the schema consumed by
:mod:`scripts.train_soma_classifier`; the remaining two are provenance
metadata that the training script ignores.
"""


# String dtype used for the persisted ``roi_qc/curated_label`` array.  An
# h5py variable-length string keeps the CSV encoding round-trippable on
# read (returns ``bytes``; we decode on read).
_CURATED_LABEL_DSET = "roi_qc/curated_label"


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 timestamp (no microseconds).

    Format: ``"YYYY-MM-DDTHH:MM:SS+00:00"``.  Stable, sortable, and
    timezone-explicit so that "latest wins" is unambiguous across machines.
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def append_curation_row(
    csv_path: Path,
    session_id: str,
    roi_index: int,
    label: str,
    curator: str,
    timestamp: str | None = None,
) -> None:
    """Append a single curation row to ``csv_path`` (creating the file if needed).

    Append-only: this function never modifies or removes existing rows.
    Re-labelling the same ROI simply adds another row with a fresh
    timestamp; :func:`load_latest_labels` resolves duplicates on read.

    Parameters
    ----------
    csv_path : Path
        Destination CSV file.  Created (with header) on first write.
    session_id : str
        Canonical session identifier in ``YYYYMMDD_HH_MM_SS_<animal_id>``
        form (the same format used by
        :mod:`scripts.train_soma_classifier`).
    roi_index : int
        Zero-based ROI index within the session's ``ca.h5`` ``dff``
        array.  Negative or non-integer values are rejected.
    label : str
        One of :data:`hm2p.extraction.soma_classifier.CLASS_NAMES`
        (``"soma"`` / ``"dend"`` / ``"artefact"``).
    curator : str
        Free-form identifier for the human curator (defaults to the
        ``USER`` env var on the page side; we accept any non-empty string
        here so unit tests can pass deterministic values).
    timestamp : str or None
        ISO-8601 timestamp.  When ``None``, the current UTC time is
        recorded.  Existing timestamps are accepted unchanged so callers
        that build a row off-line (e.g. tests) can fix the value.

    Raises
    ------
    ValueError
        ``label`` is not in :data:`CLASS_NAMES`, ``roi_index`` is negative,
        ``session_id`` is empty, or ``curator`` is empty.

    Notes
    -----
    The CSV is opened in text-append mode with newline handling per
    :pep:`305`.  This is safe for cooperating processes on Posix because
    each ``write`` is small (well under ``PIPE_BUF``) and the OS guarantees
    atomic writes for short-line appends.  We still document the contract
    explicitly so that callers don't add concurrent writers without
    thinking.
    """
    if not session_id:
        raise ValueError("session_id must be non-empty")
    if not curator:
        raise ValueError("curator must be non-empty")
    if label not in CLASS_NAMES:
        raise ValueError(f"Invalid label {label!r}; expected one of {list(CLASS_NAMES)}")
    if not isinstance(roi_index, (int, np.integer)) or int(roi_index) < 0:
        raise ValueError(f"roi_index must be a non-negative int; got {roi_index!r}")

    ts = timestamp if timestamp is not None else _utc_now_iso()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(CURATION_CSV_COLUMNS)
        writer.writerow(
            [
                session_id,
                int(roi_index),
                label,
                curator,
                ts,
            ]
        )


def _validate_curation_frame(df: pd.DataFrame, source: Path | str) -> pd.DataFrame:
    """Type-coerce a curation DataFrame and validate required columns/labels."""
    required = {"session_id", "roi_index", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Curation CSV {source} is missing required columns: {sorted(missing)}")

    valid_labels = set(CLASS_NAMES)
    bad = sorted(set(df["label"].dropna().astype(str).unique()) - valid_labels)
    if bad:
        raise ValueError(
            f"Curation CSV {source} contains unrecognised label values "
            f"{bad!r}; expected one of {sorted(valid_labels)}."
        )

    df = df.copy()
    df["session_id"] = df["session_id"].astype(str)
    df["roi_index"] = df["roi_index"].astype(np.int64)
    df["label"] = df["label"].astype(str)
    if "curator" in df.columns:
        df["curator"] = df["curator"].fillna("unknown").astype(str)
    else:
        df["curator"] = "unknown"
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].fillna("").astype(str)
    else:
        df["timestamp"] = ""
    return df


def load_latest_labels(csv_path: Path) -> pd.DataFrame:
    """Load the curation CSV and return one row per ``(session_id, roi_index)``.

    Duplicates are resolved by taking the row with the largest
    ``timestamp`` (lexicographic order is correct for ISO-8601 strings).
    Rows with an empty timestamp are kept but treated as oldest.

    Parameters
    ----------
    csv_path : Path
        CSV produced by :func:`append_curation_row`.  May not exist —
        in that case an empty DataFrame with the canonical columns is
        returned.

    Returns
    -------
    pandas.DataFrame
        Columns are :data:`CURATION_CSV_COLUMNS`.  Indexed 0..N-1.

    Raises
    ------
    ValueError
        Required columns are missing or the file contains an
        unrecognised label.
    """
    if not Path(csv_path).exists():
        return pd.DataFrame({col: [] for col in CURATION_CSV_COLUMNS}).astype(
            {"roi_index": np.int64}
        )
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return pd.DataFrame({col: [] for col in CURATION_CSV_COLUMNS}).astype(
            {"roi_index": np.int64}
        )

    df = _validate_curation_frame(df, csv_path)

    # Sort by timestamp ascending so that ``drop_duplicates(keep="last")``
    # retains the most recent row for each (session, roi) pair.  Empty
    # timestamps sort first ("" < any ISO-8601 string).
    df = df.sort_values(by="timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["session_id", "roi_index"], keep="last")
    df = df.reset_index(drop=True)
    return df[list(CURATION_CSV_COLUMNS)]


def labels_for_session(csv_path: Path, session_id: str) -> dict[int, str]:
    """Return ``{roi_index: label}`` for a single session, latest-wins.

    Convenience wrapper around :func:`load_latest_labels` for callers
    that want fast O(1) lookup by ROI index.

    Parameters
    ----------
    csv_path : Path
        Curation CSV produced by :func:`append_curation_row`.
    session_id : str
        Canonical session identifier.

    Returns
    -------
    dict[int, str]
        Empty if no rows for this session exist.  Each value is one of
        :data:`hm2p.extraction.soma_classifier.CLASS_NAMES`.
    """
    df = load_latest_labels(csv_path)
    if len(df) == 0:
        return {}
    sub = df[df["session_id"] == session_id]
    return {int(r.roi_index): str(r.label) for r in sub.itertuples(index=False)}


def apply_curation_to_ca_h5(
    csv_path: Path,
    session_id: str,
    ca_h5_path: Path,
) -> int:
    """Persist the latest curation labels for one session into ``ca.h5``.

    Reads the curation CSV, picks rows for ``session_id``, and writes a
    string array at ``roi_qc/curated_label`` of length ``n_rois``.
    Un-curated ROIs receive an empty string ``""``.

    The function does *not* upload anything to S3.  Treat it as a local
    write that the user runs deliberately when they want the latest
    labels reflected in their cached/working copy of ``ca.h5``.

    Parameters
    ----------
    csv_path : Path
        Curation CSV.
    session_id : str
        Canonical session identifier (for filtering rows).
    ca_h5_path : Path
        Local path to the session's ``ca.h5`` file (must exist).

    Returns
    -------
    int
        Number of curated ROIs written (i.e. non-empty labels).

    Raises
    ------
    FileNotFoundError
        ``ca_h5_path`` does not exist.
    KeyError
        The ``ca.h5`` file does not contain a ``dff`` dataset (so we
        cannot infer ``n_rois``).
    ValueError
        A curated ``roi_index`` exceeds the session's ROI count.
    """
    ca_h5_path = Path(ca_h5_path)
    if not ca_h5_path.exists():
        raise FileNotFoundError(f"ca.h5 not found: {ca_h5_path}")

    labels = labels_for_session(csv_path, session_id)

    with h5py.File(ca_h5_path, "r+") as f:
        if "dff" not in f:
            raise KeyError(
                f"ca.h5 at {ca_h5_path} is missing the 'dff' dataset; cannot infer n_rois."
            )
        n_rois = int(f["dff"].shape[0])

        if labels:
            max_idx = max(labels)
            if max_idx >= n_rois:
                raise ValueError(
                    f"Curation CSV references roi_index={max_idx} but "
                    f"session {session_id} has only {n_rois} ROIs."
                )

        out = np.array(
            [labels.get(i, "") for i in range(n_rois)],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        if _CURATED_LABEL_DSET in f:
            del f[_CURATED_LABEL_DSET]
        f.create_dataset(
            _CURATED_LABEL_DSET,
            data=out,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    written = sum(1 for v in labels.values() if v)
    log.info(
        "Wrote %d / %d curated labels to %s for session %s",
        written,
        n_rois,
        ca_h5_path,
        session_id,
    )
    return written


def _read_curated_label_array(roi_qc: dict | None, n_rois: int) -> np.ndarray:
    """Return an ``(n_rois,)`` string array of curated labels (empty if absent).

    Accepts either ``bytes`` arrays (h5py default for variable-length
    strings) or already-decoded ``str`` arrays.  Length-mismatched arrays
    are truncated/padded to ``n_rois``.
    """
    out = np.array([""] * n_rois, dtype=object)
    if roi_qc is None:
        return out
    raw = roi_qc.get("curated_label")
    if raw is None:
        return out
    arr = np.asarray(raw)
    if arr.dtype.kind in ("S", "O") and arr.size > 0:
        decoded = []
        for v in arr.tolist():
            if isinstance(v, bytes):
                decoded.append(v.decode("utf-8", errors="replace"))
            elif v is None:
                decoded.append("")
            else:
                decoded.append(str(v))
        decoded_arr = np.array(decoded, dtype=object)
    else:
        decoded_arr = arr.astype(object)

    n = min(len(decoded_arr), n_rois)
    out[:n] = decoded_arr[:n]
    return out


def effective_roi_label(
    roi_qc: dict | None,
    n_rois: int,
) -> np.ndarray:
    """Return per-ROI string labels, preferring curated labels over classifier argmax.

    Resolution order, for each ROI:

    1. The curated label, if ``roi_qc/curated_label[i]`` is one of
       :data:`hm2p.extraction.soma_classifier.CLASS_NAMES`.
    2. The argmax of ``p_soma`` / ``p_dend`` / ``p_artefact`` when all
       three are present and finite.
    3. ``"soma"`` as the conservative fallback (matches the runtime
       behaviour of :mod:`hm2p.calcium.run` when ``stat.npy`` is missing
       and no probabilities are available).

    Parameters
    ----------
    roi_qc : dict or None
        Mapping of QC dataset name to numpy array, as returned by the
        ``"roi_qc"`` group of ``ca.h5``.  ``None`` is treated as
        "no QC at all" — every ROI falls back to the conservative
        default.
    n_rois : int
        Number of ROIs in the session.  Used to size the output array
        and to detect length-mismatched probability arrays.

    Returns
    -------
    numpy.ndarray
        Object dtype, length ``n_rois``.  Each entry is one of
        :data:`hm2p.extraction.soma_classifier.CLASS_NAMES`.
    """
    if n_rois < 0:
        raise ValueError(f"n_rois must be non-negative; got {n_rois}")

    curated = _read_curated_label_array(roi_qc, n_rois)

    # Probabilistic fallback.
    p_soma = (
        np.asarray(roi_qc.get("p_soma"))
        if roi_qc is not None and roi_qc.get("p_soma") is not None
        else None
    )
    p_dend = (
        np.asarray(roi_qc.get("p_dend"))
        if roi_qc is not None and roi_qc.get("p_dend") is not None
        else None
    )
    p_art = (
        np.asarray(roi_qc.get("p_artefact"))
        if roi_qc is not None and roi_qc.get("p_artefact") is not None
        else None
    )

    out = np.array(["soma"] * n_rois, dtype=object)

    if (
        p_soma is not None
        and p_dend is not None
        and p_art is not None
        and len(p_soma) == n_rois
        and len(p_dend) == n_rois
        and len(p_art) == n_rois
    ):
        stack = np.stack(
            [p_soma.astype(np.float64), p_dend.astype(np.float64), p_art.astype(np.float64)],
            axis=1,
        )
        # Replace NaNs with -inf so ROIs with all-NaN rows fall back to
        # the conservative default ("soma" via the initialiser above).
        with np.errstate(invalid="ignore"):
            finite_rows = np.isfinite(stack).all(axis=1)
        idx = np.argmax(np.where(finite_rows[:, None], stack, -np.inf), axis=1)
        for i in range(n_rois):
            if finite_rows[i]:
                out[i] = CLASS_NAMES[int(idx[i])]

    valid = set(CLASS_NAMES)
    for i in range(n_rois):
        c = curated[i]
        if c in valid:
            out[i] = c

    return out
