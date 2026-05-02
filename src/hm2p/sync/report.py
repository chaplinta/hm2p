"""Stage 5b — sync report aggregator.

Walks ``derivatives/sync/`` and emits a single
``derivatives/sync_report/sync_report.parquet`` with one row per session.
The aggregator only reads root attrs (no heavy datasets) so it stays
fast even when the sync.h5 payload is large or partially corrupt.

Reading attrs only also ensures the aggregator is robust to broken
``dff`` / ``hd_deg`` arrays — see test
``tests/sync/test_report.py::test_corrupt_dff_does_not_block_aggregator``.

See ``docs/sync-pipeline-design.md`` §1.4 / §2.3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from hm2p.sync.diagnostics import (
    SYNC_DIAG_FLOAT_KEYS,
    SYNC_DIAG_INT_KEYS,
)

log = logging.getLogger(__name__)


# Column order for sync_report.parquet — the validators in
# ``hm2p.io.hdf5.validate_sync_report_parquet`` rely on every column being
# present and in this exact order for deterministic diff-checks.
_COLUMN_ORDER: tuple[str, ...] = (
    "exp_id",
    "sub",
    "ses",
    "sync_status",
    "sync_warnings",
    "sync_failures",
    "dlc_champion_id",
    "read_error",
    *SYNC_DIAG_INT_KEYS,
    *SYNC_DIAG_FLOAT_KEYS,
)


def _decode(val: Any) -> str:
    """Decode an HDF5 attr that might be bytes or str."""
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def _exp_id_from_sub_ses(sub: str, ses: str) -> str:
    """Reconstruct canonical exp_id from sub-/ses- folder names.

    ``sub-1117646`` + ``ses-20220804T135202`` → ``20220804_13_52_02_1117646``.
    """
    if sub.startswith("sub-"):
        animal = sub[4:]
    else:
        animal = sub
    if ses.startswith("ses-"):
        s = ses[4:]
    else:
        s = ses
    if "T" in s:
        date, t = s.split("T", 1)
        if len(t) >= 6:
            hh, mm, sec = t[0:2], t[2:4], t[4:6]
            return f"{date}_{hh}_{mm}_{sec}_{animal}"
    return f"{s}_{animal}"


def _empty_row(*, exp_id: str = "", sub: str = "", ses: str = "", read_error: str = "") -> dict:
    """Construct a sentinel row for sessions that fail to read."""
    row: dict[str, Any] = {
        "exp_id": exp_id,
        "sub": sub,
        "ses": ses,
        "sync_status": "",
        "sync_warnings": "[]",
        "sync_failures": "[]",
        "dlc_champion_id": "",
        "read_error": read_error,
    }
    for k in SYNC_DIAG_INT_KEYS:
        row[k] = -9999
    for k in SYNC_DIAG_FLOAT_KEYS:
        row[k] = float("nan")
    return row


def _row_from_sync_attrs(
    *,
    exp_id: str,
    sub: str,
    ses: str,
    attrs: dict[str, Any],
) -> dict:
    """Build a parquet row from the root attrs of a sync.h5."""
    if "sync_status" not in attrs:
        return _empty_row(
            exp_id=exp_id,
            sub=sub,
            ses=ses,
            read_error="schema version 0.0 — rebuild required",
        )
    row: dict[str, Any] = _empty_row(exp_id=exp_id, sub=sub, ses=ses)
    row["sync_status"] = _decode(attrs["sync_status"])
    row["sync_warnings"] = _decode(attrs.get("sync_warnings", "[]"))
    row["sync_failures"] = _decode(attrs.get("sync_failures", "[]"))
    row["dlc_champion_id"] = _decode(attrs.get("dlc_champion_id", ""))
    for k in SYNC_DIAG_INT_KEYS:
        sync_key = f"sync_diag/{k}"
        if sync_key in attrs:
            try:
                row[k] = int(attrs[sync_key])
            except (TypeError, ValueError):
                row[k] = -9999
    for k in SYNC_DIAG_FLOAT_KEYS:
        sync_key = f"sync_diag/{k}"
        if sync_key in attrs:
            try:
                row[k] = float(attrs[sync_key])
            except (TypeError, ValueError):
                row[k] = float("nan")
    return row


def build_report(
    sync_dir: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Walk ``sync_dir`` and emit ``sync_report.parquet`` with one row per session.

    The aggregator reads only root attrs from each sync.h5 — never the
    heavy datasets — so a corrupt resampled payload does not block the
    report. Sessions whose ``sync.h5`` cannot be opened or whose
    ``sync_status`` attr is missing get a row with the ``read_error``
    column populated and sentinel scalars.

    Parameters
    ----------
    sync_dir:
        Directory laid out as ``<sync_dir>/<sub>/<ses>/sync.h5``.
    output_path:
        Destination ``.parquet`` file. Parents are created.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame written to disk. Sorted by ``exp_id``.
    """
    sync_root = Path(sync_dir)
    out = Path(output_path)
    rows: list[dict] = []
    if sync_root.exists():
        for sub_dir in sorted(sync_root.iterdir()):
            if not sub_dir.is_dir():
                continue
            for ses_dir in sorted(sub_dir.iterdir()):
                if not ses_dir.is_dir():
                    continue
                sync_path = ses_dir / "sync.h5"
                if not sync_path.exists():
                    continue
                exp_id = _exp_id_from_sub_ses(sub_dir.name, ses_dir.name)
                try:
                    from hm2p.io.hdf5 import read_attrs

                    attrs = read_attrs(sync_path)
                except Exception as exc:
                    log.warning("Failed to read sync.h5 attrs at %s: %s", sync_path, exc)
                    rows.append(
                        _empty_row(
                            exp_id=exp_id,
                            sub=sub_dir.name,
                            ses=ses_dir.name,
                            read_error=str(exc),
                        )
                    )
                    continue
                rows.append(
                    _row_from_sync_attrs(
                        exp_id=exp_id,
                        sub=sub_dir.name,
                        ses=ses_dir.name,
                        attrs=attrs,
                    )
                )
    df = pd.DataFrame(rows, columns=list(_COLUMN_ORDER))
    # Sort deterministically for diff-checks.
    if not df.empty:
        df = df.sort_values("exp_id").reset_index(drop=True)
    # Force string dtypes for the textual columns to match the validator.
    for col in (
        "exp_id",
        "sub",
        "ses",
        "sync_status",
        "sync_warnings",
        "sync_failures",
        "dlc_champion_id",
        "read_error",
    ):
        df[col] = df[col].astype(object)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("sync_report.parquet written: %d rows → %s", len(df), out)
    return df


def column_order() -> tuple[str, ...]:
    """Return the canonical column order of sync_report.parquet."""
    return _COLUMN_ORDER
