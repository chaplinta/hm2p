"""Session dataclass and registry loading from metadata CSVs.

Each experimental session is identified by a canonical session ID:
    YYYYMMDD_HH_MM_SS_<animal_id>   e.g. 20220804_13_52_02_1117646

The ground-truth registry lives in two flat CSV files:
    metadata/animals.csv       — animal IDs, genotype, GCaMP indicator, cell type
    metadata/experiments.csv   — session IDs, animal ID, extractor, tracker,
                                 bad_behav_times, orientation
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Session ID helpers
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<ss>\d{2})_(?P<animal_id>\d+)$"
)


def parse_session_id(session_id: str) -> dict[str, str]:
    """Parse a canonical session ID string into its components.

    Args:
        session_id: e.g. "20220804_13_52_02_1117646"

    Returns:
        Dict with keys: date, hh, mm, ss, animal_id.

    Raises:
        ValueError: If the session ID does not match the expected format.
    """
    m = _SESSION_ID_RE.match(session_id)
    if m is None:
        raise ValueError(
            f"Invalid session ID: {session_id!r}. Expected format: YYYYMMDD_HH_MM_SS_<animal_id>"
        )
    return m.groupdict()


def session_id_to_neurobluepint(session_id: str) -> str:
    """Convert canonical session ID to NeuroBlueprint folder name.

    Args:
        session_id: e.g. "20220804_13_52_02_1117646"

    Returns:
        NeuroBlueprint-compatible name, e.g. "ses-20220804T135202"
    """
    parts = parse_session_id(session_id)
    return f"ses-{parts['date']}T{parts['hh']}{parts['mm']}{parts['ss']}"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """Immutable record for one experimental session.

    Attributes:
        session_id:       Canonical session identifier (YYYYMMDD_HH_MM_SS_<animal>).
        animal_id:        Animal identifier (matches sub-{animal_id} in NeuroBlueprint).
        celltype:         Cell population label ("penk" or "nonpenk").
        gcamp:            GCaMP indicator used (e.g. "GCaMP7f", "GCaMP8f").
        virus_id:         Viral construct ID (e.g. "ADD3", "344").
        extractor:        2P extraction backend ("suite2p" or "caiman").
        tracker:          Pose estimation backend ("dlc", "sleap", or "lp").
        orientation:      Camera rotation angle (degrees) for keypoint correction.
        bad_behav_times:  Raw string from CSV listing stuck-fibre periods (MM:SS-MM:SS, ...).
                          Empty string if none.
    """

    session_id: str
    animal_id: str
    celltype: Literal["penk", "nonpenk"]
    gcamp: str
    virus_id: str
    extractor: Literal["suite2p", "caiman"] = "suite2p"
    tracker: Literal["dlc", "sleap", "lp"] = "dlc"
    orientation: float = 0.0
    bad_behav_times: str = ""

    @property
    def neurobluepint_ses(self) -> str:
        """NeuroBlueprint session folder name, e.g. 'ses-20220804T135202'."""
        return session_id_to_neurobluepint(self.session_id)

    @property
    def neurobluepint_sub(self) -> str:
        """NeuroBlueprint subject folder name, e.g. 'sub-1117646'."""
        return f"sub-{self.animal_id}"

    def derivatives_path(self, derivative: str, root: Path) -> Path:
        """Construct path to a named derivative directory for this session.

        Args:
            derivative: Name of the derivative (e.g. "movement", "calcium", "sync").
            root: Root data directory (local or S3 mount).

        Returns:
            Path to derivatives/<derivative>/sub-{id}/ses-{date}/.
        """
        return root / "derivatives" / derivative / self.neurobluepint_sub / self.neurobluepint_ses


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------


def load_registry(
    animals_csv: Path,
    experiments_csv: Path,
) -> list[Session]:
    """Load all sessions from the metadata CSV files.

    Merges animals.csv (animal-level metadata) with experiments.csv
    (session-level metadata) on animal_id.  The canonical session
    identifier is the ``exp_id`` column in experiments.csv
    (format: YYYYMMDD_HH_MM_SS_<animal_id>); animal_id is extracted
    from the last underscore-delimited segment.

    Args:
        animals_csv: Path to metadata/animals.csv.
        experiments_csv: Path to metadata/experiments.csv.

    Returns:
        List of Session objects, one per row in experiments.csv.

    Raises:
        FileNotFoundError: If either CSV does not exist.
        KeyError: If required columns are missing from the CSVs.
    """
    animals = pd.read_csv(animals_csv, dtype=str)
    experiments = pd.read_csv(experiments_csv, dtype=str)

    # animal_id is embedded in exp_id (last segment); derive it for the merge
    experiments = experiments.copy()
    experiments["animal_id"] = experiments["exp_id"].str.split("_").str[-1]

    merged = experiments.merge(animals, on="animal_id", how="left", validate="many_to_one")

    sessions: list[Session] = []
    for _, row in merged.iterrows():
        sessions.append(
            Session(
                session_id=row["exp_id"],
                animal_id=row["animal_id"],
                celltype=row["celltype"],
                gcamp=row.get("gcamp", "GCaMP7f"),
                virus_id=row.get("virus_id", ""),
                extractor=row.get("extractor", "suite2p"),
                tracker=row.get("tracker", "dlc"),
                orientation=float(row.get("orientation", 0.0) or 0.0),
                bad_behav_times=row.get("bad_behav_times", "") or "",
            )
        )
    return sessions


def get_session(
    session_id: str,
    animals_csv: Path,
    experiments_csv: Path,
) -> Session:
    """Load a single session by ID from the registry.

    Args:
        session_id: Canonical session identifier.
        animals_csv: Path to metadata/animals.csv.
        experiments_csv: Path to metadata/experiments.csv.

    Returns:
        Session object for the specified session.

    Raises:
        KeyError: If session_id is not found in the registry.
    """
    all_sessions = load_registry(animals_csv, experiments_csv)
    by_id = {s.session_id: s for s in all_sessions}
    if session_id not in by_id:
        raise KeyError(f"Session {session_id!r} not found in registry.")
    return by_id[session_id]


# ---------------------------------------------------------------------------
# bad_behav_times parser
# ---------------------------------------------------------------------------


def parse_bad_behav_times(raw: str, total_seconds: float) -> list[tuple[float, float]]:
    """Parse bad_behav_times string from experiments.csv into (start, end) intervals.

    Format in CSV: "MM:SS-MM:SS, MM:SS-MM:SS, ..."
    Returns list of (start_s, end_s) float tuples, empty list if raw is empty.

    Args:
        raw: Raw string from experiments.csv bad_behav_times column.
        total_seconds: Total session duration (s), used to clip end times.

    Returns:
        List of (start_s, end_s) tuples in seconds.
    """
    try:
        if not raw or pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass
    # Handle literal "nan" string (CSV missing values read as string)
    if str(raw).strip().lower() in ("nan", "", "none"):
        return []

    intervals: list[tuple[float, float]] = []
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        start_str, end_str = segment.split("-")
        start_s = _mmss_to_seconds(start_str.strip())
        end_s = _mmss_to_seconds(end_str.strip())
        end_s = min(end_s, total_seconds)
        intervals.append((start_s, end_s))
    return intervals


def _mmss_to_seconds(mmss: str) -> float:
    """Convert MM:SS string to total seconds."""
    mm, ss = mmss.split(":")
    return int(mm) * 60 + float(ss)
