"""Tests for session.py — session dataclass, ID parsing, registry loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hm2p.session import (
    Session,
    _mmss_to_seconds,
    parse_bad_behav_times,
    parse_session_id,
    session_id_to_neurobluepint,
)

# ---------------------------------------------------------------------------
# parse_session_id
# ---------------------------------------------------------------------------


def test_parse_session_id_valid() -> None:
    result = parse_session_id("20220804_13_52_02_1117646")
    assert result == {
        "date": "20220804",
        "hh": "13",
        "mm": "52",
        "ss": "02",
        "animal_id": "1117646",
    }


def test_parse_session_id_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid session ID"):
        parse_session_id("not_a_session_id")


def test_parse_session_id_wrong_format_raises() -> None:
    with pytest.raises(ValueError):
        parse_session_id("20220804_13_52_02")  # missing animal_id


# ---------------------------------------------------------------------------
# session_id_to_neurobluepint
# ---------------------------------------------------------------------------


def test_session_id_to_neurobluepint() -> None:
    nb = session_id_to_neurobluepint("20220804_13_52_02_1117646")
    assert nb == "ses-20220804T135202"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


def test_session_properties(penk_session: Session) -> None:
    assert penk_session.neurobluepint_sub == "sub-1117646"
    assert penk_session.neurobluepint_ses == "ses-20220804T135202"


def test_session_derivatives_path(penk_session: Session, tmp_path: Path) -> None:
    path = penk_session.derivatives_path("movement", tmp_path)
    assert path == tmp_path / "derivatives" / "movement" / "sub-1117646" / "ses-20220804T135202"


def test_session_is_frozen(penk_session: Session) -> None:
    with pytest.raises(AttributeError):
        penk_session.session_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# parse_bad_behav_times
# ---------------------------------------------------------------------------


def test_mmss_to_seconds() -> None:
    assert _mmss_to_seconds("02:30") == 150.0
    assert _mmss_to_seconds("00:00") == 0.0
    assert _mmss_to_seconds("10:00") == 600.0


def test_parse_bad_behav_empty() -> None:
    """Empty / NaN / unknown → no exclusions."""
    assert parse_bad_behav_times("", 600.0) == []
    assert parse_bad_behav_times("nan", 600.0) == []
    assert parse_bad_behav_times("none", 600.0) == []


def test_parse_bad_behav_question_mark() -> None:
    """'?' (unknown) → treated as no exclusions."""
    assert parse_bad_behav_times("?", 600.0) == []


def test_parse_bad_behav_single_interval() -> None:
    """Single semicolon-delimited interval parses correctly."""
    result = parse_bad_behav_times("02:30-03:00", 600.0)
    assert result == [(150.0, 180.0)]


def test_parse_bad_behav_multiple_intervals_semicolon() -> None:
    """Semicolon-separated intervals (real CSV format) parse correctly."""
    result = parse_bad_behav_times("02:30-03:00;07:15-07:45", 600.0)
    assert result == [(150.0, 180.0), (435.0, 465.0)]


def test_parse_bad_behav_end_keyword() -> None:
    """'end' as end timestamp maps to total_seconds."""
    result = parse_bad_behav_times("27:00-end", 1800.0)
    assert result == [(1620.0, 1800.0)]


def test_parse_bad_behav_real_csv_row() -> None:
    """Parse the exact format found in the real experiments.csv."""
    raw = "11:10-11:30;13:20-21:00;22:30-24:40;27:00-end"
    total = 1800.0  # 30 min session
    result = parse_bad_behav_times(raw, total)
    assert result[0] == (670.0, 690.0)  # 11:10–11:30
    assert result[1] == (800.0, 1260.0)  # 13:20–21:00
    assert result[2] == (1350.0, 1480.0)  # 22:30–24:40
    assert result[3] == (1620.0, 1800.0)  # 27:00–end


def test_parse_bad_behav_clips_to_total_duration() -> None:
    """Explicit end time past session length is clipped to total_seconds."""
    result = parse_bad_behav_times("09:50-10:30", 600.0)  # 10:30 > 10:00 total
    assert result == [(590.0, 600.0)]


# ---------------------------------------------------------------------------
# load_registry (integration — uses tmp_path CSVs)
# ---------------------------------------------------------------------------


def test_load_registry(tmp_path: Path) -> None:
    from hm2p.session import load_registry

    animals_csv = tmp_path / "animals.csv"
    experiments_csv = tmp_path / "experiments.csv"

    animals_csv.write_text(
        textwrap.dedent("""\
            animal_id,strain,gcamp,celltype,virus_id
            1117646,Penk-Cre,GCaMP7f,penk,ADD3
        """)
    )
    experiments_csv.write_text(
        textwrap.dedent("""\
            exp_id,extractor,tracker,orientation,bad_behav_times
            20220804_13_52_02_1117646,suite2p,dlc,15.0,02:30-03:00
        """)
    )

    sessions = load_registry(animals_csv, experiments_csv)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "20220804_13_52_02_1117646"
    assert s.celltype == "penk"
    assert s.gcamp == "GCaMP7f"
    assert s.orientation == 15.0
    assert s.bad_behav_times == "02:30-03:00"
