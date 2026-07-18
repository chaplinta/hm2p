"""T2 — DLC snapshot selection logic: highest snapshot number is picked.

Tests the _snap_num() helper used in dlc_viewer_page and tracking_quality_page
to select the highest-numbered snapshot from finetuned model outputs.
"""

from __future__ import annotations

import re
import sys
from unittest.mock import MagicMock

# Mock streamlit before importing any frontend page module.
_st_mock = MagicMock()
_st_mock.cache_data = lambda *a, **kw: a[0] if a and callable(a[0]) else (lambda fn: fn)
_st_mock.title = MagicMock()
_st_mock.caption = MagicMock()
_st_mock.markdown = MagicMock()
sys.modules.setdefault("streamlit", _st_mock)


# ---------------------------------------------------------------------------
# _snap_num implementations under test
# These are local functions defined inside cached loaders in each page.
# We test the regex logic directly, then verify the selection behaviour.
# ---------------------------------------------------------------------------


def _snap_num(key: str) -> int:
    """Extracted verbatim from dlc_viewer_page.dl_dlc and tracking_quality_page._load_dlc_data."""
    m = re.search(r"snapshot[_-]best[_-](\d+)", key)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# TestSnapNumExtraction
# ---------------------------------------------------------------------------


class TestSnapNumExtraction:
    """T2 — _snap_num extracts the integer snapshot number from filenames."""

    def test_snapshot_best_underscore_format(self):
        """snapshot_best_100 is parsed as 100."""
        key = "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_100.h5"
        assert _snap_num(key) == 100

    def test_snapshot_best_dash_format(self):
        """snapshot-best-200 is parsed as 200."""
        key = "pose/sub-1/ses-2/DLC_Resnet_topview_snapshot-best-200.h5"
        assert _snap_num(key) == 200

    def test_large_snapshot_number(self):
        """Very large snapshot numbers (e.g., 1000000) are parsed correctly."""
        key = "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_1000000.h5"
        assert _snap_num(key) == 1_000_000

    def test_no_snapshot_pattern_returns_negative_one(self):
        """Files without a snapshot pattern return -1."""
        key = "pose/sub-1/ses-2/DLC_Hrnet_topview_full.h5"
        assert _snap_num(key) == -1

    def test_empty_string_returns_negative_one(self):
        """An empty string returns -1."""
        assert _snap_num("") == -1

    def test_non_h5_file_with_pattern(self):
        """Pattern is matched even in non-.h5 filenames (regex is not extension-gated)."""
        key = "pose/sub-1/ses-2/snapshot_best_50.csv"
        assert _snap_num(key) == 50

    def test_superanimal_baseline_has_no_snapshot_pattern(self):
        """A SuperAnimal output file without a snapshot number returns -1."""
        key = "pose/sub-1/ses-2/DLC_superanimal_topviewmouse.h5"
        assert _snap_num(key) == -1


# ---------------------------------------------------------------------------
# TestSnapNumHighestSelected
# ---------------------------------------------------------------------------


class TestSnapNumHighestSelected:
    """T2 — when multiple snapshots exist, the highest-numbered one is selected."""

    def test_selects_highest_from_two_snapshots(self):
        """Given two snapshot files, max selects the higher number."""
        finetuned_keys = [
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_100.h5",
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_200.h5",
        ]
        selected = max(finetuned_keys, key=_snap_num)
        assert _snap_num(selected) == 200

    def test_selects_highest_from_three_snapshots(self):
        """Given three snapshot files, max selects the largest."""
        finetuned_keys = [
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_50.h5",
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_300.h5",
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_150.h5",
        ]
        selected = max(finetuned_keys, key=_snap_num)
        assert _snap_num(selected) == 300

    def test_single_snapshot_is_selected(self):
        """A list with a single snapshot always selects that snapshot."""
        finetuned_keys = [
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_75.h5",
        ]
        selected = max(finetuned_keys, key=_snap_num)
        assert _snap_num(selected) == 75

    def test_falls_back_to_first_h5_when_no_finetuned(self):
        """When no finetuned files exist, the first available .h5 is used."""
        all_h5_files = [
            "pose/sub-1/ses-2/DLC_superanimal_topviewmouse.h5",
            "pose/sub-1/ses-2/DLC_superanimal_topviewmouse_filtered.h5",
        ]
        # Mimic the page logic: finetuned = [k for k in h5_files if "Hrnet" in k or "Resnet" in k]
        finetuned = [k for k in all_h5_files if "Hrnet" in k or "Resnet" in k]
        selected = max(finetuned, key=_snap_num) if finetuned else all_h5_files[0]
        assert selected == "pose/sub-1/ses-2/DLC_superanimal_topviewmouse.h5"

    def test_snapshot_order_independent_of_list_order(self):
        """Max is determined by snapshot number, not list order."""
        # Intentionally put higher number last in list
        finetuned_keys = [
            "pose/sub-1/ses-2/DLC_Resnet_topview_snapshot-best-500.h5",
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_100.h5",
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_250.h5",
        ]
        selected = max(finetuned_keys, key=_snap_num)
        assert _snap_num(selected) == 500

    def test_mixed_hrnet_resnet_snapshots(self):
        """Highest snapshot is selected across mixed Hrnet/Resnet files."""
        finetuned_keys = [
            "pose/sub-1/ses-2/DLC_Hrnet_topview_snapshot_best_200.h5",
            "pose/sub-1/ses-2/DLC_Resnet_topview_snapshot-best-350.h5",
        ]
        selected = max(finetuned_keys, key=_snap_num)
        assert _snap_num(selected) == 350

    def test_no_snapshot_files_fallback_handles_correctly(self):
        """When all finetuned files have no snapshot number, max still picks one."""
        # All return -1 — max will pick arbitrarily (first in CPython), no crash
        finetuned_keys = [
            "pose/sub-1/ses-2/DLC_Hrnet_topview_full.h5",
            "pose/sub-1/ses-2/DLC_Resnet_topview_full.h5",
        ]
        # Should not raise even when all snap_nums are -1
        selected = max(finetuned_keys, key=_snap_num)
        assert selected in finetuned_keys
