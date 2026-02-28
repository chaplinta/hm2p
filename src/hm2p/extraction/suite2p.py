"""Suite2p extractor with post-hoc soma/dendrite ROI classification.

Wraps roiextractors.Suite2pSegmentationExtractor. After loading the Suite2p
output folder, each ROI is classified as 'soma', 'dend', or 'artefact' using
shape statistics from stat.npy and pre-trained classifiers:
    - classifier_soma.npy   (existing, reused unchanged)
    - classifier_dend.npy   (existing, reused unchanged)

There is a single imaging plane — soma and dendrite ROIs co-exist.
No second Suite2p run is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.extraction.base import BaseExtractor


class Suite2pExtractor(BaseExtractor):
    """Extractor backed by Suite2p output folder."""

    def __init__(self, folder_path: Path) -> None:
        """Initialise from a Suite2p output directory.

        Args:
            folder_path: Path to the Suite2p output directory containing
                         plane0/F.npy, plane0/Fneu.npy, plane0/stat.npy, etc.
        """
        raise NotImplementedError

    # -- BaseExtractor interface --------------------------------------------

    def get_raw_traces(self) -> np.ndarray:
        raise NotImplementedError

    def get_neuropil_traces(self) -> np.ndarray | None:
        raise NotImplementedError

    def get_accepted_roi_ids(self) -> list[int]:
        raise NotImplementedError

    def get_roi_masks(self) -> np.ndarray:
        raise NotImplementedError

    def get_sampling_frequency(self) -> float:
        raise NotImplementedError

    def get_roi_types(self) -> list[str]:
        """Classify ROIs using Suite2p shape stats and pre-trained classifiers."""
        raise NotImplementedError

    @property
    def n_rois(self) -> int:
        raise NotImplementedError

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_path(cls, path: Path) -> Suite2pExtractor:
        return cls(path)


def classify_roi_types(
    stat: list[dict],  # type: ignore[type-arg]
    classifier_soma_path: Path,
    classifier_dend_path: Path,
) -> list[str]:
    """Classify each ROI as 'soma', 'dend', or 'artefact'.

    Uses shape statistics (aspect ratio, radius, compactness) from Suite2p
    stat.npy and existing pre-trained sklearn classifiers.

    Args:
        stat: List of per-ROI stat dicts loaded from stat.npy.
        classifier_soma_path: Path to classifier_soma.npy.
        classifier_dend_path: Path to classifier_dend.npy.

    Returns:
        List of strings ('soma', 'dend', 'artefact'), one per ROI.
    """
    raise NotImplementedError
