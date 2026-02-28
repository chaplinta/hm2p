"""Abstract extractor interface wrapping roiextractors.

All downstream calcium processing (Stage 4) receives a BaseExtractor instance
and calls the same methods regardless of whether Suite2p or CaImAn was used.

The concrete subclasses (Suite2pExtractor, CaimanExtractor) construct the
appropriate roiextractors class and expose unified access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseExtractor(ABC):
    """Unified interface over roiextractors SegmentationExtractor subclasses."""

    @abstractmethod
    def get_raw_traces(self) -> np.ndarray:
        """Return raw fluorescence traces.

        Returns:
            (n_rois, n_frames) float32 — raw F per ROI.
        """
        ...

    @abstractmethod
    def get_neuropil_traces(self) -> np.ndarray | None:
        """Return neuropil traces (Fneu), or None if not available.

        Returns:
            (n_rois, n_frames) float32, or None for CaImAn.
        """
        ...

    @abstractmethod
    def get_accepted_roi_ids(self) -> list[int]:
        """Return indices of ROIs passing the extractor's quality filter.

        Returns:
            List of ROI indices (0-based).
        """
        ...

    @abstractmethod
    def get_roi_masks(self) -> np.ndarray:
        """Return spatial masks for all ROIs.

        Returns:
            (n_rois, height, width) bool.
        """
        ...

    @abstractmethod
    def get_sampling_frequency(self) -> float:
        """Return imaging frame rate in Hz."""
        ...

    @abstractmethod
    def get_roi_types(self) -> list[str]:
        """Return per-ROI type labels: 'soma', 'dend', or 'artefact'.

        For Suite2p: classified post-hoc from shape statistics in stat.npy.
        For CaImAn: all accepted ROIs are labelled 'soma'.

        Returns:
            List of strings, length n_rois.
        """
        ...

    @property
    @abstractmethod
    def n_rois(self) -> int:
        """Total number of ROIs (accepted + rejected)."""
        ...

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Number of imaging frames."""
        ...

    @classmethod
    @abstractmethod
    def from_path(cls, path: Path) -> BaseExtractor:
        """Construct extractor from the native output directory or file.

        Args:
            path: Path to Suite2p output folder, CaImAn .hdf5, etc.
        """
        ...
