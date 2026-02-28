"""CaImAn extractor wrapping roiextractors.CaimanSegmentationExtractor.

CaImAn handles neuropil internally (CNMF), so get_neuropil_traces() returns
None. All accepted ROIs are labelled 'soma' (CaImAn does not detect dendrites).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.extraction.base import BaseExtractor


class CaimanExtractor(BaseExtractor):
    """Extractor backed by a CaImAn .hdf5 results file."""

    def __init__(self, hdf5_path: Path) -> None:
        """Initialise from a CaImAn analysis results HDF5 file.

        Args:
            hdf5_path: Path to the CaImAn .hdf5 output file.
        """
        raise NotImplementedError

    # -- BaseExtractor interface --------------------------------------------

    def get_raw_traces(self) -> np.ndarray:
        raise NotImplementedError

    def get_neuropil_traces(self) -> np.ndarray | None:
        """CaImAn handles neuropil internally — always returns None."""
        return None

    def get_accepted_roi_ids(self) -> list[int]:
        raise NotImplementedError

    def get_roi_masks(self) -> np.ndarray:
        raise NotImplementedError

    def get_sampling_frequency(self) -> float:
        raise NotImplementedError

    def get_roi_types(self) -> list[str]:
        """CaImAn only detects soma — all accepted ROIs labelled 'soma'."""
        raise NotImplementedError

    @property
    def n_rois(self) -> int:
        raise NotImplementedError

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    @classmethod
    def from_path(cls, path: Path) -> CaimanExtractor:
        return cls(path)
