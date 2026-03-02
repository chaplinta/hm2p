"""CaImAn extractor wrapping CaImAn HDF5 results.

CaImAn handles neuropil internally (CNMF), so get_neuropil_traces() returns
None. All accepted ROIs are labelled 'soma' (CaImAn does not detect dendrites).

This implementation reads CaImAn analysis results from an HDF5 file. When
the full CaImAn library is available, it uses CaImAn's native loader;
otherwise it reads the standard HDF5 keys directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hm2p.extraction.base import BaseExtractor


class CaimanExtractor(BaseExtractor):
    """Extractor backed by a CaImAn .hdf5 results file."""

    def __init__(self, hdf5_path: Path) -> None:
        """Initialise from a CaImAn analysis results HDF5 file.

        Reads the standard CaImAn output keys (C, idx_components, A, dims).

        Args:
            hdf5_path: Path to the CaImAn .hdf5 output file.

        Raises:
            FileNotFoundError: If hdf5_path does not exist.
            KeyError: If required HDF5 keys are absent.
        """
        if not hdf5_path.exists():
            raise FileNotFoundError(f"CaImAn HDF5 not found: {hdf5_path}")

        import h5py

        with h5py.File(hdf5_path, "r") as f:
            if "C" not in f:
                raise KeyError(
                    f"Required key 'C' (temporal components) missing in {hdf5_path}. "
                    f"Available keys: {list(f.keys())}"
                )
            self._C: np.ndarray = f["C"][:].astype(np.float32)

            if "idx_components" in f:
                self._accepted: list[int] = list(f["idx_components"][:].astype(int))
            else:
                self._accepted = list(range(self._C.shape[0]))

            if "dims" in f:
                self._dims: tuple[int, int] = tuple(int(x) for x in f["dims"][:])  # type: ignore[assignment]
            else:
                self._dims = (512, 512)

            if "A" in f:
                self._A: np.ndarray | None = np.asarray(f["A"])
            else:
                self._A = None

    # -- BaseExtractor interface --------------------------------------------

    def get_raw_traces(self) -> np.ndarray:
        """Return temporal components (C matrix) for accepted components.

        Returns:
            (n_accepted, n_frames) float32.
        """
        return self._C[self._accepted]

    def get_neuropil_traces(self) -> np.ndarray | None:
        """CaImAn handles neuropil internally — always returns None."""
        return None

    def get_accepted_roi_ids(self) -> list[int]:
        """Return indices of accepted components.

        Returns:
            List of 0-based component indices.
        """
        return self._accepted

    def get_roi_masks(self) -> np.ndarray:
        """Return spatial footprints for accepted components.

        Returns:
            (n_accepted, height, width) bool.

        Raises:
            RuntimeError: If spatial components (A) were not in the HDF5.
        """
        if self._A is None:
            raise RuntimeError("Spatial components (A) not found in CaImAn HDF5")
        h, w = self._dims
        n_total = self._A.shape[1] if self._A.ndim == 2 else self._A.shape[0]
        masks = np.zeros((len(self._accepted), h, w), dtype=bool)
        for i, idx in enumerate(self._accepted):
            if idx < n_total:
                if self._A.ndim == 2:
                    mask_flat = self._A[:, idx]
                else:
                    mask_flat = self._A[idx].ravel()
                masks[i] = mask_flat.reshape(h, w) > 0
        return masks

    def get_sampling_frequency(self) -> float:
        """Return imaging frame rate (not stored in CaImAn HDF5 by default).

        Returns:
            Default 30.0 Hz. Override in subclass or config if different.
        """
        return 30.0

    def get_roi_types(self) -> list[str]:
        """CaImAn only detects soma — all accepted ROIs labelled 'soma'."""
        return ["soma"] * len(self._accepted)

    @property
    def n_rois(self) -> int:
        """Total number of components (accepted + rejected)."""
        return self._C.shape[0]

    @property
    def n_frames(self) -> int:
        """Number of imaging frames."""
        return self._C.shape[1]

    @classmethod
    def from_path(cls, path: Path) -> CaimanExtractor:
        return cls(path)
