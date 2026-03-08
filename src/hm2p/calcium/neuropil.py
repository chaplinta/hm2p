"""Stage 4a — neuropil subtraction.

Two methods (configured in config/pipeline.yaml via neuropil_method):
    fixed  — F_corr = F - 0.7 * Fneu  (default; Suite2p only)
    fissa  — Spatial ICA on ROI masks + raw movie (accurate in dense tissue)

CaImAn handles neuropil internally — this module is a no-op for CaImAn sessions.

References:
    Pachitariu et al. 2017. "Suite2p: beyond 10,000 neurons with standard
    two-photon microscopy." doi:10.1101/061507
    https://github.com/MouseLand/suite2p

    Keemink et al. 2018. "FISSA: A neuropil decontamination toolbox for
    calcium imaging signals." Sci Rep 8:3493.
    doi:10.1038/s41598-018-21640-2
    https://github.com/rochefort-lab/fissa
"""

from __future__ import annotations

import numpy as np


def subtract_fixed_coefficient(
    F: np.ndarray,
    Fneu: np.ndarray,
    coefficient: float = 0.7,
) -> np.ndarray:
    """Apply fixed-coefficient neuropil subtraction.

    F_corr = F - coefficient * Fneu

    Args:
        F: (n_rois, n_frames) float32 — raw fluorescence traces.
        Fneu: (n_rois, n_frames) float32 — neuropil traces.
        coefficient: Neuropil mixing coefficient (default 0.7).

    Returns:
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    """
    return F - coefficient * Fneu


def subtract_fissa(
    F: np.ndarray,
    roi_masks: np.ndarray,
    tiff_path: str | None = None,
    n_components: int = 4,
) -> np.ndarray:
    """Apply FISSA spatial ICA neuropil subtraction.

    More accurate than fixed coefficient in densely labelled tissue.
    Requires the raw TIFF stack and ROI masks.

    Args:
        F: (n_rois, n_frames) float32 — raw fluorescence traces.
        roi_masks: (n_rois, height, width) bool — spatial ROI masks.
        tiff_path: Path to the raw TIFF stack. Required for FISSA.
        n_components: Number of ICA components per ROI (default 4).

    Returns:
        (n_rois, n_frames) float32 — neuropil-corrected fluorescence.
    """
    raise NotImplementedError
