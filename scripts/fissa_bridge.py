#!/usr/bin/env python3
"""Isolated-environment FISSA runner for the Stage 4 reprocessing path.

FISSA (Keemink et al. 2018) pins ``scikit-learn < 1.2``, which conflicts with
the ROI classifier's requirement of ``scikit-learn >= 1.4``. The two cannot share
one environment, so FISSA runs here in a dedicated virtual environment and hands
its neuropil-corrected traces back to the main pipeline as a plain ``.npy`` file.

This script is intentionally thin: it loads a regenerated registered movie and the
crop-aligned ROI masks from disk, calls
:func:`hm2p.calcium.neuropil.subtract_fissa_from_movie`, and writes the corrected
``(n_rois, n_frames)`` trace array. It performs no S3 or Suite2p work — the calling
driver (:mod:`run_stage4_fissa`) handles download, re-registration, mask building,
and the downstream dF/F0 / ca.h5 steps in the main environment.

Only ``numpy``, ``fissa`` (+ its scikit-learn<1.2) and ``hm2p`` installed with
``--no-deps`` are required here.

Usage (run by run_stage4_fissa.run_session_fissa via subprocess):
    /opt/fissa/bin/python fissa_bridge.py \\
        --movie movie.npy --masks masks.npz --out F_corr.npy

References
----------
Keemink SW, Lowe SC, Pakan JMP, Dylda E, van Rossum MCW, Bhatt DH. 2018.
"FISSA: A neuropil decontamination toolbox for calcium imaging signals."
Scientific Reports 8:3493. doi:10.1038/s41598-018-21640-2
https://github.com/rochefort-lab/fissa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path when run as a standalone script on EC2.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_masks(masks_npz: Path) -> list[np.ndarray]:
    """Load the per-ROI boolean masks saved by the driver via ``np.savez``.

    Parameters
    ----------
    masks_npz : Path
        ``.npz`` written with positional arrays (``np.savez(path, *masks)``), so
        keys are ``arr_0``, ``arr_1`` ... in ROI order.

    Returns
    -------
    list of np.ndarray
        Boolean masks in their original order.
    """
    with np.load(masks_npz) as data:
        keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
        return [data[k].astype(bool) for k in keys]


def main() -> None:  # pragma: no cover - thin CLI wrapper around tested code
    parser = argparse.ArgumentParser(description="Isolated-env FISSA runner")
    parser.add_argument("--movie", required=True, type=Path,
                        help="(n_frames, Ly, Lx) registered movie .npy")
    parser.add_argument("--masks", required=True, type=Path,
                        help="crop-aligned per-ROI boolean masks .npz")
    parser.add_argument("--out", required=True, type=Path,
                        help="destination (n_rois, n_frames) F_corr .npy")
    parser.add_argument("--cache", type=Path, default=None,
                        help="FISSA intermediate cache dir (default: <out>/../fissa_cache)")
    args = parser.parse_args()

    from hm2p.calcium.neuropil import subtract_fissa_from_movie

    cache_dir = args.cache or (args.out.parent / "fissa_cache")
    movie = np.load(args.movie)
    masks = load_masks(args.masks)
    F_corr = subtract_fissa_from_movie(
        movie=movie, roi_masks=masks, output_dir=cache_dir
    )
    np.save(args.out, np.asarray(F_corr, dtype=np.float32))
    print(f"Wrote {args.out} with shape {np.asarray(F_corr).shape}")


if __name__ == "__main__":  # pragma: no cover
    main()
