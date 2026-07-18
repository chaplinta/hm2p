"""Rastermap analysis — discovery of neural population structure.

Rastermap (Stringer et al. 2025, Nature Neuroscience) sorts neurons along
a 1D axis so that nearby neurons have similar activity patterns. This
reveals sequential activation, sustained states, and tuning structure
without prior knowledge of what variables drive the activity.

Reference:
    Stringer C et al. 2025. "Rastermap: a discovery method for neural
    population recordings." Nature Neuroscience 28:201-212.
    doi:10.1038/s41593-024-01783-4
    https://github.com/MouseLand/rastermap
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_rastermap(
    dff: np.ndarray,
    n_clusters: int = 100,
    n_PCs: int = 200,
    time_lag_window: int = 10,
) -> dict:
    """Sort neurons using Rastermap.

    Args:
        dff: (n_rois, n_frames) neural activity matrix.
        n_clusters: Number of k-means clusters.
        n_PCs: Number of PCs for initial dimensionality reduction.
        time_lag_window: Max time lag for asymmetric similarity.

    Returns:
        Dict with:
            isort — (n_rois,) sort order indices
            embedding — (n_rois,) 1D embedding positions
    """
    from rastermap import Rastermap

    n_rois = dff.shape[0]
    n_clusters = min(n_clusters, n_rois)
    n_PCs = min(n_PCs, n_rois, dff.shape[1])

    model = Rastermap(
        n_clusters=n_clusters,
        n_PCs=n_PCs,
        time_lag_window=time_lag_window,
    )
    model.fit(np.nan_to_num(dff.astype(np.float32)))

    return {
        "isort": model.isort,
        "embedding": model.embedding,
    }


def compute_superneurons(
    dff: np.ndarray,
    isort: np.ndarray,
    bin_size: int = 10,
) -> np.ndarray:
    """Average nearby neurons in the Rastermap sorting.

    Creates denoised "superneuron" traces by averaging groups of
    `bin_size` adjacent neurons in the sorting order.

    Args:
        dff: (n_rois, n_frames) neural activity.
        isort: (n_rois,) sort order from Rastermap.
        bin_size: Number of neurons per superneuron.

    Returns:
        (n_superneurons, n_frames) averaged traces.
    """
    sorted_dff = dff[isort]
    n_rois = sorted_dff.shape[0]
    n_super = n_rois // bin_size

    superneurons = np.zeros((n_super, sorted_dff.shape[1]), dtype=np.float32)
    for i in range(n_super):
        superneurons[i] = np.nanmean(sorted_dff[i * bin_size : (i + 1) * bin_size], axis=0)

    return superneurons


def superneuron_behaviour_correlations(
    superneurons: np.ndarray,
    hd_deg: np.ndarray | None = None,
    speed: np.ndarray | None = None,
    light_on: np.ndarray | None = None,
) -> dict:
    """Correlate superneuron activity with behavioural variables.

    Args:
        superneurons: (n_super, n_frames) from compute_superneurons.
        hd_deg: (n_frames,) head direction in degrees.
        speed: (n_frames,) speed in cm/s.
        light_on: (n_frames,) bool.

    Returns:
        Dict with per-superneuron correlation arrays.
    """
    n_super, n_frames = superneurons.shape
    result = {}

    if speed is not None:
        n = min(n_frames, len(speed))
        corrs = np.full(n_super, np.nan)
        for i in range(n_super):
            s = superneurons[i, :n]
            v = np.isfinite(s) & np.isfinite(speed[:n])
            if v.sum() > 10 and np.std(s[v]) > 0:
                corrs[i] = float(spearmanr(s[v], speed[:n][v])[0])
        result["speed_corr"] = corrs

    if hd_deg is not None:
        # Circular-linear correlation: split HD into sin/cos
        n = min(n_frames, len(hd_deg))
        hd_rad = np.deg2rad(hd_deg[:n] % 360)
        corrs = np.full(n_super, np.nan)
        for i in range(n_super):
            s = superneurons[i, :n]
            v = np.isfinite(s) & np.isfinite(hd_rad)
            if v.sum() > 10 and np.std(s[v]) > 0:
                r_sin = abs(float(spearmanr(s[v], np.sin(hd_rad[v]))[0]))
                r_cos = abs(float(spearmanr(s[v], np.cos(hd_rad[v]))[0]))
                corrs[i] = max(r_sin, r_cos)
        result["hd_corr"] = corrs

    if light_on is not None:
        n = min(n_frames, len(light_on))
        light = light_on[:n].astype(bool)
        mod = np.full(n_super, np.nan)
        for i in range(n_super):
            s = superneurons[i, :n]
            if light.sum() > 10 and (~light).sum() > 10:
                ml = np.nanmean(s[light])
                md = np.nanmean(s[~light])
                denom = ml + md
                mod[i] = (ml - md) / denom if denom > 0 else 0
        result["light_mod"] = mod

    return result
