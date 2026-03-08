"""Population-level analysis for HD cell ensembles.

PCA dimensionality, pairwise correlations, population vector analysis,
and ensemble coherence metrics. All functions are pure numpy.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def population_pca(
    signals: npt.NDArray[np.floating],
    n_components: int | None = None,
) -> dict:
    """PCA of population activity.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
        Neural signals (mean-subtracted internally).
    n_components : int or None
        Number of components to return. None = all.

    Returns
    -------
    dict
        ``"components"`` — (n_comp, n_frames) principal components.
        ``"explained_variance_ratio"`` — fraction of variance per component.
        ``"cumulative_variance"`` — cumulative variance explained.
        ``"n_components_95"`` — number of components for 95% variance.
        ``"loadings"`` — (n_cells, n_comp) loadings matrix.
    """
    n_cells, n_frames = signals.shape
    if n_components is None:
        n_components = min(n_cells, n_frames)
    n_components = min(n_components, n_cells, n_frames)

    # Mean-subtract
    X = signals - signals.mean(axis=1, keepdims=True)

    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained = (S**2) / np.sum(S**2)
    cumulative = np.cumsum(explained)

    # Number of components for 95% variance
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)
    n95 = min(n95, len(cumulative))

    return {
        "components": Vt[:n_components],
        "explained_variance_ratio": explained[:n_components],
        "cumulative_variance": cumulative[:n_components],
        "n_components_95": n95,
        "loadings": U[:, :n_components] * S[:n_components],
    }


def pairwise_correlations(
    signals: npt.NDArray[np.floating],
) -> np.ndarray:
    """Pairwise Pearson correlation matrix.

    Parameters
    ----------
    signals : (n_cells, n_frames) float

    Returns
    -------
    corr : (n_cells, n_cells) float
        Correlation matrix.
    """
    return np.corrcoef(signals)


def population_vector_correlation(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_bins: int = 36,
) -> np.ndarray:
    """Population vector correlation matrix between HD bins.

    For each pair of HD bins, computes the correlation between the
    population vectors (mean activity across cells). This reveals the
    circular structure of HD encoding.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_bins : int

    Returns
    -------
    pv_corr : (n_bins, n_bins) float
        Population vector correlation matrix.
    """
    n_cells = signals.shape[0]
    hd_mod = np.mod(hd_deg[mask], 360.0)
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_idx = np.clip(np.digitize(hd_mod, bin_edges) - 1, 0, n_bins - 1)

    # Mean population vector per HD bin
    pop_vectors = np.zeros((n_bins, n_cells), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    sig_masked = signals[:, mask]
    for i in range(len(bin_idx)):
        pop_vectors[bin_idx[i]] += sig_masked[:, i]
        counts[bin_idx[i]] += 1

    # Normalise
    occupied = counts > 0
    pop_vectors[occupied] /= counts[occupied, None]

    # Correlation matrix (only between occupied bins)
    pv_corr = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    for i in range(n_bins):
        if not occupied[i]:
            continue
        for j in range(n_bins):
            if not occupied[j]:
                continue
            vi = pop_vectors[i]
            vj = pop_vectors[j]
            std_i = np.std(vi)
            std_j = np.std(vj)
            if std_i > 0 and std_j > 0:
                pv_corr[i, j] = float(np.corrcoef(vi, vj)[0, 1])
            else:
                pv_corr[i, j] = 0.0

    return pv_corr


def ensemble_coherence(
    signals: npt.NDArray[np.floating],
    window_frames: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Population coherence over time using mean pairwise correlation.

    Parameters
    ----------
    signals : (n_cells, n_frames) float
    window_frames : int

    Returns
    -------
    centers : (n_windows,) int
        Frame centres.
    coherence : (n_windows,) float
        Mean pairwise correlation per window.
    """
    n_cells, n_frames = signals.shape
    centers = []
    coherence = []

    for start in range(0, n_frames - window_frames + 1, window_frames // 2):
        end = start + window_frames
        chunk = signals[:, start:end]
        corr = np.corrcoef(chunk)
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        if mask.sum() > 0:
            mean_corr = float(np.nanmean(corr[mask]))
        else:
            mean_corr = 0.0
        centers.append(start + window_frames // 2)
        coherence.append(mean_corr)

    return np.array(centers, dtype=int), np.array(coherence, dtype=np.float64)
