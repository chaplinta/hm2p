"""Population geometry and topology of the RSP head-direction code.

If the simultaneously imaged population carries a one-dimensional head-direction
signal, its state space should be organised as a ring. This module provides
three complementary read-outs of that structure:

1. A pure-numpy ring test on the first two principal components (angle coverage
   uniformity, radius consistency, and circular correlation between the
   embedding angle and measured head direction). It requires no optional
   dependencies.
2. Persistent homology of the embedded point cloud (a single long-lived H1
   feature indicates a ring), computed with ``ripser`` when available.
3. A CEBRA embedding, optionally supervised by circular head-direction labels.

Cell counts differ between the Penk+ and Penk-CamKII+ populations, so
:func:`decoding_at_matched_n` equalises the number of ROIs by subsampling
before decoding.

All required computation is pure numpy/scipy/scikit-learn; ``ripser`` and
``cebra`` are imported lazily inside the functions that need them.

References
----------
Chaudhuri et al. 2019. "The intrinsic attractor manifold and population
dynamics of a canonical cognitive circuit across waking and sleep."
Nature Neuroscience 22:1512-1520. doi:10.1038/s41593-019-0460-x

Tralie et al. 2018. "Ripser.py: A Lean Persistent Homology Library for Python."
Journal of Open Source Software 3(29):925. doi:10.21105/joss.00925
https://github.com/scikit-tda/ripser.py

Schneider, Lee & Mathis 2023. "Learnable latent embeddings for joint
behavioural and neural analysis." Nature 617:360-368.
doi:10.1038/s41586-023-06031-6
https://github.com/AdaptiveMotorControlLab/CEBRA

Jammalamadaka & SenGupta 2001. "Topics in Circular Statistics."
World Scientific. doi:10.1142/4031
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hm2p.analysis.comparison import rayleigh_test

_RING_KEYS = (
    "radius_cv",
    "angle_uniformity_p",
    "angle_hd_correlation",
    "mean_radius",
    "rayleigh_z",
    "n_points",
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _zscore_rows(signals: npt.NDArray[np.floating]) -> np.ndarray:
    """Z-score each row (ROI) across time without warning on constant rows.

    Parameters
    ----------
    signals : (n_rois, n_frames) float

    Returns
    -------
    (n_rois, n_frames) float
        Z-scored traces; constant rows become all-zero rows.
    """
    x = np.asarray(signals, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    good = np.isfinite(std) & (std > 0.0)
    out = np.zeros_like(x)
    np.divide(x - mean, np.where(good, std, 1.0), out=out, where=good)
    return out


def circular_correlation(
    alpha_deg: npt.NDArray[np.floating],
    beta_deg: npt.NDArray[np.floating],
) -> float:
    """Circular-circular correlation coefficient between two angular variables.

    Uses the rotation-invariant (mean-free) form of the circular correlation
    coefficient, built from pairwise angular differences:

        r = sum_ij sin(a_i - a_j) sin(b_i - b_j) /
            sqrt(sum_ij sin^2(a_i - a_j) * sum_ij sin^2(b_i - b_j))

    The double sums are evaluated in O(n) from trigonometric moments. The
    sine-of-deviation-from-the-circular-mean form of Jammalamadaka & SenGupta
    (2001) is not used here because both variables of interest — embedding
    angle and head direction — are close to uniformly distributed over the
    circle, so their sample circular means (and therefore that estimator) are
    numerically ill-defined. The mean-free form is invariant to an arbitrary
    rotation of either variable, which is exactly the required property: a PCA
    embedding fixes the ring only up to rotation and reflection.

    Parameters
    ----------
    alpha_deg, beta_deg : (n,) float
        Paired angles in degrees. Non-finite pairs are dropped.

    Returns
    -------
    float
        Coefficient in [-1, 1]; +1 for ``beta = alpha + const``, -1 for a
        reflected relationship (``beta = -alpha + const``), and NaN if fewer
        than two usable pairs remain or either variable has no angular
        dispersion.

    Raises
    ------
    ValueError
        If the two inputs have different lengths.

    References
    ----------
    Fisher & Lee 1983. "A correlation coefficient for circular data."
    Biometrika 70(2):327-332. doi:10.1093/biomet/70.2.327

    Jammalamadaka & SenGupta 2001. "Topics in Circular Statistics."
    World Scientific. doi:10.1142/4031
    """
    a = np.asarray(alpha_deg, dtype=np.float64).ravel()
    b = np.asarray(beta_deg, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"alpha_deg and beta_deg must have equal length; got {a.size}, {b.size}")

    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n < 2:
        return float("nan")
    a_rad = np.deg2rad(a[finite])
    b_rad = np.deg2rad(b[finite])

    sa, ca = np.sin(a_rad), np.cos(a_rad)
    sb, cb = np.sin(b_rad), np.cos(b_rad)

    numerator = float((sa @ sb) * (ca @ cb) - (sa @ cb) * (ca @ sb))
    disp_a = float((sa @ sa) * (ca @ ca) - (sa @ ca) ** 2)
    disp_b = float((sb @ sb) * (cb @ cb) - (sb @ cb) ** 2)

    tol = 1e-12 * float(n) ** 2
    if disp_a <= tol or disp_b <= tol:
        return float("nan")
    return float(np.clip(numerator / np.sqrt(disp_a * disp_b), -1.0, 1.0))


def _empty_ring_result() -> dict:
    """Ring-score dictionary filled with NaN, for conditions with too few frames.

    Returns
    -------
    dict
        The keys of :func:`ring_score_pca` with NaN values (``n_points`` is 0).
    """
    result = dict.fromkeys(_RING_KEYS, float("nan"))
    result["n_points"] = 0
    return result


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def population_embedding_pca(
    signals: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_components: int = 3,
    smooth_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-dimensional PCA embedding of population activity.

    Traces are optionally smoothed along time, restricted to the masked frames,
    z-scored per ROI, and projected onto their leading principal components.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    mask : (n_frames,) bool
        Frames to embed.
    n_components : int
        Number of components requested. Reduced to
        ``min(n_components, n_rois, n_valid_frames)``.
    smooth_frames : int
        Width of the Gaussian temporal smoothing kernel in frames
        (sigma = ``smooth_frames / 3``). Values <= 1 disable smoothing.
        Smoothing is applied to the full traces before masking so that the
        kernel operates on contiguous time.

    Returns
    -------
    embedding : (n_valid, n_components_used) float
        Projected population state per valid frame.
    explained_variance_ratio : (n_components_used,) float

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``n_components`` < 1, or fewer than two
        valid frames are available.
    """
    from sklearn.decomposition import PCA

    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    n_rois, n_frames = x.shape
    m = np.asarray(mask, dtype=bool)
    if m.shape != (n_frames,):
        raise ValueError(f"mask must have shape ({n_frames},); got {m.shape}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")
    n_valid = int(m.sum())
    if n_valid < 2:
        raise ValueError(f"need at least 2 valid frames to embed; got {n_valid}")

    if smooth_frames > 1:
        from scipy.ndimage import gaussian_filter1d

        x = gaussian_filter1d(x, sigma=smooth_frames / 3.0, axis=1)

    z = _zscore_rows(x[:, m])
    k = int(min(n_components, n_rois, n_valid))
    pca = PCA(n_components=k)
    embedding = pca.fit_transform(z.T)
    return np.asarray(embedding, dtype=np.float64), np.asarray(
        pca.explained_variance_ratio_, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Pure-numpy ring test
# ---------------------------------------------------------------------------


def ring_score_pca(
    embedding: npt.NDArray[np.floating],
    hd_deg_valid: npt.NDArray[np.floating] | None = None,
) -> dict:
    """Ring test on the PC1-PC2 plane of a population embedding.

    A ring manifold produces (i) angles that cover the full circle roughly
    uniformly, so the Rayleigh test of non-uniformity is *not* significant
    (large ``angle_uniformity_p``), and (ii) a near-constant radius (small
    ``radius_cv``). If head direction is supplied, the circular correlation
    between the embedding angle and head direction tests whether the ring is the
    head-direction ring rather than some other closed trajectory.

    Parameters
    ----------
    embedding : (n_points, n_components) float
        Population embedding; only the first two components are used.
    hd_deg_valid : (n_points,) float or None
        Head direction for the same frames, in degrees. If None,
        ``angle_hd_correlation`` is NaN.

    Returns
    -------
    dict
        ``"angle_deg"`` — (n_points,) embedding angle in [0, 360).
        ``"radius"`` — (n_points,) distance from the PC-plane origin.
        ``"mean_radius"`` — mean radius.
        ``"radius_cv"`` — coefficient of variation of the radius (NaN if the
        mean radius is zero); low values indicate an annulus.
        ``"rayleigh_z"`` — Rayleigh Z of the embedding angles.
        ``"angle_uniformity_p"`` — Rayleigh p-value; large = uniform coverage.
        ``"angle_hd_correlation"`` — circular correlation with head direction.
        ``"n_points"`` — number of points used.

    Raises
    ------
    ValueError
        If the embedding has fewer than two components or ``hd_deg_valid``
        does not match the number of points.

    References
    ----------
    Chaudhuri et al. 2019. "The intrinsic attractor manifold and population
    dynamics of a canonical cognitive circuit across waking and sleep."
    Nature Neuroscience 22:1512-1520. doi:10.1038/s41593-019-0460-x
    """
    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] < 2:
        raise ValueError(f"embedding must be (n_points, >=2 components); got shape {emb.shape}")
    n_points = emb.shape[0]

    angle_deg = np.mod(np.rad2deg(np.arctan2(emb[:, 1], emb[:, 0])), 360.0)
    radius = np.sqrt(emb[:, 0] ** 2 + emb[:, 1] ** 2)

    mean_radius = float(np.mean(radius)) if n_points > 0 else float("nan")
    if n_points > 0 and mean_radius > 0:
        radius_cv = float(np.std(radius) / mean_radius)
    else:
        radius_cv = float("nan")

    rayleigh = rayleigh_test(angle_deg)

    if hd_deg_valid is None:
        angle_hd_corr = float("nan")
    else:
        hd = np.asarray(hd_deg_valid, dtype=np.float64).ravel()
        if hd.size != n_points:
            raise ValueError(f"hd_deg_valid must have {n_points} entries; got {hd.size}")
        angle_hd_corr = circular_correlation(angle_deg, hd)

    return {
        "angle_deg": angle_deg,
        "radius": radius,
        "mean_radius": mean_radius,
        "radius_cv": radius_cv,
        "rayleigh_z": float(rayleigh["z"]),
        "angle_uniformity_p": float(rayleigh["p_value"]),
        "angle_hd_correlation": angle_hd_corr,
        "n_points": int(n_points),
    }


# ---------------------------------------------------------------------------
# Persistent homology
# ---------------------------------------------------------------------------


def persistence_diagram(
    embedding: npt.NDArray[np.floating],
    max_dim: int = 1,
    n_landmarks: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Persistence diagrams of a population embedding (Vietoris-Rips, ripser).

    Parameters
    ----------
    embedding : (n_points, n_components) float
        Point cloud, e.g. the output of :func:`population_embedding_pca`.
    max_dim : int
        Maximum homology dimension (1 = loops).
    n_landmarks : int or None
        If given and smaller than ``n_points``, a random subset of this many
        points is used. Rips complexes scale poorly with point count, so this
        keeps runtime tractable for long sessions.
    rng : numpy.random.Generator or None
        Generator used for landmark subsampling.

    Returns
    -------
    dict
        ``"dgms"`` — list of (n_features, 2) birth/death arrays, one per
        homology dimension.
        ``"n_points_used"`` — number of points passed to ripser.
        ``"max_dim"`` — the requested maximum dimension.

    Raises
    ------
    ImportError
        If ``ripser`` is not installed.
    ValueError
        If the embedding is not 2-D, is empty, or ``n_landmarks`` < 1.
    """
    try:
        import ripser
    except ImportError as exc:
        raise ImportError(
            "ripser is required for persistent homology; install it with 'uv pip install ripser'"
        ) from exc

    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] < 1:
        raise ValueError(f"embedding must be a non-empty 2-D array; got shape {emb.shape}")
    if n_landmarks is not None and n_landmarks < 1:
        raise ValueError(f"n_landmarks must be >= 1 or None; got {n_landmarks}")

    if n_landmarks is not None and n_landmarks < emb.shape[0]:
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(emb.shape[0], size=n_landmarks, replace=False)
        emb = emb[np.sort(idx)]

    result = ripser.ripser(emb, maxdim=max_dim)
    dgms = [np.asarray(d, dtype=np.float64) for d in result["dgms"]]
    return {"dgms": dgms, "n_points_used": int(emb.shape[0]), "max_dim": int(max_dim)}


def ring_score_persistence(dgms: list[npt.NDArray[np.floating]]) -> dict:
    """Ring score from an H1 persistence diagram.

    A ring manifold yields one H1 feature that lives much longer than all
    others, so ``ring_score`` approaches 1. Several comparable features (or
    none) give a score near 0.

    Parameters
    ----------
    dgms : list of (n_features, 2) float
        Persistence diagrams indexed by homology dimension, as returned by
        :func:`persistence_diagram`. Element 1 is the H1 diagram; infinite
        death times are ignored.

    Returns
    -------
    dict
        ``"h1_max_persistence"`` — longest H1 lifetime (0 if none).
        ``"h1_second_persistence"`` — second-longest H1 lifetime (0 if none).
        ``"ring_score"`` — (max - second) / max, in [0, 1]; 0 if no H1 feature.
        ``"n_h1_features"`` — number of finite H1 features.
        ``"n_h1_features_above"`` — features with lifetime > 0.5 * max.

    References
    ----------
    Chaudhuri et al. 2019. "The intrinsic attractor manifold and population
    dynamics of a canonical cognitive circuit across waking and sleep."
    Nature Neuroscience 22:1512-1520. doi:10.1038/s41593-019-0460-x
    """
    empty = {
        "h1_max_persistence": 0.0,
        "h1_second_persistence": 0.0,
        "ring_score": 0.0,
        "n_h1_features": 0,
        "n_h1_features_above": 0,
    }
    if dgms is None or len(dgms) < 2:
        return empty

    h1 = np.asarray(dgms[1], dtype=np.float64)
    if h1.size == 0:
        return empty
    if h1.ndim != 2 or h1.shape[1] != 2:
        raise ValueError(f"H1 diagram must have shape (n_features, 2); got {h1.shape}")

    lifetimes = h1[:, 1] - h1[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if lifetimes.size == 0:
        return empty

    ordered = np.sort(lifetimes)[::-1]
    h1_max = float(ordered[0])
    h1_second = float(ordered[1]) if ordered.size > 1 else 0.0
    ring_score = float((h1_max - h1_second) / h1_max) if h1_max > 0 else 0.0

    return {
        "h1_max_persistence": h1_max,
        "h1_second_persistence": h1_second,
        "ring_score": ring_score,
        "n_h1_features": int(ordered.size),
        "n_h1_features_above": int(np.sum(ordered > 0.5 * h1_max)),
    }


# ---------------------------------------------------------------------------
# CEBRA
# ---------------------------------------------------------------------------


def cebra_embedding(
    signals: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    hd_deg: npt.NDArray[np.floating] | None = None,
    output_dim: int = 3,
    max_iterations: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """CEBRA embedding of population activity, optionally guided by head direction.

    With ``hd_deg`` supplied the model is trained behaviour-contrastively on
    circular labels (sine and cosine of head direction); otherwise it is trained
    time-contrastively, which uses no behavioural labels at all.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    mask : (n_frames,) bool
        Frames to embed.
    hd_deg : (n_frames,) float or None
        Head direction in degrees. None selects time-contrastive training.
    output_dim : int
        Dimensionality of the embedding.
    max_iterations : int
        Training iterations.
    rng : numpy.random.Generator or None
        If given, used to derive a torch seed for reproducibility.

    Returns
    -------
    embedding : (n_valid, output_dim) float

    Raises
    ------
    ImportError
        If ``cebra`` is not installed.
    ValueError
        If shapes are inconsistent or fewer than two valid frames exist.

    References
    ----------
    Schneider, Lee & Mathis 2023. "Learnable latent embeddings for joint
    behavioural and neural analysis." Nature 617:360-368.
    doi:10.1038/s41586-023-06031-6
    https://github.com/AdaptiveMotorControlLab/CEBRA
    """
    try:
        import cebra
    except ImportError as exc:
        raise ImportError(
            "cebra is required for CEBRA embeddings; install it with "
            "'uv pip install cebra' (note: CEBRA currently pins numpy<2)"
        ) from exc

    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    n_frames = x.shape[1]
    m = np.asarray(mask, dtype=bool)
    if m.shape != (n_frames,):
        raise ValueError(f"mask must have shape ({n_frames},); got {m.shape}")
    if int(m.sum()) < 2:
        raise ValueError(f"need at least 2 valid frames; got {int(m.sum())}")

    if rng is not None:
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover - torch ships with cebra
            pass

    neural = _zscore_rows(x[:, m]).T

    if hd_deg is None:
        model = cebra.CEBRA(
            output_dimension=output_dim,
            max_iterations=max_iterations,
            conditional="time",
            verbose=False,
        )
        model.fit(neural)
    else:
        hd = np.asarray(hd_deg, dtype=np.float64)
        if hd.shape != (n_frames,):
            raise ValueError(f"hd_deg must have shape ({n_frames},); got {hd.shape}")
        hd_rad = np.deg2rad(hd[m])
        labels = np.column_stack([np.sin(hd_rad), np.cos(hd_rad)])
        model = cebra.CEBRA(
            output_dimension=output_dim,
            max_iterations=max_iterations,
            conditional="time_delta",
            verbose=False,
        )
        model.fit(neural, labels)

    return np.asarray(model.transform(neural), dtype=np.float64)


# ---------------------------------------------------------------------------
# Cell-count matched decoding
# ---------------------------------------------------------------------------


def decoding_at_matched_n(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    n_cells: int,
    n_draws: int = 20,
    rng: np.random.Generator | None = None,
    n_folds: int = 5,
) -> dict:
    """Cross-validated HD decoding from a fixed number of randomly drawn ROIs.

    Decoding accuracy grows with the number of contributing cells, so comparing
    populations of different size is confounded. Repeatedly subsampling the same
    number of ROIs equalises that.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
    n_cells : int
        Number of ROIs per draw; must be in [1, n_rois].
    n_draws : int
        Number of independent subsamples.
    rng : numpy.random.Generator or None
    n_folds : int
        Folds for :func:`hm2p.analysis.decoder.cross_validated_decode`.

    Returns
    -------
    dict
        ``"errors"`` — (n_draws,) median absolute decoding error per draw (deg).
        ``"median"`` — median of ``errors`` across draws.
        ``"iqr"`` — interquartile range of ``errors``.
        ``"n_cells"`` — cells per draw.
        ``"n_draws"`` — number of draws.

    Raises
    ------
    ValueError
        If ``n_cells`` is outside [1, n_rois] or ``n_draws`` < 1.
    """
    from hm2p.analysis.decoder import cross_validated_decode

    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    n_rois = x.shape[0]
    if not 1 <= n_cells <= n_rois:
        raise ValueError(f"n_cells must be in [1, {n_rois}]; got {n_cells}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1; got {n_draws}")
    if rng is None:
        rng = np.random.default_rng()

    errors = np.full(n_draws, np.nan, dtype=np.float64)
    for draw in range(n_draws):
        idx = rng.choice(n_rois, size=n_cells, replace=False)
        result = cross_validated_decode(
            x[np.sort(idx)],
            hd_deg,
            mask,
            n_folds=n_folds,
            rng=rng,
        )
        errors[draw] = float(result["errors"]["median_abs_error"])

    return {
        "errors": errors,
        "median": float(np.median(errors)),
        "iqr": float(np.percentile(errors, 75) - np.percentile(errors, 25)),
        "n_cells": int(n_cells),
        "n_draws": int(n_draws),
    }


# ---------------------------------------------------------------------------
# Session-level summary
# ---------------------------------------------------------------------------


def topology_summary(
    signals: npt.NDArray[np.floating],
    hd_deg: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    light_on: npt.NDArray[np.bool_],
    n_cells_matched: int | None = None,
    rng: np.random.Generator | None = None,
    n_components: int = 3,
    n_landmarks: int | None = 500,
) -> dict:
    """Ring-structure summary for one session, split by room-light condition.

    Parameters
    ----------
    signals : (n_rois, n_frames) float
    hd_deg : (n_frames,) float
    mask : (n_frames,) bool
        Valid frames.
    light_on : (n_frames,) bool
        Room-light state per frame.
    n_cells_matched : int or None
        If given, also run :func:`decoding_at_matched_n` with this many ROIs.
    rng : numpy.random.Generator or None
    n_components : int
        Components for the PCA embedding.
    n_landmarks : int or None
        Landmark subsample size for the persistence computation.

    Returns
    -------
    dict
        ``"pca_all"``, ``"pca_light"``, ``"pca_dark"`` — :func:`ring_score_pca`
        results (without the per-frame ``angle_deg``/``radius`` arrays).
        ``"explained_variance_all"`` — explained variance ratio of the all-frame
        embedding (empty array if it could not be computed).
        ``"ripser_available"`` — whether persistent homology could be run.
        ``"persistence"`` — :func:`ring_score_persistence` result, or NaN
        ``ring_score`` when ripser is unavailable.
        ``"matched_decoding"`` — :func:`decoding_at_matched_n` result or None.

    Raises
    ------
    ValueError
        If array shapes are inconsistent.
    """
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames); got shape {x.shape}")
    n_frames = x.shape[1]
    hd = np.asarray(hd_deg, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    light = np.asarray(light_on, dtype=bool)
    if hd.shape != (n_frames,) or m.shape != (n_frames,) or light.shape != (n_frames,):
        raise ValueError(
            "hd_deg, mask, and light_on must have shape (n_frames,); "
            f"got {hd.shape}, {m.shape}, {light.shape} for n_frames={n_frames}"
        )

    conditions = {"all": m, "light": m & light, "dark": m & ~light}
    summary: dict = {}
    embedding_all: np.ndarray | None = None
    summary["explained_variance_all"] = np.zeros(0, dtype=np.float64)

    for name, cond_mask in conditions.items():
        if int(cond_mask.sum()) < 2 or x.shape[0] < 2:
            summary[f"pca_{name}"] = _empty_ring_result()
            continue
        embedding, evr = population_embedding_pca(x, cond_mask, n_components=n_components)
        if embedding.shape[1] < 2:
            summary[f"pca_{name}"] = _empty_ring_result()
            continue
        ring = ring_score_pca(embedding, hd[cond_mask])
        ring.pop("angle_deg")
        ring.pop("radius")
        summary[f"pca_{name}"] = ring
        if name == "all":
            embedding_all = embedding
            summary["explained_variance_all"] = evr

    if embedding_all is None:
        summary["ripser_available"] = False
        summary["persistence"] = {**ring_score_persistence([]), "ring_score": float("nan")}
    else:
        try:
            dgm = persistence_diagram(embedding_all, max_dim=1, n_landmarks=n_landmarks, rng=rng)
        except ImportError:
            summary["ripser_available"] = False
            summary["persistence"] = {
                **ring_score_persistence([]),
                "ring_score": float("nan"),
            }
        else:
            summary["ripser_available"] = True
            summary["persistence"] = ring_score_persistence(dgm["dgms"])

    if n_cells_matched is None:
        summary["matched_decoding"] = None
    else:
        summary["matched_decoding"] = decoding_at_matched_n(
            x, hd, m, n_cells=n_cells_matched, rng=rng
        )

    return summary
