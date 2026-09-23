"""Poisson GLM encoding models for RSP neurons.

Fits per-ROI Poisson generalised linear models that predict per-frame activity
(event counts, deconvolved spike rate, or CASCADE spike rate) from behavioural
variables: head direction, angular head velocity, speed, 2-D position, light
condition, and optional behavioural syllable identity. Each variable is mapped
through a basis (raised-cosine bumps for 1-D variables, Gaussian bumps for
position, one-hot for categorical) so tuning shape is not assumed.

The model-selection design follows the multiplexed-coding analysis of

    Hardcastle et al. 2017. "A Multiplexed, Heterogeneous, and Adaptive Code
    for Navigation in Medial Entorhinal Cortex." Neuron 94:375-387.
    doi:10.1016/j.neuron.2017.03.025

i.e. forward stepwise selection on cross-validated held-out log-likelihood,
with the decision to keep a variable made by a signed-rank test across folds
(the original uses a one-sided paired test on per-fold held-out likelihood;
here the non-parametric Wilcoxon signed-rank test is used throughout, per the
project's non-parametric-only statistics policy).

Two fitting backends are available. The default uses scikit-learn's
``PoissonRegressor`` (ridge-penalised Poisson regression). The optional
``"nemos"`` backend uses NEMOS,

    Balzani et al. 2024. "NEMOS: Neural Encoding Models in Python."
    https://github.com/flatironinstitute/nemos

which is declared in the ``analysis`` extra and imported lazily.

Scientific context: the encoding profile (fraction of explained deviance
attributable to each variable) is the quantity compared between Penk+ and
Penk-CamKII+ RSP populations (hypothesis H8).

All functions are pure numpy / scikit-learn — no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import PoissonRegressor

__all__ = [
    "DesignMatrix",
    "FittedGLM",
    "build_design_matrix",
    "circular_basis",
    "cross_validated_deviance",
    "deviance_explained",
    "drop_one_contributions",
    "encoding_profile",
    "fit_poisson_glm",
    "forward_selection",
    "linear_basis",
    "onehot_basis",
    "poisson_deviance",
    "population_encoding_profiles",
    "position_basis",
    "predict_rate",
    "valid_rows",
]

# Linear-predictor clip: exp(36) ~ 4e15, far above any plausible rate, and
# keeps exp() away from overflow for pathological coefficients.
_ETA_CLIP = 36.0
# Floor for predicted rates so log(mu) stays finite.
_RATE_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class DesignMatrix:
    """Design matrix plus the column blocks belonging to each variable.

    Attributes
    ----------
    X : (n_frames, n_features) float
        Basis-expanded regressors. Rows where ``valid`` is False are zero.
        No intercept column — the intercept is handled by the model.
    groups : dict[str, (n_cols,) int]
        Column indices of each variable block, keyed by variable name
        (``"hd"``, ``"ahv"``, ``"speed"``, ``"position"``, ``"light"``,
        ``"syllable"``).
    valid : (n_frames,) bool
        Frames where every supplied continuous variable is finite.
    """

    X: npt.NDArray[np.float64]
    groups: dict[str, npt.NDArray[np.int_]] = field(default_factory=dict)
    valid: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n_frames(self) -> int:
        """Number of frames (rows of ``X``)."""
        return int(self.X.shape[0])

    @property
    def variables(self) -> list[str]:
        """Variable names, in the order their columns appear in ``X``."""
        return list(self.groups)

    def subset(self, names: list[str] | tuple[str, ...]) -> npt.NDArray[np.float64]:
        """Return the columns of ``X`` belonging to ``names``.

        Parameters
        ----------
        names : sequence of str
            Variable names; must all be present in ``groups``.

        Returns
        -------
        (n_frames, n_selected) float
            Column subset, in ``groups`` order (not the order of ``names``).

        Raises
        ------
        KeyError
            If a name is not a variable of this design matrix.
        """
        unknown = [n for n in names if n not in self.groups]
        if unknown:
            raise KeyError(f"unknown design variables: {unknown}")
        keep = [self.groups[n] for n in self.groups if n in names]
        if not keep:
            return np.zeros((self.n_frames, 0), dtype=np.float64)
        cols = np.concatenate(keep)
        return np.asarray(self.X[:, cols], dtype=np.float64)


@dataclass
class FittedGLM:
    """Coefficients of a fitted Poisson GLM with an exponential link."""

    coef: npt.NDArray[np.float64]
    intercept: float
    backend: str = "sklearn"


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------


def _raised_cosine(delta: npt.NDArray[np.floating], width: float) -> npt.NDArray[np.float64]:
    """Raised-cosine bump ``0.5 * (1 + cos(pi * delta / width))``.

    Parameters
    ----------
    delta : ndarray
        Signed offsets from the bump centre.
    width : float
        Half-width of the support; the bump is zero for ``|delta| >= width``.

    Returns
    -------
    ndarray
        Bump values in [0, 1], same shape as ``delta``.
    """
    d = np.asarray(delta, dtype=np.float64)
    out = np.zeros(d.shape, dtype=np.float64)
    inside = np.abs(d) < width
    out[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside] / width))
    return out


def _resolve_range(
    values: npt.NDArray[np.float64],
    value_range: tuple[float, float] | None,
) -> tuple[float, float]:
    """Pick a finite (lo, hi) range for a 1-D variable.

    Parameters
    ----------
    values : (n,) float
        Variable values; non-finite entries are ignored.
    value_range : (float, float) or None
        Explicit range. If None, the finite data range is used, widened to a
        unit interval when the variable is constant or fully non-finite.

    Returns
    -------
    (lo, hi) : tuple of float
        With ``hi > lo``.

    Raises
    ------
    ValueError
        If ``value_range`` is given with ``hi <= lo`` or non-finite bounds.
    """
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            raise ValueError(f"value_range must be finite with hi > lo, got {value_range}")
        return lo, hi
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0, 1.0
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo:
        return lo - 0.5, lo + 0.5
    return lo, hi


def circular_basis(
    hd_deg: npt.ArrayLike,
    n_basis: int = 12,
    width_deg: float | None = None,
) -> npt.NDArray[np.float64]:
    """Raised-cosine bumps tiling head direction over 0-360 degrees.

    Bump centres are evenly spaced; with the default width (equal to the centre
    spacing) every pair of neighbouring bumps forms a partition of unity, so
    each row sums to exactly 1 and the basis is continuous across 0/360.

    Parameters
    ----------
    hd_deg : (n_frames,) array_like
        Head direction in degrees; wrapping is handled internally.
        Non-finite entries give an all-zero row.
    n_basis : int
        Number of bumps (>= 2).
    width_deg : float, optional
        Half-width of each bump in degrees. Default ``360 / n_basis``.

    Returns
    -------
    (n_frames, n_basis) float
        Basis matrix.

    Raises
    ------
    ValueError
        If ``n_basis < 2`` or ``width_deg <= 0``.

    References
    ----------
    Hardcastle et al. 2017. "A Multiplexed, Heterogeneous, and Adaptive Code
    for Navigation in Medial Entorhinal Cortex." Neuron 94:375-387.
    doi:10.1016/j.neuron.2017.03.025
    """
    if n_basis < 2:
        raise ValueError(f"n_basis must be >= 2, got {n_basis}")
    hd = np.asarray(hd_deg, dtype=np.float64).ravel()
    spacing = 360.0 / n_basis
    width = spacing if width_deg is None else float(width_deg)
    if width <= 0:
        raise ValueError(f"width_deg must be > 0, got {width_deg}")
    centres = np.arange(n_basis, dtype=np.float64) * spacing
    finite = np.isfinite(hd)
    # Substitute a placeholder before the modulo so non-finite angles do not
    # raise invalid-value warnings; those rows are zeroed below.
    delta = np.where(finite, hd, 0.0)[:, None] - centres[None, :]
    delta = (delta + 180.0) % 360.0 - 180.0
    basis = _raised_cosine(delta, width)
    basis[~finite, :] = 0.0
    return basis


def linear_basis(
    values: npt.ArrayLike,
    n_basis: int = 8,
    value_range: tuple[float, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Raised-cosine bumps tiling a 1-D variable (e.g. AHV or speed).

    Centres span ``value_range`` inclusively and the bump half-width equals the
    centre spacing, so rows of finite, in-range values sum to exactly 1.
    Values outside the range are clipped to it; non-finite values give an
    all-zero row (use :func:`valid_rows` to obtain the corresponding mask).

    Parameters
    ----------
    values : (n_frames,) array_like
        Variable values.
    n_basis : int
        Number of bumps (>= 2).
    value_range : (float, float), optional
        Range to tile. Default: the finite data range.

    Returns
    -------
    (n_frames, n_basis) float
        Basis matrix.

    Raises
    ------
    ValueError
        If ``n_basis < 2`` or ``value_range`` is degenerate.
    """
    if n_basis < 2:
        raise ValueError(f"n_basis must be >= 2, got {n_basis}")
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = _resolve_range(v, value_range)
    centres = np.linspace(lo, hi, n_basis)
    width = float(centres[1] - centres[0])
    clipped = np.clip(v, lo, hi)
    basis = _raised_cosine(clipped[:, None] - centres[None, :], width)
    basis[~np.isfinite(v), :] = 0.0
    return basis


def _gaussian_bumps(
    values: npt.NDArray[np.float64],
    n_basis: int,
    value_range: tuple[float, float] | None,
) -> npt.NDArray[np.float64]:
    """Gaussian bumps on a 1-D variable, sigma equal to the centre spacing.

    Parameters
    ----------
    values : (n_frames,) float
        Variable values; clipped to the range, non-finite rows left as zeros.
    n_basis : int
        Number of bumps (>= 2).
    value_range : (float, float) or None
        Range to tile; default the finite data range.

    Returns
    -------
    (n_frames, n_basis) float
        Unnormalised Gaussian bump values.

    Raises
    ------
    ValueError
        If ``n_basis < 2``.
    """
    if n_basis < 2:
        raise ValueError(f"n_basis must be >= 2, got {n_basis}")
    lo, hi = _resolve_range(values, value_range)
    centres = np.linspace(lo, hi, n_basis)
    sigma = float(centres[1] - centres[0])
    clipped = np.clip(values, lo, hi)
    bumps = np.exp(-0.5 * ((clipped[:, None] - centres[None, :]) / sigma) ** 2)
    bumps[~np.isfinite(values), :] = 0.0
    return bumps


def position_basis(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n_x: int = 6,
    n_y: int = 6,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> npt.NDArray[np.float64]:
    """2-D Gaussian bumps on position, flattened to one column per bump.

    Column ``ix * n_y + iy`` is the product of the ``ix``-th x-bump and the
    ``iy``-th y-bump. Each row is normalised to sum to 1 (a soft assignment to
    the grid), so the position block has the same row scale as the
    raised-cosine blocks. Rows with a non-finite coordinate are all zero.

    Parameters
    ----------
    x, y : (n_frames,) array_like
        Position in mm.
    n_x, n_y : int
        Grid size (each >= 2).
    x_range, y_range : (float, float), optional
        Ranges to tile; default the finite data ranges.

    Returns
    -------
    (n_frames, n_x * n_y) float
        Basis matrix.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length, or a grid size is < 2.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    if xv.shape != yv.shape:
        raise ValueError(f"x and y must have the same length, got {xv.shape} and {yv.shape}")
    bx = _gaussian_bumps(xv, n_x, x_range)
    by = _gaussian_bumps(yv, n_y, y_range)
    basis = (bx[:, :, None] * by[:, None, :]).reshape(xv.shape[0], n_x * n_y)
    row_sum = basis.sum(axis=1)
    nonzero = row_sum > 0
    basis[nonzero] /= row_sum[nonzero, None]
    return basis


def onehot_basis(
    ids: npt.ArrayLike,
    n_levels: int | None = None,
    ignore: int = -1,
) -> npt.NDArray[np.float64]:
    """One-hot encode integer labels (e.g. behavioural syllable identity).

    Parameters
    ----------
    ids : (n_frames,) array_like
        Integer labels. Entries equal to ``ignore``, non-finite, or outside
        ``[0, n_levels)`` give an all-zero row.
    n_levels : int, optional
        Number of columns. Default: ``max(ids) + 1`` over the kept entries
        (1 if nothing is kept).
    ignore : int
        Label meaning "unassigned".

    Returns
    -------
    (n_frames, n_levels) float
        One-hot matrix.

    Raises
    ------
    ValueError
        If ``n_levels`` is given and < 1.
    """
    raw = np.asarray(ids, dtype=np.float64).ravel()
    keep = np.isfinite(raw) & (raw != float(ignore))
    codes = np.where(keep, raw, -1.0).astype(np.int64)
    if n_levels is None:
        n_levels = int(codes[keep].max()) + 1 if keep.any() else 1
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    keep &= (codes >= 0) & (codes < n_levels)
    out = np.zeros((raw.shape[0], n_levels), dtype=np.float64)
    rows = np.flatnonzero(keep)
    out[rows, codes[rows]] = 1.0
    return out


def valid_rows(*arrays: npt.ArrayLike | None) -> npt.NDArray[np.bool_]:
    """Rows that are finite in every supplied array.

    Parameters
    ----------
    *arrays : array_like or None
        1-D arrays of equal length; ``None`` entries are skipped. Boolean and
        integer arrays are always finite.

    Returns
    -------
    (n_frames,) bool
        True where all arrays are finite.

    Raises
    ------
    ValueError
        If no non-None array is given, or lengths differ.
    """
    present = [np.asarray(a).ravel() for a in arrays if a is not None]
    if not present:
        raise ValueError("valid_rows needs at least one non-None array")
    lengths = {a.shape[0] for a in present}
    if len(lengths) > 1:
        raise ValueError(f"arrays must have equal length, got {sorted(lengths)}")
    mask = np.ones(present[0].shape[0], dtype=bool)
    for a in present:
        if a.dtype.kind in "fc":
            mask &= np.isfinite(a)
    return mask


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def build_design_matrix(
    hd_deg: npt.ArrayLike | None = None,
    ahv_deg_s: npt.ArrayLike | None = None,
    speed_cm_s: npt.ArrayLike | None = None,
    x_mm: npt.ArrayLike | None = None,
    y_mm: npt.ArrayLike | None = None,
    light_on: npt.ArrayLike | None = None,
    syllable_id: npt.ArrayLike | None = None,
    *,
    n_hd_basis: int = 12,
    hd_width_deg: float | None = None,
    n_ahv_basis: int = 8,
    ahv_range: tuple[float, float] | None = None,
    n_speed_basis: int = 8,
    speed_range: tuple[float, float] | None = None,
    n_x: int = 6,
    n_y: int = 6,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    n_syllables: int | None = None,
    syllable_ignore: int = -1,
) -> DesignMatrix:
    """Assemble a basis-expanded design matrix from per-frame behaviour.

    Variables are included only when supplied. Block order is hd, ahv, speed,
    position, light, syllable. No intercept column is added: the intercept is
    fitted by the model. Frames with a non-finite continuous variable are
    marked invalid and their rows are zeroed; an unassigned syllable
    (``syllable_id == syllable_ignore``) gives a zero syllable block but leaves
    the frame valid.

    Parameters
    ----------
    hd_deg : (n_frames,) array_like, optional
        Head direction in degrees -> :func:`circular_basis`.
    ahv_deg_s : (n_frames,) array_like, optional
        Angular head velocity in deg/s -> :func:`linear_basis`.
    speed_cm_s : (n_frames,) array_like, optional
        Speed in cm/s -> :func:`linear_basis`.
    x_mm, y_mm : (n_frames,) array_like, optional
        Position in mm -> :func:`position_basis`. Both are required together.
    light_on : (n_frames,) array_like of bool, optional
        Room-light state -> single 0/1 column.
    syllable_id : (n_frames,) array_like of int, optional
        Behavioural syllable -> :func:`onehot_basis`.
    n_hd_basis, hd_width_deg, n_ahv_basis, ahv_range, n_speed_basis, speed_range : ...
        Basis parameters for the 1-D variables.
    n_x, n_y, x_range, y_range : ...
        Basis parameters for position.
    n_syllables : int, optional
        Number of syllable levels; default inferred from the data.
    syllable_ignore : int
        Syllable label meaning "unassigned".

    Returns
    -------
    DesignMatrix

    Raises
    ------
    ValueError
        If no variable is supplied, only one position coordinate is supplied,
        or the supplied variables differ in length.
    """
    if (x_mm is None) != (y_mm is None):
        raise ValueError("x_mm and y_mm must be supplied together")

    blocks: list[tuple[str, npt.NDArray[np.float64]]] = []
    continuous: list[npt.ArrayLike] = []

    if hd_deg is not None:
        blocks.append(("hd", circular_basis(hd_deg, n_hd_basis, hd_width_deg)))
        continuous.append(hd_deg)
    if ahv_deg_s is not None:
        blocks.append(("ahv", linear_basis(ahv_deg_s, n_ahv_basis, ahv_range)))
        continuous.append(ahv_deg_s)
    if speed_cm_s is not None:
        blocks.append(("speed", linear_basis(speed_cm_s, n_speed_basis, speed_range)))
        continuous.append(speed_cm_s)
    if x_mm is not None and y_mm is not None:
        blocks.append(("position", position_basis(x_mm, y_mm, n_x, n_y, x_range, y_range)))
        continuous.extend([x_mm, y_mm])
    if light_on is not None:
        light = np.asarray(light_on).ravel().astype(np.float64)[:, None]
        blocks.append(("light", light))
        continuous.append(light.ravel())
    if syllable_id is not None:
        blocks.append(("syllable", onehot_basis(syllable_id, n_syllables, syllable_ignore)))

    if not blocks:
        raise ValueError("build_design_matrix needs at least one variable")

    lengths = {b.shape[0] for _, b in blocks}
    if len(lengths) > 1:
        raise ValueError(f"variables must have equal length, got {sorted(lengths)}")

    valid = valid_rows(*continuous) if continuous else np.ones(blocks[0][1].shape[0], bool)

    groups: dict[str, npt.NDArray[np.int_]] = {}
    start = 0
    for name, block in blocks:
        groups[name] = np.arange(start, start + block.shape[1])
        start += block.shape[1]

    design_x = np.concatenate([b for _, b in blocks], axis=1)
    design_x[~valid, :] = 0.0
    return DesignMatrix(X=design_x, groups=groups, valid=valid)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _check_xy(
    X: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate and coerce a design matrix / response pair.

    Parameters
    ----------
    X : (n_frames, n_features) array_like
    y : (n_frames,) array_like
        Non-negative, finite response.

    Returns
    -------
    (X, y) : tuple of ndarray

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, lengths disagree, or ``y`` has non-finite or
        negative values.
    """
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).ravel()
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"X has {Xa.shape[0]} rows but y has {ya.shape[0]}")
    if not np.all(np.isfinite(ya)):
        raise ValueError("y contains non-finite values")
    if np.any(ya < 0):
        raise ValueError("y must be non-negative (Poisson response)")
    return Xa, ya


def _fit_intercept_only(
    y: npt.NDArray[np.float64], backend: str, n_features: int = 0
) -> FittedGLM:
    """Fit the constant-rate model, with all regressor coefficients at zero.

    Parameters
    ----------
    y : (n_frames,) float
        Response.
    backend : str
        Recorded on the returned model.
    n_features : int
        Number of (zero) coefficients, so the model can still be applied to a
        design matrix with columns.

    Returns
    -------
    FittedGLM
        Zero ``coef`` and ``intercept = log(mean(y))``.
    """
    rate = float(np.mean(y)) if y.size else 0.0
    intercept = float(np.log(max(rate, _RATE_FLOOR)))
    return FittedGLM(
        coef=np.zeros(int(n_features), dtype=np.float64), intercept=intercept, backend=backend
    )


def _fit_nemos(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    alpha: float,
) -> FittedGLM:
    """Fit a ridge-penalised Poisson GLM with NEMOS.

    Parameters
    ----------
    X : (n_frames, n_features) float
    y : (n_frames,) float
    alpha : float
        Ridge regularisation strength.

    Returns
    -------
    FittedGLM

    Raises
    ------
    ImportError
        If nemos is not installed.

    References
    ----------
    Balzani et al. 2024. "NEMOS: Neural Encoding Models in Python."
    https://github.com/flatironinstitute/nemos
    """
    try:
        nmo = import_module("nemos")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "backend='nemos' requires the nemos package; install the analysis "
            "extra with `uv pip install 'hm2p[analysis]'` or `uv pip install nemos`"
        ) from exc
    model = nmo.glm.GLM(
        observation_model=nmo.observation_models.PoissonObservations(),
        regularizer=nmo.regularizer.Ridge(),
        regularizer_strength=float(alpha),
    )
    model.fit(X, y)
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).ravel()[0])
    return FittedGLM(coef=coef, intercept=intercept, backend="nemos")


def fit_poisson_glm(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    alpha: float = 1e-3,
    backend: str = "sklearn",
    max_iter: int = 500,
) -> FittedGLM:
    """Fit a ridge-penalised Poisson GLM with an exponential link.

    Parameters
    ----------
    X : (n_frames, n_features) array_like
        Design matrix without an intercept column.
    y : (n_frames,) array_like
        Non-negative response (event counts or spike rate per frame).
    alpha : float
        L2 penalty strength on the coefficients.
    backend : {"sklearn", "nemos"}
        ``"sklearn"`` uses :class:`sklearn.linear_model.PoissonRegressor`;
        ``"nemos"`` lazily imports NEMOS.
    max_iter : int
        Maximum solver iterations (sklearn backend).

    Returns
    -------
    FittedGLM

    Raises
    ------
    ValueError
        If the inputs are malformed or ``backend`` is unknown.
    ImportError
        If ``backend="nemos"`` and nemos is not installed.

    References
    ----------
    Hardcastle et al. 2017. "A Multiplexed, Heterogeneous, and Adaptive Code
    for Navigation in Medial Entorhinal Cortex." Neuron 94:375-387.
    doi:10.1016/j.neuron.2017.03.025

    Balzani et al. 2024. "NEMOS: Neural Encoding Models in Python."
    https://github.com/flatironinstitute/nemos
    """
    Xa, ya = _check_xy(X, y)
    if backend not in ("sklearn", "nemos"):
        raise ValueError(f"unknown backend {backend!r}, expected 'sklearn' or 'nemos'")
    if Xa.shape[1] == 0 or not np.any(ya > 0):
        # No regressors, or an entirely silent cell: the penalised fit is the
        # constant-rate model and solvers can fail on the degenerate problem.
        return _fit_intercept_only(ya, backend, Xa.shape[1])
    if backend == "nemos":
        return _fit_nemos(Xa, ya, alpha)
    reg = PoissonRegressor(alpha=float(alpha), max_iter=int(max_iter), fit_intercept=True)
    reg.fit(Xa, ya)
    return FittedGLM(
        coef=np.asarray(reg.coef_, dtype=np.float64).ravel(),
        intercept=float(reg.intercept_),
        backend="sklearn",
    )


def predict_rate(model: FittedGLM, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Predicted per-frame rate ``exp(X @ coef + intercept)``.

    Parameters
    ----------
    model : FittedGLM
    X : (n_frames, n_features) array_like
        Must have as many columns as ``model.coef``.

    Returns
    -------
    (n_frames,) float
        Predicted rates, floored at a small positive value.

    Raises
    ------
    ValueError
        If the column count does not match the coefficients.
    """
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xa.shape}")
    if Xa.shape[1] != model.coef.shape[0]:
        raise ValueError(
            f"X has {Xa.shape[1]} columns but model has {model.coef.shape[0]} coefficients"
        )
    eta = Xa @ model.coef + model.intercept
    return np.maximum(np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP)), _RATE_FLOOR)


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------


def poisson_deviance(y: npt.ArrayLike, mu: npt.ArrayLike) -> float:
    """Poisson deviance ``2 * sum(y * log(y / mu) - (y - mu))``.

    Frames where ``y`` or ``mu`` is non-finite are dropped. ``y * log(y)`` is
    taken as 0 at ``y = 0``. Valid for non-integer ``y`` (quasi-likelihood),
    which is what deconvolved / CASCADE rates are.

    Parameters
    ----------
    y : (n_frames,) array_like
        Observed non-negative response.
    mu : (n_frames,) array_like
        Predicted rate.

    Returns
    -------
    float
        Deviance; 0.0 if no frame is usable.

    Raises
    ------
    ValueError
        If the lengths differ.
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    mua = np.asarray(mu, dtype=np.float64).ravel()
    if ya.shape != mua.shape:
        raise ValueError(f"y and mu must have equal length, got {ya.shape} and {mua.shape}")
    ok = np.isfinite(ya) & np.isfinite(mua)
    if not ok.any():
        return 0.0
    yv = ya[ok]
    mv = np.maximum(mua[ok], _RATE_FLOOR)
    term = np.zeros_like(yv)
    pos = yv > 0
    # log(y) - log(mu) rather than log(y / mu): the ratio underflows to zero
    # for denormal y, which would send the deviance to -inf.
    term[pos] = yv[pos] * (np.log(yv[pos]) - np.log(mv[pos]))
    return float(2.0 * np.sum(term - (yv - mv)))


def deviance_explained(
    y: npt.ArrayLike,
    mu: npt.ArrayLike,
    y_baseline_rate: float | npt.ArrayLike | None = None,
) -> float:
    """Pseudo-R-squared: fraction of null deviance explained by the model.

    ``1 - D(y, mu) / D(y, baseline)``, where the baseline is a constant-rate
    model. Values are <= 1; held-out values can be negative when the model
    predicts worse than the constant rate.

    Parameters
    ----------
    y : (n_frames,) array_like
        Observed response.
    mu : (n_frames,) array_like
        Model prediction.
    y_baseline_rate : float or array_like, optional
        Constant rate of the null model. Default: the mean of the finite ``y``
        (use the training mean when scoring held-out data).

    Returns
    -------
    float
        0.0 when the null deviance is not positive (e.g. a silent cell).
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    if y_baseline_rate is None:
        finite = np.isfinite(ya)
        rate = float(np.mean(ya[finite])) if finite.any() else 0.0
        baseline = np.full(ya.shape, rate, dtype=np.float64)
    else:
        baseline = np.broadcast_to(np.asarray(y_baseline_rate, dtype=np.float64), ya.shape).astype(
            np.float64
        )
    null_dev = poisson_deviance(ya, baseline)
    if not np.isfinite(null_dev) or null_dev <= 0:
        return 0.0
    return float(1.0 - poisson_deviance(ya, mu) / null_dev)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _fold_indices(
    n_frames: int,
    n_folds: int,
    block: bool = True,
    rng: np.random.Generator | None = None,
) -> list[npt.NDArray[np.int_]]:
    """Held-out index arrays for each cross-validation fold.

    Parameters
    ----------
    n_frames : int
        Number of samples.
    n_folds : int
        Number of folds (>= 2, <= ``n_frames``).
    block : bool
        True gives contiguous blocks (appropriate for autocorrelated time
        series); False gives a random partition.
    rng : numpy.random.Generator, optional
        Used only when ``block`` is False.

    Returns
    -------
    list of (n_test,) int arrays

    Raises
    ------
    ValueError
        If ``n_folds < 2`` or ``n_folds > n_frames``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_folds > n_frames:
        raise ValueError(f"n_folds ({n_folds}) exceeds number of frames ({n_frames})")
    order = np.arange(n_frames)
    if not block:
        generator = np.random.default_rng() if rng is None else rng
        order = generator.permutation(order)
    return [np.asarray(part) for part in np.array_split(order, n_folds)]


def cross_validated_deviance(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    n_folds: int = 5,
    block: bool = True,
    alpha: float = 1e-3,
    backend: str = "sklearn",
    rng: np.random.Generator | None = None,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Cross-validated deviance explained of a Poisson GLM.

    Each fold is fitted on the remaining frames and scored on the held-out
    frames against a constant-rate null whose rate is the *training* mean, so
    no information leaks from the test block.

    Parameters
    ----------
    X : (n_frames, n_features) array_like
        Design matrix; may have zero columns (constant-rate model).
    y : (n_frames,) array_like
        Non-negative response.
    n_folds : int
        Number of folds.
    block : bool
        Contiguous blocks (default) or a random partition.
    alpha : float
        Ridge penalty.
    backend : {"sklearn", "nemos"}
    rng : numpy.random.Generator, optional
        Used only when ``block`` is False.
    max_iter : int
        Solver iterations (sklearn backend).

    Returns
    -------
    dict
        ``"deviance_explained"`` — (n_folds,) held-out deviance explained.
        ``"mean_deviance_explained"`` — float, mean over folds.
        ``"prediction"`` — (n_frames,) out-of-fold predicted rate.
        ``"fold"`` — (n_frames,) int fold assignment.
        ``"n_folds"`` — int.
    """
    Xa, ya = _check_xy(X, y)
    folds = _fold_indices(ya.shape[0], n_folds, block=block, rng=rng)
    scores = np.full(len(folds), np.nan, dtype=np.float64)
    prediction = np.full(ya.shape[0], np.nan, dtype=np.float64)
    fold_id = np.full(ya.shape[0], -1, dtype=np.int64)

    for k, test_idx in enumerate(folds):
        train_mask = np.ones(ya.shape[0], dtype=bool)
        train_mask[test_idx] = False
        model = fit_poisson_glm(
            Xa[train_mask], ya[train_mask], alpha=alpha, backend=backend, max_iter=max_iter
        )
        mu_test = predict_rate(model, Xa[test_idx])
        train_rate = float(np.mean(ya[train_mask])) if train_mask.any() else 0.0
        scores[k] = deviance_explained(ya[test_idx], mu_test, train_rate)
        prediction[test_idx] = mu_test
        fold_id[test_idx] = k

    return {
        "deviance_explained": scores,
        "mean_deviance_explained": float(np.mean(scores)),
        "prediction": prediction,
        "fold": fold_id,
        "n_folds": len(folds),
    }


# ---------------------------------------------------------------------------
# Variable contributions and model selection
# ---------------------------------------------------------------------------


def _valid_subset(
    design: DesignMatrix, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Frames that are valid in the design and finite in the response.

    Parameters
    ----------
    design : DesignMatrix
    y : (n_frames,) array_like
        Response; non-finite frames are dropped.

    Returns
    -------
    keep : (n_frames,) bool
        Frames to use.
    y_valid : (n_valid,) float
        Response restricted to those frames.

    Raises
    ------
    ValueError
        If lengths disagree or no frame survives.
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.shape[0] != design.n_frames:
        raise ValueError(f"y has {ya.shape[0]} frames but design has {design.n_frames}")
    keep = design.valid & np.isfinite(ya)
    if not keep.any():
        raise ValueError("no valid frames remain after masking")
    return keep, ya[keep]


def _cv_scores_for(
    design: DesignMatrix,
    y_valid: npt.NDArray[np.float64],
    keep: npt.NDArray[np.bool_],
    names: list[str],
    n_folds: int,
    block: bool,
    alpha: float,
    backend: str,
    rng: np.random.Generator | None,
    max_iter: int,
) -> npt.NDArray[np.float64]:
    """Per-fold held-out deviance explained for a subset of variables.

    Parameters
    ----------
    design : DesignMatrix
    y_valid : (n_valid,) float
        Response restricted to valid frames.
    keep : (n_frames,) bool
        Valid-frame mask used to build ``y_valid``.
    names : list of str
        Variables to include; an empty list gives the constant-rate model.
    n_folds, block, alpha, backend, rng, max_iter
        Passed to :func:`cross_validated_deviance`.

    Returns
    -------
    (n_folds,) float
        Held-out deviance explained per fold.
    """
    X = design.subset(names)[keep]
    result = cross_validated_deviance(
        X,
        y_valid,
        n_folds=n_folds,
        block=block,
        alpha=alpha,
        backend=backend,
        rng=rng,
        max_iter=max_iter,
    )
    return np.asarray(result["deviance_explained"], dtype=np.float64)


def drop_one_contributions(
    design: DesignMatrix,
    y: npt.ArrayLike,
    n_folds: int = 5,
    alpha: float = 1e-3,
    backend: str = "sklearn",
    rng: np.random.Generator | None = None,
    block: bool = True,
    max_iter: int = 500,
) -> dict[str, float]:
    """Partial contribution of each variable, by leave-one-variable-out CV.

    The contribution of a variable is the cross-validated deviance explained of
    the full model minus that of the model with the variable's columns removed.

    Parameters
    ----------
    design : DesignMatrix
    y : (n_frames,) array_like
        Non-negative response.
    n_folds : int
    alpha : float
        Ridge penalty.
    backend : {"sklearn", "nemos"}
    rng : numpy.random.Generator, optional
    block : bool
        Contiguous-block folds (default).
    max_iter : int

    Returns
    -------
    dict[str, float]
        One entry per variable in ``design.groups``.
    """
    return _full_and_contributions(
        design,
        y,
        n_folds=n_folds,
        alpha=alpha,
        backend=backend,
        rng=rng,
        block=block,
        max_iter=max_iter,
    )[1]


def _full_and_contributions(
    design: DesignMatrix,
    y: npt.ArrayLike,
    n_folds: int,
    alpha: float,
    backend: str,
    rng: np.random.Generator | None,
    block: bool,
    max_iter: int,
) -> tuple[float, dict[str, float]]:
    """Full-model CV deviance explained and each variable's drop-one contribution.

    Parameters
    ----------
    design : DesignMatrix
    y : (n_frames,) array_like
        Non-negative response.
    n_folds, alpha, backend, rng, block, max_iter
        Passed to :func:`cross_validated_deviance`.

    Returns
    -------
    full : float
        Cross-validated deviance explained of the model with every variable.
    contributions : dict[str, float]
        ``full`` minus the score of the model without each variable.
    """
    keep, y_valid = _valid_subset(design, y)
    names = design.variables
    full = float(
        np.mean(
            _cv_scores_for(
                design, y_valid, keep, names, n_folds, block, alpha, backend, rng, max_iter
            )
        )
    )
    contributions: dict[str, float] = {}
    for name in names:
        reduced = float(
            np.mean(
                _cv_scores_for(
                    design,
                    y_valid,
                    keep,
                    [n for n in names if n != name],
                    n_folds,
                    block,
                    alpha,
                    backend,
                    rng,
                    max_iter,
                )
            )
        )
        contributions[name] = full - reduced
    return full, contributions


def _wilcoxon_greater(
    candidate: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]
) -> float:
    """One-sided Wilcoxon signed-rank p-value for candidate > reference.

    Parameters
    ----------
    candidate, reference : (n_folds,) float
        Paired per-fold scores.

    Returns
    -------
    float
        p-value; 1.0 when the test is undefined (fewer than 3 non-zero
        differences, or all differences zero).
    """
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    nonzero = diff[diff != 0]
    if nonzero.size < 3:
        return 1.0
    stat = wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
    return float(stat.pvalue)


def forward_selection(
    design: DesignMatrix,
    y: npt.ArrayLike,
    n_folds: int = 5,
    alpha: float = 1e-3,
    backend: str = "sklearn",
    rng: np.random.Generator | None = None,
    block: bool = True,
    p_threshold: float = 0.05,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Forward stepwise variable selection on held-out deviance explained.

    At each step the best remaining variable (highest mean held-out deviance
    explained when added) is compared with the current model using a one-sided
    Wilcoxon signed-rank test across folds; it is kept only if the improvement
    is significant at ``p_threshold``. Selection stops at the first rejection.
    This follows Hardcastle et al. 2017, with their parametric fold test
    replaced by the signed-rank test.

    Parameters
    ----------
    design : DesignMatrix
    y : (n_frames,) array_like
        Non-negative response.
    n_folds : int
        Number of folds; at least 3 non-zero fold differences are needed for
        the signed-rank test to reject, so use >= 5.
    alpha : float
        Ridge penalty.
    backend : {"sklearn", "nemos"}
    rng : numpy.random.Generator, optional
    block : bool
    p_threshold : float
        Significance level for keeping a variable.
    max_iter : int

    Returns
    -------
    dict
        ``"selected"`` — list of variable names, in selection order.
        ``"steps"`` — list of dicts with ``"candidate"``, ``"mean_deviance_explained"``,
        ``"p_value"``, ``"accepted"`` for each step.
        ``"deviance_explained"`` — float, mean held-out deviance explained of the
        selected model (0.0 if nothing is selected).

    References
    ----------
    Hardcastle et al. 2017. "A Multiplexed, Heterogeneous, and Adaptive Code
    for Navigation in Medial Entorhinal Cortex." Neuron 94:375-387.
    doi:10.1016/j.neuron.2017.03.025
    """
    keep, y_valid = _valid_subset(design, y)
    remaining = design.variables
    selected: list[str] = []
    steps: list[dict[str, Any]] = []

    current = _cv_scores_for(
        design, y_valid, keep, [], n_folds, block, alpha, backend, rng, max_iter
    )
    while remaining:
        scores = {
            name: _cv_scores_for(
                design,
                y_valid,
                keep,
                [*selected, name],
                n_folds,
                block,
                alpha,
                backend,
                rng,
                max_iter,
            )
            for name in remaining
        }
        best = max(scores, key=lambda n: float(np.mean(scores[n])))
        p_value = _wilcoxon_greater(scores[best], current)
        accepted = bool(p_value < p_threshold)
        steps.append(
            {
                "candidate": best,
                "mean_deviance_explained": float(np.mean(scores[best])),
                "p_value": p_value,
                "accepted": accepted,
            }
        )
        if not accepted:
            break
        selected.append(best)
        remaining = [n for n in remaining if n != best]
        current = scores[best]

    return {
        "selected": selected,
        "steps": steps,
        "deviance_explained": float(np.mean(current)) if selected else 0.0,
    }


def encoding_profile(
    design: DesignMatrix,
    y: npt.ArrayLike,
    n_folds: int = 5,
    alpha: float = 1e-3,
    backend: str = "sklearn",
    rng: np.random.Generator | None = None,
    block: bool = True,
    p_threshold: float = 0.05,
    run_selection: bool = True,
    max_iter: int = 500,
) -> dict[str, Any]:
    """Full-model fit quality, per-variable contributions, and selected variables.

    Parameters
    ----------
    design : DesignMatrix
    y : (n_frames,) array_like
        Non-negative response.
    n_folds, alpha, backend, rng, block, p_threshold, max_iter
        Passed through to :func:`drop_one_contributions` and
        :func:`forward_selection`.
    run_selection : bool
        Run forward selection as well as the drop-one contributions.

    Returns
    -------
    dict
        ``"full_dev_expl"`` — float, cross-validated deviance explained of the
        model containing every variable.
        ``"contributions"`` — dict, raw drop-one contributions (can be negative).
        ``"profile"`` — dict, contributions clipped at 0 and normalised to sum 1
        (all zeros if no variable contributes).
        ``"dominant"`` — str or None, variable with the largest profile weight.
        ``"selected"`` — list of str (empty when ``run_selection`` is False).
        ``"steps"`` — list of dicts from :func:`forward_selection` (empty when
        ``run_selection`` is False).
    """
    full, contributions = _full_and_contributions(
        design,
        y,
        n_folds=n_folds,
        alpha=alpha,
        backend=backend,
        rng=rng,
        block=block,
        max_iter=max_iter,
    )
    positive = {k: max(v, 0.0) for k, v in contributions.items()}
    total = sum(positive.values())
    profile = (
        {k: v / total for k, v in positive.items()} if total > 0 else dict.fromkeys(positive, 0.0)
    )
    dominant = max(profile, key=lambda k: profile[k]) if total > 0 else None

    selected: list[str] = []
    steps: list[dict[str, Any]] = []
    if run_selection:
        sel = forward_selection(
            design,
            y,
            n_folds=n_folds,
            alpha=alpha,
            backend=backend,
            rng=rng,
            block=block,
            p_threshold=p_threshold,
            max_iter=max_iter,
        )
        selected = list(sel["selected"])
        steps = list(sel["steps"])

    return {
        "full_dev_expl": full,
        "contributions": contributions,
        "profile": profile,
        "dominant": dominant,
        "selected": selected,
        "steps": steps,
    }


def population_encoding_profiles(
    signals: npt.ArrayLike,
    design: DesignMatrix,
    n_folds: int = 5,
    alpha: float = 1e-3,
    backend: str = "sklearn",
    rng: np.random.Generator | None = None,
    block: bool = True,
    p_threshold: float = 0.05,
    run_selection: bool = True,
    max_iter: int = 500,
) -> pd.DataFrame:
    """Encoding profile of every ROI in a session.

    Parameters
    ----------
    signals : (n_rois, n_frames) array_like
        Non-negative per-frame activity (event counts or spike rate).
    design : DesignMatrix
        Shared behavioural design matrix.
    n_folds, alpha, backend, rng, block, p_threshold, run_selection, max_iter
        Passed to :func:`encoding_profile`.

    Returns
    -------
    pandas.DataFrame
        One row per ROI with columns ``roi``, ``full_dev_expl``,
        ``part_<variable>`` for each variable in ``design.groups``,
        ``dominant``, and ``selected`` (comma-joined, empty string if none).

    Raises
    ------
    ValueError
        If ``signals`` is not 2-D or its frame count does not match ``design``.
    """
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim != 2:
        raise ValueError(f"signals must be 2-D (n_rois, n_frames), got shape {sig.shape}")
    if sig.shape[1] != design.n_frames:
        raise ValueError(f"signals has {sig.shape[1]} frames but design has {design.n_frames}")
    rows: list[dict[str, Any]] = []
    for roi in range(sig.shape[0]):
        result = encoding_profile(
            design,
            sig[roi],
            n_folds=n_folds,
            alpha=alpha,
            backend=backend,
            rng=rng,
            block=block,
            p_threshold=p_threshold,
            run_selection=run_selection,
            max_iter=max_iter,
        )
        row: dict[str, Any] = {"roi": roi, "full_dev_expl": result["full_dev_expl"]}
        for name in design.variables:
            row[f"part_{name}"] = result["profile"][name]
        row["dominant"] = result["dominant"]
        row["selected"] = ",".join(result["selected"])
        rows.append(row)
    columns = (
        ["roi", "full_dev_expl"]
        + [f"part_{n}" for n in design.variables]
        + ["dominant", "selected"]
    )
    return pd.DataFrame(rows, columns=columns)
