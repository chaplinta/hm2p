"""Dispersion and separability tests between two labelled cell populations.

Hypothesis H6 states that the Penk-CamKII+ population, being a mixture of
transcriptomic types, is more heterogeneous than the Penk+ population,
which corresponds to a single type. Heterogeneity is assessed in three
complementary ways:

1. **Dispersion** — mean distance of cells to their own group centroid in
   standardised feature space (a multivariate spread measure, as in
   Anderson's PERMDISP).
2. **Distributional difference** — the multivariate energy distance
   between the two groups, and the number of Gaussian mixture components
   favoured by BIC within each group.
3. **Separability** — leave-one-animal-out cross-validated classification
   of cell type from the feature vector, with a permutation test for the
   balanced accuracy.

All permutation nulls shuffle the cell-type label at the **animal** level
(every cell of an animal moves together), because cell type is an
animal-level property here: the two populations were imaged in different
mice, so shuffling individual cells would produce an anticonservative
null that ignores between-animal variance.

References
----------
Anderson 2006. "Distance-based tests for homogeneity of multivariate
    dispersions." Biometrics 62(1):245-253.
    doi:10.1111/j.1541-0420.2005.00440.x
Szekely & Rizzo 2013. "Energy statistics: A class of statistics based on
    distances." Journal of Statistical Planning and Inference
    143(8):1249-1272. doi:10.1016/j.jspi.2013.03.018
Ojala & Garriga 2010. "Permutation Tests for Studying Classifier
    Performance." Journal of Machine Learning Research 11:1833-1863.
    https://www.jmlr.org/papers/v11/ojala10a.html
Pedregosa et al. 2011. "Scikit-learn: Machine Learning in Python."
    Journal of Machine Learning Research 12:2825-2830.
    https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "prepare_feature_matrix",
    "group_dispersion",
    "dispersion_ratio_test",
    "energy_distance_test",
    "gmm_components",
    "gmm_components_by_group",
    "loao_classifier",
    "classifier_permutation_test",
    "penk_like_fraction",
    "variance_ratio_permutation",
]

#: Default group labels for the two retrosplenial populations.
GROUP_A: str = "nonpenk"
GROUP_B: str = "penk"

#: Metadata columns carried through :func:`prepare_feature_matrix`.
META_COLUMNS: tuple[str, ...] = ("animal_id", "session_id", "celltype", "roi_idx", "roi_type")


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return a numpy Generator from a Generator, seed, or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_valid_fraction: float = 0.8,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Select, clean and standardise a feature matrix from a feature table.

    Columns whose fraction of finite values falls below
    *min_valid_fraction*, and columns with zero variance, are dropped.
    Rows with any remaining non-finite value are then dropped. The
    surviving matrix is z-scored column-wise across all cells (both
    groups together), so that Euclidean distances weight every feature
    equally.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-cell feature table; must contain the metadata columns used
        downstream (``animal_id``, ``celltype``).
    feature_cols : list of str
        Candidate feature column names.
    min_valid_fraction : float
        Minimum fraction of finite values a column must have to be kept.

    Returns
    -------
    X : (n_cells, n_features) float
        Standardised feature matrix.
    meta : pandas.DataFrame
        Metadata columns for the retained rows, index reset.
    used_cols : list of str
        Feature columns retained, in matrix column order.

    Raises
    ------
    ValueError
        If no requested feature column is present in *df*, or if every
        column or every row is dropped.
    """
    present = [c for c in feature_cols if c in df.columns]
    if not present:
        raise ValueError("none of the requested feature columns are present in df")

    values = df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)

    keep_col = finite.mean(axis=0) >= min_valid_fraction
    if not keep_col.any():
        raise ValueError(f"no feature column has at least {min_valid_fraction:.0%} finite values")
    values = values[:, keep_col]
    cols = [c for c, k in zip(present, keep_col, strict=True) if k]

    keep_row = np.isfinite(values).all(axis=1)
    if not keep_row.any():
        raise ValueError("no row has finite values for every retained feature column")
    values = values[keep_row]

    sd = values.std(axis=0)
    keep_col2 = sd > 0
    if not keep_col2.any():
        raise ValueError("all retained feature columns have zero variance")
    values = values[:, keep_col2]
    cols = [c for c, k in zip(cols, keep_col2, strict=True) if k]

    x = (values - values.mean(axis=0)) / values.std(axis=0)

    meta_cols = [c for c in df.columns if c in META_COLUMNS]
    meta = df.loc[keep_row, meta_cols].reset_index(drop=True)

    return x, meta, cols


def group_dispersion(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
) -> dict[str, float]:
    """Spread of each group around its own centroid.

    Follows the multivariate-dispersion logic of Anderson 2006
    (doi:10.1111/j.1541-0420.2005.00440.x), using Euclidean distances in
    the standardised feature space.

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Group label per cell.

    Returns
    -------
    dict
        For each label ``g``: ``f"{g}_mean"`` (mean distance to the group
        centroid), ``f"{g}_median"`` and ``f"{g}_n"`` (number of cells).
    """
    x = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)

    out: dict[str, float] = {}
    for group in np.unique(lab):
        members = x[lab == group]
        centroid = members.mean(axis=0)
        distances = np.linalg.norm(members - centroid, axis=1)
        out[f"{group}_mean"] = float(np.mean(distances))
        out[f"{group}_median"] = float(np.median(distances))
        out[f"{group}_n"] = float(members.shape[0])
    return out


def _permute_labels_by_animal(
    labels: npt.NDArray[np.str_] | list[str],
    animal_ids: npt.NDArray[np.str_] | list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle group labels between animals, keeping each animal's cells together.

    Parameters
    ----------
    labels : (n_cells,) sequence of str
        Group label per cell. Every cell of an animal must share one label.
    animal_ids : (n_cells,) sequence of str
        Animal identifier per cell.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    (n_cells,) object ndarray
        Permuted per-cell labels.
    """
    lab = np.asarray(labels)
    animals = np.asarray(animal_ids)

    unique_animals = np.unique(animals)
    animal_labels = np.array([lab[animals == a][0] for a in unique_animals])
    permuted = rng.permutation(animal_labels)

    mapping = dict(zip(unique_animals, permuted, strict=True))
    return np.array([mapping[a] for a in animals])


def _two_sided_ratio_p(observed: float, null: npt.NDArray[np.floating]) -> float:
    """Permutation p-value for a ratio statistic, two-sided on the log scale.

    Parameters
    ----------
    observed : float
        Observed ratio (must be positive and finite to give a usable p-value).
    null : (n_perms,) float
        Null-distribution ratios.

    Returns
    -------
    float
        ``(1 + #{|log null| >= |log observed|}) / (1 + n_perms)``, or NaN if
        the observed ratio is not positive and finite.
    """
    if not np.isfinite(observed) or observed <= 0:
        return float("nan")
    null = np.asarray(null, dtype=np.float64)
    usable = null[np.isfinite(null) & (null > 0)]
    if usable.size == 0:
        return float("nan")
    extreme = int(np.count_nonzero(np.abs(np.log(usable)) >= abs(np.log(observed))))
    return float((1 + extreme) / (1 + usable.size))


def _one_sided_p(observed: float, null: npt.NDArray[np.floating]) -> float:
    """Permutation p-value for a statistic where larger means more extreme."""
    null = np.asarray(null, dtype=np.float64)
    usable = null[np.isfinite(null)]
    if not np.isfinite(observed) or usable.size == 0:
        return float("nan")
    extreme = int(np.count_nonzero(usable >= observed))
    return float((1 + extreme) / (1 + usable.size))


def dispersion_ratio_test(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    animal_ids: npt.NDArray[np.str_] | list[str],
    n_perms: int = 1000,
    rng: np.random.Generator | int | None = None,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Test whether one group is more dispersed than the other.

    The statistic is ``dispersion(group_a) / dispersion(group_b)`` using the
    mean distance to the group centroid. The null distribution comes from
    permuting the group label at the animal level.

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Group label per cell.
    animal_ids : (n_cells,) sequence of str
        Animal identifier per cell.
    n_perms : int
        Number of animal-level label permutations.
    rng : Generator, int or None
        Random generator or seed.
    group_a, group_b : str
        Numerator and denominator groups.

    Returns
    -------
    dict
        ``observed`` — the dispersion ratio.
        ``dispersion`` — the full :func:`group_dispersion` output.
        ``null`` — (n_perms,) array of permuted ratios.
        ``p_value`` — two-sided permutation p-value on the log ratio.
        ``n_perms`` — number of permutations actually run.

    Raises
    ------
    ValueError
        If either group is absent from *labels*.
    """
    x = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)
    generator = _as_rng(rng)

    for group in (group_a, group_b):
        if not np.any(lab == group):
            raise ValueError(f"group {group!r} is not present in labels")

    dispersion = group_dispersion(x, lab)
    observed = _dispersion_ratio(x, lab, group_a, group_b)

    null = np.empty(n_perms, dtype=np.float64)
    for i in range(n_perms):
        perm = _permute_labels_by_animal(lab, animal_ids, generator)
        null[i] = _dispersion_ratio(x, perm, group_a, group_b)

    return {
        "observed": observed,
        "dispersion": dispersion,
        "null": null,
        "p_value": _two_sided_ratio_p(observed, null),
        "n_perms": int(n_perms),
    }


def _dispersion_ratio(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_],
    group_a: str,
    group_b: str,
) -> float:
    """Ratio of mean centroid distances for two groups, NaN when undefined."""
    disp = group_dispersion(x, labels)
    a = disp.get(f"{group_a}_mean", float("nan"))
    b = disp.get(f"{group_b}_mean", float("nan"))
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("nan")
    return float(a / b)


def _energy_distance_from_matrix(
    dist: npt.NDArray[np.floating],
    mask_a: npt.NDArray[np.bool_],
    mask_b: npt.NDArray[np.bool_],
) -> float:
    """Multivariate energy distance from a precomputed pairwise distance matrix.

    ``E = 2 * mean(d_ab) - mean(d_aa) - mean(d_bb)``, the multivariate
    generalisation of the energy statistic (Szekely & Rizzo 2013,
    doi:10.1016/j.jspi.2013.03.018). ``scipy.stats.energy_distance`` only
    handles the one-dimensional case, so the statistic is computed here
    directly from pairwise Euclidean distances.

    Parameters
    ----------
    dist : (n, n) float
        Pairwise Euclidean distances between all cells.
    mask_a, mask_b : (n,) bool
        Group membership masks.

    Returns
    -------
    float
        Energy distance (non-negative), or NaN if either group is empty.
    """
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        return float("nan")

    d_ab = float(dist[np.ix_(mask_a, mask_b)].mean())
    d_aa = float(dist[np.ix_(mask_a, mask_a)].mean())
    d_bb = float(dist[np.ix_(mask_b, mask_b)].mean())
    return float(2.0 * d_ab - d_aa - d_bb)


def energy_distance_test(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    animal_ids: npt.NDArray[np.str_] | list[str],
    n_perms: int = 1000,
    rng: np.random.Generator | int | None = None,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """Multivariate energy-distance test between two groups of cells.

    Szekely & Rizzo 2013. "Energy statistics: A class of statistics based
    on distances." Journal of Statistical Planning and Inference
    143(8):1249-1272. doi:10.1016/j.jspi.2013.03.018

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Group label per cell.
    animal_ids : (n_cells,) sequence of str
        Animal identifier per cell.
    n_perms : int
        Number of animal-level label permutations.
    rng : Generator, int or None
        Random generator or seed.
    group_a, group_b : str
        The two groups to compare.

    Returns
    -------
    dict
        ``observed`` — energy distance.
        ``null`` — (n_perms,) array of permuted energy distances.
        ``p_value`` — one-sided permutation p-value (larger = more different).
        ``n_perms`` — number of permutations run.

    Raises
    ------
    ValueError
        If either group is absent from *labels*.
    """
    x = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)
    generator = _as_rng(rng)

    for group in (group_a, group_b):
        if not np.any(lab == group):
            raise ValueError(f"group {group!r} is not present in labels")

    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    observed = _energy_distance_from_matrix(dist, lab == group_a, lab == group_b)

    null = np.empty(n_perms, dtype=np.float64)
    for i in range(n_perms):
        perm = _permute_labels_by_animal(lab, animal_ids, generator)
        null[i] = _energy_distance_from_matrix(dist, perm == group_a, perm == group_b)

    return {
        "observed": observed,
        "null": null,
        "p_value": _one_sided_p(observed, null),
        "n_perms": int(n_perms),
    }


def gmm_components(
    x: npt.NDArray[np.floating],
    max_components: int = 6,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Select the number of Gaussian mixture components by BIC.

    A group whose feature distribution is best described by more than one
    component is multimodal, which is one operationalisation of
    heterogeneity.

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    max_components : int
        Largest number of components to try (clipped to the number of cells).
    rng : Generator, int or None
        Random generator or seed (used to seed scikit-learn's estimator).

    Returns
    -------
    dict
        ``best_k`` — component count minimising BIC (NaN if no fit succeeded).
        ``bic`` — list of BIC values, one per tried component count.
        ``n_components_tried`` — list of component counts tried.
        ``n_cells`` — number of cells.

    Raises
    ------
    ValueError
        If *max_components* is less than 1.
    """
    from sklearn.mixture import GaussianMixture

    if max_components < 1:
        raise ValueError(f"max_components must be >= 1; got {max_components}")

    x = np.asarray(x, dtype=np.float64)
    generator = _as_rng(rng)
    seed = int(generator.integers(0, 2**31 - 1))

    n_cells = x.shape[0]
    ks = list(range(1, min(max_components, n_cells) + 1))

    bics: list[float] = []
    for k in ks:
        try:
            model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
            model.fit(x)
            bics.append(float(model.bic(x)))
        except ValueError:
            bics.append(float("nan"))

    finite = [b for b in bics if np.isfinite(b)]
    if finite:
        best_k = int(ks[int(np.nanargmin(np.asarray(bics, dtype=np.float64)))])
    else:
        best_k = -1

    return {
        "best_k": best_k,
        "bic": bics,
        "n_components_tried": ks,
        "n_cells": int(n_cells),
    }


def gmm_components_by_group(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    max_components: int = 6,
    rng: np.random.Generator | int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run :func:`gmm_components` separately within each group.

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Group label per cell.
    max_components : int
        Largest number of components to try.
    rng : Generator, int or None
        Random generator or seed.

    Returns
    -------
    dict
        Group label to :func:`gmm_components` output.
    """
    lab = np.asarray(labels)
    generator = _as_rng(rng)
    return {
        str(group): gmm_components(
            np.asarray(x, dtype=np.float64)[lab == group], max_components, generator
        )
        for group in np.unique(lab)
    }


def _make_classifier(model: str, seed: int) -> Any:
    """Instantiate the requested scikit-learn classifier.

    Parameters
    ----------
    model : {"logistic", "gbm"}
        ``"logistic"`` gives an L2 logistic regression with balanced class
        weights; ``"gbm"`` gives a histogram gradient-boosting classifier.
    seed : int
        Random seed for the estimator.

    Returns
    -------
    sklearn estimator

    Raises
    ------
    ValueError
        If *model* is not recognised.
    """
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    if model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=100, class_weight="balanced", random_state=seed
        )
    raise ValueError(f"model must be 'logistic' or 'gbm'; got {model!r}")


def _loao_predictions(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_],
    animal_ids: npt.NDArray[np.str_],
    model: str,
    seed: int,
    compute_importance: bool,
    n_repeats: int,
) -> dict[str, Any]:
    """Leave-one-animal-out fit/predict loop shared by the public functions.

    Parameters
    ----------
    x : (n_cells, n_features) float
    labels : (n_cells,) str
    animal_ids : (n_cells,) str
    model : {"logistic", "gbm"}
    seed : int
    compute_importance : bool
        Whether to run ``sklearn.inspection.permutation_importance`` per fold.
    n_repeats : int
        Repeats for the permutation importance.

    Returns
    -------
    dict
        ``y_true``, ``y_pred`` (pooled over folds), ``per_animal_accuracy``,
        ``n_folds`` and ``importance`` ((n_features,) array, NaN if not
        computed).
    """
    from sklearn.inspection import permutation_importance

    animals = np.unique(animal_ids)
    y_true: list[str] = []
    y_pred: list[str] = []
    per_animal: dict[str, float] = {}
    importances: list[np.ndarray] = []
    n_folds = 0

    for animal in animals:
        test = animal_ids == animal
        train = ~test
        if np.unique(labels[train]).size < 2 or test.sum() == 0:
            per_animal[str(animal)] = float("nan")
            continue

        clf = _make_classifier(model, seed)
        clf.fit(x[train], labels[train])
        pred = clf.predict(x[test])

        y_true.extend(labels[test].tolist())
        y_pred.extend(np.asarray(pred).tolist())
        per_animal[str(animal)] = float(np.mean(np.asarray(pred) == labels[test]))
        n_folds += 1

        if compute_importance:
            result = permutation_importance(
                clf,
                x[test],
                labels[test],
                scoring="accuracy",
                n_repeats=n_repeats,
                random_state=seed,
            )
            importances.append(np.asarray(result.importances_mean, dtype=np.float64))

    if importances:
        importance = np.mean(np.vstack(importances), axis=0)
    else:
        importance = np.full(x.shape[1], np.nan, dtype=np.float64)

    return {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "per_animal_accuracy": per_animal,
        "n_folds": n_folds,
        "importance": importance,
    }


def _balanced_accuracy(y_true: npt.NDArray[np.str_], y_pred: npt.NDArray[np.str_]) -> float:
    """Balanced accuracy, or NaN when no predictions were made."""
    if y_true.size == 0:
        return float("nan")
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def loao_classifier(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    animal_ids: npt.NDArray[np.str_] | list[str],
    model: str = "logistic",
    rng: np.random.Generator | int | None = None,
    feature_names: list[str] | None = None,
    n_repeats: int = 5,
) -> dict[str, Any]:
    """Leave-one-animal-out classification of cell type from cell features.

    Held-out folds are whole animals, so no cell from a test animal ever
    appears in training. This is the appropriate cross-validation scheme
    when the class label is an animal-level property: a cell-level split
    would let the classifier learn animal identity instead of cell type.

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Cell-type label per cell.
    animal_ids : (n_cells,) sequence of str
        Animal identifier per cell.
    model : {"logistic", "gbm"}
        Classifier to use.
    rng : Generator, int or None
        Random generator or seed.
    feature_names : list of str, optional
        Names matching the columns of *x*; when given, the returned
        ``importance_by_feature`` maps name to mean importance.
    n_repeats : int
        Repeats used by ``sklearn.inspection.permutation_importance``.

    Returns
    -------
    dict
        ``balanced_accuracy`` — pooled over held-out animals.
        ``per_animal_accuracy`` — accuracy for each held-out animal
        (NaN for animals that could not be tested because the remaining
        training set had a single class).
        ``n_folds`` — number of usable folds.
        ``importance`` — (n_features,) mean permutation importance.
        ``importance_by_feature`` — name to importance (empty without names).
        ``y_true``, ``y_pred`` — pooled held-out labels and predictions.
        ``model`` — the model name used.
    """
    x = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels).astype(str)
    animals = np.asarray(animal_ids).astype(str)
    seed = int(_as_rng(rng).integers(0, 2**31 - 1))

    folds = _loao_predictions(x, lab, animals, model, seed, True, n_repeats)
    importance = folds["importance"]

    by_feature: dict[str, float] = {}
    if feature_names is not None:
        by_feature = {
            name: float(value) for name, value in zip(feature_names, importance, strict=False)
        }

    return {
        "balanced_accuracy": _balanced_accuracy(folds["y_true"], folds["y_pred"]),
        "per_animal_accuracy": folds["per_animal_accuracy"],
        "n_folds": folds["n_folds"],
        "importance": importance,
        "importance_by_feature": by_feature,
        "y_true": folds["y_true"],
        "y_pred": folds["y_pred"],
        "model": model,
    }


def classifier_permutation_test(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    animal_ids: npt.NDArray[np.str_] | list[str],
    n_perms: int = 200,
    model: str = "logistic",
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Permutation test for the leave-one-animal-out balanced accuracy.

    The whole cross-validation is repeated with animal-level label
    shuffles, giving the null distribution of balanced accuracy under no
    association between cell type and features.

    Ojala & Garriga 2010. "Permutation Tests for Studying Classifier
    Performance." Journal of Machine Learning Research 11:1833-1863.
    https://www.jmlr.org/papers/v11/ojala10a.html

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Cell-type label per cell.
    animal_ids : (n_cells,) sequence of str
        Animal identifier per cell.
    n_perms : int
        Number of animal-level label permutations.
    model : {"logistic", "gbm"}
        Classifier to use.
    rng : Generator, int or None
        Random generator or seed.

    Returns
    -------
    dict
        ``observed`` — balanced accuracy on the true labels.
        ``null`` — (n_perms,) array of permuted balanced accuracies.
        ``p_value`` — one-sided permutation p-value.
        ``n_perms`` — number of permutations run.
    """
    x = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels).astype(str)
    animals = np.asarray(animal_ids).astype(str)
    generator = _as_rng(rng)
    seed = int(generator.integers(0, 2**31 - 1))

    observed_folds = _loao_predictions(x, lab, animals, model, seed, False, 1)
    observed = _balanced_accuracy(observed_folds["y_true"], observed_folds["y_pred"])

    null = np.empty(n_perms, dtype=np.float64)
    for i in range(n_perms):
        perm = _permute_labels_by_animal(lab, animals, generator).astype(str)
        folds = _loao_predictions(x, perm, animals, model, seed, False, 1)
        null[i] = _balanced_accuracy(folds["y_true"], folds["y_pred"])

    return {
        "observed": observed,
        "null": null,
        "p_value": _one_sided_p(observed, null),
        "n_perms": int(n_perms),
    }


def penk_like_fraction(
    x: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    k: int = 10,
    target_group: str = GROUP_A,
    reference_group: str = GROUP_B,
) -> dict[str, Any]:
    """Nearest-neighbour estimate of a reference-like sub-mode in the target group.

    For every cell of *target_group*, the fraction of its *k* nearest
    neighbours (over all cells, excluding itself) that belong to
    *reference_group*. If the mixed population contains a sub-mode that
    resembles the reference population, a subset of its cells will have
    predominantly reference neighbours.

    Parameters
    ----------
    x : (n_cells, n_features) float
        Standardised feature matrix.
    labels : (n_cells,) sequence of str
        Group label per cell.
    k : int
        Neighbourhood size (clipped to the number of other cells).
    target_group : str
        Group whose cells are examined.
    reference_group : str
        Group whose presence in the neighbourhood is counted.

    Returns
    -------
    dict
        ``fractions`` — (n_target,) array of neighbour fractions.
        ``median_fraction`` — median over target cells.
        ``fraction_majority_reference`` — fraction of target cells whose
        neighbourhood is more than half reference cells.
        ``k`` — neighbourhood size actually used.
        ``n_target`` — number of target cells.
        ``baseline`` — overall fraction of reference cells in the data,
        the value expected when the groups are fully intermingled.

    Raises
    ------
    ValueError
        If *k* is less than 1, or either group is absent from *labels*.
    """
    from sklearn.neighbors import NearestNeighbors

    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    x = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels).astype(str)
    for group in (target_group, reference_group):
        if not np.any(lab == group):
            raise ValueError(f"group {group!r} is not present in labels")

    n_cells = x.shape[0]
    k_eff = int(min(k, n_cells - 1))
    if k_eff < 1:
        raise ValueError("at least two cells are required to compute neighbourhoods")

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(x)
    target_idx = np.flatnonzero(lab == target_group)
    _, indices = nn.kneighbors(x[target_idx])
    # Column 0 is the query point itself.
    neighbour_labels = lab[indices[:, 1:]]
    fractions = (neighbour_labels == reference_group).mean(axis=1)

    return {
        "fractions": fractions.astype(np.float64),
        "median_fraction": float(np.median(fractions)) if fractions.size else float("nan"),
        "fraction_majority_reference": (
            float(np.mean(fractions > 0.5)) if fractions.size else float("nan")
        ),
        "k": k_eff,
        "n_target": int(target_idx.size),
        "baseline": float(np.mean(lab == reference_group)),
    }


def _variance_ratio(
    values: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_],
    group_a: str,
    group_b: str,
) -> float:
    """Ratio of within-group variances, NaN when either group is too small."""
    a = values[labels == group_a]
    b = values[labels == group_b]
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_b = float(np.var(b, ddof=1))
    if var_b <= 0:
        return float("nan")
    return float(np.var(a, ddof=1) / var_b)


def variance_ratio_permutation(
    values: npt.NDArray[np.floating],
    labels: npt.NDArray[np.str_] | list[str],
    animal_ids: npt.NDArray[np.str_] | list[str],
    n_perms: int = 1000,
    rng: np.random.Generator | int | None = None,
    group_a: str = GROUP_A,
    group_b: str = GROUP_B,
) -> dict[str, Any]:
    """One-dimensional variance-ratio test with an animal-level permutation null.

    The univariate counterpart of :func:`dispersion_ratio_test`: it asks
    whether a single feature is more variable in one group than in the
    other. A permutation null is used instead of an F test because the
    F test assumes normality and independent observations, neither of
    which holds for cells nested within animals.

    Parameters
    ----------
    values : (n_cells,) float
        Feature values; non-finite entries are dropped.
    labels : (n_cells,) sequence of str
        Group label per cell.
    animal_ids : (n_cells,) sequence of str
        Animal identifier per cell.
    n_perms : int
        Number of animal-level label permutations.
    rng : Generator, int or None
        Random generator or seed.
    group_a, group_b : str
        Numerator and denominator groups.

    Returns
    -------
    dict
        ``observed`` — variance ratio ``var(group_a) / var(group_b)``.
        ``var_a``, ``var_b`` — the two within-group variances.
        ``null`` — (n_perms,) array of permuted ratios.
        ``p_value`` — two-sided permutation p-value on the log ratio.
        ``n_perms`` — number of permutations run.
        ``n_cells`` — number of finite values used.
    """
    vals = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels).astype(str)
    animals = np.asarray(animal_ids).astype(str)
    generator = _as_rng(rng)

    finite = np.isfinite(vals)
    vals, lab, animals = vals[finite], lab[finite], animals[finite]

    observed = _variance_ratio(vals, lab, group_a, group_b)
    var_a = float(np.var(vals[lab == group_a], ddof=1)) if (lab == group_a).sum() > 1 else np.nan
    var_b = float(np.var(vals[lab == group_b], ddof=1)) if (lab == group_b).sum() > 1 else np.nan

    null = np.empty(n_perms, dtype=np.float64)
    for i in range(n_perms):
        perm = _permute_labels_by_animal(lab, animals, generator).astype(str)
        null[i] = _variance_ratio(vals, perm, group_a, group_b)

    return {
        "observed": observed,
        "var_a": var_a,
        "var_b": var_b,
        "null": null,
        "p_value": _two_sided_ratio_p(observed, null),
        "n_perms": int(n_perms),
        "n_cells": int(vals.size),
    }
