"""Low-rheobase (LR) vs regular-spiking (RS) classification of RSC L2/3 neurons.

Hypothesis H1: Penk+ neurons of dorsal granular retrosplenial cortex layer 2/3
are the low-rheobase (LR) cell type described by Brennan et al. 2020. LR neurons
are characterised by high input resistance, low rheobase, little or no
spike-frequency adaptation, a spike half-width between fast-spiking and
regular-spiking values, small somata, and sustained high firing rates.

This module provides:

- :data:`LR_CRITERIA` — threshold set with documented provenance, applied by
  :func:`classify_lr_rs` to give each cell an ``lr_score`` and ``lr_class``.
- Spike-train descriptors (:func:`adaptation_index`,
  :func:`first_spike_latency_s`, :func:`spike_frequency_adaptation_ratio`) that
  take spike times, so they are usable and testable without eFEL.
- :func:`data_driven_lr_axis` — unsupervised alternative that does not assume
  thresholds: PCA on z-scored ephys metrics plus a Gaussian-mixture BIC check
  for bimodality of PC1.
- :func:`enrichment_test` — is the LR class enriched in one cell type, tested
  with an animal-stratified permutation (both cell types are patched in every
  animal, so labels are permuted within animal).
- :func:`compare_to_reference` — per-group medians against published LR/RS
  reference ranges.

Threshold caveat
----------------
All absolute thresholds and reference ranges here are dataset-dependent: they
shift with recording temperature, internal solution, series-resistance
compensation and the measurement protocol. The defaults are approximate values
transcribed from the references below and **must be checked against the papers
and against this dataset's protocols before publication**. They are function
parameters precisely so they can be overridden.

References
----------
Brennan et al. 2020. "Hyperexcitable Neurons Enable Precise and Persistent
Information Encoding in the Superficial Retrosplenial Cortex." Cell Reports
30:1598-1612. doi:10.1016/j.celrep.2019.12.093

Yousuf et al. 2020. "Modular Organization of Murine Retrosplenial Cortex."
Frontiers in Neural Circuits 14:576504. doi:10.3389/fncir.2020.576504

Brennan et al. 2021. "Thalamus and claustrum control parallel layer 1 circuits
in retrosplenial cortex." eLife 10:e62207. doi:10.7554/eLife.62207

Druckmann et al. 2007. "A novel multiple objective optimization framework for
constraining conductance-based neuron models by experimental data." Frontiers
in Neuroscience 1:7-18. doi:10.3389/neuro.01.1.1.001.2007 (adaptation index as
implemented by eFEL, https://github.com/BlueBrain/eFEL)

Benjamini & Hochberg 1995. "Controlling the False Discovery Rate: A Practical
and Powerful Approach to Multiple Testing." Journal of the Royal Statistical
Society B 57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from hm2p.patching.statistics import ordered_groups

# Column names in ``results/patching/analysis/metrics.csv``.
COL_RIN = "ephys_passive_rin"
COL_RHEOBASE = "ephys_passive_rhreo"
COL_MAX_SPIKES = "ephys_passive_maxsp"
COL_HALF_WIDTH = "ephys_active_halfWidth"
COL_ADAPTATION = "adaptation_index"

LR_CLASS = "LR"
RS_CLASS = "RS"
AMBIGUOUS_CLASS = "ambiguous"


@dataclass(frozen=True)
class LRCriteria:
    """Thresholds separating LR from RS neurons, with the column each applies to.

    Each threshold is a single-sided rule: a cell meets the criterion when its
    value is above (``*_min``) or below (``*_max``) the threshold. Values are
    approximate and dataset-dependent — see the module docstring.

    Attributes
    ----------
    rin_min_mohm : float
        Input resistance above this (MΩ) is LR-like. Brennan et al. 2020 report
        markedly higher input resistance in LR than RS neurons.
    rheobase_max_pa : float
        Rheobase below this (pA) is LR-like — the defining LR property.
    half_width_max_ms : float
        Spike half-width below this (ms) is LR-like: LR spikes are narrower than
        regular-spiking but broader than fast-spiking interneurons. Temperature
        dependent.
    adaptation_index_max : float
        Adaptation index below this is LR-like (little or no spike-frequency
        adaptation). Only applied when the column is present.
    max_spikes_min : float
        Spike count in the maximum current step above this is LR-like
        (sustained high firing).
    rin_col, rheobase_col, half_width_col, adaptation_col, max_spikes_col : str
        Column names the thresholds are read from.
    source : str
        Provenance string recorded with the results.
    """

    rin_min_mohm: float = 150.0
    rheobase_max_pa: float = 100.0
    half_width_max_ms: float = 2.0
    adaptation_index_max: float = 0.3
    max_spikes_min: float = 15.0

    rin_col: str = COL_RIN
    rheobase_col: str = COL_RHEOBASE
    half_width_col: str = COL_HALF_WIDTH
    adaptation_col: str = COL_ADAPTATION
    max_spikes_col: str = COL_MAX_SPIKES

    source: str = (
        'Approximate values from Brennan et al. 2020. "Hyperexcitable Neurons '
        "Enable Precise and Persistent Information Encoding in the Superficial "
        'Retrosplenial Cortex." Cell Reports 30:1598-1612. '
        "doi:10.1016/j.celrep.2019.12.093 — verify against the paper and against "
        "this dataset's recording conditions before publication."
    )

    def rules(self) -> list[tuple[str, str, str, float]]:
        """Return the criteria as ``(name, column, direction, threshold)`` tuples.

        Returns
        -------
        list of tuple
            ``direction`` is ``"gt"`` (value must exceed the threshold) or
            ``"lt"`` (value must fall below it).
        """
        return [
            ("rin", self.rin_col, "gt", self.rin_min_mohm),
            ("rheobase", self.rheobase_col, "lt", self.rheobase_max_pa),
            ("half_width", self.half_width_col, "lt", self.half_width_max_ms),
            ("adaptation", self.adaptation_col, "lt", self.adaptation_index_max),
            ("max_spikes", self.max_spikes_col, "gt", self.max_spikes_min),
        ]


#: Default LR criteria (see :class:`LRCriteria` for provenance and caveats).
LR_CRITERIA = LRCriteria()


@dataclass(frozen=True)
class ReferenceRanges:
    """Published LR and RS reference ranges, per metric column.

    Attributes
    ----------
    ranges : mapping
        ``{"LR": {column: (low, high)}, "RS": {column: (low, high)}}``.
    source : str
        Citation and caveat.
    """

    ranges: Mapping[str, Mapping[str, tuple[float, float]]] = field(default_factory=dict)
    source: str = ""


#: Approximate LR / RS reference ranges from Brennan et al. 2020 (Cell Reports
#: 30:1598-1612, doi:10.1016/j.celrep.2019.12.093). Transcribed as approximate
#: ranges and NOT yet verified against the paper's tables; they must be checked
#: (and adjusted for recording temperature and internal solution) before any
#: published claim of correspondence.
BRENNAN_2020_REFERENCE = ReferenceRanges(
    ranges={
        LR_CLASS: {
            COL_RIN: (150.0, 500.0),
            COL_RHEOBASE: (5.0, 100.0),
            COL_HALF_WIDTH: (0.5, 1.5),
            COL_MAX_SPIKES: (15.0, 60.0),
            COL_ADAPTATION: (-0.05, 0.30),
        },
        RS_CLASS: {
            COL_RIN: (40.0, 150.0),
            COL_RHEOBASE: (100.0, 500.0),
            COL_HALF_WIDTH: (1.0, 2.5),
            COL_MAX_SPIKES: (2.0, 20.0),
            COL_ADAPTATION: (0.20, 0.80),
        },
    },
    source=(
        'Approximate, UNVERIFIED transcription of Brennan et al. 2020. "Hyperexcitable '
        "Neurons Enable Precise and Persistent Information Encoding in the Superficial "
        'Retrosplenial Cortex." Cell Reports 30:1598-1612. '
        "doi:10.1016/j.celrep.2019.12.093 — check against the paper before publication."
    ),
)


# ============================================================================
# Threshold classification
# ============================================================================


def _criterion_met(values: pd.Series, direction: str, threshold: float) -> pd.Series:
    """Evaluate one single-sided criterion, propagating NaN as "not evaluable".

    Parameters
    ----------
    values : Series
        Metric values.
    direction : {"gt", "lt"}
        Whether the value must exceed or fall below *threshold*.
    threshold : float
        Threshold value.

    Returns
    -------
    Series of object
        ``True`` (met), ``False`` (not met) or ``pd.NA`` (value missing).

    Raises
    ------
    ValueError
        If *direction* is not ``"gt"`` or ``"lt"``.
    """
    if direction not in ("gt", "lt"):
        raise ValueError(f"direction must be 'gt' or 'lt', got {direction!r}")
    numeric = pd.to_numeric(values, errors="coerce")
    met = numeric > threshold if direction == "gt" else numeric < threshold
    return met.astype("object").where(numeric.notna(), other=pd.NA)


def classify_lr_rs(
    df: pd.DataFrame,
    criteria: LRCriteria | None = None,
    min_criteria_met: int = 3,
) -> pd.DataFrame:
    """Score each cell against the LR criteria and assign an LR/RS/ambiguous class.

    A cell is ``"LR"`` when at least *min_criteria_met* criteria are met,
    ``"RS"`` when at least *min_criteria_met* evaluable criteria are failed, and
    ``"ambiguous"`` otherwise (including when too few metrics are available).
    Criteria whose column is missing from *df*, or whose value is NaN for that
    cell, are not evaluated and count towards neither side.

    Parameters
    ----------
    df : DataFrame
        Cell-level metrics table.
    criteria : LRCriteria, optional
        Thresholds; defaults to :data:`LR_CRITERIA`.
    min_criteria_met : int
        Number of criteria required for a confident call (default 3).

    Returns
    -------
    DataFrame
        Copy of *df* with added columns: ``lr_<name>`` (bool or NA) per
        criterion, ``lr_score`` (int, criteria met), ``lr_n_evaluable`` (int),
        ``lr_fraction_met`` (float, NaN when nothing evaluable) and ``lr_class``
        in {``"LR"``, ``"RS"``, ``"ambiguous"``}.

    Raises
    ------
    ValueError
        If *min_criteria_met* is not positive, or none of the criterion columns
        are present in *df*.

    References
    ----------
    Brennan et al. 2020. "Hyperexcitable Neurons Enable Precise and Persistent
    Information Encoding in the Superficial Retrosplenial Cortex." Cell Reports
    30:1598-1612. doi:10.1016/j.celrep.2019.12.093
    """
    if min_criteria_met < 1:
        raise ValueError(f"min_criteria_met must be >= 1, got {min_criteria_met}")
    crit = criteria if criteria is not None else LR_CRITERIA

    rules = [r for r in crit.rules() if r[1] in df.columns]
    if not rules:
        raise ValueError(
            "none of the LR criterion columns are present in the DataFrame: "
            f"{[r[1] for r in crit.rules()]}"
        )

    out = df.copy()
    met_cols: list[str] = []
    for name, column, direction, threshold in rules:
        col = f"lr_{name}"
        out[col] = _criterion_met(out[column], direction, threshold)
        met_cols.append(col)

    met_frame = out[met_cols]
    n_met = met_frame.apply(lambda row: sum(v is True for v in row), axis=1).astype(int)
    n_failed = met_frame.apply(lambda row: sum(v is False for v in row), axis=1).astype(int)
    n_evaluable = n_met + n_failed

    out["lr_score"] = n_met
    out["lr_n_evaluable"] = n_evaluable
    out["lr_fraction_met"] = n_met / n_evaluable.replace(0, np.nan)

    lr_class = np.full(len(out), AMBIGUOUS_CLASS, dtype=object)
    lr_class[(n_met >= min_criteria_met).to_numpy()] = LR_CLASS
    lr_class[((n_failed >= min_criteria_met) & (n_met < min_criteria_met)).to_numpy()] = RS_CLASS
    out["lr_class"] = lr_class
    return out


# ============================================================================
# Spike-train descriptors
# ============================================================================


def _interspike_intervals(spike_times_s: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return finite, sorted inter-spike intervals in seconds.

    Parameters
    ----------
    spike_times_s : array-like
        Spike times in seconds.

    Returns
    -------
    ndarray
        Inter-spike intervals, shape (n_spikes - 1,); empty if fewer than
        2 finite spike times.
    """
    times = np.asarray(list(spike_times_s), dtype=float)
    times = np.sort(times[np.isfinite(times)])
    if times.size < 2:
        return np.zeros(0, dtype=float)
    return np.diff(times)


def adaptation_index(spike_times_s: Sequence[float] | np.ndarray) -> float:
    """Mean normalised change in successive inter-spike intervals.

    ``mean_i (ISI_{i+1} - ISI_i) / (ISI_{i+1} + ISI_i)``. Positive values mean
    the train slows down (adaptation), zero means no adaptation, negative means
    acceleration. LR neurons show little or no adaptation (Brennan et al. 2020).

    Parameters
    ----------
    spike_times_s : array-like
        Spike times in seconds, in any order.

    Returns
    -------
    float
        Adaptation index, or NaN if fewer than 3 spikes (2 ISIs) are available
        or an ISI pair sums to zero.

    References
    ----------
    Druckmann et al. 2007. "A novel multiple objective optimization framework
    for constraining conductance-based neuron models by experimental data."
    Frontiers in Neuroscience 1:7-18. doi:10.3389/neuro.01.1.1.001.2007
    """
    isis = _interspike_intervals(spike_times_s)
    if isis.size < 2:
        return float("nan")
    numerator = isis[1:] - isis[:-1]
    denominator = isis[1:] + isis[:-1]
    if np.any(denominator == 0):
        return float("nan")
    return float(np.mean(numerator / denominator))


def first_spike_latency_s(
    spike_times_s: Sequence[float] | np.ndarray,
    stim_onset_s: float,
) -> float:
    """Latency from stimulus onset to the first spike at or after onset.

    Late-spiking behaviour (a long first-spike latency) is absent in LR neurons
    (Brennan et al. 2020), so this separates LR from late-spiking types.

    Parameters
    ----------
    spike_times_s : array-like
        Spike times in seconds.
    stim_onset_s : float
        Current-step onset in seconds.

    Returns
    -------
    float
        Latency in seconds, or NaN if there is no spike at or after onset, or
        *stim_onset_s* is not finite.
    """
    if not np.isfinite(stim_onset_s):
        return float("nan")
    times = np.asarray(list(spike_times_s), dtype=float)
    times = times[np.isfinite(times)]
    after = times[times >= stim_onset_s]
    if after.size == 0:
        return float("nan")
    return float(after.min() - stim_onset_s)


def spike_frequency_adaptation_ratio(spike_times_s: Sequence[float] | np.ndarray) -> float:
    """Ratio of the last to the first inter-spike interval.

    1.0 means no adaptation; > 1 means the train slows down. Complements
    :func:`adaptation_index`, which is sensitive to the whole train rather than
    only its ends.

    Parameters
    ----------
    spike_times_s : array-like
        Spike times in seconds.

    Returns
    -------
    float
        ``ISI_last / ISI_first``, or NaN if fewer than 3 spikes are available or
        the first ISI is zero.
    """
    isis = _interspike_intervals(spike_times_s)
    if isis.size < 2 or isis[0] == 0:
        return float("nan")
    return float(isis[-1] / isis[0])


def adaptation_from_efel(features: Mapping[str, Any]) -> float:
    """Read eFEL's ``adaptation_index2`` from a feature dictionary.

    Thin adapter so eFEL-derived features can be fed into the same analysis
    without importing eFEL here (it is not a dependency of this package). eFEL
    returns each feature as an array of per-trace values; the mean of the finite
    entries is returned.

    Parameters
    ----------
    features : mapping
        eFEL feature dictionary, e.g. one element of
        ``efel.get_feature_values(traces, ["adaptation_index2"])``.

    Returns
    -------
    float
        Mean ``adaptation_index2``, or NaN when the key is absent, None, empty
        or all-NaN.

    References
    ----------
    Druckmann et al. 2007. "A novel multiple objective optimization framework
    for constraining conductance-based neuron models by experimental data."
    Frontiers in Neuroscience 1:7-18. doi:10.3389/neuro.01.1.1.001.2007
    (eFEL implementation: https://github.com/BlueBrain/eFEL)
    """
    value = features.get("adaptation_index2")
    if value is None:
        return float("nan")
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


# ============================================================================
# Unsupervised LR axis
# ============================================================================


def data_driven_lr_axis(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    group_col: str = "cell_type",
) -> dict[str, Any]:
    """Threshold-free LR axis: PCA on z-scored metrics plus a bimodality check.

    Complements :func:`classify_lr_rs`, which depends on absolute thresholds
    that may not transfer between recording conditions. Metrics are z-scored,
    PCA is run on complete cases, and PC1 is treated as the candidate
    excitability axis. A 2-component Gaussian mixture is compared with a
    1-component mixture by BIC to ask whether PC1 is bimodal, i.e. whether the
    data themselves support two cell classes.

    A BIC difference favouring two components is suggestive, not a test: with
    tens of cells, mixture BIC is noisy and can favour two components for
    skewed unimodal data. Treat it as a descriptive check.

    Parameters
    ----------
    df : DataFrame
        Cell-level metrics table.
    metric_cols : sequence of str
        Numeric columns forming the axis (ephys metrics for the LR question).
    group_col : str
        Column with cell-type labels, reported per group but not used to fit.

    Returns
    -------
    dict
        ``metric_cols``, ``n_cells``, ``pc1_loadings`` (metric -> loading),
        ``explained_variance_ratio`` (list), ``pc1_scores`` (ndarray),
        ``pc2_scores`` (ndarray or None), ``groups`` (ndarray of labels aligned
        with the scores), ``pc1_by_group`` (group -> ndarray of scores),
        ``pc1_median_by_group`` (group -> float), ``bic_1``, ``bic_2``,
        ``bic_delta`` (``bic_1 - bic_2``; positive favours two components) and
        ``bimodal`` (bool, ``bic_delta > 0``). Mixture entries are NaN/False
        when there are too few cells to fit two components.

    Raises
    ------
    ValueError
        If *metric_cols* is empty, a column is missing, or fewer than 3 complete
        cases remain.

    References
    ----------
    Brennan et al. 2020. "Hyperexcitable Neurons Enable Precise and Persistent
    Information Encoding in the Superficial Retrosplenial Cortex." Cell Reports
    30:1598-1612. doi:10.1016/j.celrep.2019.12.093
    """
    metric_cols = list(metric_cols)
    if not metric_cols:
        raise ValueError("metric_cols must not be empty")
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"metric columns not found in DataFrame: {missing}")

    keep_cols = list(metric_cols)
    has_group = group_col in df.columns
    if has_group:
        keep_cols = keep_cols + [group_col]

    sub = df[keep_cols].dropna(subset=metric_cols)
    if len(sub) < 3:
        raise ValueError(f"need at least 3 complete cases for PCA, got {len(sub)}")

    matrix = sub[metric_cols].to_numpy(dtype=float)
    sd = matrix.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0  # constant metric contributes nothing rather than NaN
    z = (matrix - matrix.mean(axis=0)) / sd

    n_components = min(2, z.shape[0], z.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(z)
    pc1 = scores[:, 0]
    pc2 = scores[:, 1] if scores.shape[1] > 1 else None

    groups = sub[group_col].to_numpy() if has_group else np.full(len(sub), "all", dtype=object)
    pc1_by_group = {str(g): pc1[groups == g] for g in ordered_groups(groups)}

    bic_1 = bic_2 = float("nan")
    if len(pc1) >= 4:
        x = pc1.reshape(-1, 1)
        bic_1 = float(GaussianMixture(n_components=1, random_state=0).fit(x).bic(x))
        bic_2 = float(GaussianMixture(n_components=2, random_state=0).fit(x).bic(x))
    bic_delta = bic_1 - bic_2

    return {
        "metric_cols": metric_cols,
        "n_cells": int(len(sub)),
        "pc1_loadings": dict(zip(metric_cols, pca.components_[0].tolist(), strict=True)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_scores": pc1,
        "pc2_scores": pc2,
        "groups": groups,
        "pc1_by_group": pc1_by_group,
        "pc1_median_by_group": {
            g: float(np.median(v)) if v.size else float("nan") for g, v in pc1_by_group.items()
        },
        "bic_1": bic_1,
        "bic_2": bic_2,
        "bic_delta": bic_delta,
        "bimodal": bool(np.isfinite(bic_delta) and bic_delta > 0),
    }


# ============================================================================
# Enrichment of the LR class in one cell type
# ============================================================================


def _lr_fraction(is_lr: np.ndarray, is_group: np.ndarray) -> float:
    """Fraction of LR cells within one group.

    Parameters
    ----------
    is_lr : ndarray of bool
        LR membership per cell.
    is_group : ndarray of bool
        Group membership per cell.

    Returns
    -------
    float
        ``mean(is_lr[is_group])``, NaN if the group is empty.
    """
    if not is_group.any():
        return float("nan")
    return float(np.mean(is_lr[is_group]))


def enrichment_test(
    df: pd.DataFrame,
    class_col: str = "lr_class",
    group_col: str = "cell_type",
    animal_col: str = "animal_id",
    n_perms: int = 2000,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Test whether the LR class is enriched in one cell type, within animals.

    Both cell types are patched in every animal, so cell-type labels are
    permuted **within** each animal. This preserves each animal's cell counts
    per type and removes animal-level differences in yield or recording quality
    from the null, which a plain Fisher exact test on pooled cells does not do.
    The Fisher p-value is reported as a descriptive statistic only.

    Parameters
    ----------
    df : DataFrame
        Cell-level table with *class_col* (output of :func:`classify_lr_rs`),
        *group_col* and *animal_col*.
    class_col : str
        Column holding ``"LR"`` / ``"RS"`` / ``"ambiguous"`` labels.
    group_col : str
        Cell-type column with exactly two groups.
    animal_col : str
        Animal identity column.
    n_perms : int
        Number of within-animal label permutations.
    rng : numpy Generator, int or None
        Random source or seed.

    Returns
    -------
    dict
        ``groups`` (the two labels in reporting order), ``n_cells``,
        ``n_animals``, ``n_<group>``, ``n_lr_<group>``, ``frac_lr_<group>``,
        ``observed_diff`` (first minus second fraction), ``fisher_odds_ratio``,
        ``fisher_p_descriptive``, ``p_perm``, ``n_perms``, ``null_median``.

    Raises
    ------
    ValueError
        If a required column is missing, *n_perms* is not positive, or
        *group_col* does not contain exactly two groups.
    """
    for col in (class_col, group_col, animal_col):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not found in DataFrame columns")
    if n_perms < 1:
        raise ValueError(f"n_perms must be >= 1, got {n_perms}")

    sub = df[[class_col, group_col, animal_col]].dropna()
    groups = ordered_groups(sub[group_col])
    if len(groups) != 2:
        raise ValueError(
            f"enrichment test requires exactly 2 groups in {group_col!r}, "
            f"found {len(groups)}: {groups}"
        )
    g1, g2 = groups
    generator = np.random.default_rng(rng)

    is_lr = (sub[class_col] == LR_CLASS).to_numpy()
    is_g1 = (sub[group_col] == g1).to_numpy()
    animal_names, animal_codes = np.unique(sub[animal_col].to_numpy(), return_inverse=True)

    frac_g1 = _lr_fraction(is_lr, is_g1)
    frac_g2 = _lr_fraction(is_lr, ~is_g1)
    observed = frac_g1 - frac_g2

    table = np.array(
        [
            [int(np.sum(is_lr & is_g1)), int(np.sum(~is_lr & is_g1))],
            [int(np.sum(is_lr & ~is_g1)), int(np.sum(~is_lr & ~is_g1))],
        ]
    )
    odds_ratio, fisher_p = stats.fisher_exact(table)

    index_groups = [np.flatnonzero(animal_codes == c) for c in range(animal_names.size)]
    null = np.empty(n_perms, dtype=float)
    for i in range(n_perms):
        permuted = is_g1.copy()
        for idx in index_groups:
            permuted[idx] = generator.permutation(is_g1[idx])
        null[i] = _lr_fraction(is_lr, permuted) - _lr_fraction(is_lr, ~permuted)

    finite_null = null[np.isfinite(null)]
    if np.isfinite(observed) and finite_null.size:
        n_extreme = int(np.sum(np.abs(finite_null) >= abs(observed)))
        p_perm = (n_extreme + 1) / (finite_null.size + 1)
        null_median = float(np.median(finite_null))
    else:
        p_perm = float("nan")
        null_median = float("nan")

    return {
        "groups": [str(g1), str(g2)],
        "n_cells": int(len(sub)),
        "n_animals": int(animal_names.size),
        f"n_{g1}": int(is_g1.sum()),
        f"n_{g2}": int((~is_g1).sum()),
        f"n_lr_{g1}": int(np.sum(is_lr & is_g1)),
        f"n_lr_{g2}": int(np.sum(is_lr & ~is_g1)),
        f"frac_lr_{g1}": frac_g1,
        f"frac_lr_{g2}": frac_g2,
        "observed_diff": observed,
        "fisher_odds_ratio": float(odds_ratio),
        "fisher_p_descriptive": float(fisher_p),
        "p_perm": p_perm,
        "n_perms": int(n_perms),
        "null_median": null_median,
    }


# ============================================================================
# Comparison with published reference values
# ============================================================================


def compare_to_reference(
    df: pd.DataFrame,
    reference: ReferenceRanges | Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    metric_cols: Sequence[str] | None = None,
    group_col: str = "cell_type",
) -> pd.DataFrame:
    """Compare per-group medians with published LR and RS reference ranges.

    Parameters
    ----------
    df : DataFrame
        Cell-level metrics table.
    reference : ReferenceRanges or mapping, optional
        ``{"LR": {column: (low, high)}, "RS": {column: (low, high)}}`` — the
        LR and RS ranges. Defaults to :data:`BRENNAN_2020_REFERENCE`.
    metric_cols : sequence of str, optional
        Metrics to compare. Defaults to every column present in both the
        reference and *df*.
    group_col : str
        Cell-type column.

    Returns
    -------
    DataFrame
        One row per (metric, group): ``metric``, ``group``, ``n``, ``median``,
        ``lr_low``, ``lr_high``, ``rs_low``, ``rs_high``, ``in_lr_range``,
        ``in_rs_range`` and ``verdict`` in {``"LR"``, ``"RS"``, ``"both"``,
        ``"neither"``} (``"both"`` when the reference ranges overlap at that
        median, so the metric does not discriminate there).

    Raises
    ------
    ValueError
        If *group_col* is missing, or no reference metric is present in *df*.

    Notes
    -----
    The default ranges are approximate and unverified — see
    :data:`BRENNAN_2020_REFERENCE`.

    References
    ----------
    Brennan et al. 2020. "Hyperexcitable Neurons Enable Precise and Persistent
    Information Encoding in the Superficial Retrosplenial Cortex." Cell Reports
    30:1598-1612. doi:10.1016/j.celrep.2019.12.093
    """
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} not found in DataFrame columns")

    ref = BRENNAN_2020_REFERENCE if reference is None else reference
    ranges = ref.ranges if isinstance(ref, ReferenceRanges) else ref
    lr_ranges = ranges.get(LR_CLASS, {})
    rs_ranges = ranges.get(RS_CLASS, {})

    if metric_cols is None:
        candidates = list(dict.fromkeys(list(lr_ranges) + list(rs_ranges)))
        metric_cols = [c for c in candidates if c in df.columns]
    else:
        metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        raise ValueError("no reference metric columns are present in the DataFrame")

    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        lr_low, lr_high = lr_ranges.get(metric, (np.nan, np.nan))
        rs_low, rs_high = rs_ranges.get(metric, (np.nan, np.nan))
        for group in ordered_groups(df[group_col]):
            vals = pd.to_numeric(df.loc[df[group_col] == group, metric], errors="coerce").dropna()
            median = float(vals.median()) if len(vals) else float("nan")
            in_lr = bool(np.isfinite(median) and lr_low <= median <= lr_high)
            in_rs = bool(np.isfinite(median) and rs_low <= median <= rs_high)
            if in_lr and in_rs:
                verdict = "both"
            elif in_lr:
                verdict = LR_CLASS
            elif in_rs:
                verdict = RS_CLASS
            else:
                verdict = "neither"
            rows.append(
                {
                    "metric": metric,
                    "group": str(group),
                    "n": int(len(vals)),
                    "median": median,
                    "lr_low": float(lr_low),
                    "lr_high": float(lr_high),
                    "rs_low": float(rs_low),
                    "rs_high": float(rs_high),
                    "in_lr_range": in_lr,
                    "in_rs_range": in_rs,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def with_thresholds(**overrides: Any) -> LRCriteria:
    """Return a copy of :data:`LR_CRITERIA` with selected thresholds replaced.

    Parameters
    ----------
    **overrides
        Field names of :class:`LRCriteria` to override (thresholds, column names
        or the provenance string).

    Returns
    -------
    LRCriteria
        New criteria object; :data:`LR_CRITERIA` is frozen and unchanged.

    Raises
    ------
    TypeError
        If an unknown field name is given.
    """
    return replace(LR_CRITERIA, **overrides)
