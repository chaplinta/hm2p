"""Spike-frequency adaptation from current-step (IV) sweeps.

Adaptation is measured on two sweeps per cell so that cells firing at
different rates are compared fairly: the sweep with the most spikes, and the
lowest-current sweep reaching a fixed spike count. Measures:

- adaptation index: mean of (ISI_{i+1} - ISI_i) / (ISI_{i+1} + ISI_i)
  over successive intervals (0 = no adaptation, positive = slowing). This is
  the ``adaptation_index`` used by the Allen Cell Types electrophysiology
  feature set and eFEL ("adaptation_index2").
  Gouwens et al. 2019. "Classification of electrophysiological and
  morphological neuron types in the mouse visual cortex." Nature
  Neuroscience 22:1182-1195. doi:10.1038/s41593-019-0417-0
- SFA ratio: last ISI / first ISI.
- ISI coefficient of variation.
- early/late rate ratio: firing rate over the first third of the spike train
  divided by the rate over the last third.

Spike detection reuses :func:`hm2p.patching.ephys.detect_spikes`, with
detected peaks required to exceed ``min_peak_mv`` so that subthreshold
sweeps do not yield spurious events.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from hm2p.patching.ephys import detect_spikes
from hm2p.patching.lr_classify import adaptation_index, spike_frequency_adaptation_ratio


def sweep_spike_times(
    trace: npt.ArrayLike, sample_rate_hz: float, min_peak_mv: float = -10.0
) -> np.ndarray:
    """Spike peak times (s) in one sweep; peaks must exceed *min_peak_mv*."""
    v = np.asarray(trace, dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).any() or np.nanmax(v) < min_peak_mv:
        return np.zeros(0)
    idx = np.asarray(detect_spikes(v), dtype=np.int64)
    idx = idx[v[idx] > min_peak_mv] if idx.size else idx
    return idx.astype(np.float64) / float(sample_rate_hz)


def _early_late_rate_ratio(spike_times_s: np.ndarray) -> float:
    """Rate over the first third of the spike train divided by the last third."""
    t = np.asarray(spike_times_s, dtype=np.float64)
    if t.size < 6:
        return float("nan")
    span = t[-1] - t[0]
    if span <= 0:
        return float("nan")
    third = span / 3.0
    early = np.sum(t < t[0] + third)
    late = np.sum(t > t[-1] - third)
    if late == 0:
        return float("nan")
    return float(early / late)


def sweep_adaptation(spike_times_s: npt.ArrayLike) -> dict[str, float]:
    """Adaptation measures for one spike train (times in seconds)."""
    t = np.asarray(spike_times_s, dtype=np.float64)
    isi = np.diff(t)
    nan = float("nan")
    return {
        "n_spikes": float(t.size),
        "adaptation_index": float(adaptation_index(t)) if t.size >= 3 else nan,
        "sfa_ratio": float(spike_frequency_adaptation_ratio(t)) if t.size >= 3 else nan,
        "isi_cv": float(np.std(isi) / np.mean(isi)) if isi.size >= 2 and np.mean(isi) > 0 else nan,
        "first_isi_ms": float(isi[0] * 1e3) if isi.size >= 1 else nan,
        "early_late_rate_ratio": _early_late_rate_ratio(t),
    }


def cell_adaptation(
    traces: npt.ArrayLike,
    stim: npt.ArrayLike,
    sample_rate_hz: float,
    target_spikes: int = 6,
    min_peak_mv: float = -10.0,
) -> dict[str, Any]:
    """Adaptation on the max-spike sweep and on the first sweep with ``target_spikes``.

    Parameters
    ----------
    traces : (n_samples, n_sweeps) array of membrane potential (mV).
    stim : (n_sweeps,) injected current per sweep.
    sample_rate_hz : sampling rate.
    target_spikes : spike count defining the rate-matched sweep.

    Returns
    -------
    dict
        Keys prefixed ``max_`` (max-spike sweep) and ``matched_`` (first,
        lowest-current sweep with >= ``target_spikes`` spikes), plus
        ``fi_slope_spikes_per_pa`` (least-squares slope of spike count on
        current over suprathreshold sweeps; descriptive) and
        ``matched_current``.
    """
    v = np.asarray(traces, dtype=np.float64)
    i_inj = np.asarray(stim, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != i_inj.size:
        raise ValueError("traces must be (n_samples, n_sweeps) matching stim length")
    trains = [sweep_spike_times(v[:, k], sample_rate_hz, min_peak_mv) for k in range(v.shape[1])]
    counts = np.array([t.size for t in trains])
    out: dict[str, Any] = {}
    nan_block = sweep_adaptation(np.zeros(0))
    k_max = int(np.argmax(counts)) if counts.size else -1
    max_block = sweep_adaptation(trains[k_max]) if k_max >= 0 and counts[k_max] else nan_block
    out.update({f"max_{k}": val for k, val in max_block.items()})
    order = np.argsort(i_inj)
    k_match = next((int(k) for k in order if counts[k] >= target_spikes), None)
    if k_match is None:
        out.update({f"matched_{k}": val for k, val in nan_block.items()})
        out["matched_current"] = float("nan")
    else:
        out.update({f"matched_{k}": val for k, val in sweep_adaptation(trains[k_match]).items()})
        out["matched_current"] = float(i_inj[k_match])
    supra = counts > 0
    if supra.sum() >= 2 and np.ptp(i_inj[supra]) > 0:
        out["fi_slope_spikes_per_pa"] = float(np.polyfit(i_inj[supra], counts[supra], 1)[0])
    else:
        out["fi_slope_spikes_per_pa"] = float("nan")
    return out
