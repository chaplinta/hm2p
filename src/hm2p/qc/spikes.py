"""Calcium-signal and spike-extraction quality summary for one session.

Covers the Stage 4 outputs in ``ca.h5``: the dF/F baseline (F0), noise level,
the two event detectors (Voigts & Harnett noise-probability events and the
SD-threshold events), CASCADE spike-rate inference, and how much the three
agree frame by frame.

Methods summarised (implementations in :mod:`hm2p.calcium`):

- Voigts J, Harnett MT. 2020. "Somatic and dendritic encoding of spatial
  variables in retrosplenial cortex differs during 2D navigation." Neuron
  105:237-245. doi:10.1016/j.neuron.2019.10.016.
- Zong W, Obenhaus HA, Skytøen ER, et al. 2022. "Large-scale two-photon
  calcium imaging in freely moving mice." Cell 185:1240-1256.
  doi:10.1016/j.cell.2022.02.017.
- Rupprecht P, Carta S, Hoffmann A, et al. 2021. "A database and deep
  learning toolbox for noise-optimized, generalized spike inference from
  calcium imaging." Nature Neuroscience 24:1324-1337.
  doi:10.1038/s41593-021-00895-5. https://github.com/HelmchenLabSoftware/Cascade
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

from hm2p.calcium.qc import (
    ACTIVE_FRAC_MIN,
    BLEACH_MAX_LOSS,
    FNEU_CORR_MAX,
    SNR_MIN,
    TAU_MAX_S,
    TAU_MIN_S,
)
from hm2p.qc.common import bin_reduce, encode_i16, fnum, hist, mask_intervals, rounded, run_lengths

ROI_TYPE_NAMES = {0: "soma", 1: "dend", 2: "artefact"}  # ca.h5 roi_types code
WINDOW_S = 90.0
OVERVIEW_BIN_S = 2.0
SPIKE_PRESENT = 0.2  # spikes/s summed over an event counted as "CASCADE saw it"


def noise_sd(trace: npt.ArrayLike) -> float:
    """Robust noise SD from the first difference: MAD(diff) / (0.6745 * sqrt 2).

    Insensitive to slow baseline changes and to sparse transients.
    """
    d = np.diff(np.asarray(trace, dtype=np.float64))
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    return float(np.median(np.abs(d - np.median(d))) / (0.6745 * np.sqrt(2.0)))


def jaccard(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Frame-level Jaccard index of two boolean masks (NaN when both are empty)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def event_count(mask: npt.ArrayLike) -> int:
    """Number of contiguous True runs in an event mask."""
    v, _, _ = run_lengths(np.asarray(mask, dtype=bool))
    return int(v.sum())


def busiest_window(mask: npt.ArrayLike, win: int) -> int:
    """Start index of the *win*-frame window with most event frames (0 if none)."""
    m = np.asarray(mask, dtype=np.float64)
    if m.size <= win:
        return 0
    c = np.convolve(m, np.ones(win), mode="valid")
    return int(np.argmax(c)) if c.max() > 0 else int((m.size - win) // 2)


def _roi_metrics(
    dff: np.ndarray,
    vh: np.ndarray | None,
    sd: np.ndarray | None,
    spk: np.ndarray | None,
    f0: np.ndarray | None,
    fcorr: np.ndarray | None,
    dur_min: float,
) -> dict:
    nsd = noise_sd(dff)
    r: dict = {
        "noise_sd": fnum(nsd, 4),
        "snr_p99": fnum(
            np.nanpercentile(dff, 99) / nsd if nsd and np.isfinite(nsd) and nsd > 0 else np.nan, 2
        ),
        "dff_nan": fnum(np.mean(~np.isfinite(dff)), 4),
    }
    if vh is not None:
        r["rate_vh"] = fnum(event_count(vh) / dur_min, 3)
        r["frac_vh"] = fnum(vh.mean(), 4)
        amps = [np.nanmax(dff[a:b]) for a, b in mask_intervals(vh)]
        r["amp_vh"] = fnum(np.median(amps), 3) if amps else None
    if sd is not None:
        r["rate_sd"] = fnum(event_count(sd) / dur_min, 3)
    if vh is not None and sd is not None:
        r["jaccard_vh_sd"] = fnum(jaccard(vh, sd), 3)
    if spk is not None:
        ok = np.isfinite(spk)
        r["spk_nan"] = fnum(np.mean(~ok), 4)
        r["spk_rate"] = fnum(np.nanmean(spk), 4) if ok.any() else None
        both = ok & np.isfinite(dff)
        r["spk_dff_rho"] = (
            fnum(stats.spearmanr(spk[both], dff[both]).statistic, 3)
            if both.sum() > 20 and np.nanstd(spk[both]) > 0
            else None
        )
        tot = np.nansum(spk)
        if vh is not None or sd is not None:
            any_ev = np.zeros(spk.size, dtype=bool)
            if vh is not None:
                any_ev |= vh
            if sd is not None:
                any_ev |= sd
            r["spk_in_events"] = fnum(np.nansum(spk[any_ev]) / tot, 3) if tot > 0 else None
        if vh is not None:
            ivs = mask_intervals(vh)
            if ivs:
                seen = [np.nansum(spk[a:b]) > SPIKE_PRESENT for a, b in ivs]
                r["vh_with_spikes"] = fnum(np.mean(seen), 3)
    if f0 is not None:
        f0f = f0[np.isfinite(f0)]
        if f0f.size:
            med = np.median(f0f)
            r["f0_min_rel"] = fnum(f0f.min() / med if med > 0 else np.nan, 3)
            r["f0_le0"] = fnum(np.mean(f0f <= 0), 4)
            k = max(1, f0f.size // 20)
            r["f0_end_over_start"] = fnum(
                np.median(f0f[-k:]) / np.median(f0f[:k]) if np.median(f0f[:k]) > 0 else np.nan, 3
            )
    if fcorr is not None and f0 is not None:
        r["frac_below_f0"] = fnum(np.mean(fcorr < f0), 3)
    return r


def _get(ca: dict, k: str) -> np.ndarray | None:
    return None if k not in ca else np.asarray(ca[k])


def summarise_spikes(
    ca: dict[str, npt.ArrayLike], attrs: dict | None = None, light_on: npt.ArrayLike | None = None
) -> dict:
    """JSON-ready calcium / spike-extraction summary of one session.

    Parameters
    ----------
    ca
        Datasets read from ``ca.h5``: ``dff`` (n_rois, T) required; optional
        ``F_corr``, ``F0_rolling`` / ``F0_percentile``, ``event_masks``,
        ``event_masks_sd``, ``spikes``, ``roi_types``, ``frame_times`` and
        ``roi_qc/<name>`` entries.
    attrs
        Root attributes of ``ca.h5`` (``fps_imaging``, ``f0_method`` ...).
    light_on
        Optional per-imaging-frame room-light flag (from ``sync.h5``).
    """
    attrs = attrs or {}
    dff = np.asarray(ca["dff"], dtype=np.float64)
    if dff.ndim != 2:
        raise ValueError("dff must be (n_rois, n_frames)")
    n_rois, T = dff.shape
    fps = float(attrs.get("fps_imaging", 0) or 0)
    ft = _get(ca, "frame_times")
    if fps <= 0 and ft is not None and ft.size > 2:
        fps = 1.0 / float(np.median(np.diff(ft)))
    if fps <= 0:
        raise ValueError("cannot determine imaging frame rate")
    f0_method = attrs.get("f0_method", "rolling")
    if isinstance(f0_method, bytes):
        f0_method = f0_method.decode()
    f0_all = _get(ca, "F0_percentile" if f0_method == "percentile" else "F0_rolling")
    fcorr_all = _get(ca, "F_corr")
    vh_all = _get(ca, "event_masks")
    sd_all = _get(ca, "event_masks_sd")
    spk_all = _get(ca, "spikes")
    types = _get(ca, "roi_types")
    types = (
        types.astype(int)
        if types is not None and types.size == n_rois
        else np.zeros(n_rois, dtype=int)
    )
    dur_min = T / fps / 60.0

    def row(a: np.ndarray | None, i: int, dtype: type = np.float64) -> np.ndarray | None:
        return None if a is None or a.shape[0] != n_rois else np.asarray(a[i], dtype=dtype)

    win = min(T, round(WINDOW_S * fps))
    ov = max(1, round(OVERVIEW_BIN_S * fps))
    rois = []
    for i in range(n_rois):
        dff_i = dff[i]
        vh = row(vh_all, i, bool)
        sd = row(sd_all, i, bool)
        spk = row(spk_all, i)
        f0 = row(f0_all, i)
        fc = row(fcorr_all, i)
        m = _roi_metrics(dff_i, vh, sd, spk, f0, fc, dur_min)
        m["i"] = i
        m["type"] = ROI_TYPE_NAMES.get(int(types[i]), str(types[i]))
        if m["type"] != "artefact":
            a = busiest_window(vh if vh is not None else np.zeros(T, bool), win)
            sl = slice(a, a + win)
            tr: dict = {"start": a, "dff": encode_i16(dff_i[sl])}
            if spk is not None:
                tr["spikes"] = encode_i16(spk[sl])
            if vh is not None:
                tr["vh"] = mask_intervals(vh[sl])
            if sd is not None:
                tr["sd"] = mask_intervals(sd[sl])
            if fc is not None:
                tr["fcorr_ov"] = encode_i16(bin_reduce(fc, ov))
            if f0 is not None:
                tr["f0_ov"] = encode_i16(bin_reduce(f0, ov))
            m["trace"] = tr
        rois.append(m)

    qc = {
        k.split("/", 1)[1]: np.asarray(v, dtype=float)
        for k, v in ca.items()
        if k.startswith("roi_qc/")
        and np.asarray(v).size == n_rois
        and np.asarray(v).dtype.kind in "fiu"
    }
    if qc:
        for k, v in qc.items():
            for r, val in zip(rois, v, strict=True):
                r["qc_" + k] = fnum(val, 4)
    out: dict = {
        "n_rois": n_rois,
        "n_frames": int(T),
        "fps": fnum(fps, 4),
        "duration_min": fnum(dur_min, 2),
        "f0_method": f0_method,
        "neuropil_method": _s(attrs.get("neuropil_method")),
        "spikes_model": _s(attrs.get("spikes_model")),
        "spikes_units": _s(attrs.get("spikes_units")),
        "has": {
            "spikes": spk_all is not None,
            "vh": vh_all is not None,
            "sd": sd_all is not None,
            "f0": f0_all is not None,
        },
        "window_s": WINDOW_S,
        "overview_bin_s": OVERVIEW_BIN_S,
        "counts": {name: int((types == k).sum()) for k, name in ROI_TYPE_NAMES.items()},
        "qc_fail": _qc_fail_fractions(qc, types),
        "rois": rois,
        "population": _population(dff, spk_all, types, ov, light_on),
    }
    return out


def _s(v: object) -> str | None:
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


def _qc_fail_fractions(qc: dict[str, np.ndarray], types: np.ndarray) -> dict:
    """Fraction of soma ROIs failing each threshold in :mod:`hm2p.calcium.qc`."""
    soma = types == 0
    if not soma.any() or not qc:
        return {}
    tests = {
        "snr_event": lambda v: v < SNR_MIN,
        "decay_tau_s": lambda v: (v < TAU_MIN_S) | (v > TAU_MAX_S),
        "fneu_dff_corr": lambda v: v > FNEU_CORR_MAX,
        "bleach_slope": lambda v: v < BLEACH_MAX_LOSS,
        "active_fraction": lambda v: v < ACTIVE_FRAC_MIN,
    }
    out = {}
    for k, f in tests.items():
        if k in qc:
            v = qc[k][soma]
            with np.errstate(invalid="ignore"):
                out[k] = fnum(np.mean(f(v) | ~np.isfinite(v)), 3)
    return out


def _population(
    dff: np.ndarray,
    spk: np.ndarray | None,
    types: np.ndarray,
    ov: int,
    light_on: npt.ArrayLike | None,
) -> dict:
    """Soma-mean dF/F and spike rate over the session (detects shared artefacts)."""
    soma = types == 0
    sel = soma if soma.any() else np.ones(types.size, dtype=bool)
    with np.errstate(all="ignore"):
        mean_dff = np.nanmean(dff[sel], axis=0)
    p: dict = {"mean_dff": encode_i16(bin_reduce(mean_dff, ov))}
    if spk is not None and spk.shape == dff.shape:
        with np.errstate(all="ignore"):
            p["mean_spk"] = encode_i16(bin_reduce(np.nanmean(spk[sel], axis=0), ov))
    if dff[sel].shape[0] >= 2:
        sub = dff[sel][:, np.all(np.isfinite(dff[sel]), axis=0)]
        if sub.shape[1] > 10:
            c = np.corrcoef(sub)
            iu = np.triu_indices_from(c, k=1)
            p["pair_corr_hist"] = hist(c[iu], -0.2, 1.0, 48)
    if light_on is not None:
        lo = np.asarray(light_on, dtype=float).ravel()
        if lo.size == dff.shape[1]:
            p["light"] = rounded(bin_reduce(lo, ov), 2)
    return p


def overview_row(s: dict) -> dict:
    """The few numbers shown per session in the cross-session table."""
    soma = [r for r in s["rois"] if r["type"] == "soma"]

    def med(k: str) -> float | None:
        v = [r[k] for r in soma if r.get(k) is not None]
        return fnum(np.median(v), 4) if v else None

    return {
        "n_soma": len(soma),
        "noise_sd": med("noise_sd"),
        "snr_p99": med("snr_p99"),
        "rate_vh": med("rate_vh"),
        "rate_sd": med("rate_sd"),
        "jaccard_vh_sd": med("jaccard_vh_sd"),
        "spk_rate": med("spk_rate"),
        "spk_dff_rho": med("spk_dff_rho"),
        "vh_with_spikes": med("vh_with_spikes"),
        "f0_end_over_start": med("f0_end_over_start"),
        "has_spikes": s["has"]["spikes"],
    }
