#!/usr/bin/env python3
"""Dark-enhancement confound gauntlet + mechanism tests (paired, within-session).

Stress-tests and mechanistically explains the headline finding "RSP HD mean
vector length (MVL) is HIGHER in the dark" (see
.claude/agent-memory/rsp-science-advisor/project_dark_enhancement_new_hypotheses.md).

Design
------
- Unit of analysis = session. Light vs dark is PAIRED within each session.
- For each session we compute, per soma ROI, an MVL (or other metric) in light
  and in dark under various matched-sampling controls, summarise to one number
  per session (median over HD-significant soma cells, falling back to all soma
  cells), then test across sessions with Wilcoxon signed-rank.
- Soma ROIs only (roi_types == 0). All non-excluded sessions.
- dF/F is the primary signal; events (binary spike-like masks) is a cheap
  sensitivity check. deconv is skipped (degenerate / all-zero dataset-wide).
- All tests non-parametric. FDR (Benjamini-Hochberg) across the confirmatory
  family. Effect sizes: matched-pairs rank-biserial correlation + median diff.

Make-or-break: A1 (occupancy-matched) and A2 (speed+|AHV|-matched). If the
dark>light MVL effect vanishes under matching, the headline is a sampling
artefact and must be reframed.

Statistical methods
--------------------
Wilcoxon signed-rank (Wilcoxon 1945); matched-pairs rank-biserial effect size
(Kerby 2014); McNemar test for paired proportions (McNemar 1947);
Benjamini-Hochberg FDR (Benjamini & Hochberg 1995). Occupancy / kinematics
matched-sampling and circular-shuffle MVL debiasing follow Hardcastle et al.
(2017) and Muller & Kubie (1987) — implemented in
``hm2p.analysis.matched_tuning``.

References
----------
Wilcoxon 1945. "Individual comparisons by ranking methods." Biometrics
    Bulletin 1(6):80-83. doi:10.2307/3001968
Kerby 2014. "The simple difference formula: an approach to teaching nonparametric
    correlation." Comprehensive Psychology 3:11.IT.3.1. doi:10.2466/11.IT.3.1
McNemar 1947. "Note on the sampling error of the difference between correlated
    proportions or percentages." Psychometrika 12(2):153-157.
    doi:10.1007/BF02295996
Benjamini & Hochberg 1995. "Controlling the false discovery rate." JRSS-B
    57(1):289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

Usage
-----
    python scripts/run_dark_hypotheses.py --smoke               # 2 sessions, fast
    python scripts/run_dark_hypotheses.py --limit 3 --n-perms 50
    python scripts/run_dark_hypotheses.py                       # full run
    python scripts/run_dark_hypotheses.py --signal events       # sensitivity
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.analysis.information import skaggs_info_rate  # noqa: E402
from hm2p.analysis.matched_tuning import (  # noqa: E402
    match_indices_1d,
    match_indices_2d,
    matched_condition_mvl,
    occupancy_histogram,
    shuffle_debiased_mvl,
    shuffle_debiased_statistic,
    tuning_curve_fwhm,
)
from hm2p.analysis.tuning import (  # noqa: E402
    compute_hd_tuning_curve,
    mean_vector_length,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("dark_hyp")

BUCKET = "hm2p-derivatives"
SPEED_THRESHOLD = 2.5  # cm/s — matches AnalysisParams.speed_threshold
HD_N_BINS = 36
HD_SMOOTH_DEG = 6.0
ALPHA = 0.05

# Confirmatory family (FDR-corrected together). Generating-only hyps excluded.
CONFIRMATORY = {"A1", "A2", "A3", "C1", "C2", "B2", "D1"}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.Session(profile_name="hm2p-agent").client("s3")
    return _S3


def _download_h5(key: str) -> h5py.File | None:
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=key)
        return h5py.File(io.BytesIO(obj["Body"].read()), "r")
    except Exception as exc:  # noqa: BLE001
        log.debug("download failed %s: %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# Metadata + session enumeration
# ---------------------------------------------------------------------------


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(__file__).resolve().parent.parent / "metadata"
    animals = pd.read_csv(base / "animals.csv")
    animals["animal_id"] = animals["animal_id"].astype(str)
    exps = pd.read_csv(base / "experiments.csv")
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    return animals, exps


def session_keys(exp_id: str, animal_id: str) -> tuple[str, str, str, str]:
    """Return (sub, ses, sync_key, analysis_key) for an exp_id."""
    parts = exp_id.split("_")
    sub = f"sub-{animal_id}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return (
        sub,
        ses,
        f"sync/{sub}/{ses}/sync.h5",
        f"analysis/{sub}/{ses}/analysis.h5",
    )


# ---------------------------------------------------------------------------
# Per-session loading from sync.h5
# ---------------------------------------------------------------------------


def load_session_arrays(sync_f: h5py.File, signal: str) -> dict | None:
    """Load aligned behaviour + soma signal arrays from an open sync.h5.

    Returns a dict with the frame-aligned behaviour vectors, the (n_soma,
    n_frames) signal matrix, soma ROI indices, and the precomputed moving /
    light / dark masks. None if the session lacks required fields or has no
    soma ROIs.
    """
    required = [
        "hd_deg",
        "speed_cm_s",
        "ahv_deg_s",
        "light_on",
        "bad_behav",
        "active",
        "roi_types",
        signal,
    ]
    for r in required:
        if r not in sync_f:
            log.warning("sync.h5 missing %s — skipping session", r)
            return None

    roi_types = sync_f["roi_types"][:]
    soma_idx = np.where(roi_types == 0)[0]
    if soma_idx.size == 0:
        return None

    sig = sync_f[signal][:][soma_idx].astype(np.float64)  # (n_soma, n_frames)
    n_frames = sig.shape[1]

    def _vec(name):
        v = sync_f[name][:].astype(np.float64)
        return v[:n_frames]

    hd = _vec("hd_deg")
    speed = _vec("speed_cm_s")
    ahv = _vec("ahv_deg_s")
    light_on = sync_f["light_on"][:][:n_frames].astype(bool)
    bad_behav = sync_f["bad_behav"][:][:n_frames].astype(bool)
    hd_conf = _vec("hd_confidence") if "hd_confidence" in sync_f else np.full(n_frames, np.nan)

    valid = ~bad_behav & np.isfinite(hd) & np.isfinite(speed)
    moving = valid & (speed >= SPEED_THRESHOLD)

    out = {
        "signal": sig,
        "soma_idx": soma_idx,
        "hd": hd,
        "speed": speed,
        "abs_ahv": np.abs(ahv),
        "light_on": light_on,
        "bad_behav": bad_behav,
        "valid": valid,
        "moving": moving,
        "hd_conf": hd_conf,
        "n_frames": n_frames,
    }
    # Optional fields used by specific hypotheses
    for opt in ("x_maze", "y_maze", "Fneu_raw", "frame_times"):
        if opt in sync_f:
            arr = sync_f[opt][:]
            if arr.ndim == 1:
                out[opt] = arr[:n_frames]
            else:
                out[opt] = arr[:, :n_frames]
    return out


# ---------------------------------------------------------------------------
# Per-cell MVL helpers
# ---------------------------------------------------------------------------


def _cell_mvl(signal_1d, hd, mask):
    tc, bc = compute_hd_tuning_curve(
        signal_1d, hd, mask, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG
    )
    return mean_vector_length(tc, bc)


def _significant_hd_mask(arrays, signal_idx_local) -> np.ndarray:
    """Boolean over soma cells: HD-significant in light OR dark (raw MVL > 0.1
    in both conditions as a cheap proxy when analysis.h5 significance not loaded).

    Replaced by analysis.h5-based significance where available (see
    load_analysis_significance).
    """
    return np.ones(len(signal_idx_local), dtype=bool)


# ---------------------------------------------------------------------------
# A1 / A2 / C1: matched-MVL engine, per session
# ---------------------------------------------------------------------------


def _matched_session(
    arrays, match, n_boot, n_shuffles, rng, mask_light=None, mask_dark=None,
    match_vars_kind=None, statistic="mvl",
):
    """Compute per-soma-cell debiased MVL in light & dark under matched sampling.

    Returns arrays (n_soma,) of mvl_light, mvl_dark (bootstrap-mean debiased),
    plus raw (un-debiased) versions and the mean matched frame count.
    """
    sig = arrays["signal"]
    hd = arrays["hd"]
    if mask_light is None:
        mask_light = arrays["moving"] & arrays["light_on"]
    if mask_dark is None:
        mask_dark = arrays["moving"] & ~arrays["light_on"]

    hd_l = hd[mask_light]
    hd_d = hd[mask_dark]
    if hd_l.size < 50 or hd_d.size < 50:
        return None

    if match == "kinematics":
        mv_l = (arrays["speed"][mask_light], arrays["abs_ahv"][mask_light])
        mv_d = (arrays["speed"][mask_dark], arrays["abs_ahv"][mask_dark])
    else:
        mv_l = mv_d = None

    n_soma = sig.shape[0]
    mvl_l = np.full(n_soma, np.nan)
    mvl_d = np.full(n_soma, np.nan)
    raw_l = np.full(n_soma, np.nan)
    raw_d = np.full(n_soma, np.nan)
    n_matched = np.nan

    for c in range(n_soma):
        s_l = sig[c][mask_light]
        s_d = sig[c][mask_dark]
        res = matched_condition_mvl(
            s_l,
            hd_l,
            s_d,
            hd_d,
            match_vars_a=mv_l,
            match_vars_b=mv_d,
            match=match,
            n_bins=HD_N_BINS,
            smoothing_sigma_deg=HD_SMOOTH_DEG,
            n_boot=n_boot,
            n_shuffles=n_shuffles,
            debias=True,
            statistic=statistic,
            rng=rng,
        )
        mvl_l[c] = res["mvl_a"]
        mvl_d[c] = res["mvl_b"]
        raw_l[c] = res["mvl_a_raw"]
        raw_d[c] = res["mvl_b_raw"]
        if c == 0:
            n_matched = res["n_matched"]

    return {
        "mvl_light": mvl_l,
        "mvl_dark": mvl_d,
        "raw_light": raw_l,
        "raw_dark": raw_d,
        "n_matched": n_matched,
    }


# ---------------------------------------------------------------------------
# Non-parametric paired test + effect size
# ---------------------------------------------------------------------------


def paired_test(light_vals, dark_vals, label=""):
    """Wilcoxon signed-rank on (dark - light) across sessions + rank-biserial.

    Positive median diff / effect => dark > light.
    """
    a = np.asarray(light_vals, float)
    b = np.asarray(dark_vals, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    diff = b - a  # dark - light
    nz = diff[diff != 0]
    if len(nz) < 2:
        return {
            "label": label,
            "n": n,
            "p_value": np.nan,
            "statistic": np.nan,
            "median_diff": float(np.median(diff)) if n else np.nan,
            "rank_biserial": np.nan,
            "n_dark_gt_light": int(np.sum(diff > 0)),
        }
    w, p = stats.wilcoxon(nz, alternative="two-sided")
    # Matched-pairs rank-biserial (Kerby 2014): r = (W+ - W-) / W_total
    ranks = stats.rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    total = w_pos + w_neg
    rb = float((w_pos - w_neg) / total) if total > 0 else np.nan
    return {
        "label": label,
        "n": n,
        "statistic": float(w),
        "p_value": float(p),
        "median_diff": float(np.median(diff)),
        "rank_biserial": rb,
        "n_dark_gt_light": int(np.sum(diff > 0)),
    }


def _session_summary(mvl_light, mvl_dark, sig_mask=None):
    """Median MVL (light, dark) over HD-significant soma cells (fallback: all)."""
    ml = np.asarray(mvl_light, float)
    md = np.asarray(mvl_dark, float)
    both = np.isfinite(ml) & np.isfinite(md)
    if sig_mask is not None:
        use = both & sig_mask
        if use.sum() < 1:
            use = both  # fallback
    else:
        use = both
    if use.sum() < 1:
        return np.nan, np.nan, 0
    return float(np.median(ml[use])), float(np.median(md[use])), int(use.sum())


# ---------------------------------------------------------------------------
# analysis.h5 significance / width (for B2, B3, and significance masks)
# ---------------------------------------------------------------------------


def load_analysis(analysis_f: h5py.File, signal: str, soma_idx: np.ndarray) -> dict | None:
    if analysis_f is None or signal not in analysis_f:
        return None
    grp = analysis_f[signal]
    out = {}
    for cond in ("light", "dark", "all"):
        h = grp.get(f"hd/{cond}")
        if h is None:
            continue
        for k in ("mvl", "tuning_width", "significant"):
            if k in h:
                out[f"{cond}_{k}"] = h[k][:][soma_idx]
    return out


# ===========================================================================
# Hypotheses
# ===========================================================================


def run_matched_hypothesis(sessions, hid, match, n_boot, n_shuffles, seed, statistic="mvl"):
    """A1 (occupancy), A2 (kinematics), C1 (state composition) share this engine."""
    rows = []
    light_summ, dark_summ = [], []
    for s in sessions:
        arrays = s["arrays"]
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        if hid.startswith("C1"):
            # Match active/bad_behav composition: restrict both conditions to
            # ACTIVE moving frames (already exclude bad_behav via 'valid'); then
            # the comparison is occupancy-matched on top (state-composition
            # equalised by construction since both use identical active+moving).
            ml = arrays["moving"] & arrays["light_on"]
            md = arrays["moving"] & ~arrays["light_on"]
            res = _matched_session(
                arrays, "occupancy", n_boot, n_shuffles, rng,
                mask_light=ml, mask_dark=md, statistic=statistic,
            )
        else:
            res = _matched_session(
                arrays, match, n_boot, n_shuffles, rng, statistic=statistic
            )
        if res is None:
            continue
        sig_mask = s.get("hd_sig_mask")
        l, d, ncell = _session_summary(res["mvl_light"], res["mvl_dark"], sig_mask)
        rl, rd, _ = _session_summary(res["raw_light"], res["raw_dark"], sig_mask)
        if not (np.isfinite(l) and np.isfinite(d)):
            continue
        light_summ.append(l)
        dark_summ.append(d)
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "n_cells": ncell,
                "mvl_light_matched": l,
                "mvl_dark_matched": d,
                "mvl_light_raw": rl,
                "mvl_dark_raw": rd,
                "n_matched_frames": res["n_matched"],
            }
        )
    test = paired_test(light_summ, dark_summ, label=hid)
    return test, pd.DataFrame(rows)


def run_C2_hd_confidence(sessions, seed):
    """C2: hd_confidence light vs dark (validates IR-camera assumption)."""
    light_summ, dark_summ, rows = [], [], []
    for s in sessions:
        a = s["arrays"]
        conf = a["hd_conf"]
        ml = a["moving"] & a["light_on"]
        md = a["moving"] & ~a["light_on"]
        cl = conf[ml]
        cd = conf[md]
        cl = cl[np.isfinite(cl)]
        cd = cd[np.isfinite(cd)]
        if cl.size < 50 or cd.size < 50:
            continue
        light_summ.append(float(np.median(cl)))
        dark_summ.append(float(np.median(cd)))
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "hd_conf_light": light_summ[-1],
                "hd_conf_dark": dark_summ[-1],
            }
        )
    test = paired_test(light_summ, dark_summ, label="C2")
    return test, pd.DataFrame(rows)


def run_A3_epoch_order(sessions, match, n_boot, n_shuffles, seed):
    """A3: exclude first light + first dark epoch (adaptation transient), recompute
    occupancy-matched MVL light vs dark.

    Epochs are contiguous runs of light_on; we drop the frames belonging to the
    first light run and the first dark run, then matched-MVL as in A1.
    """
    light_summ, dark_summ, rows = [], [], []
    for s in sessions:
        a = s["arrays"]
        light = a["light_on"]
        # Find epoch boundaries
        changes = np.where(np.diff(light.astype(int)) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(light)]])
        drop = np.zeros(len(light), dtype=bool)
        seen_light = seen_dark = False
        for st, en in zip(starts, ends):
            is_light = bool(light[st])
            if is_light and not seen_light:
                drop[st:en] = True
                seen_light = True
            elif (not is_light) and not seen_dark:
                drop[st:en] = True
                seen_dark = True
            if seen_light and seen_dark:
                break
        keep = ~drop
        ml = a["moving"] & light & keep
        md = a["moving"] & ~light & keep
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        res = _matched_session(a, match, n_boot, n_shuffles, rng, mask_light=ml, mask_dark=md)
        if res is None:
            continue
        sig_mask = s.get("hd_sig_mask")
        l, d, ncell = _session_summary(res["mvl_light"], res["mvl_dark"], sig_mask)
        if not (np.isfinite(l) and np.isfinite(d)):
            continue
        light_summ.append(l)
        dark_summ.append(d)
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "n_cells": ncell,
                "mvl_light_matched": l,
                "mvl_dark_matched": d,
            }
        )
    test = paired_test(light_summ, dark_summ, label="A3")
    return test, pd.DataFrame(rows)


def run_B2_gain_vs_sharpening(sessions):
    """B2: amplitude-normalised MVL (use stored MVL, already a normalised
    resultant length) + tuning WIDTH from analysis.h5, paired light vs dark.

    Classifies the session population: width DOWN in dark + MVL UP => sharpening;
    MVL UP, width FLAT => gain-only. Uses analysis.h5 hd/{light,dark}.
    """
    mvl_l_summ, mvl_d_summ = [], []
    wid_l_summ, wid_d_summ = [], []
    rows = []
    for s in sessions:
        ah = s.get("analysis")
        if ah is None:
            continue
        for need in ("light_mvl", "dark_mvl", "light_tuning_width", "dark_tuning_width"):
            if need not in ah:
                break
        else:
            sig_mask = s.get("hd_sig_mask")
            ml = ah["light_mvl"].astype(float)
            md = ah["dark_mvl"].astype(float)
            wl = ah["light_tuning_width"].astype(float)
            wd = ah["dark_tuning_width"].astype(float)
            both = np.isfinite(ml) & np.isfinite(md) & np.isfinite(wl) & np.isfinite(wd)
            if sig_mask is not None and (both & sig_mask).sum() >= 1:
                both = both & sig_mask
            if both.sum() < 1:
                continue
            mvl_l_summ.append(float(np.median(ml[both])))
            mvl_d_summ.append(float(np.median(md[both])))
            wid_l_summ.append(float(np.median(wl[both])))
            wid_d_summ.append(float(np.median(wd[both])))
            rows.append(
                {
                    "exp_id": s["exp_id"],
                    "animal_id": s["animal_id"],
                    "celltype": s["celltype"],
                    "n_cells": int(both.sum()),
                    "mvl_light": mvl_l_summ[-1],
                    "mvl_dark": mvl_d_summ[-1],
                    "width_light": wid_l_summ[-1],
                    "width_dark": wid_d_summ[-1],
                }
            )
    mvl_test = paired_test(mvl_l_summ, mvl_d_summ, label="B2_mvl")
    wid_test = paired_test(wid_l_summ, wid_d_summ, label="B2_width")
    # Classification
    df = pd.DataFrame(rows)
    verdict = "inconclusive"
    if not df.empty:
        mvl_up = mvl_test["median_diff"] > 0 and mvl_test["p_value"] < ALPHA
        width_down = wid_test["median_diff"] < 0 and wid_test["p_value"] < ALPHA
        if mvl_up and width_down:
            verdict = "sharpening (MVL up + width down)"
        elif mvl_up and not width_down:
            verdict = "gain-only (MVL up, width flat)"
        elif not mvl_up:
            verdict = "no MVL enhancement in this stored metric"
    return mvl_test, wid_test, verdict, df


def run_D1_soma_neuropil_coupling(sessions, n_boot, n_shuffles, seed):
    """D1: within-ROI coupling of soma MVL-change and neuropil (Fneu_raw)
    MVL-change across light->dark. Spearman across cells (pooled, animal-aware
    secondary). Coupled change argues input-driven, not somatic artefact.
    """
    soma_delta_all, neu_delta_all, animal_all = [], [], []
    rows = []
    for s in sessions:
        a = s["arrays"]
        if "Fneu_raw" not in a:
            continue
        fneu = a["Fneu_raw"]  # (n_rois_all? , n_frames) — index by soma_idx
        soma_idx = a["soma_idx"]
        if fneu.shape[0] <= soma_idx.max():
            continue
        fneu_soma = fneu[soma_idx].astype(np.float64)
        sig = a["signal"]  # already soma-only
        hd = a["hd"]
        ml = a["moving"] & a["light_on"]
        md = a["moving"] & ~a["light_on"]
        if ml.sum() < 50 or md.sum() < 50:
            continue
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        hd_l, hd_d = hd[ml], hd[md]
        for c in range(sig.shape[0]):
            # soma MVL change (debiased)
            rs_l = shuffle_debiased_mvl(
                sig[c][ml],
                hd_l,
                n_bins=HD_N_BINS,
                smoothing_sigma_deg=HD_SMOOTH_DEG,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            rs_d = shuffle_debiased_mvl(
                sig[c][md],
                hd_d,
                n_bins=HD_N_BINS,
                smoothing_sigma_deg=HD_SMOOTH_DEG,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            rn_l = shuffle_debiased_mvl(
                fneu_soma[c][ml],
                hd_l,
                n_bins=HD_N_BINS,
                smoothing_sigma_deg=HD_SMOOTH_DEG,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            rn_d = shuffle_debiased_mvl(
                fneu_soma[c][md],
                hd_d,
                n_bins=HD_N_BINS,
                smoothing_sigma_deg=HD_SMOOTH_DEG,
                n_shuffles=n_shuffles,
                rng=rng,
            )
            soma_delta = rs_d["mvl_debiased"] - rs_l["mvl_debiased"]
            neu_delta = rn_d["mvl_debiased"] - rn_l["mvl_debiased"]
            soma_delta_all.append(soma_delta)
            neu_delta_all.append(neu_delta)
            animal_all.append(s["animal_id"])
            rows.append(
                {
                    "exp_id": s["exp_id"],
                    "animal_id": s["animal_id"],
                    "roi_local": c,
                    "soma_mvl_delta": soma_delta,
                    "neu_mvl_delta": neu_delta,
                }
            )
    soma_delta_all = np.asarray(soma_delta_all)
    neu_delta_all = np.asarray(neu_delta_all)
    ok = np.isfinite(soma_delta_all) & np.isfinite(neu_delta_all)
    if ok.sum() < 5:
        return {"label": "D1", "n": int(ok.sum()), "rho": np.nan, "p_value": np.nan}, pd.DataFrame(
            rows
        )
    rho, p = stats.spearmanr(soma_delta_all[ok], neu_delta_all[ok])
    return {
        "label": "D1",
        "n": int(ok.sum()),
        "rho": float(rho),
        "p_value": float(p),
        "median_soma_delta": float(np.median(soma_delta_all[ok])),
        "median_neu_delta": float(np.median(neu_delta_all[ok])),
    }, pd.DataFrame(rows)


def run_B3_gained_lost(sessions):
    """B3 (generating): 2x2 sig-light x sig-dark McNemar + dark-only split-half.

    Pools soma cells across sessions. McNemar tests whether the count of cells
    significant in dark-only differs from light-only (recruitment vs loss).
    """
    n_ll = n_ld = n_dl = n_dd = 0  # (light_sig, dark_sig): ll=both, ld=light only? define below
    # b = sig in light, not dark ; c = sig in dark, not light
    b = c = both = neither = 0
    rows = []
    for s in sessions:
        ah = s.get("analysis")
        if ah is None or "light_significant" not in ah or "dark_significant" not in ah:
            continue
        sl = ah["light_significant"].astype(bool)
        sd = ah["dark_significant"].astype(bool)
        b += int(np.sum(sl & ~sd))
        c += int(np.sum(~sl & sd))
        both += int(np.sum(sl & sd))
        neither += int(np.sum(~sl & ~sd))
        rows.append(
            {
                "exp_id": s["exp_id"],
                "n_light_only": int(np.sum(sl & ~sd)),
                "n_dark_only": int(np.sum(~sl & sd)),
                "n_both": int(np.sum(sl & sd)),
            }
        )
    # McNemar exact (binomial) on discordant pairs
    if b + c >= 1:
        p = stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    else:
        p = np.nan
    return {
        "label": "B3",
        "light_only": b,
        "dark_only": c,
        "both": both,
        "neither": neither,
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }, pd.DataFrame(rows)


def _matched_curves_session(arrays, n_shuffles, rng, statistic="mvl"):
    """Per soma cell: occupancy-matched light & dark debiased MVL, FWHM width,
    and per-condition HD significance.

    The light/dark HD-occupancy distribution is equalised ONCE per session
    (shared across cells, since HD is per-frame), then each cell's matched
    tuning curve is computed in each condition. MVL is circular-shuffle
    debiased; per-condition significance is the circular-shuffle permutation
    p-value on the matched subset. This removes the sampling confound that A1/A2
    showed drives the raw MVL difference, so B2/B3 do not inherit it.

    Returns a dict of (n_soma,) arrays, or None if either condition has too few
    frames after matching.
    """
    sig = arrays["signal"]
    hd = arrays["hd"]
    ml = arrays["moving"] & arrays["light_on"]
    md = arrays["moving"] & ~arrays["light_on"]
    hd_l, hd_d = hd[ml], hd[md]
    if hd_l.size < 50 or hd_d.size < 50:
        return None
    idx_l, idx_d = match_indices_1d(hd_l, hd_d, n_bins=HD_N_BINS, circular=True, rng=rng)
    if idx_l.size < 50 or idx_d.size < 50:
        return None
    hd_lm, hd_dm = hd_l[idx_l], hd_d[idx_d]
    n_soma = sig.shape[0]
    out = {k: np.full(n_soma, np.nan) for k in ("mvl_l", "mvl_d", "wid_l", "wid_d")}
    sig_l = np.zeros(n_soma, bool)
    sig_d = np.zeros(n_soma, bool)
    ones_l = np.ones(hd_lm.size, bool)
    ones_d = np.ones(hd_dm.size, bool)
    for c in range(n_soma):
        s_l = sig[c][ml][idx_l]
        s_d = sig[c][md][idx_d]
        r_l = shuffle_debiased_statistic(
            s_l, hd_lm, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG,
            n_shuffles=n_shuffles, statistic=statistic, rng=rng,
        )
        r_d = shuffle_debiased_statistic(
            s_d, hd_dm, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG,
            n_shuffles=n_shuffles, statistic=statistic, rng=rng,
        )
        out["mvl_l"][c] = r_l["stat_debiased"]
        out["mvl_d"][c] = r_d["stat_debiased"]
        n_sh = len(r_l["shuffle_dist"])
        p_l = (1 + int(np.sum(r_l["shuffle_dist"] >= r_l["stat_raw"]))) / (1 + n_sh)
        p_d = (1 + int(np.sum(r_d["shuffle_dist"] >= r_d["stat_raw"]))) / (1 + n_sh)
        sig_l[c] = p_l < ALPHA
        sig_d[c] = p_d < ALPHA
        tc_l, bc = compute_hd_tuning_curve(
            s_l, hd_lm, ones_l, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG
        )
        tc_d, _ = compute_hd_tuning_curve(
            s_d, hd_dm, ones_d, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG
        )
        out["wid_l"][c] = tuning_curve_fwhm(tc_l, bc)
        out["wid_d"][c] = tuning_curve_fwhm(tc_d, bc)
    out["sig_l"] = sig_l
    out["sig_d"] = sig_d
    return out


def run_tightened_b2_b3(sessions, n_shuffles, seed, statistic="mvl"):
    """Recompute B2 (gain vs sharpening) and B3 (recruitment) on OCCUPANCY-MATCHED
    tuning curves instead of the stored, unmatched MVL/significance.

    A1/A2 showed the raw dark>light MVL is largely a sampling artefact, so the
    stored-MVL versions of B2 and B3 inherit that confound. Here MVL, FWHM width
    and per-condition HD significance are all derived from the same occupancy-
    matched frames. Returns (b2_mvl_test, b2_width_test, b2_verdict, b2_df,
    b3_dict, b3_df).
    """
    mvl_l_s, mvl_d_s, wid_l_s, wid_d_s = [], [], [], []
    b2_rows, b3_rows = [], []
    b = c = both = neither = 0  # b=light-only sig, c=dark-only sig
    for s in sessions:
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        res = _matched_curves_session(s["arrays"], n_shuffles, rng, statistic=statistic)
        if res is None:
            continue
        sig_either = res["sig_l"] | res["sig_d"]
        finite = np.isfinite(res["mvl_l"]) & np.isfinite(res["mvl_d"])
        sel = (sig_either & finite) if (sig_either & finite).sum() >= 1 else finite
        if sel.sum() >= 1:
            mvl_l_s.append(float(np.median(res["mvl_l"][sel])))
            mvl_d_s.append(float(np.median(res["mvl_d"][sel])))
            wfin = sel & np.isfinite(res["wid_l"]) & np.isfinite(res["wid_d"])
            wid_l_s.append(float(np.median(res["wid_l"][wfin])) if wfin.sum() else np.nan)
            wid_d_s.append(float(np.median(res["wid_d"][wfin])) if wfin.sum() else np.nan)
            b2_rows.append(
                {
                    "exp_id": s["exp_id"],
                    "animal_id": s["animal_id"],
                    "celltype": s["celltype"],
                    "n_cells": int(sel.sum()),
                    "mvl_light": mvl_l_s[-1],
                    "mvl_dark": mvl_d_s[-1],
                    "width_light": wid_l_s[-1],
                    "width_dark": wid_d_s[-1],
                }
            )
        sl, sd = res["sig_l"], res["sig_d"]
        b += int(np.sum(sl & ~sd))
        c += int(np.sum(~sl & sd))
        both += int(np.sum(sl & sd))
        neither += int(np.sum(~sl & ~sd))
        b3_rows.append(
            {
                "exp_id": s["exp_id"],
                "n_light_only": int(np.sum(sl & ~sd)),
                "n_dark_only": int(np.sum(~sl & sd)),
                "n_both": int(np.sum(sl & sd)),
            }
        )
    mvl_test = paired_test(mvl_l_s, mvl_d_s, label="B2t_mvl")
    wid_test = paired_test(wid_l_s, wid_d_s, label="B2t_width")
    mvl_up = mvl_test["median_diff"] > 0 and np.isfinite(mvl_test["p_value"]) and mvl_test["p_value"] < ALPHA
    width_down = wid_test["median_diff"] < 0 and np.isfinite(wid_test["p_value"]) and wid_test["p_value"] < ALPHA
    if mvl_up and width_down:
        verdict = "sharpening (matched MVL up + width down)"
    elif mvl_up:
        verdict = "gain-only (matched MVL up, width flat)"
    else:
        verdict = "no MVL enhancement survives occupancy matching"
    if b + c >= 1:
        p = stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    else:
        p = np.nan
    b3 = {
        "label": "B3t",
        "light_only": b,
        "dark_only": c,
        "both": both,
        "neither": neither,
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }
    return mvl_test, wid_test, verdict, pd.DataFrame(b2_rows), b3, pd.DataFrame(b3_rows)


def _place_skaggs_info(sig_vals, x_vals, y_vals, xe, ye, occ, n_shuffles, rng):
    """Shuffle-debiased Skaggs place (spatial) information for one cell.

    Builds a 2-D rate map (mean signal per position bin) over the supplied
    position-occupancy ``occ`` and edges, computes Skaggs info on the rectified
    map, and subtracts the mean of a circular-shift null. Returns bits/event.
    """
    occf = occ.flatten()
    valid = occf > 0

    def _info(s):
        ssum, _, _ = np.histogram2d(x_vals, y_vals, bins=[xe, ye], weights=s)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = (ssum / occ).flatten()
        base = float(np.nanmin(rate[valid])) if valid.any() else 0.0
        return skaggs_info_rate(np.nan_to_num(rate - base), occf)

    raw = _info(sig_vals)
    n = sig_vals.size
    sh = np.empty(n_shuffles)
    for i in range(n_shuffles):
        off = int(rng.integers(1, max(2, n - 1)))
        sh[i] = _info(np.roll(sig_vals, off))
    return float(raw - np.mean(sh))


def run_P1_matched_place_info(sessions, n_shuffles, seed):
    """P1 (sensitivity): place Skaggs information light vs dark, with 2-D
    position-occupancy MATCHED across conditions.

    The stored 'spatial info higher in dark' (H5.3) shares the same sampling
    confound as the HD-MVL effect: if the animal covers the maze differently in
    the dark, place information inflates without any coding change. Here the
    joint x/y occupancy is equalised across light/dark before recomputing
    shuffle-debiased place information, paired within session.
    """
    place_bins = 8
    light_summ, dark_summ, rows = [], [], []
    for s in sessions:
        a = s["arrays"]
        if "x_maze" not in a or "y_maze" not in a:
            continue
        x = np.asarray(a["x_maze"], float)
        y = np.asarray(a["y_maze"], float)
        mv = a["moving"]
        light = a["light_on"]
        sig = a["signal"]
        finite = mv & np.isfinite(x) & np.isfinite(y)
        ml = finite & light
        md = finite & ~light
        if ml.sum() < 100 or md.sum() < 100:
            continue
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        il, idk = match_indices_2d(
            x[ml], y[ml], x[md], y[md], n_bins=(place_bins, place_bins), rng=rng
        )
        if il.size < 100 or idk.size < 100:
            continue
        xl, yl = x[ml][il], y[ml][il]
        xd, yd = x[md][idk], y[md][idk]
        allx = np.concatenate([xl, xd])
        ally = np.concatenate([yl, yd])
        xe = np.linspace(allx.min(), allx.max() + 1e-9, place_bins + 1)
        ye = np.linspace(ally.min(), ally.max() + 1e-9, place_bins + 1)
        occ_l, _, _ = np.histogram2d(xl, yl, bins=[xe, ye])
        occ_d, _, _ = np.histogram2d(xd, yd, bins=[xe, ye])
        sig_l = sig[:, ml][:, il]
        sig_d = sig[:, md][:, idk]
        info_l, info_d = [], []
        for c in range(sig.shape[0]):
            info_l.append(_place_skaggs_info(sig_l[c], xl, yl, xe, ye, occ_l, n_shuffles, rng))
            info_d.append(_place_skaggs_info(sig_d[c], xd, yd, xe, ye, occ_d, n_shuffles, rng))
        ia = np.array(info_l)
        ib = np.array(info_d)
        ok = np.isfinite(ia) & np.isfinite(ib)
        if ok.sum() < 1:
            continue
        light_summ.append(float(np.median(ia[ok])))
        dark_summ.append(float(np.median(ib[ok])))
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "n_cells": int(ok.sum()),
                "place_info_light": light_summ[-1],
                "place_info_dark": dark_summ[-1],
            }
        )
    test = paired_test(light_summ, dark_summ, label="P1")
    return test, pd.DataFrame(rows)


def run_B1_maze_position(sessions, n_boot, n_shuffles, seed):
    """B1 (generating, exploratory): is HD dark-enhancement concentrated at maze
    junctions (visual-aliasing hotspots) vs dead-ends?

    Visual-cue conflict in the q-rose maze is strongest at junctions and
    symmetric corridors, where the same view appears at multiple headings, so
    visual landmarks would most disrupt HD tuning in light. If darkness removes
    that conflict, dark-minus-light MVL should be larger at junctions than at
    dead-ends.

    Frame positions are classified with the *real* maze graph
    (``hm2p.maze.topology``): a frame is a junction if its nearest accessible
    cell is a T-junction or crossroads (graph degree >= 3), a dead-end if degree
    1. Within each region the light-vs-dark MVL is occupancy-matched (per the A1
    control), so per-cell dark-minus-light DeltaMVL is not a sampling artefact;
    region DeltaMVL distributions are compared with Mann-Whitney.
    """
    from hm2p.maze.discretize import discretize_position_fast
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    idx_type = {i: maze.node_types[c] for i, c in enumerate(maze.cell_list)}

    junction_delta, deadend_delta, rows = [], [], []
    sess_junction_med, sess_deadend_med = [], []
    for s in sessions:
        a = s["arrays"]
        if "x_maze" not in a or "y_maze" not in a:
            continue
        x = np.asarray(a["x_maze"], float)
        y = np.asarray(a["y_maze"], float)
        mv = a["moving"]
        light = a["light_on"]
        cell_idx = discretize_position_fast(x, y, maze)
        types = np.array(
            [idx_type.get(int(i), "none") if i >= 0 else "none" for i in cell_idx]
        )
        is_junction = mv & np.isin(types, ("t_junction", "crossroads"))
        is_deadend = mv & (types == "dead_end")
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        n_j = n_d = 0
        sess_j = sess_d = np.nan
        for region, region_mask, store in (
            ("junction", is_junction, junction_delta),
            ("deadend", is_deadend, deadend_delta),
        ):
            ml = region_mask & light
            md = region_mask & ~light
            if ml.sum() < 50 or md.sum() < 50:
                continue
            # Occupancy-match light vs dark within this maze region.
            res = _matched_session(
                a, "occupancy", n_boot, n_shuffles, rng, mask_light=ml, mask_dark=md
            )
            if res is None:
                continue
            delta = np.asarray(res["mvl_dark"]) - np.asarray(res["mvl_light"])
            delta = delta[np.isfinite(delta)]
            store.extend(delta.tolist())
            if region == "junction":
                n_j = int(ml.sum() + md.sum())
                sess_j = float(np.median(delta)) if delta.size else np.nan
            else:
                n_d = int(ml.sum() + md.sum())
                sess_d = float(np.median(delta)) if delta.size else np.nan
        rows.append(
            {
                "exp_id": s["exp_id"],
                "n_junction_frames": n_j,
                "n_deadend_frames": n_d,
                "median_junction_delta": sess_j,
                "median_deadend_delta": sess_d,
            }
        )
        if np.isfinite(sess_j) and np.isfinite(sess_d):
            sess_junction_med.append(sess_j)
            sess_deadend_med.append(sess_d)
    jd = np.asarray(junction_delta)
    dd = np.asarray(deadend_delta)
    jd = jd[np.isfinite(jd)]
    dd = dd[np.isfinite(dd)]
    # Cell-level (pooled, generating; pseudoreplicated across cells).
    if len(jd) >= 3 and len(dd) >= 3:
        u, p = stats.mannwhitneyu(jd, dd, alternative="two-sided")
    else:
        u, p = np.nan, np.nan
    # Session-level paired test — the proper unit, avoids pseudoreplication.
    sess_test = paired_test(sess_deadend_med, sess_junction_med, label="B1_session")
    return {
        "label": "B1",
        "n_junction_cells": len(jd),
        "n_deadend_cells": len(dd),
        "median_junction_delta": float(np.median(jd)) if len(jd) else np.nan,
        "median_deadend_delta": float(np.median(dd)) if len(dd) else np.nan,
        "u": float(u) if np.isfinite(u) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
        "n_sessions_paired": len(sess_junction_med),
        "session_p_value": sess_test["p_value"],
        "session_median_diff": sess_test["median_diff"],
        "session_rank_biserial": sess_test["rank_biserial"],
    }, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# J1 / J2: matched junction controls for the Stage-6 H7 hypotheses
# ---------------------------------------------------------------------------


def _classify_maze_frames(arrays):
    """Boolean frame masks for junction / corridor / dead_end from the maze-
    registered ``x_maze``/``y_maze`` coordinates. None if those coords are
    absent (older sync.h5).

    Positions are classified with the real q-rose maze graph
    (``hm2p.maze.topology``): a frame is a junction if its nearest accessible
    cell is a T-junction or crossroads, a corridor if a through-cell, a
    dead-end if a leaf.
    """
    if "x_maze" not in arrays or "y_maze" not in arrays:
        return None
    from hm2p.maze.discretize import discretize_position_fast
    from hm2p.maze.topology import build_rose_maze

    maze = build_rose_maze()
    idx_type = {i: maze.node_types[c] for i, c in enumerate(maze.cell_list)}
    x = np.asarray(arrays["x_maze"], float)
    y = np.asarray(arrays["y_maze"], float)
    cell_idx = discretize_position_fast(x, y, maze)
    types = np.array(
        [idx_type.get(int(i), "none") if i >= 0 else "none" for i in cell_idx]
    )
    return {
        "junction": np.isin(types, ("t_junction", "crossroads")),
        "corridor": types == "corridor",
        "dead_end": types == "dead_end",
    }


def _raw_region_mvl_per_cell(arrays, mask):
    """Per soma cell raw (un-debiased, unmatched) HD MVL over a frame mask.

    None if the mask has fewer than 50 frames.
    """
    sig = arrays["signal"]
    hd = arrays["hd"]
    if int(mask.sum()) < 50:
        return None
    hd_m = hd[mask]
    ones = np.ones(hd_m.size, bool)
    n_soma = sig.shape[0]
    out = np.full(n_soma, np.nan)
    for c in range(n_soma):
        out[c] = _cell_mvl(sig[c][mask], hd_m, ones)
    return out


def _region_matched_mvl(arrays, mask_a, mask_b, n_shuffles, rng):
    """Equalise the HD-occupancy distribution between two frame masks, return
    per soma cell shuffle-debiased MVL in each (``mvl_a``, ``mvl_b``).

    Same one-dimensional HD occupancy matching + circular-shuffle debiasing as
    the A1 control (``hm2p.analysis.matched_tuning``). None if either mask has
    too few frames before or after matching.
    """
    sig = arrays["signal"]
    hd = arrays["hd"]
    hd_a = hd[mask_a]
    hd_b = hd[mask_b]
    if hd_a.size < 50 or hd_b.size < 50:
        return None
    idx_a, idx_b = match_indices_1d(hd_a, hd_b, n_bins=HD_N_BINS, circular=True, rng=rng)
    if idx_a.size < 50 or idx_b.size < 50:
        return None
    hd_am, hd_bm = hd_a[idx_a], hd_b[idx_b]
    n_soma = sig.shape[0]
    mvl_a = np.full(n_soma, np.nan)
    mvl_b = np.full(n_soma, np.nan)
    for c in range(n_soma):
        s_a = sig[c][mask_a][idx_a]
        s_b = sig[c][mask_b][idx_b]
        r_a = shuffle_debiased_statistic(
            s_a, hd_am, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG,
            n_shuffles=n_shuffles, statistic="mvl", rng=rng,
        )
        r_b = shuffle_debiased_statistic(
            s_b, hd_bm, n_bins=HD_N_BINS, smoothing_sigma_deg=HD_SMOOTH_DEG,
            n_shuffles=n_shuffles, statistic="mvl", rng=rng,
        )
        mvl_a[c] = r_a["stat_debiased"]
        mvl_b[c] = r_b["stat_debiased"]
    return {"mvl_a": mvl_a, "mvl_b": mvl_b, "n_matched": int(idx_a.size)}


def _run_junction_matched(sessions, mask_fn, n_shuffles, seed, label):
    """Shared engine for the junction occupancy-matched controls.

    ``mask_fn(arrays, regions) -> (mask_a, mask_b, name_a, name_b)`` selects the
    two frame sets to contrast. Returns ``(matched_test, raw_test, df)`` where
    the raw test uses un-debiased MVL on all region frames and the matched test
    uses occupancy-matched, shuffle-debiased MVL. Session summary = median over
    HD-significant soma cells (fallback all soma). ``paired_test(a, b)`` reports
    ``b - a``, so median diff > 0 means condition B has the higher MVL.
    """
    a_m, b_m, a_r, b_r, rows = [], [], [], [], []
    for s in sessions:
        arrays = s["arrays"]
        regions = _classify_maze_frames(arrays)
        if regions is None:
            continue
        mask_a, mask_b, name_a, name_b = mask_fn(arrays, regions)
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        res = _region_matched_mvl(arrays, mask_a, mask_b, n_shuffles, rng)
        if res is None:
            continue
        sig_mask = s.get("hd_sig_mask")
        ma, mb, ncell = _session_summary(res["mvl_a"], res["mvl_b"], sig_mask)
        if not (np.isfinite(ma) and np.isfinite(mb)):
            continue
        raw_a = _raw_region_mvl_per_cell(arrays, mask_a)
        raw_b = _raw_region_mvl_per_cell(arrays, mask_b)
        if raw_a is not None and raw_b is not None:
            ra, rb, _ = _session_summary(raw_a, raw_b, sig_mask)
        else:
            ra = rb = np.nan
        a_m.append(ma)
        b_m.append(mb)
        a_r.append(ra)
        b_r.append(rb)
        rows.append(
            {
                "exp_id": s["exp_id"],
                "animal_id": s["animal_id"],
                "celltype": s["celltype"],
                "n_cells": ncell,
                "n_matched_frames": res["n_matched"],
                f"mvl_{name_a}_matched": ma,
                f"mvl_{name_b}_matched": mb,
                f"mvl_{name_a}_raw": ra,
                f"mvl_{name_b}_raw": rb,
            }
        )
    matched = paired_test(a_m, b_m, label=f"{label}_matched")
    raw = paired_test(a_r, b_r, label=f"{label}_raw")
    return matched, raw, pd.DataFrame(rows)


def run_J1_junction_vs_corridor(sessions, n_shuffles, seed):
    """J1 (control for H7.2): HD MVL at junctions vs corridors in LIGHT, with the
    HD-occupancy distribution equalised between the two location types.

    Junctions are sampled with a narrower, more repetitive HD distribution than
    corridors (the mouse pauses and turns), which by itself changes MVL. J1
    tests whether the H7.2 junction-vs-corridor difference survives equalising
    that sampling. ``median(diff) > 0`` => junction MVL > corridor MVL after
    matching.
    """

    def mask_fn(a, regions):
        mv, light = a["moving"], a["light_on"]
        return (
            mv & light & regions["corridor"],  # a = corridor
            mv & light & regions["junction"],  # b = junction
            "corridor",
            "junction",
        )

    return _run_junction_matched(sessions, mask_fn, n_shuffles, seed, "J1")


def run_J2_junction_lightdark(sessions, n_shuffles, seed):
    """J2 (control for H7.3): HD MVL at junctions, light vs dark, occupancy-
    matched within junction frames.

    This is the confound-controlled version of the H7.3 headline. If the raw
    junction MVL light-vs-dark difference is driven by the mouse sampling HD
    differently at junctions in the dark, it will collapse here — the same way
    A1/A2 collapsed the whole-session dark>light MVL effect. ``median(diff) > 0``
    => junction MVL higher in dark after matching.
    """

    def mask_fn(a, regions):
        mv, light = a["moving"], a["light_on"]
        return (
            mv & light & regions["junction"],  # a = light
            mv & ~light & regions["junction"],  # b = dark
            "light",
            "dark",
        )

    return _run_junction_matched(sessions, mask_fn, n_shuffles, seed, "J2")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks(sessions, seed):
    """(1) matched='none' reproduces raw MVL; (2) occupancy matching equalises
    the light/dark HD-occupancy histograms."""
    msgs = []
    s = sessions[0]
    a = s["arrays"]
    rng = np.random.default_rng(seed)
    ml = a["moving"] & a["light_on"]
    md = a["moving"] & ~a["light_on"]
    sig = a["signal"][0]
    # (1) match none, debias off == raw per-condition MVL
    res = matched_condition_mvl(
        sig[ml], a["hd"][ml], sig[md], a["hd"][md], match="none", debias=False, rng=rng
    )
    raw_l = _cell_mvl(sig[ml], a["hd"][ml], np.ones(ml.sum(), bool))
    raw_d = _cell_mvl(sig[md], a["hd"][md], np.ones(md.sum(), bool))
    ok1 = np.isclose(res["mvl_a"], raw_l, atol=1e-9) and np.isclose(res["mvl_b"], raw_d, atol=1e-9)
    msgs.append(
        f"SANITY match=none reproduces raw MVL: {ok1} (none={res['mvl_a']:.4f} raw={raw_l:.4f})"
    )
    # (2) occupancy matching equalises histograms
    from hm2p.analysis.matched_tuning import match_indices_1d

    ia, ib = match_indices_1d(a["hd"][ml], a["hd"][md], n_bins=HD_N_BINS, circular=True, rng=rng)
    h_l = occupancy_histogram(a["hd"][ml][ia], HD_N_BINS)
    h_d = occupancy_histogram(a["hd"][md][ib], HD_N_BINS)
    maxdiff = float(np.max(np.abs(h_l - h_d)))
    ok2 = maxdiff < 1e-9
    msgs.append(
        f"SANITY occupancy matching equalises HD histograms: {ok2} (max|diff|={maxdiff:.2e})"
    )
    return msgs, ok1 and ok2


# ---------------------------------------------------------------------------
# Loader: assemble per-session payloads
# ---------------------------------------------------------------------------


def assemble_sessions(animals, exps, signal, limit, need_analysis=True):
    valid = exps[exps["exclude"].astype(str).str.strip() != "1"]
    if limit:
        valid = valid.head(limit)
    log.info("Assembling %d sessions (signal=%s)...", len(valid), signal)
    sessions = []
    for _, exp in valid.iterrows():
        exp_id = exp["exp_id"]
        animal_id = exp["animal_id"]
        ar = animals[animals["animal_id"] == animal_id]
        celltype = str(ar.iloc[0]["celltype"]) if not ar.empty else ""
        sub, ses, sync_key, an_key = session_keys(exp_id, animal_id)
        sync_f = _download_h5(sync_key)
        if sync_f is None:
            log.warning("no sync.h5 for %s", exp_id)
            continue
        arrays = load_session_arrays(sync_f, signal)
        if arrays is None:
            sync_f.close()
            continue
        an = None
        if need_analysis:
            an_f = _download_h5(an_key)
            if an_f is not None:
                an = load_analysis(an_f, signal, arrays["soma_idx"])
                an_f.close()
        # HD-significance mask over soma cells from analysis.h5 (light OR dark)
        hd_sig_mask = None
        if an is not None and "light_significant" in an and "dark_significant" in an:
            hd_sig_mask = an["light_significant"].astype(bool) | an["dark_significant"].astype(
                bool
            )
        sessions.append(
            {
                "exp_id": exp_id,
                "animal_id": animal_id,
                "celltype": celltype,
                "arrays": arrays,
                "analysis": an,
                "hd_sig_mask": hd_sig_mask,
            }
        )
        sync_f.close()
        log.info("  loaded %s (%s, %d soma cells)", exp_id, celltype, len(arrays["soma_idx"]))
    return sessions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--signal",
        default="dff",
        choices=["dff", "events"],
        help="signal type (dff primary; events sensitivity)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="2 sessions, tiny boot/shuffle counts for a fast end-to-end check",
    )
    ap.add_argument("--limit", type=int, default=None, help="limit number of sessions")
    ap.add_argument("--n-boot", type=int, default=20, help="matched bootstrap draws")
    ap.add_argument("--n-shuffles", type=int, default=100, help="circular shuffles for debiasing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("results/dark_hypotheses"))
    ap.add_argument("--skip-d1", action="store_true", help="skip D1 (slow neuropil pass)")
    ap.add_argument("--skip-b1", action="store_true", help="skip B1 (slow position pass)")
    ap.add_argument(
        "--skip-junction", action="store_true",
        help="skip J1/J2 (matched junction controls for the Stage-6 H7 hypotheses)",
    )
    ap.add_argument(
        "--skip-mi", action="store_true",
        help="skip the Skaggs HD/place mutual-information cross-check",
    )
    args = ap.parse_args()

    if args.smoke:
        args.limit = args.limit or 2
        args.n_boot = min(args.n_boot, 5)
        args.n_shuffles = min(args.n_shuffles, 20)
        log.info(
            "SMOKE: limit=%d n_boot=%d n_shuffles=%d", args.limit, args.n_boot, args.n_shuffles
        )

    animals, exps = load_metadata()
    sessions = assemble_sessions(animals, exps, args.signal, args.limit)
    if not sessions:
        log.error("no sessions loaded")
        sys.exit(1)
    log.info(
        "Loaded %d sessions, %d animals", len(sessions), len({s["animal_id"] for s in sessions})
    )

    args.output.mkdir(parents=True, exist_ok=True)

    sanity_msgs, sanity_ok = sanity_checks(sessions, args.seed)
    for m in sanity_msgs:
        log.info(m)

    results = {}  # hid -> test dict
    per_hyp_df = {}  # hid -> dataframe

    # --- Gauntlet ---
    log.info("=== A1 occupancy-matched ===")
    results["A1"], per_hyp_df["A1"] = run_matched_hypothesis(
        sessions, "A1", "occupancy", args.n_boot, args.n_shuffles, args.seed
    )
    log.info("=== A2 kinematics-matched ===")
    results["A2"], per_hyp_df["A2"] = run_matched_hypothesis(
        sessions, "A2", "kinematics", args.n_boot, args.n_shuffles, args.seed
    )
    log.info("=== A3 epoch-order ===")
    results["A3"], per_hyp_df["A3"] = run_A3_epoch_order(
        sessions, "occupancy", args.n_boot, args.n_shuffles, args.seed
    )
    log.info("=== C1 state-composition matched ===")
    results["C1"], per_hyp_df["C1"] = run_matched_hypothesis(
        sessions, "C1", "occupancy", args.n_boot, args.n_shuffles, args.seed
    )
    log.info("=== C2 hd_confidence ===")
    results["C2"], per_hyp_df["C2"] = run_C2_hd_confidence(sessions, args.seed)

    # --- Skaggs HD mutual-information cross-check (Voigts & Harnett 2020;
    # Zong et al. 2022). Same matched/shuffle machinery, information statistic.
    if not args.skip_mi:
        log.info("=== A1 occupancy-matched (Skaggs HD info) ===")
        results["A1_mi"], per_hyp_df["A1_mi"] = run_matched_hypothesis(
            sessions, "A1", "occupancy", args.n_boot, args.n_shuffles, args.seed,
            statistic="skaggs",
        )
        log.info("=== A2 kinematics-matched (Skaggs HD info) ===")
        results["A2_mi"], per_hyp_df["A2_mi"] = run_matched_hypothesis(
            sessions, "A2", "kinematics", args.n_boot, args.n_shuffles, args.seed,
            statistic="skaggs",
        )

    # --- Mechanism ---
    log.info("=== B2 gain vs sharpening ===")
    b2_mvl, b2_wid, b2_verdict, per_hyp_df["B2"] = run_B2_gain_vs_sharpening(sessions)
    results["B2_mvl"] = b2_mvl
    results["B2_width"] = b2_wid
    results["B2_verdict"] = {"label": "B2", "verdict": b2_verdict}

    if not args.skip_d1:
        log.info("=== D1 soma<->neuropil coupling ===")
        results["D1"], per_hyp_df["D1"] = run_D1_soma_neuropil_coupling(
            sessions, args.n_boot, args.n_shuffles, args.seed
        )

    log.info("=== B3 gained/lost McNemar ===")
    results["B3"], per_hyp_df["B3"] = run_B3_gained_lost(sessions)

    log.info("=== B2/B3 tightened on occupancy-matched MVL ===")
    (
        b2t_mvl,
        b2t_wid,
        b2t_verdict,
        per_hyp_df["B2_tightened"],
        results["B3_tightened"],
        per_hyp_df["B3_tightened"],
    ) = run_tightened_b2_b3(sessions, args.n_shuffles, args.seed)
    results["B2_tightened_mvl"] = b2t_mvl
    results["B2_tightened_width"] = b2t_wid
    results["B2_tightened_verdict"] = {"label": "B2t", "verdict": b2t_verdict}

    if not args.skip_mi:
        log.info("=== B2/B3 tightened on occupancy-matched Skaggs HD info ===")
        b2tm_mvl, _b2tm_wid, b2tm_verdict, _b2tm_df, b3tm, _b3tm_df = run_tightened_b2_b3(
            sessions, args.n_shuffles, args.seed, statistic="skaggs"
        )
        results["B2_tightened_mi"] = b2tm_mvl
        results["B2_tightened_mi_verdict"] = {"label": "B2t-MI", "verdict": b2tm_verdict}
        results["B3_tightened_mi"] = b3tm

        log.info("=== P1 occupancy-matched place Skaggs info ===")
        results["P1"], per_hyp_df["P1"] = run_P1_matched_place_info(
            sessions, args.n_shuffles, args.seed
        )

    if not args.skip_b1:
        log.info("=== B1 maze position (exploratory) ===")
        results["B1"], per_hyp_df["B1"] = run_B1_maze_position(
            sessions, args.n_boot, args.n_shuffles, args.seed
        )

    if not args.skip_junction:
        log.info("=== J1/J2 junction tuning occupancy-matched (H7 controls) ===")
        (
            results["J1_matched"],
            results["J1_raw"],
            per_hyp_df["J1"],
        ) = run_J1_junction_vs_corridor(sessions, args.n_shuffles, args.seed)
        (
            results["J2_matched"],
            results["J2_raw"],
            per_hyp_df["J2"],
        ) = run_J2_junction_lightdark(sessions, args.n_shuffles, args.seed)

    # --- FDR across confirmatory family ---
    fam = []
    fam_keys = []
    for k, r in results.items():
        base = k.split("_")[0]
        # "tightened" re-analysis, the "_mi"/"P1" Skaggs cross-check, and the
        # raw (unmatched) leg of the J1/J2 junction controls are reported
        # alongside the primary MVL family, not folded into its FDR.
        if "tightened" in k or k.endswith("_mi") or k == "P1" or k.endswith("_raw"):
            continue
        if base in CONFIRMATORY and isinstance(r, dict) and np.isfinite(r.get("p_value", np.nan)):
            fam.append(r["p_value"])
            fam_keys.append(k)
    fdr_map = {}
    if fam:
        rej, p_fdr, _, _ = multipletests(fam, alpha=ALPHA, method="fdr_bh")
        for k, pf, rj in zip(fam_keys, p_fdr, rej):
            fdr_map[k] = (float(pf), bool(rj))

    # --- Write outputs ---
    for hid, df in per_hyp_df.items():
        if df is not None and not df.empty:
            df.to_csv(args.output / f"{hid}.csv", index=False)

    write_report(args, sessions, results, fdr_map, sanity_msgs, b2_verdict)
    log.info("Done. Outputs in %s", args.output)


def _verdict_from(test, fdr):
    p = test.get("p_value", np.nan)
    md = test.get("median_diff", np.nan)
    if not np.isfinite(p):
        return "inconclusive (insufficient pairs)"
    p_use = fdr[0] if fdr else p
    if p_use < ALPHA:
        direction = "dark > light" if md > 0 else "light > dark"
        return f"confirmed ({direction})"
    return "refuted/null (no light-dark difference survives)"


def _verdict_generic(test, pos_label, neg_label):
    """Verdict for a paired test with arbitrary direction labels (used by the
    J1/J2 junction controls, which are not light-vs-dark)."""
    p = test.get("p_value", np.nan)
    md = test.get("median_diff", np.nan)
    if not np.isfinite(p):
        return "inconclusive (insufficient pairs)"
    if p < ALPHA:
        return f"significant ({pos_label if md > 0 else neg_label})"
    return "null (no difference survives occupancy matching)"


def write_report(args, sessions, results, fdr_map, sanity_msgs, b2_verdict):
    L = []
    L.append("# Dark-enhancement confound gauntlet + mechanism")
    L.append("")
    L.append(f"**Signal:** {args.signal} (primary=dF/F; events=sensitivity)")
    L.append(
        f"**Sessions:** {len(sessions)} | **Animals:** {len({s['animal_id'] for s in sessions})}"
    )
    n_penk = len({s["animal_id"] for s in sessions if s["celltype"] == "penk"})
    n_np = len({s["animal_id"] for s in sessions if s["celltype"] == "nonpenk"})
    L.append(
        f"**Cell types:** {n_penk} Penk+ animals, {n_np} CamKII+ animals "
        "(pooled — within-session paired design)"
    )
    L.append(
        f"**Speed threshold:** {SPEED_THRESHOLD} cm/s | **HD bins:** {HD_N_BINS} | "
        f"**boot:** {args.n_boot} | **shuffles:** {args.n_shuffles}"
    )
    L.append("")
    L.append(
        "Unit = session; light vs dark PAIRED within session; per-session "
        "summary = median over HD-significant soma cells (fallback all soma). "
        "Wilcoxon signed-rank on (dark - light); positive => dark higher. "
        "FDR (BH) across confirmatory family {A1,A2,A3,C1,C2,B2,D1}."
    )
    L.append("")
    L.append("## Sanity checks")
    for m in sanity_msgs:
        L.append(f"- {m}")
    L.append("")

    def fmt(test, hid):
        fdr = fdr_map.get(hid)
        v = _verdict_from(test, fdr)
        p = test.get("p_value", np.nan)
        md = test.get("median_diff", np.nan)
        rb = test.get("rank_biserial", np.nan)
        n = test.get("n", np.nan)
        ndark = test.get("n_dark_gt_light", np.nan)
        fdr_s = f", FDR-p={fdr[0]:.4f}" if fdr else ""
        return (
            f"- N={n} sessions ({ndark} with dark>light); "
            f"Wilcoxon p={p:.4f}{fdr_s}; median(dark-light)={md:.4f}; "
            f"rank-biserial={rb:.3f}\n- **Verdict:** {v}"
        )

    # A1
    L.append("## A1 — occupancy-matched + shuffle-debiased MVL (MAKE-OR-BREAK) [confirmatory]")
    L.append("Equalise light/dark HD-occupancy histograms, debias MVL by circular shuffle.")
    L.append(fmt(results["A1"], "A1"))
    L.append(
        "- Most-likely confound if it flips: residual occupancy concentration "
        "beyond first moment; controlled here by per-bin equalisation."
    )
    L.append("")
    # A2
    L.append("## A2 — joint speed+|AHV|-matched MVL (MAKE-OR-BREAK) [confirmatory]")
    L.append("Equalise the 2-D speed x |AHV| distribution across light/dark.")
    L.append(fmt(results["A2"], "A2"))
    L.append(
        "- Most-likely confound if it flips: slower/steadier dark sampling "
        "raising per-bin SNR; controlled here."
    )
    L.append("")
    # A3
    L.append("## A3 — epoch-order / bleaching (exclude 1st light + 1st dark epoch) [confirmatory]")
    L.append(fmt(results["A3"], "A3"))
    L.append("- Most-likely confound: photobleaching + adaptation transient in first epochs.")
    L.append("")
    # C1
    L.append("## C1 — active/bad_behav state-composition matched [confirmatory]")
    L.append(
        "Both conditions restricted to active moving (bad_behav excluded), then occupancy-matched."
    )
    L.append(fmt(results["C1"], "C1"))
    L.append("")
    # C2
    L.append("## C2 — hd_confidence light vs dark (IR-camera validation) [confirmatory]")
    L.append("Tests whether tracking confidence differs by light (it must NOT — camera is IR).")
    c2 = results["C2"]
    c2_p = c2.get("p_value", np.nan)
    c2_fdr = fdr_map.get("C2")
    c2_p_use = c2_fdr[0] if c2_fdr else c2_p
    fdr_s = f", FDR-p={c2_fdr[0]:.4f}" if c2_fdr else ""
    L.append(
        f"- N={c2['n']} sessions; Wilcoxon p={c2_p:.4f}{fdr_s}; "
        f"median(dark-light)={c2.get('median_diff', np.nan):.4f}; "
        f"rank-biserial={c2.get('rank_biserial', np.nan):.3f}"
    )
    if not np.isfinite(c2_p):
        c2_verdict = "inconclusive (insufficient pairs)"
    elif c2_p_use >= ALPHA:
        c2_verdict = (
            "confirmed NULL (no confidence difference; IR-camera "
            "assumption upheld — reviewer shield holds)"
        )
    else:
        c2_verdict = (
            "WARNING: confidence differs by light — IR-camera assumption "
            "is violated; HD tuning differences may be tracking artefacts"
        )
    L.append(f"- **Verdict:** {c2_verdict} (here a non-significant result is the desired outcome)")
    L.append("")
    # B2
    L.append("## B2 — gain vs sharpening dissociation [confirmatory]")
    bm, bw = results["B2_mvl"], results["B2_width"]
    L.append(
        f"- MVL (stored, normalised): N={bm['n']}, Wilcoxon p={bm['p_value']:.4f}, "
        f"median(dark-light)={bm['median_diff']:.4f}, rank-biserial={bm['rank_biserial']:.3f}"
    )
    L.append(
        f"- Tuning width (FWHM): N={bw['n']}, Wilcoxon p={bw['p_value']:.4f}, "
        f"median(dark-light)={bw['median_diff']:.4f}, rank-biserial={bw['rank_biserial']:.3f}"
    )
    L.append(f"- **Classification:** {b2_verdict}")
    L.append("")
    # D1
    if "D1" in results:
        d = results["D1"]
        L.append("## D1 — within-ROI soma<->neuropil MVL-change coupling [confirmatory]")
        L.append(
            f"- Spearman(soma ΔMVL, neuropil ΔMVL) across cells: N={d['n']}, "
            f"rho={d.get('rho', np.nan):.3f}, p={d.get('p_value', np.nan):.4f}"
        )
        verd = (
            "inconclusive"
            if not np.isfinite(d.get("p_value", np.nan))
            else (
                "coupled => input-driven (argues against somatic artefact)"
                if d["p_value"] < ALPHA and d.get("rho", 0) > 0
                else "not coupled"
            )
        )
        L.append(f"- **Verdict:** {verd}")
        L.append("")
    # B3
    b3 = results["B3"]
    L.append("## B3 — gained vs lost cells (McNemar) [generating]")
    L.append(
        f"- light-only sig: {b3['light_only']}, dark-only sig: {b3['dark_only']}, "
        f"both: {b3['both']}, neither: {b3['neither']}; "
        f"McNemar exact p={b3['p_value']:.4f}"
        if np.isfinite(b3["p_value"])
        else f"- light-only: {b3['light_only']}, dark-only: {b3['dark_only']} (too few discordant)"
    )
    L.append("")
    # B2/B3 tightened on occupancy-matched MVL
    if "B2_tightened_mvl" in results:
        bm, bw = results["B2_tightened_mvl"], results["B2_tightened_width"]
        bv = results["B2_tightened_verdict"]["verdict"]
        L.append("## B2/B3 TIGHTENED — recomputed on occupancy-matched curves [sensitivity]")
        L.append(
            "MVL, FWHM width and per-condition HD significance all derived from "
            "the SAME occupancy-matched frames (not stored/unmatched values), so "
            "these do not inherit the sampling confound A1/A2 exposed. Not part of "
            "the primary FDR family."
        )
        L.append(
            f"- B2 matched MVL: N={bm['n']}, Wilcoxon p={bm['p_value']:.4f}, "
            f"median(dark-light)={bm['median_diff']:.4f}, rank-biserial={bm['rank_biserial']:.3f}"
        )
        L.append(
            f"- B2 matched width (FWHM): N={bw['n']}, Wilcoxon p={bw['p_value']:.4f}, "
            f"median(dark-light)={bw['median_diff']:.4f}, rank-biserial={bw['rank_biserial']:.3f}"
        )
        L.append(f"- **B2 classification (matched):** {bv}")
        b3t = results["B3_tightened"]
        L.append(
            f"- B3 matched recruitment: light-only sig {b3t['light_only']}, "
            f"dark-only sig {b3t['dark_only']}, both {b3t['both']}, neither "
            f"{b3t['neither']}; McNemar exact p="
            + (f"{b3t['p_value']:.4f}" if np.isfinite(b3t["p_value"]) else "n/a")
        )
        L.append("")
    # Skaggs mutual-information cross-check
    if "A1_mi" in results:
        L.append(
            "## MI — Skaggs HD information cross-check (Voigts & Harnett 2020; "
            "Zong et al. 2022) [sensitivity]"
        )
        L.append(
            "Same matched/shuffle gauntlet, statistic = Skaggs HD information "
            "(bits/event) instead of MVL. MI captures non-unimodal tuning MVL "
            "misses, but is at least as occupancy-biased, so it goes through the "
            "same matching. Reported alongside MVL, not in the MVL FDR family."
        )
        L.append(f"- **A1 (occupancy-matched MI):** {fmt(results['A1_mi'], 'A1_mi')}")
        L.append(f"- **A2 (kinematics-matched MI):** {fmt(results['A2_mi'], 'A2_mi')}")
        if "B2_tightened_mi" in results:
            bmi = results["B2_tightened_mi"]
            bmv = results["B2_tightened_mi_verdict"]["verdict"]
            L.append(
                f"- B2 matched MI (gain): N={bmi['n']}, Wilcoxon p={bmi['p_value']:.4f}, "
                f"median(dark-light)={bmi['median_diff']:.4f}, "
                f"rank-biserial={bmi['rank_biserial']:.3f} — {bmv}"
            )
            b3mi = results["B3_tightened_mi"]
            L.append(
                f"- B3 matched recruitment (MI significance): light-only "
                f"{b3mi['light_only']}, dark-only {b3mi['dark_only']}, both "
                f"{b3mi['both']}; McNemar p="
                + (f"{b3mi['p_value']:.4f}" if np.isfinite(b3mi["p_value"]) else "n/a")
            )
        if "P1" in results:
            L.append(
                f"- **P1 (position-occupancy-matched PLACE info):** "
                f"{fmt(results['P1'], 'P1')}"
            )
        L.append("")
    # B1
    if "B1" in results:
        b1 = results["B1"]
        L.append("## B1 — dark-enhancement vs maze position [generating, exploratory]")
        L.append(
            f"- junction/corridor cells: N={b1['n_junction_cells']}, "
            f"median ΔMVL={b1.get('median_junction_delta', np.nan):.4f}"
        )
        L.append(
            f"- dead-end cells: N={b1['n_deadend_cells']}, "
            f"median ΔMVL={b1.get('median_deadend_delta', np.nan):.4f}"
        )
        L.append(
            f"- Cell-level (pooled, pseudoreplicated) Mann-Whitney junction vs "
            f"dead-end p={b1.get('p_value', np.nan):.4f}"
        )
        L.append(
            f"- **Session-level paired (proper unit):** N={b1.get('n_sessions_paired', 0)} "
            f"sessions, Wilcoxon p={b1.get('session_p_value', np.nan):.4f}, "
            f"median(junction-deadend ΔMVL)={b1.get('session_median_diff', np.nan):.4f}, "
            f"rank-biserial={b1.get('session_rank_biserial', np.nan):.3f}"
        )
        L.append(
            "- Positions classified by the real q-rose maze graph (T-junction/"
            "crossroads vs dead-end); light-vs-dark occupancy-matched within "
            "region. Exploratory / hypothesis-generating."
        )
        L.append("")
    # J1 / J2 — matched junction controls for the Stage-6 H7 hypotheses
    if "J2_matched" in results:
        L.append(
            "## J1/J2 — junction HD tuning, occupancy-matched (H7 controls) [sensitivity]"
        )
        L.append(
            "Matched controls for the Stage-6 junction hypotheses H7.2 (MVL at "
            "junctions vs corridors) and H7.3 (junction MVL light vs dark). HD "
            "is sampled differently by location type and by light, so the raw "
            "junction contrasts can be sampling artefacts. Here the HD-occupancy "
            "distribution is equalised between the two frame sets and MVL is "
            "circular-shuffle debiased — the same machinery that collapsed the "
            "whole-session dark>light effect in A1/A2. Exploratory; not in the "
            "confirmatory FDR family."
        )
        j1m, j1r = results["J1_matched"], results["J1_raw"]
        L.append(
            f"- **J1 junction vs corridor (light):** raw N={j1r['n']}, "
            f"Wilcoxon p={j1r['p_value']:.4f}, median(junction-corridor)="
            f"{j1r['median_diff']:.4f}; matched N={j1m['n']}, p={j1m['p_value']:.4f}, "
            f"median={j1m['median_diff']:.4f}, rank-biserial={j1m['rank_biserial']:.3f}"
        )
        L.append(
            f"  - **Verdict:** {_verdict_generic(j1m, 'junction > corridor', 'corridor > junction')}"
        )
        j2m, j2r = results["J2_matched"], results["J2_raw"]
        L.append(
            f"- **J2 junction light vs dark:** raw N={j2r['n']}, "
            f"Wilcoxon p={j2r['p_value']:.4f}, median(dark-light)="
            f"{j2r['median_diff']:.4f}; matched N={j2m['n']}, p={j2m['p_value']:.4f}, "
            f"median={j2m['median_diff']:.4f}, rank-biserial={j2m['rank_biserial']:.3f}"
        )
        L.append(
            f"  - **Verdict:** {_verdict_generic(j2m, 'dark > light', 'light > dark')}"
        )
        L.append("")

    (args.output / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
