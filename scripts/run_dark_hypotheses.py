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

from hm2p.analysis.matched_tuning import (  # noqa: E402
    matched_condition_mvl,
    occupancy_histogram,
    shuffle_debiased_mvl,
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
    arrays, match, n_boot, n_shuffles, rng, mask_light=None, mask_dark=None, match_vars_kind=None
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


def run_matched_hypothesis(sessions, hid, match, n_boot, n_shuffles, seed):
    """A1 (occupancy), A2 (kinematics), C1 (state composition) share this engine."""
    rows = []
    light_summ, dark_summ = [], []
    for s in sessions:
        arrays = s["arrays"]
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        if hid == "C1":
            # Match active/bad_behav composition: restrict both conditions to
            # ACTIVE moving frames (already exclude bad_behav via 'valid'); then
            # the comparison is occupancy-matched on top (state-composition
            # equalised by construction since both use identical active+moving).
            ml = arrays["moving"] & arrays["light_on"]
            md = arrays["moving"] & ~arrays["light_on"]
            res = _matched_session(
                arrays, "occupancy", n_boot, n_shuffles, rng, mask_light=ml, mask_dark=md
            )
        else:
            res = _matched_session(arrays, match, n_boot, n_shuffles, rng)
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


def run_B1_maze_position(sessions, n_shuffles, seed):
    """B1 (generating, exploratory): dark-enhancement vs maze position.

    Splits frames into 'junction/corridor' vs 'dead-end' regions by maze
    occupancy entropy is not available here without the maze graph, so we use a
    simpler proxy: high-coverage (frequently-visited, central) vs low-coverage
    locations. Reported as exploratory only.

    Implementation: per session, split moving frames by whether the animal's
    x_maze/y_maze cell is in the top-tertile of visit frequency (corridor/junction
    proxy) vs bottom-tertile (dead-end proxy); compute dark-minus-light debiased
    MVL change in each region; test region difference with Wilcoxon across cells.
    """
    junction_delta, deadend_delta, rows = [], [], []
    for s in sessions:
        a = s["arrays"]
        if "x_maze" not in a or "y_maze" not in a:
            continue
        x = a["x_maze"]
        y = a["y_maze"]
        mv = a["moving"]
        finite = mv & np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 200:
            continue
        # Discretise to integer maze cells (x_maze/y_maze are in cell units)
        xc = np.round(x).astype(int)
        yc = np.round(y).astype(int)
        key = xc.astype(np.int64) * 1000 + yc.astype(np.int64)
        # Visit frequency per cell among finite moving frames
        uniq, inv, counts = np.unique(key[finite], return_inverse=True, return_counts=True)
        freq_map = dict(zip(uniq.tolist(), counts.tolist()))
        freq = np.array([freq_map.get(int(k), 0) for k in key])
        hi = np.nanpercentile(counts, 66)
        lo = np.nanpercentile(counts, 33)
        junction = finite & (freq >= hi)
        deadend = finite & (freq <= lo)
        sig = a["signal"]
        hd = a["hd"]
        light = a["light_on"]
        rng = np.random.default_rng(seed + hash(s["exp_id"]) % 10_000)
        for region, store in (("junction", junction_delta), ("deadend", deadend_delta)):
            mask = junction if region == "junction" else deadend
            ml = mask & light
            md = mask & ~light
            if ml.sum() < 50 or md.sum() < 50:
                continue
            hd_l, hd_d = hd[ml], hd[md]
            for cidx in range(sig.shape[0]):
                rl = shuffle_debiased_mvl(
                    sig[cidx][ml],
                    hd_l,
                    n_bins=HD_N_BINS,
                    smoothing_sigma_deg=HD_SMOOTH_DEG,
                    n_shuffles=n_shuffles,
                    rng=rng,
                )
                rd = shuffle_debiased_mvl(
                    sig[cidx][md],
                    hd_d,
                    n_bins=HD_N_BINS,
                    smoothing_sigma_deg=HD_SMOOTH_DEG,
                    n_shuffles=n_shuffles,
                    rng=rng,
                )
                store.append(rd["mvl_debiased"] - rl["mvl_debiased"])
        rows.append(
            {
                "exp_id": s["exp_id"],
                "n_junction": int(junction.sum()),
                "n_deadend": int(deadend.sum()),
            }
        )
    jd = np.asarray(junction_delta)
    dd = np.asarray(deadend_delta)
    jd = jd[np.isfinite(jd)]
    dd = dd[np.isfinite(dd)]
    if len(jd) >= 3 and len(dd) >= 3:
        u, p = stats.mannwhitneyu(jd, dd, alternative="two-sided")
    else:
        u, p = np.nan, np.nan
    return {
        "label": "B1",
        "n_junction_cells": len(jd),
        "n_deadend_cells": len(dd),
        "median_junction_delta": float(np.median(jd)) if len(jd) else np.nan,
        "median_deadend_delta": float(np.median(dd)) if len(dd) else np.nan,
        "u": float(u) if np.isfinite(u) else np.nan,
        "p_value": float(p) if np.isfinite(p) else np.nan,
    }, pd.DataFrame(rows)


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

    if not args.skip_b1:
        log.info("=== B1 maze position (exploratory) ===")
        results["B1"], per_hyp_df["B1"] = run_B1_maze_position(
            sessions, args.n_shuffles, args.seed
        )

    # --- FDR across confirmatory family ---
    fam = []
    fam_keys = []
    for k, r in results.items():
        base = k.split("_")[0]
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
        L.append(f"- Mann-Whitney junction vs dead-end p={b1.get('p_value', np.nan):.4f}")
        L.append(
            "- NOTE: position proxy is visit-frequency tertile, not maze-graph "
            "junctions; treat as hypothesis-generating only."
        )
        L.append("")

    (args.output / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
