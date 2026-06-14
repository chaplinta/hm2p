"""H-N10: Neuropil HD tuning control + independent dark>light test.

Reads sync.h5 (which already contains Fneu_raw copied from ca.h5, plus the
imaging-rate-aligned hd_deg / speed_cm_s / light_on / bad_behav / roi_types),
so soma and neuropil signals share the same frame axis and HD alignment as the
somatic dff used in the main analyses.

Two neuropil dF/F definitions are computed, both using the project rolling
min-max baseline (Pachitariu et al. 2017) followed by compute_dff, identical to
the somatic pipeline:

  - session-level neuropil trace:  mean(Fneu_raw across soma ROIs) -> dF/F.
    Used for the session light-vs-dark neuropil MVL test (less noisy field-level
    afferent readout).
  - per-ROI local neuropil trace:  each soma ROI's own Fneu_raw -> dF/F.
    Used for the per-ROI soma-vs-neuropil MVL contamination control, because the
    relevant comparison is each soma against its own Suite2p annular surround.

Soma dF/F is read directly from sync.h5 ("dff", the FISSA-corrected somatic
signal) for the per-ROI comparison so soma and neuropil are processed on the
exact same frames/mask.

References
----------
Kerr et al. 2005. "Imaging input and output of neocortical networks in vivo."
    PNAS 102(39):14063-14068. doi:10.1073/pnas.0506029102
Margetts-Smith et al. 2025. bioRxiv 2025.02.06.636939v1 (ubiquitous ATN->RSP).
Pachitariu et al. 2017. "Suite2p." doi:10.1101/061507 (F0 estimation).
Muller et al. 1987. J Neurosci 7(7):1951-1968 (circular-shift null).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import boto3
import h5py
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from hm2p.analysis.significance import hd_tuning_significance
from hm2p.analysis.tuning import compute_hd_tuning_curve, mean_vector_length
from hm2p.calcium.dff import compute_baseline, compute_dff

BUCKET = "hm2p-derivatives"
REGION = "ap-southeast-2"
SPEED_THRESH = 2.5  # cm/s
N_BINS = 36
SMOOTH_DEG = 6.0
N_SHUFFLES = 1000
SEED = 12345
OUTDIR = Path("/workspace/results/neuropil_control")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def expid_to_subses(eid: str) -> tuple[str, str]:
    m = re.match(r"(\d{8})_(\d{2})_(\d{2})_(\d{2})_(\d+)", eid)
    d, H, M, S, a = m.groups()
    return f"sub-{a}", f"ses-{d}T{H}{M}{S}"


def load_metadata() -> pd.DataFrame:
    exp = pd.read_csv("/workspace/metadata/experiments.csv")
    ani = pd.read_csv("/workspace/metadata/animals.csv")
    ani["animal_id"] = ani["animal_id"].astype(str)
    exp["sub"], exp["ses"] = zip(*exp["exp_id"].map(expid_to_subses))
    exp["animal_id"] = exp["exp_id"].str.extract(r"_(\d+)$")[0]
    m = exp.merge(
        ani[["animal_id", "celltype", "virus_id"]], on="animal_id", how="left"
    )
    m["equip"] = m["fibre"].astype(str) + "/" + m["lens"].astype(str)
    return m


# ---------------------------------------------------------------------------
# Per-session computation
# ---------------------------------------------------------------------------
def fneu_dff(fneu_trace_2d: np.ndarray, fps: float) -> np.ndarray:
    """dF/F of a (n, T) fluorescence array via rolling baseline (project method)."""
    f0 = compute_baseline(fneu_trace_2d.astype(np.float32), fps=fps)
    return compute_dff(fneu_trace_2d.astype(np.float32), f0)


def base_mask(speed: np.ndarray, hd: np.ndarray, bad: np.ndarray) -> np.ndarray:
    return (speed > SPEED_THRESH) & np.isfinite(hd) & np.isfinite(speed) & (~bad)


def mvl_for_mask(signal: np.ndarray, hd: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 50:  # too few frames for a stable tuning curve
        return np.nan
    tc, bc = compute_hd_tuning_curve(
        signal, hd, mask, n_bins=N_BINS, smoothing_sigma_deg=SMOOTH_DEG
    )
    return mean_vector_length(tc, bc)


def process_session(s3, sub: str, ses: str, rng) -> dict:
    o = s3.get_object(Bucket=BUCKET, Key=f"sync/{sub}/{ses}/sync.h5")
    with h5py.File(io.BytesIO(o["Body"].read()), "r") as f:
        fps = float(f.attrs.get("fps_imaging", 9.643))
        hd = f["hd_deg"][:].astype(np.float64)
        speed = f["speed_cm_s"][:].astype(np.float64)
        light = f["light_on"][:].astype(bool)
        bad = f["bad_behav"][:].astype(bool) if "bad_behav" in f else np.zeros_like(light)
        roi_types = f["roi_types"][:]
        fneu = f["Fneu_raw"][:].astype(np.float32)
        soma_dff = f["dff"][:].astype(np.float32)  # FISSA-corrected somatic
        snr = f["roi_qc/snr_event"][:] if "roi_qc/snr_event" in f else None

    soma_idx = np.where(roi_types == 0)[0]
    n_soma = len(soma_idx)
    if n_soma == 0:
        return {"sub": sub, "ses": ses, "n_soma": 0}

    m_all = base_mask(speed, hd, bad)
    m_light = m_all & light
    m_dark = m_all & (~light)

    # ----- session-level neuropil trace: mean Fneu across soma ROIs -----
    fneu_mean = fneu[soma_idx].mean(axis=0, keepdims=True)  # (1, T)
    np_dff = fneu_dff(fneu_mean, fps)[0]

    np_mvl_all = mvl_for_mask(np_dff, hd, m_all)
    np_mvl_light = mvl_for_mask(np_dff, hd, m_light)
    np_mvl_dark = mvl_for_mask(np_dff, hd, m_dark)

    # significance of session neuropil HD tuning (all-frames)
    np_sig = hd_tuning_significance(
        np_dff, hd, m_all, n_shuffles=N_SHUFFLES, metric="mvl",
        n_bins=N_BINS, smoothing_sigma_deg=SMOOTH_DEG, rng=rng,
    )

    # ----- per-ROI: soma vs its own local neuropil -----
    local_np_dff = fneu_dff(fneu[soma_idx], fps)  # (n_soma, T)
    roi_rows = []
    for k, ridx in enumerate(soma_idx):
        soma_mvl = mvl_for_mask(soma_dff[ridx], hd, m_all)
        np_loc_mvl = mvl_for_mask(local_np_dff[k], hd, m_all)
        soma_mvl_l = mvl_for_mask(soma_dff[ridx], hd, m_light)
        soma_mvl_d = mvl_for_mask(soma_dff[ridx], hd, m_dark)
        # soma HD significance (so we can report soma>neuropil among HD cells)
        soma_sig = hd_tuning_significance(
            soma_dff[ridx], hd, m_all, n_shuffles=N_SHUFFLES, metric="mvl",
            n_bins=N_BINS, smoothing_sigma_deg=SMOOTH_DEG, rng=rng,
        )
        roi_rows.append(dict(
            sub=sub, ses=ses, roi=int(ridx),
            soma_mvl=soma_mvl, local_np_mvl=np_loc_mvl,
            soma_gt_np=(np.isfinite(soma_mvl) and np.isfinite(np_loc_mvl)
                        and soma_mvl > np_loc_mvl),
            soma_hd_sig=soma_sig["p_value"] < 0.05,
            soma_mvl_light=soma_mvl_l, soma_mvl_dark=soma_mvl_d,
            snr=float(snr[ridx]) if snr is not None else np.nan,
        ))

    sess = dict(
        sub=sub, ses=ses, fps=fps, n_soma=n_soma,
        n_light_frames=int(m_light.sum()), n_dark_frames=int(m_dark.sum()),
        np_mvl_all=np_mvl_all, np_mvl_light=np_mvl_light, np_mvl_dark=np_mvl_dark,
        np_mvl_dark_minus_light=(np_mvl_dark - np_mvl_light),
        np_dark_light_ratio=(np_mvl_dark / np_mvl_light
                             if np_mvl_light and np_mvl_light > 0 else np.nan),
        np_hd_p=np_sig["p_value"], np_hd_sig=np_sig["p_value"] < 0.05,
    )
    return sess, roi_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    s3 = boto3.client("s3", region_name=REGION)
    md = load_metadata()

    sess_rows, roi_rows = [], []
    for _, r in md.iterrows():
        sub, ses = r["sub"], r["ses"]  # bracket access: 'sub' shadows a Series method
        print(f"  {sub}/{ses} ({r['celltype']}, {r['equip']})", flush=True)
        out = process_session(s3, sub, ses, rng)
        if isinstance(out, dict):  # no soma
            continue
        sess, rois = out
        meta = dict(animal_id=r["animal_id"], celltype=r["celltype"], lens=r["lens"],
                    fibre=r["fibre"], equip=r["equip"],
                    primary=int(r["primary_exp"] == 1 and r["exclude"] == 0))
        sess.update(meta)
        sess_rows.append(sess)
        for rr in rois:
            rr.update(meta)
            roi_rows.append(rr)

    sess_df = pd.DataFrame(sess_rows)
    roi_df = pd.DataFrame(roi_rows)
    sess_df.to_csv(OUTDIR / "session_neuropil_mvl.csv", index=False)
    roi_df.to_csv(OUTDIR / "roi_soma_vs_neuropil.csv", index=False)
    print("\nSaved session + ROI tables.")
    analyse(sess_df, roi_df)


def _mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 2 * U / (len(a) * len(b)) - 1  # rank-biserial
    return U, p, rbc


def analyse(sess_df: pd.DataFrame, roi_df: pd.DataFrame) -> None:
    lines = []
    def L(s=""):
        print(s); lines.append(s)

    for cohort_name, sdf, rdf in [
        ("PRIMARY (primary_exp=1 & exclude=0)",
         sess_df[sess_df.primary == 1], roi_df[roi_df.primary == 1]),
        ("ALL non-excluded sessions", sess_df, roi_df),
    ]:
        L(f"\n{'='*70}\nCOHORT: {cohort_name}\n{'='*70}")
        L(f"sessions: {len(sdf)}  (penk={sum(sdf.celltype=='penk')}, "
          f"nonpenk={sum(sdf.celltype=='nonpenk')})  "
          f"animals: {sdf.animal_id.nunique()}")

        # (1) neuropil carries HD tuning?
        sig = sdf.np_hd_sig.sum()
        L(f"\n(1) Neuropil HD significance (circular shuffle, p<0.05): "
          f"{sig}/{len(sdf)} sessions significant "
          f"({100*sig/len(sdf):.0f}%)")
        L(f"    neuropil MVL (all): median={sdf.np_mvl_all.median():.4f} "
          f"[IQR {sdf.np_mvl_all.quantile(.25):.4f}-{sdf.np_mvl_all.quantile(.75):.4f}]")

        # (2) neuropil dark vs light (Wilcoxon paired across sessions)
        pair = sdf.dropna(subset=["np_mvl_light", "np_mvl_dark"])
        if len(pair) >= 6:
            w, p = wilcoxon(pair.np_mvl_dark, pair.np_mvl_light)
            n_dark_gt = int((pair.np_mvl_dark > pair.np_mvl_light).sum())
            L(f"\n(2) Neuropil MVL dark vs light (Wilcoxon, N={len(pair)} sessions): "
              f"W={w:.1f}, p={p:.4f}")
            L(f"    median light={pair.np_mvl_light.median():.4f}, "
              f"dark={pair.np_mvl_dark.median():.4f}, "
              f"median(dark-light)={ (pair.np_mvl_dark-pair.np_mvl_light).median():.4f}")
            L(f"    dark>light in {n_dark_gt}/{len(pair)} sessions")
            direction = ("DARK>LIGHT (matches somatic headline)"
                         if pair.np_mvl_dark.median() > pair.np_mvl_light.median()
                         else "LIGHT>=DARK (does NOT match somatic headline)")
            L(f"    --> Independent headline test direction: {direction}")
            L(f"    --> If p<0.05 AND dark>light: input-driven (argues AGAINST "
              f"per-cell somatic sampling artifact).")
            L(f"    --> If NOT dark>light: consistent with a somatic sampling artifact.")

        # (3) per-ROI soma vs local neuropil
        rr = rdf.dropna(subset=["soma_mvl", "local_np_mvl"])
        frac_all = rr.soma_gt_np.mean()
        rr_hd = rr[rr.soma_hd_sig]
        frac_hd = rr_hd.soma_gt_np.mean() if len(rr_hd) else np.nan
        L(f"\n(3) Per-ROI soma>neuropil MVL (contamination control), "
          f"N={len(rr)} soma ROIs:")
        L(f"    soma>neuropil overall: {rr.soma_gt_np.sum()}/{len(rr)} "
          f"({100*frac_all:.1f}%)")
        L(f"    soma>neuropil among HD-significant soma ({len(rr_hd)} ROIs): "
          f"{100*frac_hd:.1f}%")
        L(f"    median soma MVL={rr.soma_mvl.median():.4f}, "
          f"median local-neuropil MVL={rr.local_np_mvl.median():.4f}")
        # paired Wilcoxon soma vs neuropil per ROI (animal-level safe? report cell-level + animal-level)
        if len(rr) >= 6:
            w, p = wilcoxon(rr.soma_mvl, rr.local_np_mvl)
            L(f"    paired Wilcoxon (cell-level) soma vs neuropil MVL: "
              f"W={w:.0f}, p={p:.2e}")
        # animal-level (avoid pseudoreplication)
        am = rr.groupby("animal_id").agg(
            soma=("soma_mvl", "median"), npl=("local_np_mvl", "median")).dropna()
        if len(am) >= 6:
            w, p = wilcoxon(am.soma, am.npl)
            L(f"    paired Wilcoxon (animal-level medians, N={len(am)}): "
              f"W={w:.0f}, p={p:.4f}; "
              f"soma>neuropil in {int((am.soma>am.npl).sum())}/{len(am)} animals")

        # (4) between cell-type neuropil comparison (animal-level medians)
        L(f"\n(4) Penk+ vs Penk-CamKII+ neuropil tuning (animal-level medians, "
          f"hypothesis-generating; 4 nonpenk animals):")
        for metric, col in [("neuropil MVL (all)", "np_mvl_all"),
                            ("neuropil dark/light ratio", "np_dark_light_ratio"),
                            ("neuropil dark-light", "np_mvl_dark_minus_light")]:
            am = sdf.groupby(["animal_id", "celltype"])[col].median().reset_index()
            penk = am[am.celltype == "penk"][col]
            non = am[am.celltype == "nonpenk"][col]
            U, p, rbc = _mwu(penk, non)
            L(f"    {metric}: penk median={np.nanmedian(penk):.4f} (n={penk.notna().sum()}), "
              f"nonpenk median={np.nanmedian(non):.4f} (n={non.notna().sum()}); "
              f"MWU p={p:.3f}, rank-biserial r={rbc:.3f}")

        # (5) equipment split (f4mm vs f6mm)
        L(f"\n(5) Equipment split (lens — different PSF/neuropil contamination):")
        for lens, g in sdf.groupby("lens"):
            sig = g.np_hd_sig.sum()
            pair = g.dropna(subset=["np_mvl_light", "np_mvl_dark"])
            dl = (pair.np_mvl_dark - pair.np_mvl_light).median() if len(pair) else np.nan
            L(f"    {lens}: {len(g)} sessions, neuropil-HD-sig {sig}/{len(g)}, "
              f"median np MVL={g.np_mvl_all.median():.4f}, "
              f"median(dark-light)={dl:.4f}")

    (OUTDIR / "neuropil_control_summary.txt").write_text("\n".join(lines))
    print(f"\nWrote summary to {OUTDIR/'neuropil_control_summary.txt'}")


if __name__ == "__main__":
    main()
