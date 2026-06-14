#!/usr/bin/env python3
"""Bayes factors (BF10) for the cell-type between-group nulls.

Implements the Bayesian supplement required by docs/neural-hypotheses.md
Section 0.1: for non-significant between-group (Penk+ vs Penk-CamKII+)
comparisons, report BF10 to distinguish "evidence for the null" from
"inconclusive".

Method (stated honestly)
------------------------
The exact non-parametric Bayesian Mann-Whitney of van Doorn et al. 2020
(rank-likelihood with data-augmented Gibbs sampling) is implemented in JASP /
the R `BayesFactor`+`rankBF` stack, NOT in pingouin. pingouin 0.6.1 provides
only the parametric JZS Bayesian t-test (`bayesfactor_ttest`, Rouder et al.
2009, Cauchy prior r=0.707).

We therefore use the **rank-based normal approximation** to the Bayesian MWU:
rank-transform the pooled animal-level summaries across both groups, then apply
the JZS Bayesian t-test to the ranks. This is the same approximation strategy
used for non-parametric Bayesian inference when a full Gibbs sampler is
unavailable; it inherits the robustness of rank statistics while using the
analytic JZS BF. It is an approximation to van Doorn et al. 2020, not that exact
method, and is labelled as such in the output.

Unit of analysis: animal-level **median** of each cell metric (Section 0.3:
between-group tests use animal-level summaries, medians not means, to avoid
pseudoreplication). Primary cohort = non-excluded sessions, soma ROIs only.
With 11 Penk+ and 4 Penk-CamKII+ animals, most BFs are expected to be
inconclusive (1/3 < BF10 < 3); this is reported honestly.

Interpretation thresholds (van Doorn et al. 2020; Jeffreys 1961):
    BF10 < 1/3   -> evidence for the null (equivalence)
    1/3 <= BF10 <= 3 -> inconclusive
    BF10 > 3     -> evidence for an effect

References
----------
van Doorn, J., Ly, A., Marsman, M., Wagenmakers, E.-J. 2020. "Bayesian rank-based
    hypothesis testing for the rank sum test, the signed rank test, and Spearman's
    rho." Journal of Applied Statistics 47(16), 2984-3006.
    doi:10.1080/02664763.2019.1709053
Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G. 2009. "Bayesian
    t tests for accepting and rejecting the null hypothesis." Psychonomic Bulletin
    & Review 16(2), 225-237. doi:10.3758/PBR.16.2.225
pingouin: https://pingouin-stats.org
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

BUCKET = "hm2p-derivatives"
SIGNAL = "dff"  # the primary signal the nulls were originally reported on

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3
        _S3 = boto3.Session(profile_name="hm2p-agent").client("s3")
    return _S3


def _download_h5(key: str):
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=key)
        return h5py.File(io.BytesIO(obj["Body"].read()), "r")
    except Exception:
        return None


def load_cells(signal: str = SIGNAL) -> pd.DataFrame:
    """Build per-soma-cell DataFrame with HD metrics + animal/celltype labels."""
    base = Path(__file__).resolve().parent.parent / "metadata"
    animals = pd.read_csv(base / "animals.csv")
    animals["animal_id"] = animals["animal_id"].astype(str)
    exps = pd.read_csv(base / "experiments.csv")
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    valid = exps[exps["exclude"].astype(str).str.strip() != "1"]

    rows = []
    for _, exp in valid.iterrows():
        animal_id = exp["animal_id"]
        parts = exp["exp_id"].split("_")
        sub = f"sub-{animal_id}"
        ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
        arow = animals[animals["animal_id"] == animal_id]
        if arow.empty:
            continue
        celltype = str(arow.iloc[0].get("celltype", ""))

        sync = _download_h5(f"sync/{sub}/{ses}/sync.h5")
        f = _download_h5(f"analysis/{sub}/{ses}/analysis.h5")
        if f is None or signal not in f:
            if sync is not None:
                sync.close()
            continue
        grp = f[signal]
        n = grp["hd/all/mvl"].shape[0] if "hd/all/mvl" in grp else 0
        roi_types = sync["roi_types"][:] if (sync is not None and "roi_types" in sync) else np.zeros(n)
        for roi in range(n):
            if int(roi_types[roi]) != 0:  # soma only
                continue
            rec = {"animal_id": animal_id, "celltype": celltype}
            rec["hd_all_mvl"] = float(grp["hd/all/mvl"][roi])
            rec["hd_all_tuning_width"] = float(grp["hd/all/tuning_width"][roi])
            rec["hd_all_significant"] = bool(grp["hd/all/significant"][roi])
            rec["hd_comp_mvl_ratio"] = float(grp["hd/comparison/mvl_ratio"][roi])
            rows.append(rec)
        f.close()
        if sync is not None:
            sync.close()
    return pd.DataFrame(rows)


def animal_level(df: pd.DataFrame, metric: str, agg: str = "median") -> pd.DataFrame:
    """Collapse cells to one value per animal (median), keep celltype."""
    sub = df.dropna(subset=[metric])
    if agg == "median":
        g = sub.groupby(["animal_id", "celltype"])[metric].median()
    elif agg == "mean":
        g = sub.groupby(["animal_id", "celltype"])[metric].mean()
    else:
        raise ValueError(agg)
    return g.reset_index()


def hd_fraction_per_animal(df: pd.DataFrame) -> pd.DataFrame:
    """Per-animal fraction of soma cells with significant HD tuning."""
    g = df.groupby(["animal_id", "celltype"])["hd_all_significant"].mean()
    return g.reset_index().rename(columns={"hd_all_significant": "hd_fraction"})


def rank_bayes_mwu(x: np.ndarray, y: np.ndarray, r: float = 0.707) -> dict:
    """Rank-based normal-approximation Bayesian MWU.

    Rank-transform the pooled (x, y), run a JZS Bayesian independent t-test on
    the ranks. Approximation to van Doorn et al. 2020.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    pooled = np.concatenate([x, y])
    ranks = stats.rankdata(pooled)
    rx, ry = ranks[:nx], ranks[nx:]
    # Welch t on ranks -> t statistic for the BF
    t, p = stats.ttest_ind(rx, ry, equal_var=False)
    bf10 = float(pg.bayesfactor_ttest(t, nx, ny, paired=False, r=r))
    # also the frequentist MWU for reference
    u, p_mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "n_penk": nx, "n_nonpenk": ny,
        "penk_median_of_animal_summaries": float(np.median(x)),
        "nonpenk_median_of_animal_summaries": float(np.median(y)),
        "t_on_ranks": float(t),
        "BF10": bf10,
        "BF01": float(1.0 / bf10) if bf10 > 0 else np.inf,
        "mwu_U": float(u),
        "mwu_p": float(p_mwu),
    }


def interpret(bf10: float) -> str:
    if bf10 < 1 / 3:
        return "evidence for null (equivalence)"
    if bf10 > 3:
        return "evidence for an effect"
    return "inconclusive"


def main() -> None:
    df = load_cells(SIGNAL)
    print(f"Loaded {len(df)} soma cells, "
          f"{df['animal_id'].nunique()} animals "
          f"({df[df.celltype=='penk'].animal_id.nunique()} penk, "
          f"{df[df.celltype=='nonpenk'].animal_id.nunique()} nonpenk)")

    comparisons = []

    # H-N2 MVL
    al = animal_level(df, "hd_all_mvl")
    comparisons.append(("H-N2", "HD MVL (animal-median)", al, "hd_all_mvl"))

    # H-N2 tuning width
    al = animal_level(df, "hd_all_tuning_width")
    comparisons.append(("H-N2", "HD tuning width (animal-median, deg)", al, "hd_all_tuning_width"))

    # H-N4 MVL-ratio / VDI
    al = animal_level(df, "hd_comp_mvl_ratio")
    comparisons.append(("H-N4", "Visual dependence (MVL light/dark ratio, animal-median)",
                        al, "hd_comp_mvl_ratio"))

    # H-N2 HD fraction
    al = hd_fraction_per_animal(df)
    comparisons.append(("H-N2", "HD-cell fraction per animal", al, "hd_fraction"))

    out_rows = []
    for hid, label, al, col in comparisons:
        penk = al.loc[al.celltype == "penk", col].values
        nonpenk = al.loc[al.celltype == "nonpenk", col].values
        res = rank_bayes_mwu(penk, nonpenk)
        res["hypothesis"] = hid
        res["comparison"] = label
        res["metric"] = col
        res["interpretation"] = interpret(res["BF10"])
        out_rows.append(res)
        print(f"\n{hid}  {label}")
        print(f"  Penk+ median={res['penk_median_of_animal_summaries']:.4f} "
              f"(n={res['n_penk']}), CamKII+ median={res['nonpenk_median_of_animal_summaries']:.4f} "
              f"(n={res['n_nonpenk']})")
        print(f"  MWU U={res['mwu_U']:.1f}, p={res['mwu_p']:.4f}")
        print(f"  BF10={res['BF10']:.3f}  (BF01={res['BF01']:.3f})  -> {res['interpretation']}")

    out = pd.DataFrame(out_rows)[[
        "hypothesis", "comparison", "metric", "n_penk", "n_nonpenk",
        "penk_median_of_animal_summaries", "nonpenk_median_of_animal_summaries",
        "mwu_U", "mwu_p", "t_on_ranks", "BF10", "BF01", "interpretation",
    ]]
    outpath = Path(__file__).resolve().parent.parent / "results" / "bayes" / "bayes_nulls.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outpath, index=False)
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    sys.exit(main())
