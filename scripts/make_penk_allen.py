#!/usr/bin/env python3
"""Allen reference data on Penk in RSP layer 2/3 for the Penk results page.

Reads the outputs of ``scripts/allen_penk_query.py`` (EC2 job, downloaded to
``results/allen_penk/``) and writes a compact ``docs/figures/penk/data/allen.json``:

- Allen Brain Cell Atlas 10x (Yao et al. 2023): fraction of RSP L2/3 IT
  neurons expressing Penk, by transcriptomic supertype and by isocortical
  area; Kruskal-Wallis epsilon-squared of Penk by supertype and cluster;
  Silverman mode test on expressing cells; genes whose expression co-varies
  with Penk (Spearman, partial on genes detected per cell), with layer 2 vs
  layer 3 marker genes.
- MERFISH (Yao et al. 2023; Zhang et al. 2023): fraction expressing by RSP
  subdivision and Spearman correlation with CCF coordinates.
- Patch-seq (Scala et al. 2021, mouse motor cortex): Spearman of Penk with
  intrinsic properties in L2/3 IT cells; Allen Cell Types Database Penk-Cre
  comparison when present.

References
----------
Yao Z et al. 2023. "A high-resolution transcriptomic and spatial atlas of cell
types in the whole mouse brain." Nature 624:317-332. doi:10.1038/s41586-023-06812-z

Zhang M et al. 2023. "Molecularly defined and spatially resolved cell atlas of
the whole mouse brain." Nature 624:343-354. doi:10.1038/s41586-023-06808-9

Scala F et al. 2021. "Phenotypic variation of transcriptomic cell types in
mouse motor cortex." Nature 598:144-150. doi:10.1038/s41586-020-2907-3

Gouwens NW et al. 2019. "Classification of electrophysiological and
morphological neuron types in the mouse visual cortex." Nat Neurosci
22:1182-1195. doi:10.1038/s41593-019-0417-0
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "allen_penk"
OUT = REPO / "docs" / "figures" / "penk" / "data" / "allen.json"
DEPTH_MARKERS = ("Lamp5", "Calb1", "Stard8", "Ccbe1", "Cux2", "Otof", "Cdh13", "Rorb", "Cxcl14")


def _read(src: Path, name: str) -> pd.DataFrame | None:
    p = src / name
    return pd.read_csv(p) if p.exists() else None


def tenx_summary(src: Path) -> dict[str, Any]:
    """Penk in RSP L2/3 IT from the 10x atlas."""
    out: dict[str, Any] = {}
    dist = _read(src, "l23it_10x_penk_distribution.csv")
    if dist is not None:
        pooled = dist[(dist.scope == "RSP L2/3 IT") & (dist.grouping == "pooled")]
        if len(pooled):
            r = pooled.iloc[0]
            out["n_cells"] = int(r.n)
            out["frac_expressing"] = float(r.frac_expressing)
            out["silverman_p"] = float(r.silverman_p_unimodal)
        areas = dist[(dist.scope == "isocortex L2/3 IT") & (dist.grouping == "area")]
        out["areas"] = (
            areas[["group", "n", "frac_expressing"]]
            .sort_values("frac_expressing", ascending=False)
            .rename(columns={"group": "area"})
            .to_dict(orient="records")
        )
    tax = _read(src, "l23it_10x_penk_by_taxon.csv")
    if tax is not None:
        st = tax[(tax.scope == "RSP L2/3 IT") & (tax.level == "supertype") & (tax.n >= 100)]
        out["supertypes"] = (
            st[["taxon", "n", "frac_expressing", "share_of_expressing", "median_expressing"]]
            .sort_values("frac_expressing", ascending=False)
            .to_dict(orient="records")
        )
    kw = _read(src, "l23it_10x_penk_kruskal.csv")
    if kw is not None:
        rsp = kw[(kw.scope == "RSP L2/3 IT") & (kw.stratum == "ALL")]
        out["epsilon_sq"] = {
            str(r.factor): float(r.epsilon_sq) for r in rsp.itertuples() if pd.notna(r.epsilon_sq)
        }
    hist = _read(src, "l23it_10x_penk_histogram.csv")
    if hist is not None:
        h = hist[
            (hist.scope == "RSP L2/3 IT")
            & (hist.grouping == "pooled")
            & (hist.population == "expressing")
        ]
        out["hist_expressing"] = {
            "bin_lo": h.bin_lo.tolist(),
            "count": h["count"].astype(int).tolist(),
        }
    return out


def gene_summary(src: Path, n_top: int = 12) -> dict[str, Any]:
    """Genes co-varying with Penk in RSP L2/3 IT (pooled chemistries, partial rho)."""
    out: dict[str, Any] = {}
    top = _read(src, "rsp_l23it_10x_penk_gene_corr_top.csv")
    if top is not None:
        t = top[top.library_method == "ALL"]
        for d in ("positive", "negative"):
            out[d] = (
                t[t.direction == d]
                .head(n_top)[["gene", "rho", "rho_partial"]]
                .to_dict(orient="records")
            )
    mk = _read(src, "rsp_l23it_10x_penk_depth_markers.csv")
    if mk is not None:
        m = mk[(mk.library_method == "ALL") & mk.gene.isin(DEPTH_MARKERS)]
        out["markers"] = m[
            ["gene", "frac_expressing", "rho", "rho_partial", "q_bh_partial"]
        ].to_dict(orient="records")
    return out


def merfish_summary(src: Path) -> dict[str, Any]:
    """Penk by RSP subdivision and spatial correlations in the MERFISH datasets."""
    out: dict[str, Any] = {}
    sub = _read(src, "rsp_merfish_l23it_penk_by_subdivision.csv")
    if sub is not None:
        s = sub[sub.grouping == "datasetxrsp_subdivision"]
        out["subdivisions"] = s[["dataset", "rsp_subdivision", "n", "frac_expressing"]].to_dict(
            orient="records"
        )
    sp = _read(src, "rsp_merfish_l23it_penk_spatial_corr.csv")
    if sp is not None:
        keep = sp[
            sp.variable.isin(["x_ccf", "y_ccf", "ccf_ml_from_midline", "layer_ordinal"])
            & sp.rsp_subdivision.isna()
        ]
        out["spatial"] = keep[["dataset", "variable", "axis_assumed", "n", "rho", "q_bh"]].to_dict(
            orient="records"
        )
    dist = _read(src, "rsp_merfish_l23it_penk_distribution.csv")
    if dist is not None:
        d = dist[dist.grouping == "dataset"]
        out["datasets"] = (
            d[["group", "n", "frac_expressing", "silverman_p_unimodal"]]
            .rename(columns={"group": "dataset"})
            .to_dict(orient="records")
        )
    return out


def patchseq_summary(src: Path) -> dict[str, Any]:
    """Patch-seq Penk-ephys correlations (L2/3 IT) and the Cell Types DB Penk-Cre comparison."""
    out: dict[str, Any] = {}
    sp = _read(src, "patchseq_penk_ephys_spearman.csv")
    if sp is not None:
        s = sp[sp.ttype_group == "L2/3 IT"]
        out["l23it"] = s[["dataset", "variable", "n", "rho", "p", "q_bh"]].to_dict(
            orient="records"
        )
    dist = _read(src, "patchseq_penk_distribution.csv")
    if dist is not None:
        out["frac_expressing"] = {
            str(r.group): float(r.frac_expressing) for r in dist.itertuples()
        }
    for name in (
        "ctdb_penk_cre_comparison.csv",
        "ctdb_penk_summary.csv",
        "celltypes_penk_cre_summary.csv",
    ):
        c = _read(src, name)
        if c is not None:
            out["ctdb_file"] = name
            out["ctdb"] = c.head(60).to_dict(orient="records")
            break
    return out


def build(src: Path = SRC) -> dict[str, Any]:
    """All Allen summaries for the page."""
    return {
        "tenx": tenx_summary(src),
        "genes": gene_summary(src),
        "merfish": merfish_summary(src),
        "patchseq": patchseq_summary(src),
    }


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def main() -> None:  # pragma: no cover - file I/O entry point
    data = _clean(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":  # pragma: no cover
    main()
