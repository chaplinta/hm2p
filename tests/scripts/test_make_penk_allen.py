"""Tests for ``scripts/make_penk_allen.py`` (synthetic tables only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import make_penk_allen as ma  # noqa: E402


def _write(src: Path) -> None:
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "scope": ["RSP L2/3 IT", "isocortex L2/3 IT", "isocortex L2/3 IT"],
            "grouping": ["pooled", "area", "area"],
            "group": ["ALL", "RSP", "VIS"],
            "n": [100, 100, 50],
            "frac_expressing": [0.2, 0.2, 0.1],
            "silverman_p_unimodal": [0.6, 0.6, 0.9],
        }
    ).to_csv(src / "l23it_10x_penk_distribution.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["RSP L2/3 IT"] * 3,
            "level": ["supertype", "supertype", "cluster"],
            "taxon": ["A", "B", "c1"],
            "n": [500, 50, 300],
            "frac_expressing": [0.5, 0.9, 0.4],
            "share_of_expressing": [0.8, 0.1, 0.5],
            "median_expressing": [7.0, 6.0, 6.5],
        }
    ).to_csv(src / "l23it_10x_penk_by_taxon.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["RSP L2/3 IT", "RSP L2/3 IT"],
            "factor": ["supertype", "cluster"],
            "stratum": ["ALL", "ALL"],
            "epsilon_sq": [0.26, 0.29],
        }
    ).to_csv(src / "l23it_10x_penk_kruskal.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["RSP L2/3 IT"] * 2,
            "grouping": ["pooled"] * 2,
            "population": ["expressing", "all"],
            "bin_lo": [0.0, 0.0],
            "count": [5, 9],
        }
    ).to_csv(src / "l23it_10x_penk_histogram.csv", index=False)
    pd.DataFrame(
        {
            "library_method": ["ALL"] * 3 + ["10Xv2"],
            "direction": ["positive", "negative", "positive", "positive"],
            "gene": ["Lamp5", "Cxcl14", "Ntm", "X"],
            "rho": [0.4, -0.37, 0.43, 0.1],
            "rho_partial": [0.37, -0.32, 0.39, 0.1],
        }
    ).to_csv(src / "rsp_l23it_10x_penk_gene_corr_top.csv", index=False)
    pd.DataFrame(
        {
            "library_method": ["ALL", "ALL"],
            "gene": ["Lamp5", "Foo"],
            "frac_expressing": [0.6, 0.1],
            "rho": [0.4, 0.0],
            "rho_partial": [0.37, 0.0],
            "q_bh_partial": [0.0, 1.0],
        }
    ).to_csv(src / "rsp_l23it_10x_penk_depth_markers.csv", index=False)
    pd.DataFrame(
        {
            "grouping": ["datasetxrsp_subdivision", "dataset"],
            "dataset": ["M", "M"],
            "rsp_subdivision": ["RSPd", np.nan],
            "n": [10, 20],
            "frac_expressing": [0.7, 0.5],
        }
    ).to_csv(src / "rsp_merfish_l23it_penk_by_subdivision.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["M", "M"],
            "variable": ["x_ccf", "z_reconstructed"],
            "axis_assumed": ["AP", "z"],
            "n": [10, 10],
            "rho": [-0.3, 0.3],
            "q_bh": [0.0, 0.0],
            "rsp_subdivision": [np.nan, np.nan],
        }
    ).to_csv(src / "rsp_merfish_l23it_penk_spatial_corr.csv", index=False)
    pd.DataFrame(
        {
            "ttype_group": ["L2/3 IT", "L5 IT"],
            "dataset": ["s", "s"],
            "variable": ["rheobase", "rheobase"],
            "n": [80, 70],
            "rho": [0.28, 0.0],
            "p": [0.01, 0.9],
            "q_bh": [0.05, 0.9],
        }
    ).to_csv(src / "patchseq_penk_ephys_spearman.csv", index=False)
    pd.DataFrame(
        {
            "line_name": ["Penk-IRES2-Cre-neo", "Penk-IRES2-Cre-neo", "Rbp4-Cre", np.nan],
            "structure__acronym": ["VISp6a", "RSPagl6a", "VISp5", np.nan],
        }
    ).to_csv(src / "patchseq_penk_ephys.csv", index=False)


def test_build(tmp_path: Path) -> None:
    _write(tmp_path)
    out = ma.build(tmp_path)
    t = out["tenx"]
    assert t["frac_expressing"] == 0.2 and [a["area"] for a in t["areas"]] == ["RSP", "VIS"]
    assert [s["taxon"] for s in t["supertypes"]] == ["A"] and t["epsilon_sq"]["cluster"] == 0.29
    assert t["hist_expressing"]["count"] == [5]
    g = out["genes"]
    assert [x["gene"] for x in g["positive"]] == ["Lamp5", "Ntm"] and g["markers"][0][
        "gene"
    ] == "Lamp5"
    m = out["merfish"]
    assert m["subdivisions"][0]["rsp_subdivision"] == "RSPd" and len(m["spatial"]) == 1
    assert out["patchseq"]["l23it"][0]["variable"] == "rheobase"
    assert out["patchseq"]["ctdb_penk_cre"] == {
        "n_cells": 2,
        "n_other_spiny": 1,
        "structures": {"VISp6a": 1, "RSPagl6a": 1},
    }


def test_build_empty_and_clean(tmp_path: Path) -> None:
    out = ma.build(tmp_path)
    assert out["tenx"] == {} and out["patchseq"] == {}
    assert ma._clean({"a": [float("nan"), np.int64(3)]}) == {"a": [None, 3]}
