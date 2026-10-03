"""Tests for the pure helpers in ``scripts/allen_penk_query.py`` (synthetic data, no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import allen_penk_query as apq

# --------------------------------------------------------------------------- genes


def test_gene_list_contents() -> None:
    assert apq.GENES[0] == "Penk"
    assert len(apq.GENES) == len(set(apq.GENES)) == 33
    for g in ("Cxcl14", "Oprd1", "Cacna1h", "Tshz2"):
        assert g in apq.GENES


def test_select_genes_present_preserves_order() -> None:
    present, missing = apq.select_genes_present(["Gad1", "Penk", "X"], ["Penk", "Sst", "Gad1"])
    assert present == ["Penk", "Gad1"]
    assert missing == ["Sst"]


def test_resolve_gene_columns_from_symbol_column() -> None:
    var = pd.DataFrame({"gene_symbol": ["Sst", "Penk", "Penk"]}, index=["E1", "E2", "E3"])
    pos, present, missing = apq.resolve_gene_columns(var, ["Penk", "Sst", "Vip"])
    assert pos == [1, 0]
    assert present == ["Penk", "Sst"]
    assert missing == ["Vip"]


def test_resolve_gene_columns_from_id_map_and_index() -> None:
    var = pd.DataFrame(index=["ENS1", "ENS2"])
    pos, present, _ = apq.resolve_gene_columns(var, ["Penk"], {"ENS2": "Penk"})
    assert pos == [1] and present == ["Penk"]
    var2 = pd.DataFrame(index=["Penk", "Gad1"])
    pos, present, missing = apq.resolve_gene_columns(var2, ["Gad1", "Vip"])
    assert pos == [1] and present == ["Gad1"] and missing == ["Vip"]


# --------------------------------------------------------------------------- region filters


def test_rsp_roi_mask() -> None:
    roi = pd.Series(["RSP", "ACA-RSP", "VIS", None, "rsp"])
    assert apq.rsp_roi_mask(roi).tolist() == [True, True, False, False, True]


def test_rsp_parcellation_mask_structure_and_substructure() -> None:
    df = pd.DataFrame(
        {
            "parcellation_structure": ["RSPd", "VISp", None, "RSPv", "ACAd"],
            "parcellation_substructure": ["RSPd1", "VISp2/3", "RSPagl5", "RSPv6a", "ACAd5"],
        }
    )
    assert apq.rsp_parcellation_mask(df).tolist() == [True, False, True, True, False]
    only_sub = df[["parcellation_substructure"]]
    assert apq.rsp_parcellation_mask(only_sub).tolist() == [True, False, True, True, False]
    assert not apq.rsp_parcellation_mask(pd.DataFrame({"x": [1]})).any()


@pytest.mark.parametrize(
    ("sub", "layer"),
    [("RSPd2/3", "L2/3"), ("RSPv1", "L1"), ("RSPagl6a", "L6a"), ("RSPd5", "L5"), ("RSPd", None)],
)
def test_parse_layer(sub: str, layer: str | None) -> None:
    assert apq.parse_layer(sub) == layer


def test_parse_layer_non_string() -> None:
    assert apq.parse_layer(np.nan) is None


def test_rsp_structure_from_substructure() -> None:
    assert apq.rsp_structure_from_substructure("RSPagl2/3") == "RSPagl"
    assert apq.rsp_structure_from_substructure("RSPv5") == "RSPv"
    assert apq.rsp_structure_from_substructure("VISp1") is None
    assert apq.rsp_structure_from_substructure(None) is None


# --------------------------------------------------------------------------- taxonomy / files


def test_annotate_with_taxonomy_index_and_dtype_mismatch() -> None:
    cells = pd.DataFrame({"cell_label": ["a", "b", "c"], "cluster_alias": [1, 2, 99]})
    pivot = pd.DataFrame(
        {
            "class": ["C1", "C1"],
            "subclass": ["007 L2/3 IT CTX Glut", "S2"],
            "supertype": ["T1", "T2"],
            "cluster": ["K1", "K2"],
            "neurotransmitter": ["Glut", "GABA"],
        },
        index=pd.Index(["1", "2"], name="cluster_alias"),
    )
    out = apq.annotate_with_taxonomy(cells, pivot)
    assert out["subclass"].tolist()[:2] == ["007 L2/3 IT CTX Glut", "S2"]
    assert pd.isna(out["subclass"].iloc[2])
    assert len(out) == 3


def test_annotate_with_taxonomy_requires_alias() -> None:
    with pytest.raises(KeyError):
        apq.annotate_with_taxonomy(
            pd.DataFrame({"cluster_alias": [1]}), pd.DataFrame({"subclass": ["x"]})
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("007 L2/3 IT CTX Glut", True),
        ("L2/3 IT RSP Glut", True),
        ("006 L4/5 IT CTX Glut", False),
        ("L2/3 IT GABA", False),
        (None, False),
    ],
)
def test_is_l23_it_like(name: object, expected: bool) -> None:
    assert apq.is_l23_it_like(name) is expected


def test_tenx_directory_for_label() -> None:
    assert apq.tenx_directory_for_label("WMB-10Xv3-Isocortex-1") == "WMB-10Xv3"
    assert apq.tenx_directory_for_label("WMB-10Xv2-TH") == "WMB-10Xv2"
    assert apq.tenx_directory_for_label("WMB-10XMulti") == "WMB-10XMulti"
    with pytest.raises(ValueError):
        apq.tenx_directory_for_label("MERFISH")


def test_pick_data_file() -> None:
    files = ["WMB-10Xv3-Isocortex-1/raw", "WMB-10Xv3-Isocortex-1/log2", "WMB-10Xv3-TH/log2"]
    assert apq.pick_data_file(files, "WMB-10Xv3-Isocortex-1") == "WMB-10Xv3-Isocortex-1/log2"
    assert apq.pick_data_file(["x-Isocortex-2-log2-v1"], "Isocortex-2") == "x-Isocortex-2-log2-v1"
    assert apq.pick_data_file(files, "WMB-10Xv3-HY") is None
    assert apq.pick_data_file(["C57BL6J-638850/log2"], "", "log2") == "C57BL6J-638850/log2"


# --------------------------------------------------------------------------- summaries


def _tenx_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subclass": ["007 L2/3 IT CTX Glut"] * 4 + ["Sst Gaba"] * 2,
            "supertype": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "cluster": ["K1", "K1", "K2", "K2", "K3", "K3"],
            "library_method": ["10Xv3", "10Xv2", "10Xv3", "10Xv3", "10Xv3", "10Xv3"],
            "Penk": [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "Cxcl14": [4.0, 0.0, 2.0, 1.0, 3.0, 3.0],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_penk_group_summary_values() -> None:
    out = apq.penk_group_summary(_tenx_frame(), genes=("Penk", "Cxcl14", "Absent"))
    sub = out[(out["level"] == "subclass") & (out["group"] == "007 L2/3 IT CTX Glut")].iloc[0]
    assert sub["n_cells"] == 4 and sub["n_penk_pos"] == 2
    assert sub["frac_penk_pos"] == pytest.approx(0.5)
    assert sub["mean_log2_penk"] == pytest.approx(0.75)
    assert sub["Cxcl14_mean_penk_pos"] == pytest.approx(3.0)
    assert sub["Cxcl14_mean_penk_neg"] == pytest.approx(0.5)
    assert bool(sub["l23_it_like"])
    sst = out[(out["level"] == "subclass") & (out["group"] == "Sst Gaba")].iloc[0]
    assert sst["n_penk_pos"] == 0 and np.isnan(sst["Cxcl14_mean_penk_pos"])
    assert not bool(sst["l23_it_like"])
    assert "Absent_mean_penk_pos" not in out.columns


def test_penk_group_summary_levels_and_pooled_row() -> None:
    out = apq.penk_group_summary(_tenx_frame(), genes=("Penk",))
    assert set(out["level"]) == {
        "subclass",
        "supertype",
        "cluster",
        "library_methodxsubclass",
        "l23_it_like_pooled",
    }
    combo = out[out["level"] == "library_methodxsubclass"]
    assert "10Xv2 | 007 L2/3 IT CTX Glut" in set(combo["group"])
    pooled = out[out["level"] == "l23_it_like_pooled"].iloc[0]
    assert pooled["n_cells"] == 4
    t1 = out[(out["level"] == "supertype") & (out["group"] == "T1")].iloc[0]
    assert t1["modal_subclass"] == "007 L2/3 IT CTX Glut"


def test_penk_group_summary_requires_penk() -> None:
    with pytest.raises(KeyError):
        apq.penk_group_summary(pd.DataFrame({"subclass": ["a"], "Gad1": [1.0]}))


def test_positive_fraction_summary_with_marginal() -> None:
    df = pd.DataFrame(
        {
            "dataset": ["m"] * 5,
            "rsp_subdivision": ["RSPd"] * 3 + ["RSPv"] * 2,
            "layer": ["L2/3", "L2/3", "L5", "L2/3", "L2/3"],
            "subclass": ["IT", "Sst", "IT", "IT", "IT"],
            "Penk": [1.0, 0.0, 0.0, 2.0, 0.0],
        }
    )
    out = apq.positive_fraction_summary(df, ["dataset", "rsp_subdivision", "layer", "subclass"])
    row = out[(out.rsp_subdivision == "RSPv") & (out.subclass == "IT")].iloc[0]
    assert row["n_cells"] == 2 and row["n_penk_pos"] == 1 and row["frac_penk_pos"] == 0.5
    allrow = out[(out.rsp_subdivision == "RSPd") & (out.layer == "L2/3") & (out.subclass == "ALL")]
    assert allrow["n_cells"].iloc[0] == 2 and allrow["n_penk_pos"].iloc[0] == 1
    assert out["n_cells"][out.subclass != "ALL"].sum() == 5
    assert out["n_cells"][out.subclass == "ALL"].sum() == 5


def test_positive_fraction_summary_without_marginal() -> None:
    df = pd.DataFrame({"a": ["x", "x"], "Penk": [0.0, 1.0]})
    out = apq.positive_fraction_summary(df, ["a"], marginal_key=None)
    assert len(out) == 1 and out["frac_penk_pos"].iloc[0] == 0.5


# --------------------------------------------------------------------------- statistics


def _cliff_brute(x: list[float], y: list[float]) -> float:
    s = sum((xi > yi) - (xi < yi) for xi in x for yi in y)
    return s / (len(x) * len(y))


def test_cliffs_delta_known_values() -> None:
    assert apq.cliffs_delta([3, 4], [1, 2]) == 1.0
    assert apq.cliffs_delta([1, 2], [3, 4]) == -1.0
    assert apq.cliffs_delta([1, 2], [1, 2]) == 0.0
    assert np.isnan(apq.cliffs_delta([], [1.0]))
    assert apq.cliffs_delta([1.0, np.nan], [0.0]) == 1.0


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(-5, 5), min_size=1, max_size=15),
    st.lists(st.integers(-5, 5), min_size=1, max_size=15),
)
def test_cliffs_delta_matches_brute_force(x: list[int], y: list[int]) -> None:
    d = apq.cliffs_delta(x, y)
    assert d == pytest.approx(_cliff_brute(x, y))
    assert -1.0 <= d <= 1.0
    assert apq.cliffs_delta(y, x) == pytest.approx(-d)


def test_bh_fdr() -> None:
    q = apq.bh_fdr([0.01, 0.04, np.nan, 0.03])
    assert np.isnan(q[2])
    assert q[0] == pytest.approx(0.03)
    assert q[1] == pytest.approx(0.04)
    assert q[3] == pytest.approx(0.04)
    assert np.isnan(apq.bh_fdr([np.nan])).all()


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=30))
def test_bh_fdr_properties(p: list[float]) -> None:
    q = apq.bh_fdr(p)
    assert np.all(q >= np.asarray(p) - 1e-12)
    assert np.all(q <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_cpm_and_orientation() -> None:
    genes_by_cells = pd.DataFrame({"c1": [1, 3], "c2": [0, 0]}, index=["Penk", "Gad1"])
    cells = apq.orient_cells_by_genes(genes_by_cells)
    assert list(cells.columns) == ["Penk", "Gad1"]
    out = apq.cpm(cells)
    assert out.loc["c1", "Penk"] == pytest.approx(250_000)
    assert out.loc["c2"].isna().all()
    assert apq.orient_cells_by_genes(cells) is cells
    with pytest.raises(KeyError):
        apq.orient_cells_by_genes(pd.DataFrame({"a": [1]}, index=["Gad1"]))


def test_penk_split_methods() -> None:
    v = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])
    nz = apq.penk_split(v, "nonzero")
    assert nz.iloc[:4].tolist() == [False, True, True, True]
    assert pd.isna(nz.iloc[4])
    tq = apq.penk_split(v, "topq", q=0.75)
    assert tq.iloc[:4].tolist() == [False, False, False, True]
    zeros = apq.penk_split(pd.Series([0.0, 0.0]), "topq")
    assert not zeros.astype(bool).any()
    with pytest.raises(ValueError):
        apq.penk_split(v, "bad")


def test_compare_penk_groups() -> None:
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame(
        {
            "dataset": ["d"] * (2 * n) + ["e"] * 4,
            "ttype_group": ["L2/3 IT"] * (2 * n) + ["L5 IT"] * 4,
            "split_a": [True] * n + [False] * n + [True, False, True, False],
            "split_b": [np.nan] * (2 * n) + [True, True, False, False],
            "rin": np.r_[rng.normal(200, 5, n), rng.normal(100, 5, n), [1, 2, 3, 4]],
        }
    )
    out = apq.compare_penk_groups(df, ["rin", "missing"], ["split_a", "split_b"])
    row = out[(out.dataset == "d") & (out.group == "L2/3 IT") & (out.split == "split_a")].iloc[0]
    assert row["n_penk_pos"] == n and row["n_penk_neg"] == n
    assert row["cliffs_delta"] == 1.0
    assert row["mannwhitney_p"] < 1e-6
    assert row["q_bh"] >= row["mannwhitney_p"]
    assert {"ALL", "L2/3 IT"} <= set(out[out.dataset == "d"]["group"])
    small = out[(out.dataset == "e") & (out.split == "split_a") & (out.group == "L5 IT")].iloc[0]
    assert np.isnan(small["mannwhitney_p"]) and small["n_penk_pos"] == 2
    # dataset d has no values for split_b -> no rows
    assert out[(out.dataset == "d") & (out.split == "split_b")].empty
    assert "missing" not in set(out["feature"])


def test_compare_penk_groups_empty() -> None:
    df = pd.DataFrame({"dataset": ["d"], "ttype_group": ["x"], "s": [np.nan], "f": [1.0]})
    assert apq.compare_penk_groups(df, ["f"], ["s"]).empty


# --------------------------------------------------------------------------- patch-seq helpers


def test_map_ephys_columns_mini_atlas_style() -> None:
    cols = [
        "Rheobase (pA)",
        "Input resistance (MOhm)",
        "AP width (ms)",
        "ISI adaptation index",
        "Max number of APs",
        "Sag ratio",
        "Resting membrane potential (mV)",
        "Latency (ms)",
        "Membrane time constant (ms)",
        "AP threshold (mV)",
        "Afterhyperpolarization (mV)",
        "AP amplitude (mV)",
    ]
    m = apq.map_ephys_columns(cols)
    assert m["rheobase"] == "Rheobase (pA)"
    assert m["input_resistance"] == "Input resistance (MOhm)"
    assert m["ap_width"] == "AP width (ms)"
    assert m["adaptation_index"] == "ISI adaptation index"
    assert m["max_firing"] == "Max number of APs"
    assert m["sag"] == "Sag ratio"
    assert m["resting_vm"] == "Resting membrane potential (mV)"
    assert m["latency"] == "Latency (ms)"
    assert m["tau"] == "Membrane time constant (ms)"
    assert m["ap_threshold"] == "AP threshold (mV)"
    assert m["ahp"] == "Afterhyperpolarization (mV)"
    assert m["ap_amplitude"] == "AP amplitude (mV)"
    assert len(set(m.values())) == len(m)


def test_map_ephys_columns_ctdb_style() -> None:
    cols = [
        "ef__ri",
        "ef__vrest",
        "ef__threshold_i_long_square",
        "ef__upstroke_downstroke_ratio_long_square",
        "ef__tau",
        "ef__f_i_curve_slope",
        "ef__adaptation",
        "ef__avg_firing_rate",
        "ef__fast_trough_v_long_square",
        "ef__peak_t_ramp",
    ]
    m = apq.map_ephys_columns(cols)
    assert m["input_resistance"] == "ef__ri"
    assert m["resting_vm"] == "ef__vrest"
    assert m["rheobase"] == "ef__threshold_i_long_square"
    assert m["upstroke_downstroke_ratio"] == "ef__upstroke_downstroke_ratio_long_square"
    assert m["tau"] == "ef__tau"
    assert m["fi_slope"] == "ef__f_i_curve_slope"
    assert m["adaptation_index"] == "ef__adaptation"
    assert m["max_firing"] == "ef__avg_firing_rate"
    assert m["ahp"] == "ef__fast_trough_v_long_square"
    assert "ap_width" not in m


@pytest.mark.parametrize(
    ("t", "g"),
    [
        ("L2/3 IT_1", "L2/3 IT"),
        ("L4/5 IT_2", "L4/5 IT"),
        ("L5 IT_1", "L5 IT"),
        ("L6 IT Car3", "L6 IT"),
        ("Sst Chodl", None),
        (np.nan, None),
    ],
)
def test_ttype_group(t: object, g: str | None) -> None:
    assert apq.ttype_group(t) == g


def test_extract_links() -> None:
    html = """
    <a href="data/patchseq_metadata.csv">m</a>
    <a href='https://x.org/ephys_features.csv.gz?dl=1'>e</a>
    <a href="page.html">p</a>
    <a href="data/patchseq_metadata.csv">dup</a>
    <a href="/other/morph.zip">z</a>
    """
    links = apq.extract_links(html, "https://portal.example/a/b")
    assert links == [
        "https://portal.example/a/data/patchseq_metadata.csv",
        "https://x.org/ephys_features.csv.gz?dl=1",
        "https://portal.example/other/morph.zip",
    ]
    filtered = apq.extract_links(html, "https://portal.example/a/b", apq.ALLEN_LINK_PATTERN)
    assert "https://portal.example/other/morph.zip" not in filtered
    assert len(filtered) == 2


def test_ctdb_to_table() -> None:
    recs = [
        {"specimen__id": 1, "line_name": "Penk-IRES2-Cre-neo", "tag__dendrite_type": "spiny",
         "structure__layer": "2/3", "ef__ri": 200.0},
        {"specimen__id": 2, "line_name": "Cux2-CreERT2", "tag__dendrite_type": "spiny",
         "structure__layer": "2/3", "ef__ri": 150.0},
        {"specimen__id": 3, "line_name": "Penk-IRES2-Cre-neo", "tag__dendrite_type": "aspiny",
         "structure__layer": "2/3", "ef__ri": 400.0},
    ]  # fmt: skip
    df = apq.ctdb_to_table(recs)
    assert df["cell_id"].tolist() == ["1", "2"]
    assert df["penk_cre"].tolist() == [True, False]
    assert set(df["ttype_group"]) == {"layer 2/3"}
    assert set(df["dataset"]) == {"allen_ctdb_cre"}
    assert apq.ctdb_to_table([]).empty


def test_find_and_id_column(tmp_path: Path) -> None:
    files = [tmp_path / "m1_patchseq_meta_data.csv", tmp_path / "m1_patchseq_exon_counts.csv.gz"]
    assert apq._find(files, "exon", "count") == files[1]
    assert apq._find(files, "ephys") is None
    assert apq._id_column(pd.DataFrame({"x": [1], "Cell": [2]})) == "Cell"
    assert apq._id_column(pd.DataFrame({"x": [1]})) == "x"


def test_gene_symbol_map() -> None:
    g = pd.DataFrame({"gene_symbol": ["Penk"]}, index=pd.Index(["E1"], name="gene_identifier"))
    assert apq._gene_symbol_map(g) == {"E1": "Penk"}
    assert apq._gene_symbol_map(None) is None
    assert apq._gene_symbol_map(pd.DataFrame({"a": [1]})) is None


def test_url_log() -> None:
    u = apq.UrlLog()
    u.add("https://a", "200", "ok", 10)
    u.add("https://b", "404", "Not Found")
    f = u.frame()
    assert f["status"].tolist() == ["200", "404"]
    assert list(f.columns) == ["url", "status", "detail", "size_bytes"]
    assert apq.UrlLog().frame().empty


# --------------------------------------------------------------------------- bookkeeping


def test_parse_s3_uri_and_key() -> None:
    assert apq.parse_s3_uri("s3://bkt/allen/penk/") == ("bkt", "allen/penk")
    assert apq.parse_s3_uri("s3://bkt") == ("bkt", "")
    with pytest.raises(ValueError):
        apq.parse_s3_uri("bkt/x")
    with pytest.raises(ValueError):
        apq.parse_s3_uri("s3:///x")
    assert apq.s3_key("a/b", "f.csv") == "a/b/f.csv"
    assert apq.s3_key("", "f.csv") == "f.csv"


def test_run_part_success_and_failure(tmp_path: Path) -> None:
    ctx = apq.Context(outdir=tmp_path, cache_dir=tmp_path)

    def good(c: apq.Context, r: apq.PartResult) -> None:
        apq.write_table(pd.DataFrame({"a": [1, 2]}), c.outdir / "t.csv", r)
        r.note("hello")

    def bad(c: apq.Context, r: apq.PartResult) -> None:
        apq.write_table(pd.DataFrame({"a": [1]}), c.outdir / "partial.csv.gz", r)
        raise RuntimeError("boom")

    ok = apq.run_part("A", good, ctx)
    assert ok.status == "ok" and ok.error is None
    assert ok.outputs[0].file == "t.csv" and ok.outputs[0].n_rows == 2
    assert ok.outputs[0].size_bytes > 0 and ok.notes == ["hello"]
    fail = apq.run_part("B", bad, ctx)
    assert fail.status == "failed" and "boom" in fail.error
    assert "RuntimeError" in fail.traceback
    assert fail.outputs[0].file == "partial.csv.gz"
    assert fail.duration_s is not None and fail.finished is not None


def test_manifest_roundtrip_and_done_status(tmp_path: Path) -> None:
    path = tmp_path / apq.MANIFEST_NAME
    m = apq.load_manifest(path)
    assert m["parts"] == {}
    m = apq.update_manifest(m, apq.PartResult(part="A", status="ok"))
    m = apq.update_manifest(m, apq.PartResult(part="B", status="failed", error="x"))
    path.write_text(json.dumps(m))
    m2 = apq.load_manifest(path)
    assert m2["parts"]["A"]["status"] == "ok"
    assert m2["genes"][0] == "Penk"
    m2 = apq.update_manifest(m2, apq.PartResult(part="B", status="ok"))
    assert m2["parts"]["B"]["status"] == "ok"
    assert apq.done_status(m2, ["A", "B"])["status"] == "ok"
    d = apq.done_status(m, ["A", "B", "C"])
    assert d["status"] == "partial" and d["parts"]["C"] == "missing"
    assert apq.done_status({"parts": {}}, ["A"])["status"] == "failed"
    assert apq.done_status({"parts": {}}, [])["status"] == "failed"
    path.write_text("{not json")
    assert apq.load_manifest(path)["parts"] == {}


def test_parser_defaults_and_choices() -> None:
    args = apq._build_arg_parser().parse_args([])
    assert args.parts == ["A", "B", "C"]
    assert args.s3_uri is None and args.write_done is None
    args = apq._build_arg_parser().parse_args(["--parts", "B", "--max-files", "1"])
    assert args.parts == ["B"] and args.max_files == 1
    with pytest.raises(SystemExit):
        apq._build_arg_parser().parse_args(["--parts", "D"])


def test_part_registry() -> None:
    assert set(apq.PART_FUNCS) == set(apq.ALL_PARTS)


# --------------------------------------------------------------------------- continuum helpers


def test_continuum_module_importable() -> None:
    mod = apq.continuum_module()
    assert hasattr(mod, "silverman_mode_test") and hasattr(mod, "shift_function")


def test_continuum_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apq, "_continuum_module", None)
    monkeypatch.setattr(apq, "_CONTINUUM_IMPORT_ERROR", ImportError("nope"))
    with pytest.raises(ImportError, match="PYTHONPATH"):
        apq.continuum_module()


def test_depth_markers_and_params() -> None:
    assert len(apq.DEPTH_MARKERS) == len(set(apq.DEPTH_MARKERS)) == 9
    assert {"Cux2", "Otof", "Cdh13", "Ccbe1", "Cxcl14"} <= set(apq.DEPTH_MARKERS)
    assert apq.HIST_EDGES[0] == 0.0 and apq.HIST_EDGES[-1] == 16.0
    p = apq._continuum_params()
    assert p["silverman"] == {"k": 1, "max_n": 3000, "n_boot": 200}
    assert p["gene_corr"]["min_frac_expressing"] == 0.05
    json.dumps(apq._json_safe(p))


def test_run_substeps_isolates_failures() -> None:
    r = apq.PartResult(part="A")
    ran: list[str] = []

    def ok() -> None:
        ran.append("ok")

    def bad() -> None:
        raise ValueError("x")

    with pytest.raises(RuntimeError, match="bad"):
        apq.run_substeps(r, [("bad", bad), ("ok", ok)])
    assert ran == ["ok"]
    assert any("bad failed: ValueError" in n for n in r.notes)
    apq.run_substeps(r, [("ok", ok)])  # no failure -> no raise


def test_write_json_and_json_safe(tmp_path: Path) -> None:
    r = apq.PartResult(part="A")
    rec = apq.write_json(
        {"a": np.float64(np.nan), "b": np.arange(2), "c": (1.0, np.inf), 3: np.int64(4)},
        tmp_path / "x.json",
        r,
    )
    data = json.loads((tmp_path / "x.json").read_text())
    assert data == {"a": None, "b": [0, 1], "c": [1.0, None], "3": 4}
    assert rec.n_rows is None and r.outputs[0].file == "x.json"


def test_subsample_rows() -> None:
    df = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 3, "v": range(13)})
    assert apq.subsample_rows(df, 20) is df
    s = apq.subsample_rows(df, 5, seed=1)
    assert len(s) == 5 and s.index.is_monotonic_increasing
    assert apq.subsample_rows(df, 5, seed=1).equals(s)
    by = apq.subsample_rows(df, 4, by="g")
    assert by["g"].value_counts().to_dict() == {"a": 4, "b": 3}
    assert apq.subsample_rows(df.iloc[0:0], 4, by="g").empty


def test_subclass_by_alias_and_mask() -> None:
    pivot = pd.DataFrame(
        {"subclass": ["007 L2/3 IT CTX Glut", "Sst Gaba"]},
        index=pd.Index([1, 2], name="cluster_alias"),
    )
    m = apq.subclass_by_alias(pivot)
    assert m == {"1": "007 L2/3 IT CTX Glut", "2": "Sst Gaba"}
    mask = apq.l23_it_alias_mask(pd.Series([1, 2, 99, None]), m)
    assert mask.tolist() == [True, False, False, False]
    with pytest.raises(KeyError):
        apq.subclass_by_alias(pd.DataFrame({"x": [1]}))


def test_isocortex_mask() -> None:
    df = pd.DataFrame({apq.ISOCORTEX_COL: ["Isocortex", "HPF", None]})
    assert apq.isocortex_mask(df).tolist() == [True, False, False]
    assert apq.isocortex_mask(pd.DataFrame({"x": [1, 2]})).all()


@pytest.mark.parametrize(
    ("roi", "sub"),
    [("RSP", "RSP"), ("RSPv", "RSPv"), ("RSPagl", "RSPagl"), ("ACA-RSP", "RSP"), ("VIS", None),
     (None, None)],
)  # fmt: skip
def test_rsp_subregion_from_roi(roi: object, sub: str | None) -> None:
    assert apq.rsp_subregion_from_roi(roi) == sub


def test_fixed_histogram() -> None:
    h = apq.fixed_histogram([0.0, 0.1, 0.5, 100.0, -1.0, np.nan], np.array([0.0, 0.25, 1.0]))
    assert h["count"].tolist() == [3, 2]  # -1 clipped into first bin, 100 into last
    assert h["frac"].sum() == pytest.approx(1.0)
    assert list(h.columns) == ["bin_lo", "bin_hi", "count", "frac"]
    empty = apq.fixed_histogram([], np.array([0.0, 1.0]))
    assert empty["count"].tolist() == [0] and empty["frac"].isna().all()


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(0, 20, allow_nan=False), min_size=0, max_size=40))
def test_fixed_histogram_conserves_counts(v: list[float]) -> None:
    h = apq.fixed_histogram(v)
    assert h["count"].sum() == len(v)
    assert len(h) == len(apq.HIST_EDGES) - 1


def test_expression_distribution() -> None:
    d = apq.expression_distribution([0.0, 0.0, 1.0, 3.0, np.nan])
    assert d["n"] == 4 and d["n_zero"] == 2 and d["n_expressing"] == 2
    assert d["frac_zero"] == 0.5 and d["frac_expressing"] == 0.5
    assert d["median_expressing"] == 2.0 and d["median_all"] == 0.5
    e = apq.expression_distribution([])
    assert e["n"] == 0 and np.isnan(e["frac_zero"]) and np.isnan(e["median_expressing"])


def test_silverman_expressing_bimodal_vs_unimodal() -> None:
    rng = np.random.default_rng(0)
    bi = np.r_[np.zeros(50), rng.normal(2, 0.3, 300), rng.normal(8, 0.3, 300)]
    uni = np.r_[np.zeros(50), rng.normal(5, 1.0, 600)]
    rb = apq.silverman_expressing(bi, max_n=400, n_boot=50)
    ru = apq.silverman_expressing(uni, max_n=400, n_boot=50)
    assert rb["n_silverman"] == 400 and rb["silverman_subsampled"]
    assert rb["silverman_p_unimodal"] < 0.05
    assert ru["silverman_p_unimodal"] > 0.05
    assert not apq.silverman_expressing([1.0, 2.0], n_boot=5)["silverman_subsampled"]


def test_groups_with_pooling() -> None:
    df = pd.DataFrame({"d": ["x", "x", "y"], "g": ["a", "b", "a"], "v": [1, 2, 3]})
    gs = apq._groups(df, ["d", "g"], pool_col="g")
    keys = [kv for kv, _ in gs]
    assert {"d": "x", "g": "ALL"} in keys and {"d": "y", "g": "ALL"} in keys
    assert len(gs) == 5
    pooled = dict((tuple(kv.values()), len(s)) for kv, s in gs)
    assert pooled[("x", "ALL")] == 2
    assert apq._groups(df, [])[0][0] == {}


def test_penk_distribution_tables() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "Penk": np.r_[np.zeros(20), rng.uniform(1, 5, 40)],
            "library_method": ["v3"] * 30 + ["v2"] * 30,
        }
    )
    s, h = apq.penk_distribution_tables(df, [[], ["library_method"], ["absent"]], "test", n_boot=5)
    assert s["grouping"].tolist() == ["pooled", "library_method", "library_method"]
    pooled = s.iloc[0]
    assert pooled["group"] == "ALL" and pooled["n"] == 60 and pooled["n_zero"] == 20
    assert "silverman_p_unimodal" in s.columns
    assert set(h["population"]) == {"all", "expressing"}
    pooled_all = h[(h.grouping == "pooled") & (h.population == "all")]
    assert pooled_all["count"].sum() == 60
    pooled_expr = h[(h.grouping == "pooled") & (h.population == "expressing")]
    assert pooled_expr["count"].sum() == 40
    s2, _ = apq.penk_distribution_tables(df, [[]], "t", silverman=False)
    assert "silverman_p_unimodal" not in s2.columns


def test_group_expression_table() -> None:
    df = pd.DataFrame({"k": ["a"] * 4 + ["b"] * 4 + ["c"], "Penk": [1, 1, 1, 0, 0, 0, 0, 2, 5]})
    t = apq.group_expression_table(df, ["k"], min_n=2)
    assert t["k"].tolist() == ["a", "b"]
    a = t.iloc[0]
    assert a["n_expressing"] == 3 and a["frac_expressing"] == 0.75
    assert a["share_of_expressing"] == pytest.approx(0.75)
    assert t["share_of_expressing"].sum() == pytest.approx(1.0)
    assert apq.group_expression_table(df.iloc[0:0], ["k"]).empty


def test_kruskal_epsilon_squared_matches_scipy() -> None:
    from scipy.stats import kruskal

    rng = np.random.default_rng(2)
    v = np.r_[rng.normal(0, 1, 20), rng.normal(1, 1, 25), np.zeros(10), [9.0] * 3]
    g = ["a"] * 20 + ["b"] * 25 + ["c"] * 10 + ["d"] * 3
    r = apq.kruskal_epsilon_squared(v, g, min_n=5)
    ref = kruskal(v[:20], v[20:45], v[45:55])
    assert r["k"] == 3 and r["n"] == 55 and r["n_groups_dropped"] == 1
    assert r["H"] == pytest.approx(ref.statistic)
    assert r["p"] == pytest.approx(ref.pvalue)
    assert r["epsilon_sq"] == pytest.approx(ref.statistic / 54)


def test_kruskal_epsilon_squared_extremes() -> None:
    perfect = apq.kruskal_epsilon_squared([1, 2, 3, 10, 11, 12], list("aaabbb"), min_n=3)
    assert perfect["epsilon_sq"] == pytest.approx(13.5 / 17.5)  # SSB / SST of ranks 1..6
    tied_within = apq.kruskal_epsilon_squared([1, 1, 1, 2, 2, 2], list("aaabbb"), min_n=3)
    assert tied_within["epsilon_sq"] == pytest.approx(1.0)
    tied = apq.kruskal_epsilon_squared([0.0] * 6, list("aaabbb"), min_n=3)
    assert np.isnan(tied["H"]) and np.isnan(tied["epsilon_sq"])
    one = apq.kruskal_epsilon_squared([1.0, 2.0, 3.0], ["a", "a", None], min_n=1)
    assert one["k"] == 1 and np.isnan(one["p"])


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.integers(0, 4), min_size=6, max_size=40),
    st.lists(st.sampled_from(["a", "b", "c"]), min_size=40, max_size=40),
)
def test_kruskal_epsilon_squared_equals_rank_eta_squared(v: list[int], g: list[str]) -> None:
    from scipy.stats import rankdata

    g = g[: len(v)]
    r = apq.kruskal_epsilon_squared(v, g, min_n=1)
    if np.isnan(r["epsilon_sq"]):
        return
    ranks = rankdata(v)
    s = pd.Series(ranks).groupby(pd.Series(g))
    ssb = (s.size() * (s.mean() - ranks.mean()) ** 2).sum()
    sst = ((ranks - ranks.mean()) ** 2).sum()
    assert r["epsilon_sq"] == pytest.approx(ssb / sst)
    assert 0.0 <= r["epsilon_sq"] <= 1.0 + 1e-12


def test_taxonomy_variance_table() -> None:
    df = pd.DataFrame(
        {
            "cluster": ["k1"] * 6 + ["k2"] * 6 + ["k3"] * 6,
            "Penk": [3, 4, 5, 3, 4, 5] + [0] * 6 + [0, 0, 0, 0, 0, 1],
            "library_method": ["v3", "v2"] * 9,
        }
    )
    t = apq.taxonomy_variance_table(df, ["cluster", "absent"], "s", strata_col="library_method",
                                    min_n=3)  # fmt: skip
    assert t["stratum"].tolist() == ["ALL", "v2", "v3"]
    allrow = t.iloc[0]
    assert allrow["k"] == 3 and allrow["epsilon_sq"] > 0.5 and allrow["p"] < 0.01
    assert allrow["top3_share_of_expressing"] == pytest.approx(1.0)
    assert allrow["max_group_frac_expressing"] == 1.0
    assert allrow["min_group_frac_expressing"] == 0.0
    assert pd.isna(allrow["stratum_col"]) and t.iloc[1]["stratum_col"] == "library_method"


def test_spearman_against_and_rho_p() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    x = rng.normal(size=50)
    m = np.c_[x**3, -x, rng.normal(size=50), np.ones(50)]
    rho = apq.spearman_against(x, m)
    assert rho[0] == pytest.approx(1.0) and rho[1] == pytest.approx(-1.0)
    assert rho[2] == pytest.approx(spearmanr(x, m[:, 2]).statistic)
    assert np.isnan(rho[3])
    r, p, n = apq.spearman_rho_p(x, m[:, 2])
    ref = spearmanr(x, m[:, 2])
    assert (r, n) == (pytest.approx(ref.statistic), 50)
    assert p == pytest.approx(ref.pvalue)
    r2, p2, n2 = apq.spearman_rho_p([1.0, np.nan, 2.0], [1.0, 2.0, np.nan])
    assert n2 == 1 and np.isnan(r2) and np.isnan(p2)
    with pytest.raises(ValueError):
        apq.spearman_against([1.0, 2.0], np.ones((3, 1)))
    with pytest.raises(ValueError):
        apq.spearman_against([1.0, np.nan], np.ones((2, 1)))


def test_spearman_p_edge_cases() -> None:
    assert np.isnan(apq._spearman_p(np.array([0.5]), 2)).all()
    assert apq._spearman_p(np.array([1.0]), 10)[0] == 0.0
    assert np.isnan(apq._spearman_p(np.array([np.nan]), 10)[0])


def test_partial_rho() -> None:
    # x and y both driven by z only -> partial correlation zero
    assert apq.partial_rho(np.array([0.25]), 0.5, np.array([0.5]))[0] == pytest.approx(0.0)
    assert apq.partial_rho(np.array([0.3]), 0.0, np.array([0.0]))[0] == pytest.approx(0.3)
    assert np.isnan(apq.partial_rho(np.array([0.3]), 1.0, np.array([0.2]))[0])


def _gene_matrix(seed: int = 4) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    n = 200
    penk = np.where(rng.random(n) < 0.6, rng.uniform(1, 6, n), 0.0)
    up = penk + rng.normal(0, 0.3, n)
    down = -penk + rng.normal(0, 0.3, n) + 10
    noise = rng.uniform(0, 5, n)
    rare = np.where(np.arange(n) < 4, 1.0, 0.0)  # 2 % expressing
    otof = np.where(np.arange(n) < 2, penk, 0.0)  # 1 % expressing (depth marker)
    m = np.c_[penk, up, down, noise, rare, otof, penk]  # last: duplicate symbol
    return m, ["Penk", "Up", "Down", "Noise", "Rare", "Otof", "Up"]


def test_gene_correlations_dense_and_sparse() -> None:
    import scipy.sparse as sp

    m, genes = _gene_matrix()
    out = apq.gene_correlations(m, genes)
    assert out["gene"].iloc[0] == "Up" and out.set_index("gene").loc["Down", "rho"] < -0.8
    assert "Penk" not in set(out["gene"]) and "Rare" not in set(out["gene"])
    otof = out.set_index("gene").loc["Otof"]
    assert not otof["passes_min_frac"] and np.isnan(otof["q_bh"])
    assert out.loc[out.passes_min_frac, "q_bh"].notna().all()
    assert (out["n_cells"] == 200).all()
    sp_out = apq.gene_correlations(sp.csr_matrix(m), genes)
    pd.testing.assert_frame_equal(out, sp_out)
    with pytest.raises(KeyError):
        apq.gene_correlations(m, ["A"] * 7)


def test_gene_correlations_partial_on_covariate() -> None:
    rng = np.random.default_rng(5)
    n = 300
    depth = rng.uniform(0, 1, n)
    penk = depth + rng.normal(0, 0.05, n) + 1
    g = depth + rng.normal(0, 0.05, n) + 1
    out = apq.gene_correlations(np.c_[penk, g], ["Penk", "G"], covariate=depth)
    row = out.iloc[0]
    assert row["rho"] > 0.8
    assert abs(row["rho_partial"]) < 0.3
    assert {"p_partial", "q_bh_partial"} <= set(out.columns)


def test_top_correlated_genes_and_markers() -> None:
    corr = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E"],
            "passes_min_frac": [True, True, True, True, False],
            "rho": [0.9, 0.5, -0.2, -0.7, 0.99],
            "p": [0.01] * 5,
        }
    )
    top = apq.top_correlated_genes(corr, n=1)
    assert top["gene"].tolist() == ["A", "D"]
    assert top["direction"].tolist() == ["positive", "negative"] and top["rank"].tolist() == [1, 1]
    mk = apq.marker_correlations(corr, ["E", "Z", "A"])
    assert mk["gene"].tolist() == ["E", "Z", "A"]
    assert mk["in_data"].tolist() == [True, False, True]
    assert np.isnan(mk.loc[1, "rho"]) and mk.loc[0, "rho"] == 0.99


def test_spearman_table() -> None:
    rng = np.random.default_rng(6)
    n = 30
    x = rng.normal(size=2 * n)
    df = pd.DataFrame(
        {
            "dataset": ["d"] * (2 * n),
            "grp": ["a"] * n + ["b"] * n,
            "penk": x,
            "up": x * 2,
            "few": [1.0, 2.0] + [np.nan] * (2 * n - 2),
        }
    )
    t = apq.spearman_table(df, "penk", ["up", "few", "missing", "penk"], ["dataset", "grp"],
                           pool_col="grp")  # fmt: skip
    assert set(t["grp"]) == {"a", "b", "ALL"}
    assert set(t["variable"]) == {"up", "few"}
    up_all = t[(t.grp == "ALL") & (t.variable == "up")].iloc[0]
    assert up_all["rho"] == pytest.approx(1.0) and up_all["n"] == 2 * n
    few = t[(t.grp == "a") & (t.variable == "few")].iloc[0]
    assert np.isnan(few["rho"]) and np.isnan(few["q_bh"])
    assert (t["q_bh"].dropna() >= t.loc[t["q_bh"].notna(), "p"] - 1e-12).all()
    assert apq.spearman_table(df, "penk", ["missing"], ["dataset"]).empty


def test_shift_table() -> None:
    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame(
        {
            "dataset": ["d"] * (2 * n),
            "grp": ["L2/3 IT"] * (2 * n),
            "pos": [True] * n + [False] * n,
            "rin": np.r_[rng.normal(10, 1, n), rng.normal(0, 1, n)],
            "tiny": [1.0] * 3 + [np.nan] * (2 * n - 3),
        }
    )
    t = apq.shift_table(df, ["rin", "tiny", "absent"], "pos", ["dataset", "grp"], pool_col="grp",
                        n_boot=50)  # fmt: skip
    assert set(t["feature"]) == {"rin"}
    assert set(t["grp"]) == {"L2/3 IT", "ALL"}
    assert len(t) == 2 * 9
    assert (t["diff"] > 5).all() and (t["lo"] <= t["diff"]).all() and (t["hi"] >= t["diff"]).all()
    assert (t["n_pos"] == n).all()


def test_distance_from_midline_units() -> None:
    assert apq.distance_from_midline([5.7, 4.7, 6.2]) == pytest.approx([0.0, 1.0, 0.5])
    um = apq.distance_from_midline([5700.0, 4700.0, np.nan])
    assert um[:2] == pytest.approx([0.0, 1000.0]) and np.isnan(um[2])


def test_depth_columns_and_spatial_proxies() -> None:
    cols = ["x_ccf", "cortical_depth", "distance_to_pia", "Penk", "layer", "z_ccf"]
    assert apq.find_depth_columns(cols) == ["cortical_depth", "distance_to_pia"]
    df = pd.DataFrame({"layer": ["L2/3", "L5", None], "z_ccf": [5.7, 5.2, 6.7]})
    out = apq.add_spatial_proxies(df)
    assert out["layer_ordinal"].tolist()[:2] == [2.0, 4.0] and np.isnan(out["layer_ordinal"][2])
    assert out["ccf_ml_from_midline"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert "layer_ordinal" not in apq.add_spatial_proxies(pd.DataFrame({"a": [1]})).columns
    sv = apq.spatial_variables([*cols, "layer_ordinal", "ccf_ml_from_midline"])
    assert sv[:2] == ["cortical_depth", "distance_to_pia"]
    assert {"x_ccf", "z_ccf", "layer_ordinal", "ccf_ml_from_midline"} <= set(sv)
    assert "layer" not in sv


def test_align_sparse_blocks() -> None:
    import scipy.sparse as sp

    b1 = (sp.csr_matrix(np.array([[1.0, 2.0, 3.0]])), ["c1"], ["A", "B", "C"])
    b2 = (np.array([[30.0, 10.0], [0.0, 1.0]]), ["c2", "c3"], ["C", "A"])
    m, cells, genes = apq.align_sparse_blocks([b1, b2])
    assert genes == ["A", "C"] and cells == ["c1", "c2", "c3"]
    assert m.toarray().tolist() == [[1.0, 3.0], [10.0, 30.0], [1.0, 0.0]]
    dup = (np.array([[1.0, 2.0]]), ["c"], ["A", "A"])
    m2, _, g2 = apq.align_sparse_blocks([dup])
    assert g2 == ["A"] and m2.toarray().tolist() == [[1.0]]
    with pytest.raises(ValueError):
        apq.align_sparse_blocks([])


def test_var_symbols() -> None:
    var = pd.DataFrame({"gene_symbol": ["Penk"]}, index=["E1"])
    assert apq.var_symbols(var).tolist() == ["Penk"]
    assert apq.var_symbols(pd.DataFrame(index=["E1", "E2"]), {"E1": "Penk"}).tolist() == [
        "Penk",
        "E2",
    ]
    assert apq.var_symbols(pd.DataFrame(index=["Penk"])).tolist() == ["Penk"]


def test_merfish_l23_it() -> None:
    cells = pd.DataFrame(
        {
            "subclass": ["007 L2/3 IT CTX Glut", "007 L2/3 IT CTX Glut", "Sst Gaba"],
            "Penk": [1.0, np.nan, 2.0],
            "layer": ["L2/3", "L1", "L2/3"],
            "z_ccf": [5.0, 5.0, 5.0],
        }
    )
    out = apq.merfish_l23_it(cells)
    assert len(out) == 1 and out["layer_ordinal"].iloc[0] == 2.0
    assert out["ccf_ml_from_midline"].iloc[0] == pytest.approx(0.7)
    with pytest.raises(KeyError):
        apq.merfish_l23_it(pd.DataFrame({"Penk": [1.0]}))


def test_patchseq_penk_continuous() -> None:
    cells = pd.DataFrame({"penk_cpm": [0.0, 3.0, np.nan], "x": [1, 2, 3]})
    out = apq.patchseq_penk_continuous(cells)
    assert out["log2_penk_cpm"].tolist() == [0.0, 2.0]
    none = apq.patchseq_penk_continuous(pd.DataFrame({"x": [1]}))
    assert none.empty and "log2_penk_cpm" in none.columns


def test_alias_keys() -> None:
    out = apq.alias_keys(pd.Series([1.0, 2, None, "a7", "3"]))
    assert out.tolist()[:2] == ["1", "2"] and out.tolist()[3:] == ["a7", "3"]
    cells = pd.DataFrame({"cluster_alias": [1.0, np.nan]})
    pivot = pd.DataFrame({"cluster_alias": ["1"], "subclass": ["S"]})
    assert apq.annotate_with_taxonomy(cells, pivot)["subclass"].tolist()[0] == "S"


# ------------------------------------------------------------------ continuum orchestration


def _tenx_cells(n: int, roi: str, seed: int, prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cluster = rng.choice(["K1", "K2", "K3"], n)
    penk = np.where(cluster == "K1", rng.uniform(2, 6, n), np.where(rng.random(n) < 0.2, 1.0, 0.0))
    return pd.DataFrame(
        {
            "cell_label": [f"{prefix}{i}" for i in range(n)],
            "library_method": rng.choice(["10Xv2", "10Xv3"], n),
            "roi": roi,
            "subclass": "007 L2/3 IT CTX Glut",
            "supertype": np.where(cluster == "K3", "T2", "T1"),
            "cluster": cluster,
            "Penk": penk,
            "Otof": rng.uniform(0, 1, n),
        }
    )


def test_part_a_continuum_end_to_end(tmp_path: Path) -> None:
    import scipy.sparse as sp

    rsp = _tenx_cells(120, "RSP", 0, "r")
    rsp = pd.concat([rsp, rsp.iloc[:5].assign(subclass="Sst Gaba", cell_label=list("abcde"))])
    area = pd.concat([_tenx_cells(60, "VISp", 1, "v"), _tenx_cells(60, "MOp", 2, "m")])
    rng = np.random.default_rng(3)
    l23 = rsp.iloc[:120]
    mat = np.c_[l23["Penk"], l23["Penk"] + rng.normal(0, 0.1, 120), rng.uniform(0, 3, (120, 3))]
    genes = ["Penk", "Up", "G1", "G2", "Otof"]
    blocks = [
        (sp.csr_matrix(mat[:70]), l23["cell_label"].tolist()[:70], genes),
        (sp.csr_matrix(mat[70:]), l23["cell_label"].tolist()[70:], genes),
    ]
    ctx = apq.Context(outdir=tmp_path, cache_dir=tmp_path)
    r = apq.PartResult(part="A")
    apq._part_a_continuum(ctx, r, rsp, area, blocks)
    files = {o.file for o in r.outputs}
    assert files == {
        "continuum_params_A.json",
        "l23it_10x_penk_distribution.csv",
        "l23it_10x_penk_histogram.csv",
        "l23it_10x_penk_by_taxon.csv",
        "l23it_10x_penk_kruskal.csv",
        "l23it_10x_penk_by_region.csv",
        "l23it_10x_cells.csv.gz",
        "rsp_l23it_10x_penk_gene_corr.csv.gz",
        "rsp_l23it_10x_penk_gene_corr_top.csv",
        "rsp_l23it_10x_penk_depth_markers.csv",
    }
    dist = pd.read_csv(tmp_path / "l23it_10x_penk_distribution.csv")
    assert set(dist["scope"]) == {"RSP L2/3 IT", "isocortex L2/3 IT"}
    areas = dist[(dist.scope == "isocortex L2/3 IT") & (dist.grouping == "area")]
    assert set(areas["group"]) == {"RSP", "VISp", "MOp"}
    assert dist.loc[dist.grouping == "pooled", "n"].iloc[0] == 120
    kw = pd.read_csv(tmp_path / "l23it_10x_penk_kruskal.csv")
    clus = kw[(kw.scope == "RSP L2/3 IT") & (kw.factor == "cluster") & (kw.stratum == "ALL")]
    assert clus["epsilon_sq"].iloc[0] > 0.5
    corr = pd.read_csv(tmp_path / "rsp_l23it_10x_penk_gene_corr.csv.gz")
    assert set(corr["library_method"]) == {"ALL", "10Xv2", "10Xv3"}
    top = corr[corr.library_method == "ALL"].iloc[0]
    assert top["gene"] == "Up" and "rho_partial" in corr.columns
    marks = pd.read_csv(tmp_path / "rsp_l23it_10x_penk_depth_markers.csv")
    assert len(marks) == 3 * len(apq.DEPTH_MARKERS)
    assert marks.loc[marks.gene == "Otof", "in_data"].all()


def test_part_a_continuum_reports_missing_blocks(tmp_path: Path) -> None:
    rsp = _tenx_cells(30, "RSP", 0, "r")
    r = apq.PartResult(part="A")
    ctx = apq.Context(outdir=tmp_path, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="gene correlations"):
        apq._part_a_continuum(ctx, r, rsp, pd.DataFrame(), [])
    assert "l23it_10x_penk_distribution.csv" in {o.file for o in r.outputs}


def test_part_b_continuum_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    n = 200
    z = rng.uniform(5.0, 6.4, n)
    cells = pd.DataFrame(
        {
            "cell_label": [f"c{i}" for i in range(n)],
            "dataset": rng.choice(["M1", "M2"], n),
            "rsp_subdivision": rng.choice(["RSPd", "RSPv"], n),
            "layer": rng.choice(["L1", "L2/3", "L5"], n, p=[0.1, 0.8, 0.1]),
            "subclass": "007 L2/3 IT CTX Glut",
            "supertype": rng.choice(["T1", "T2"], n),
            "cluster": rng.choice(["K1", "K2"], n),
            "x_ccf": rng.uniform(8, 10, n),
            "y_ccf": rng.uniform(1, 2, n),
            "z_ccf": z,
            "Penk": np.abs(z - 5.7) * 4,
        }
    )
    ctx = apq.Context(outdir=tmp_path, cache_dir=tmp_path)
    r = apq.PartResult(part="B")
    apq._part_b_continuum(ctx, r, cells)
    assert {o.file for o in r.outputs} == {
        "continuum_params_B.json",
        "rsp_merfish_l23it_penk_distribution.csv",
        "rsp_merfish_l23it_penk_histogram.csv",
        "rsp_merfish_l23it_penk_spatial_corr.csv",
        "rsp_merfish_l23it_penk_by_subdivision.csv",
        "rsp_merfish_l23it_penk_kruskal.csv",
        "rsp_merfish_l23it_penk_map.csv.gz",
    }
    sc = pd.read_csv(tmp_path / "rsp_merfish_l23it_penk_spatial_corr.csv")
    ml = sc[(sc.variable == "ccf_ml_from_midline") & sc["rsp_subdivision"].isna()]
    assert len(ml) == 2 and (ml["rho"] > 0.99).all()
    assert set(sc["axis_assumed"].dropna()) >= {"CCF left-right", "CCF layer (L1=1 ... L6b=6)"}
    kw = pd.read_csv(tmp_path / "rsp_merfish_l23it_penk_kruskal.csv")
    assert set(kw["dataset"]) == {"M1", "M2"}
    m = pd.read_csv(tmp_path / "rsp_merfish_l23it_penk_map.csv.gz")
    assert len(m) == n and "Penk" in m.columns


def test_part_c_continuum_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 40
    cpm = np.where(np.arange(n) < 20, rng.uniform(10, 1000, n), 0.0)
    cells = pd.DataFrame(
        {
            "dataset": ["scala2021_mop"] * n + ["allen_ctdb_cre"] * 3,
            "ttype_group": ["L2/3 IT"] * 30 + ["L5 IT"] * 10 + ["layer 2/3"] * 3,
            "penk_cpm": np.r_[cpm, [np.nan] * 3],
            "penk_pos_nonzero": np.r_[cpm > 0, [np.nan] * 3],
            "adaptation_index": np.r_[np.log1p(cpm) + rng.normal(0, 0.1, n), [1.0] * 3],
            "rheobase": rng.uniform(50, 300, n + 3),
        }
    )
    ctx = apq.Context(outdir=tmp_path, cache_dir=tmp_path)
    r = apq.PartResult(part="C")
    apq._part_c_continuum(ctx, r, cells, ["adaptation_index", "rheobase"])
    assert {o.file for o in r.outputs} == {
        "continuum_params_C.json",
        "patchseq_penk_ephys_spearman.csv",
        "patchseq_penk_ephys_shift.csv",
        "patchseq_penk_distribution.csv",
        "patchseq_penk_histogram.csv",
    }
    sp_ = pd.read_csv(tmp_path / "patchseq_penk_ephys_spearman.csv")
    assert set(sp_["dataset"]) == {"scala2021_mop"}
    row = sp_[(sp_.ttype_group == "L2/3 IT") & (sp_.variable == "adaptation_index")].iloc[0]
    assert row["rho"] > 0.8 and row["n"] == 30
    sh = pd.read_csv(tmp_path / "patchseq_penk_ephys_shift.csv")
    assert set(sh["ttype_group"]) == {"L2/3 IT", "ALL"}  # L5 IT has no Penk+ cells


def test_part_c_continuum_without_penk(tmp_path: Path) -> None:
    r = apq.PartResult(part="C")
    ctx = apq.Context(outdir=tmp_path, cache_dir=tmp_path)
    apq._part_c_continuum(ctx, r, pd.DataFrame({"dataset": ["x"]}), ["rheobase"])
    assert r.outputs == [] and any("skipped" in n for n in r.notes)
