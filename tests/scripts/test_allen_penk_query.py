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
