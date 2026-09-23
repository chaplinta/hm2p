"""Tests for ``scripts/run_patching_celltype.py`` — cell-type programme runner.

All data are synthetic: a small table with 6 animals, both cell types in each
animal, and penkpos cells given LR-like ephys values (Brennan et al. 2020,
Cell Reports 30:1598-1612, doi:10.1016/j.celrep.2019.12.093).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import run_patching_celltype as rpc  # noqa: E402


def make_metrics_df(seed: int = 0, n_per_type: int = 3) -> pd.DataFrame:
    """Synthetic metrics table shaped like results/patching/analysis/metrics.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for animal in range(6):
        offset = rng.normal(0.0, 5.0)
        for cell_type in ("penkpos", "penkneg"):
            lr_like = cell_type == "penkpos"
            for k in range(n_per_type):
                rows.append(
                    {
                        "cell_index": len(rows) + 1,
                        "animal_id": f"CAA-{animal}",
                        "slice_id": f"S{k}",
                        "cell_type": cell_type,
                        "hemisphere": "L",
                        "layer": "L23",
                        "ephys_passive_rin": (250.0 if lr_like else 90.0)
                        + offset
                        + rng.normal(0, 8),
                        "ephys_passive_rhreo": (50.0 if lr_like else 180.0) + rng.normal(0, 6),
                        "ephys_passive_maxsp": (25.0 if lr_like else 8.0) + rng.normal(0, 1.5),
                        "ephys_passive_tau": 15.0 + rng.normal(0, 2),
                        "ephys_active_halfWidth": (1.2 if lr_like else 2.4) + rng.normal(0, 0.05),
                        "morph_api_len": 1500.0 + rng.normal(0, 100),
                        "morph_bas_len": 900.0 + rng.normal(0, 80),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def metrics_df() -> pd.DataFrame:
    return make_metrics_df()


@pytest.fixture()
def metrics_csv(tmp_path: Path, metrics_df: pd.DataFrame) -> Path:
    path = tmp_path / "metrics.csv"
    metrics_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


class TestMetricColumns:
    def test_ephys_columns_selected(self, metrics_df: pd.DataFrame) -> None:
        cols = rpc.metric_columns(metrics_df, rpc.EPHYS_PREFIXES)
        assert "ephys_passive_rin" in cols
        assert "morph_api_len" not in cols
        assert "cell_type" not in cols

    def test_morph_columns_selected(self, metrics_df: pd.DataFrame) -> None:
        cols = rpc.metric_columns(metrics_df, rpc.MORPH_PREFIXES)
        assert set(cols) == {"morph_api_len", "morph_bas_len"}

    def test_non_numeric_columns_excluded(self, metrics_df: pd.DataFrame) -> None:
        df = metrics_df.copy()
        df["ephys_passive_note"] = "text"
        assert "ephys_passive_note" not in rpc.metric_columns(df, rpc.EPHYS_PREFIXES)

    def test_no_match_gives_empty(self, metrics_df: pd.DataFrame) -> None:
        assert rpc.metric_columns(metrics_df, ("nothing_",)) == []


class TestMissingColumns:
    def test_reports_absent(self, metrics_df: pd.DataFrame) -> None:
        assert rpc.missing_columns(metrics_df, ["cell_type", "nope"]) == ["nope"]

    def test_empty_when_all_present(self, metrics_df: pd.DataFrame) -> None:
        assert rpc.missing_columns(metrics_df, ["cell_type", "animal_id"]) == []


class TestValidateInput:
    def test_counts_and_groups(self, metrics_df: pd.DataFrame) -> None:
        info = rpc.validate_input(metrics_df)
        assert info["n_cells"] == len(metrics_df)
        assert info["n_animals"] == 6
        assert info["groups"] == ["penkpos", "penkneg"]
        assert info["group_counts"] == {"penkpos": 18, "penkneg": 18}
        assert "ephys_passive_rin" in info["ephys_cols"]
        assert info["morph_cols"] == ["morph_api_len", "morph_bas_len"]
        assert "ephys_passive_rin" in info["lr_axis_cols"]
        assert info["missing"] == []

    def test_missing_required_column_raises(self, metrics_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            rpc.validate_input(metrics_df.drop(columns=["animal_id"]))


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------


class TestJsonSafe:
    def test_numpy_scalars(self) -> None:
        out = rpc._json_safe({"a": np.float64(1.5), "b": np.int64(3), "c": np.bool_(True)})
        assert out == {"a": 1.5, "b": 3, "c": True}
        assert json.dumps(out)

    def test_nan_becomes_none(self) -> None:
        assert rpc._json_safe(float("nan")) is None
        assert rpc._json_safe(np.inf) is None

    def test_nested_containers_and_arrays(self) -> None:
        out = rpc._json_safe({"x": [np.float64(1.0), (np.int64(2),)], "y": np.arange(2)})
        assert out == {"x": [1.0, [2]], "y": [0, 1]}

    def test_passthrough_types(self) -> None:
        assert rpc._json_safe("text") == "text"
        assert rpc._json_safe(None) is None


# ---------------------------------------------------------------------------
# Mixed model wrapper
# ---------------------------------------------------------------------------


class TestRunMixedModel:
    def test_skips_gracefully_without_statsmodels(
        self, metrics_df: pd.DataFrame, monkeypatch
    ) -> None:
        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'statsmodels'")

        monkeypatch.setattr("hm2p.patching.statistics.mixed_model_comparison", raise_import_error)
        assert rpc.run_mixed_model(metrics_df, ["ephys_passive_rin"]) is None

    def test_returns_table_when_available(self, metrics_df: pd.DataFrame) -> None:
        pytest.importorskip("statsmodels")
        out = rpc.run_mixed_model(metrics_df, ["ephys_passive_rin"])
        assert out is not None
        assert "lmm_p_supplementary" in out.columns


# ---------------------------------------------------------------------------
# run_analysis / write_outputs / format_report
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results() -> dict:
    return rpc.run_analysis(make_metrics_df(), n_perms=200, seed=0)


class TestRunAnalysis:
    def test_keys_present(self, results: dict) -> None:
        for key in (
            "info",
            "summary",
            "mann_whitney",
            "cluster_permutation",
            "animal_level",
            "classified",
            "enrichment",
            "lr_axis",
            "reference",
        ):
            assert key in results

    def test_metrics_covered_in_every_table(self, results: dict) -> None:
        n_metrics = len(results["info"]["ephys_cols"]) + len(results["info"]["morph_cols"])
        assert len(results["summary"]) == n_metrics
        assert len(results["mann_whitney"]) == n_metrics
        assert len(results["cluster_permutation"]) == n_metrics
        assert len(results["animal_level"]) == n_metrics

    def test_lr_effect_recovered(self, results: dict) -> None:
        enrich = results["enrichment"]
        assert enrich["frac_lr_penkpos"] > enrich["frac_lr_penkneg"]
        assert enrich["p_perm"] < 0.05
        rin = results["cluster_permutation"].set_index("metric").loc["ephys_passive_rin"]
        assert rin["observed_diff"] > 0
        assert rin["p_perm"] < 0.05

    def test_classification_columns_added(self, results: dict) -> None:
        assert "lr_class" in results["classified"].columns
        assert "lr_score" in results["classified"].columns


class TestWriteOutputs:
    def test_files_written(self, results: dict, tmp_path: Path) -> None:
        written = rpc.write_outputs(results, tmp_path / "out")
        names = {p.name for p in written}
        assert {
            "summary_stats.csv",
            "mann_whitney_cell_level.csv",
            "cluster_permutation.csv",
            "animal_level.csv",
            "lr_classification.csv",
            "reference_comparison.csv",
            "lr_enrichment.json",
            "lr_axis.json",
            "report.md",
        } <= names
        assert all(p.exists() for p in written)

    def test_json_is_valid_and_axis_arrays_dropped(self, results: dict, tmp_path: Path) -> None:
        rpc.write_outputs(results, tmp_path)
        enrich = json.loads((tmp_path / "lr_enrichment.json").read_text())
        assert enrich["groups"] == ["penkpos", "penkneg"]
        axis = json.loads((tmp_path / "lr_axis.json").read_text())
        assert "pc1_scores" not in axis
        assert "pc1_loadings" in axis

    def test_mixed_model_file_only_when_fitted(self, results: dict, tmp_path: Path) -> None:
        no_lmm = dict(results, mixed_model=None)
        written = rpc.write_outputs(no_lmm, tmp_path / "a")
        assert not any(p.name == "mixed_model_supplementary.csv" for p in written)

        with_lmm = dict(results, mixed_model=pd.DataFrame({"metric": ["x"]}))
        written = rpc.write_outputs(with_lmm, tmp_path / "b")
        assert any(p.name == "mixed_model_supplementary.csv" for p in written)


class TestFormatReport:
    def test_contains_key_sections(self, results: dict) -> None:
        text = rpc.format_report(results)
        assert "# Penk+ vs Penk- cell-type programme" in text
        assert "Cluster permutation (primary" in text
        assert "Animal-level paired Wilcoxon" in text
        assert "descriptive, pseudoreplicated" in text
        assert "Brennan" in text
        assert "ephys_passive_rin" in text

    def test_notes_missing_mixed_model(self, results: dict) -> None:
        text = rpc.format_report(dict(results, mixed_model=None))
        assert "statsmodels not installed" in text

    def test_handles_empty_result_tables(self, results: dict) -> None:
        empty = dict(results, cluster_permutation=pd.DataFrame())
        assert "(no results)" in rpc.format_report(empty)

    def test_handles_all_nan_p_values(self, results: dict) -> None:
        nan_table = results["animal_level"].copy()
        nan_table["p_value"] = np.nan
        text = rpc.format_report(dict(results, animal_level=nan_table))
        assert "no p-values could be computed" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self) -> None:
        args = rpc.build_parser().parse_args([])
        assert args.input == rpc.DEFAULT_INPUT
        assert args.output == rpc.DEFAULT_OUTPUT
        assert args.n_perms == 5000
        assert args.seed == 0
        assert args.min_criteria == 3
        assert args.dry_run is False

    def test_overrides(self) -> None:
        args = rpc.build_parser().parse_args(
            ["--input", "a.csv", "--output", "out", "--n-perms", "10", "--dry-run"]
        )
        assert args.input == Path("a.csv")
        assert args.output == Path("out")
        assert args.n_perms == 10
        assert args.dry_run is True


class TestDescribeDryRun:
    def test_mentions_every_analysis(self, metrics_df: pd.DataFrame) -> None:
        args = rpc.build_parser().parse_args(["--dry-run", "--n-perms", "7"])
        text = rpc.describe_dry_run(rpc.validate_input(metrics_df), args)
        for name in (
            "compute_summary_stats",
            "mann_whitney_comparison",
            "cluster_permutation_comparison",
            "animal_level_comparison",
            "mixed_model_comparison",
            "classify_lr_rs",
            "enrichment_test",
            "data_driven_lr_axis",
            "compare_to_reference",
        ):
            assert name in text
        assert "permutations: 7" in text
        assert "nothing written" in text

    def test_reports_absent_axis_metrics(self, metrics_df: pd.DataFrame) -> None:
        df = metrics_df.drop(columns=list(rpc.LR_AXIS_COLS))
        args = rpc.build_parser().parse_args([])
        text = rpc.describe_dry_run(rpc.validate_input(df), args)
        assert "(none present)" in text


class TestMain:
    def test_dry_run_writes_nothing(self, metrics_csv: Path, tmp_path: Path, capsys) -> None:
        out_dir = tmp_path / "out"
        code = rpc.main(["--input", str(metrics_csv), "--output", str(out_dir), "--dry-run"])
        assert code == 0
        assert not out_dir.exists()
        assert "dry run: nothing written" in capsys.readouterr().out

    def test_full_run_writes_outputs(self, metrics_csv: Path, tmp_path: Path, capsys) -> None:
        out_dir = tmp_path / "out"
        code = rpc.main(
            [
                "--input",
                str(metrics_csv),
                "--output",
                str(out_dir),
                "--n-perms",
                "50",
                "--seed",
                "1",
            ]
        )
        assert code == 0
        assert (out_dir / "report.md").exists()
        assert (out_dir / "cluster_permutation.csv").exists()
        table = pd.read_csv(out_dir / "cluster_permutation.csv")
        assert "p_perm" in table.columns
        assert "cell-type programme" in capsys.readouterr().out

    def test_missing_input_returns_error_code(self, tmp_path: Path) -> None:
        assert rpc.main(["--input", str(tmp_path / "nope.csv")]) == 1

    def test_invalid_table_returns_error_code(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1.0]}).to_csv(bad, index=False)
        assert rpc.main(["--input", str(bad), "--output", str(tmp_path / "o")]) == 1
