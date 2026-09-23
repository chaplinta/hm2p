"""Tests for hm2p.patching.lr_classify — LR/RS classification of RSC L2/3 cells.

All data are synthetic. "LR-like" cells are given high input resistance, low
rheobase, narrow spikes and high maximum spike counts; "RS-like" cells the
opposite, following Brennan et al. 2020 (Cell Reports 30:1598-1612,
doi:10.1016/j.celrep.2019.12.093).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hm2p.patching import lr_classify as lr_mod
from hm2p.patching.lr_classify import (
    AMBIGUOUS_CLASS,
    BRENNAN_2020_REFERENCE,
    LR_CLASS,
    LR_CRITERIA,
    RS_CLASS,
    LRCriteria,
    ReferenceRanges,
    _criterion_met,
    _interspike_intervals,
    _lr_fraction,
    adaptation_from_efel,
    adaptation_index,
    classify_lr_rs,
    compare_to_reference,
    data_driven_lr_axis,
    enrichment_test,
    first_spike_latency_s,
    spike_frequency_adaptation_ratio,
    with_thresholds,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_patching_df(seed: int = 0, n_per_type: int = 3, lr_effect: bool = True) -> pd.DataFrame:
    """Synthetic patching table: 6 animals, both cell types in each animal.

    With ``lr_effect=True`` the penkpos cells are LR-like and the penkneg cells
    RS-like; with ``lr_effect=False`` both types are RS-like.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for animal in range(6):
        animal_offset = rng.normal(0.0, 5.0)
        for cell_type in ("penkpos", "penkneg"):
            lr_like = lr_effect and cell_type == "penkpos"
            for k in range(n_per_type):
                rows.append(
                    {
                        "cell_index": len(rows) + 1,
                        "animal_id": f"CAA-{animal}",
                        "cell_type": cell_type,
                        "ephys_passive_rin": (
                            (250.0 if lr_like else 90.0) + animal_offset + rng.normal(0, 8)
                        ),
                        "ephys_passive_rhreo": ((50.0 if lr_like else 180.0) + rng.normal(0, 6)),
                        "ephys_passive_maxsp": (25.0 if lr_like else 8.0) + rng.normal(0, 1.5),
                        "ephys_passive_tau": 15.0 + rng.normal(0, 2),
                        "ephys_active_halfWidth": (
                            (1.2 if lr_like else 2.4) + rng.normal(0, 0.05)
                        ),
                        "morph_api_len": 1500.0 + rng.normal(0, 100) + (200.0 if lr_like else 0.0),
                        "cell_slice_id": k,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def patching_df() -> pd.DataFrame:
    return make_patching_df()


# ---------------------------------------------------------------------------
# LRCriteria / thresholds
# ---------------------------------------------------------------------------


class TestCriteria:
    def test_default_rules_shape(self) -> None:
        rules = LR_CRITERIA.rules()
        assert len(rules) == 5
        names = [r[0] for r in rules]
        assert names == ["rin", "rheobase", "half_width", "adaptation", "max_spikes"]
        assert all(r[2] in ("gt", "lt") for r in rules)

    def test_documented_default_thresholds(self) -> None:
        assert LR_CRITERIA.rin_min_mohm == 150.0
        assert LR_CRITERIA.rheobase_max_pa == 100.0
        assert LR_CRITERIA.half_width_max_ms == 2.0
        assert LR_CRITERIA.adaptation_index_max == 0.3
        assert LR_CRITERIA.max_spikes_min == 15.0
        assert "Brennan" in LR_CRITERIA.source

    def test_with_thresholds_returns_copy(self) -> None:
        custom = with_thresholds(rin_min_mohm=300.0)
        assert custom.rin_min_mohm == 300.0
        assert custom.rheobase_max_pa == LR_CRITERIA.rheobase_max_pa
        # The module-level default is frozen and untouched.
        assert LR_CRITERIA.rin_min_mohm == 150.0

    def test_with_thresholds_unknown_field_raises(self) -> None:
        with pytest.raises(TypeError):
            with_thresholds(not_a_field=1.0)

    def test_reference_ranges_default_is_empty(self) -> None:
        empty = ReferenceRanges()
        assert empty.ranges == {}
        assert empty.source == ""


class TestCriterionMet:
    def test_greater_than(self) -> None:
        out = _criterion_met(pd.Series([200.0, 100.0]), "gt", 150.0)
        assert list(out) == [True, False]

    def test_less_than(self) -> None:
        out = _criterion_met(pd.Series([50.0, 250.0]), "lt", 100.0)
        assert list(out) == [True, False]

    def test_nan_is_not_evaluable(self) -> None:
        out = _criterion_met(pd.Series([np.nan, 200.0]), "gt", 150.0)
        assert out.iloc[0] is pd.NA
        assert out.iloc[1] is True

    def test_non_numeric_is_not_evaluable(self) -> None:
        out = _criterion_met(pd.Series(["bad", "200"]), "gt", 150.0)
        assert out.iloc[0] is pd.NA
        assert out.iloc[1] is True

    def test_bad_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _criterion_met(pd.Series([1.0]), "eq", 1.0)


# ---------------------------------------------------------------------------
# classify_lr_rs
# ---------------------------------------------------------------------------


class TestClassifyLrRs:
    def test_lr_and_rs_calls(self, patching_df: pd.DataFrame) -> None:
        out = classify_lr_rs(patching_df)
        penkpos = out.loc[out["cell_type"] == "penkpos", "lr_class"]
        penkneg = out.loc[out["cell_type"] == "penkneg", "lr_class"]
        assert (penkpos == LR_CLASS).all()
        assert (penkneg == RS_CLASS).all()

    def test_score_and_evaluable_columns(self, patching_df: pd.DataFrame) -> None:
        out = classify_lr_rs(patching_df)
        # adaptation_index column is absent → 4 evaluable criteria.
        assert (out["lr_n_evaluable"] == 4).all()
        assert out.loc[out["cell_type"] == "penkpos", "lr_score"].eq(4).all()
        assert out.loc[out["cell_type"] == "penkneg", "lr_score"].eq(0).all()
        assert out.loc[out["cell_type"] == "penkpos", "lr_fraction_met"].eq(1.0).all()

    def test_adaptation_column_used_when_present(self) -> None:
        df = pd.DataFrame(
            {
                "ephys_passive_rin": [300.0],
                "ephys_passive_rhreo": [40.0],
                "ephys_active_halfWidth": [1.0],
                "ephys_passive_maxsp": [30.0],
                "adaptation_index": [0.05],
            }
        )
        out = classify_lr_rs(df)
        assert out.loc[0, "lr_n_evaluable"] == 5
        assert out.loc[0, "lr_score"] == 5

    def test_ambiguous_when_too_few_evaluable(self) -> None:
        df = pd.DataFrame({"ephys_passive_rin": [300.0], "ephys_passive_rhreo": [np.nan]})
        out = classify_lr_rs(df)
        assert out.loc[0, "lr_class"] == AMBIGUOUS_CLASS
        assert out.loc[0, "lr_n_evaluable"] == 1

    def test_ambiguous_when_criteria_conflict(self) -> None:
        df = pd.DataFrame(
            {
                "ephys_passive_rin": [300.0],  # LR
                "ephys_passive_rhreo": [40.0],  # LR
                "ephys_active_halfWidth": [2.5],  # RS
                "ephys_passive_maxsp": [5.0],  # RS
            }
        )
        out = classify_lr_rs(df)
        assert out.loc[0, "lr_score"] == 2
        assert out.loc[0, "lr_class"] == AMBIGUOUS_CLASS

    def test_fraction_nan_when_nothing_evaluable(self) -> None:
        df = pd.DataFrame({"ephys_passive_rin": [np.nan]})
        out = classify_lr_rs(df)
        assert np.isnan(out.loc[0, "lr_fraction_met"])
        assert out.loc[0, "lr_class"] == AMBIGUOUS_CLASS

    def test_custom_criteria_change_the_call(self, patching_df: pd.DataFrame) -> None:
        strict = with_thresholds(rin_min_mohm=1e6, rheobase_max_pa=0.0, max_spikes_min=1e6)
        out = classify_lr_rs(patching_df, criteria=strict)
        assert (out["lr_class"] != LR_CLASS).all()

    def test_min_criteria_met_controls_calls(self, patching_df: pd.DataFrame) -> None:
        out = classify_lr_rs(patching_df, min_criteria_met=4)
        assert (out.loc[out["cell_type"] == "penkpos", "lr_class"] == LR_CLASS).all()
        out5 = classify_lr_rs(patching_df, min_criteria_met=5)
        assert (out5["lr_class"] == AMBIGUOUS_CLASS).all()

    def test_input_not_mutated(self, patching_df: pd.DataFrame) -> None:
        before = list(patching_df.columns)
        classify_lr_rs(patching_df)
        assert list(patching_df.columns) == before

    def test_bad_min_criteria_raises(self, patching_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="min_criteria_met"):
            classify_lr_rs(patching_df, min_criteria_met=0)

    def test_no_criterion_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="none of the LR criterion columns"):
            classify_lr_rs(pd.DataFrame({"foo": [1.0]}))

    @settings(deadline=None, max_examples=25)
    @given(
        rin=st.floats(0, 1000),
        rheo=st.floats(0, 1000),
        half_width=st.floats(0.1, 5.0),
        max_sp=st.floats(0, 100),
    )
    def test_score_never_exceeds_evaluable(
        self, rin: float, rheo: float, half_width: float, max_sp: float
    ) -> None:
        df = pd.DataFrame(
            {
                "ephys_passive_rin": [rin],
                "ephys_passive_rhreo": [rheo],
                "ephys_active_halfWidth": [half_width],
                "ephys_passive_maxsp": [max_sp],
            }
        )
        out = classify_lr_rs(df)
        assert 0 <= out.loc[0, "lr_score"] <= out.loc[0, "lr_n_evaluable"] == 4
        assert out.loc[0, "lr_class"] in (LR_CLASS, RS_CLASS, AMBIGUOUS_CLASS)


# ---------------------------------------------------------------------------
# Spike-train descriptors
# ---------------------------------------------------------------------------


class TestInterspikeIntervals:
    def test_sorted_diffs(self) -> None:
        isis = _interspike_intervals([0.3, 0.1, 0.2])
        assert np.allclose(isis, [0.1, 0.1])

    def test_too_few_spikes_gives_empty(self) -> None:
        assert _interspike_intervals([0.1]).size == 0
        assert _interspike_intervals([]).size == 0

    def test_non_finite_dropped(self) -> None:
        isis = _interspike_intervals([0.0, np.nan, 0.5, np.inf])
        assert np.allclose(isis, [0.5])


class TestAdaptationIndex:
    def test_regular_train_is_zero(self) -> None:
        assert adaptation_index([0.0, 0.1, 0.2, 0.3]) == pytest.approx(0.0)

    def test_slowing_train_is_positive(self) -> None:
        assert adaptation_index([0.0, 0.1, 0.3, 0.7]) > 0

    def test_accelerating_train_is_negative(self) -> None:
        assert adaptation_index([0.0, 0.4, 0.6, 0.7]) < 0

    def test_known_value(self) -> None:
        # ISIs 0.1, 0.3 → (0.3-0.1)/(0.3+0.1) = 0.5
        assert adaptation_index([0.0, 0.1, 0.4]) == pytest.approx(0.5)

    def test_fewer_than_three_spikes_is_nan(self) -> None:
        assert np.isnan(adaptation_index([0.0, 0.1]))
        assert np.isnan(adaptation_index([]))

    def test_zero_isi_pair_is_nan(self) -> None:
        assert np.isnan(adaptation_index([0.0, 0.0, 0.0, 0.5]))

    @settings(deadline=None, max_examples=25)
    @given(
        st.lists(st.floats(0.001, 1.0), min_size=3, max_size=20),
    )
    def test_bounded_by_one(self, intervals: list[float]) -> None:
        times = np.concatenate([[0.0], np.cumsum(intervals)])
        value = adaptation_index(times)
        assert -1.0 <= value <= 1.0


class TestFirstSpikeLatency:
    def test_basic_latency(self) -> None:
        assert first_spike_latency_s([0.25, 0.4], 0.2) == pytest.approx(0.05)

    def test_spikes_before_onset_ignored(self) -> None:
        assert first_spike_latency_s([0.05, 0.5], 0.2) == pytest.approx(0.3)

    def test_no_spike_after_onset_is_nan(self) -> None:
        assert np.isnan(first_spike_latency_s([0.05], 0.2))
        assert np.isnan(first_spike_latency_s([], 0.2))

    def test_non_finite_onset_is_nan(self) -> None:
        assert np.isnan(first_spike_latency_s([0.3], np.nan))


class TestAdaptationRatio:
    def test_regular_train_is_one(self) -> None:
        assert spike_frequency_adaptation_ratio([0.0, 0.1, 0.2, 0.3]) == pytest.approx(1.0)

    def test_adapting_train_above_one(self) -> None:
        assert spike_frequency_adaptation_ratio([0.0, 0.1, 0.5]) == pytest.approx(4.0)

    def test_fewer_than_three_spikes_is_nan(self) -> None:
        assert np.isnan(spike_frequency_adaptation_ratio([0.0, 0.2]))

    def test_zero_first_isi_is_nan(self) -> None:
        assert np.isnan(spike_frequency_adaptation_ratio([0.0, 0.0, 0.5]))


class TestAdaptationFromEfel:
    def test_reads_mean_of_array(self) -> None:
        assert adaptation_from_efel({"adaptation_index2": [0.1, 0.3]}) == pytest.approx(0.2)

    def test_scalar_value(self) -> None:
        assert adaptation_from_efel({"adaptation_index2": 0.25}) == pytest.approx(0.25)

    def test_missing_key_is_nan(self) -> None:
        assert np.isnan(adaptation_from_efel({}))

    def test_none_value_is_nan(self) -> None:
        assert np.isnan(adaptation_from_efel({"adaptation_index2": None}))

    def test_all_nan_is_nan(self) -> None:
        assert np.isnan(adaptation_from_efel({"adaptation_index2": [np.nan, np.nan]}))

    def test_empty_array_is_nan(self) -> None:
        assert np.isnan(adaptation_from_efel({"adaptation_index2": []}))


# ---------------------------------------------------------------------------
# data_driven_lr_axis
# ---------------------------------------------------------------------------

AXIS_COLS = [
    "ephys_passive_rin",
    "ephys_passive_rhreo",
    "ephys_passive_maxsp",
    "ephys_active_halfWidth",
]


class TestDataDrivenLrAxis:
    def test_keys_and_shapes(self, patching_df: pd.DataFrame) -> None:
        out = data_driven_lr_axis(patching_df, AXIS_COLS)
        assert out["n_cells"] == len(patching_df)
        assert set(out["pc1_loadings"]) == set(AXIS_COLS)
        assert len(out["explained_variance_ratio"]) == 2
        assert out["pc1_scores"].shape == (len(patching_df),)
        assert out["pc2_scores"].shape == (len(patching_df),)
        assert set(out["pc1_by_group"]) == {"penkpos", "penkneg"}

    def test_pc1_separates_the_two_types(self, patching_df: pd.DataFrame) -> None:
        out = data_driven_lr_axis(patching_df, AXIS_COLS)
        medians = out["pc1_median_by_group"]
        assert abs(medians["penkpos"] - medians["penkneg"]) > 1.0
        assert out["bimodal"] is True
        assert out["bic_delta"] > 0

    def test_unimodal_data_not_flagged_bimodal(self) -> None:
        rng = np.random.default_rng(3)
        n = 60
        df = pd.DataFrame(
            {
                "cell_type": ["penkpos"] * (n // 2) + ["penkneg"] * (n // 2),
                **{col: rng.normal(0, 1, n) for col in AXIS_COLS},
            }
        )
        out = data_driven_lr_axis(df, AXIS_COLS)
        assert out["bimodal"] is False

    def test_missing_group_column_uses_single_group(self, patching_df: pd.DataFrame) -> None:
        out = data_driven_lr_axis(patching_df[AXIS_COLS], AXIS_COLS)
        assert list(out["pc1_by_group"]) == ["all"]

    def test_constant_column_does_not_produce_nan(self, patching_df: pd.DataFrame) -> None:
        df = patching_df.copy()
        df["ephys_passive_rin"] = 100.0
        out = data_driven_lr_axis(df, AXIS_COLS)
        assert np.all(np.isfinite(out["pc1_scores"]))

    def test_too_few_cells_for_mixture(self) -> None:
        df = pd.DataFrame({col: [1.0, 2.0, 4.0] for col in AXIS_COLS})
        out = data_driven_lr_axis(df, AXIS_COLS)
        assert np.isnan(out["bic_1"])
        assert out["bimodal"] is False

    def test_rows_with_nan_metrics_dropped(self, patching_df: pd.DataFrame) -> None:
        df = patching_df.copy()
        df.loc[0, "ephys_passive_rin"] = np.nan
        out = data_driven_lr_axis(df, AXIS_COLS)
        assert out["n_cells"] == len(df) - 1

    def test_empty_metric_cols_raises(self, patching_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="metric_cols"):
            data_driven_lr_axis(patching_df, [])

    def test_missing_metric_col_raises(self, patching_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            data_driven_lr_axis(patching_df, ["nope"])

    def test_too_few_rows_raises(self) -> None:
        df = pd.DataFrame({col: [1.0, 2.0] for col in AXIS_COLS})
        with pytest.raises(ValueError, match="at least 3 complete cases"):
            data_driven_lr_axis(df, AXIS_COLS)


# ---------------------------------------------------------------------------
# enrichment_test
# ---------------------------------------------------------------------------


class TestLrFraction:
    def test_fraction(self) -> None:
        is_lr = np.array([True, False, True, True])
        is_group = np.array([True, True, False, False])
        assert _lr_fraction(is_lr, is_group) == pytest.approx(0.5)

    def test_empty_group_is_nan(self) -> None:
        assert np.isnan(_lr_fraction(np.array([True]), np.array([False])))


class TestEnrichmentTest:
    def test_strong_enrichment_detected(self, patching_df: pd.DataFrame) -> None:
        classified = classify_lr_rs(patching_df)
        out = enrichment_test(classified, n_perms=500, rng=0)
        assert out["groups"] == ["penkpos", "penkneg"]
        assert out["frac_lr_penkpos"] == pytest.approx(1.0)
        assert out["frac_lr_penkneg"] == pytest.approx(0.0)
        assert out["observed_diff"] == pytest.approx(1.0)
        assert out["p_perm"] < 0.01
        assert out["fisher_p_descriptive"] < 0.01
        assert out["n_animals"] == 6
        assert out["n_cells"] == len(patching_df)

    def test_no_enrichment_gives_large_p(self) -> None:
        df = make_patching_df(seed=1, lr_effect=False)
        classified = classify_lr_rs(df)
        # Make half of each type LR-like so the classes are independent of type.
        classified.loc[classified.index % 2 == 0, "lr_class"] = LR_CLASS
        out = enrichment_test(classified, n_perms=500, rng=0)
        assert out["p_perm"] > 0.05

    def test_reproducible_with_seed(self, patching_df: pd.DataFrame) -> None:
        classified = classify_lr_rs(patching_df)
        a = enrichment_test(classified, n_perms=200, rng=7)
        b = enrichment_test(classified, n_perms=200, rng=7)
        assert a["p_perm"] == b["p_perm"]

    def test_counts_are_consistent(self, patching_df: pd.DataFrame) -> None:
        classified = classify_lr_rs(patching_df)
        out = enrichment_test(classified, n_perms=100, rng=0)
        assert out["n_penkpos"] + out["n_penkneg"] == out["n_cells"]
        assert out["n_lr_penkpos"] <= out["n_penkpos"]

    def test_missing_column_raises(self, patching_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="lr_class"):
            enrichment_test(patching_df, n_perms=10)

    def test_bad_n_perms_raises(self, patching_df: pd.DataFrame) -> None:
        classified = classify_lr_rs(patching_df)
        with pytest.raises(ValueError, match="n_perms"):
            enrichment_test(classified, n_perms=0)

    def test_three_groups_raises(self, patching_df: pd.DataFrame) -> None:
        classified = classify_lr_rs(patching_df)
        classified.loc[0, "cell_type"] = "other"
        with pytest.raises(ValueError, match="exactly 2 groups"):
            enrichment_test(classified, n_perms=10)

    def test_empty_group_gives_nan_p(self) -> None:
        df = pd.DataFrame(
            {
                "lr_class": [LR_CLASS, RS_CLASS],
                "cell_type": ["penkpos", "penkneg"],
                "animal_id": ["a", "a"],
            }
        )
        out = enrichment_test(df, n_perms=10, rng=0)
        assert np.isfinite(out["observed_diff"])
        assert 0 < out["p_perm"] <= 1


# ---------------------------------------------------------------------------
# compare_to_reference
# ---------------------------------------------------------------------------


class TestCompareToReference:
    def test_default_reference_flags_lr_group(self, patching_df: pd.DataFrame) -> None:
        out = compare_to_reference(patching_df)
        rin = out[out["metric"] == "ephys_passive_rin"].set_index("group")
        assert rin.loc["penkpos", "verdict"] == LR_CLASS
        assert rin.loc["penkneg", "verdict"] == RS_CLASS

    def test_columns_and_row_count(self, patching_df: pd.DataFrame) -> None:
        out = compare_to_reference(patching_df)
        expected_metrics = {
            m for m in BRENNAN_2020_REFERENCE.ranges[LR_CLASS] if m in patching_df.columns
        }
        assert set(out["metric"]) == expected_metrics
        assert len(out) == 2 * len(expected_metrics)
        for col in ("n", "median", "lr_low", "rs_high", "in_lr_range", "in_rs_range", "verdict"):
            assert col in out.columns

    def test_overlapping_ranges_give_both(self) -> None:
        df = pd.DataFrame({"cell_type": ["penkpos", "penkpos"], "m": [1.0, 1.0]})
        ref = {LR_CLASS: {"m": (0.0, 2.0)}, RS_CLASS: {"m": (0.5, 3.0)}}
        out = compare_to_reference(df, reference=ref)
        assert out.loc[0, "verdict"] == "both"

    def test_value_outside_both_ranges(self) -> None:
        df = pd.DataFrame({"cell_type": ["penkpos"], "m": [99.0]})
        ref = {LR_CLASS: {"m": (0.0, 2.0)}, RS_CLASS: {"m": (2.0, 3.0)}}
        out = compare_to_reference(df, reference=ref)
        assert out.loc[0, "verdict"] == "neither"

    def test_reference_ranges_object_accepted(self) -> None:
        df = pd.DataFrame({"cell_type": ["penkpos"], "m": [1.0]})
        ref = ReferenceRanges(ranges={LR_CLASS: {"m": (0.0, 2.0)}}, source="synthetic")
        out = compare_to_reference(df, reference=ref)
        assert out.loc[0, "verdict"] == LR_CLASS

    def test_metric_cols_filter(self, patching_df: pd.DataFrame) -> None:
        out = compare_to_reference(patching_df, metric_cols=["ephys_passive_rin", "absent"])
        assert set(out["metric"]) == {"ephys_passive_rin"}

    def test_all_nan_group_median_is_nan(self) -> None:
        df = pd.DataFrame({"cell_type": ["penkpos"], "ephys_passive_rin": [np.nan]})
        out = compare_to_reference(df)
        assert np.isnan(out.loc[0, "median"])
        assert out.loc[0, "verdict"] == "neither"

    def test_missing_group_col_raises(self, patching_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="group_col"):
            compare_to_reference(patching_df.drop(columns=["cell_type"]))

    def test_no_reference_metrics_present_raises(self) -> None:
        df = pd.DataFrame({"cell_type": ["penkpos"], "other": [1.0]})
        with pytest.raises(ValueError, match="no reference metric columns"):
            compare_to_reference(df)

    def test_reference_source_is_cited_and_caveated(self) -> None:
        assert "Brennan" in BRENNAN_2020_REFERENCE.source
        assert "10.1016/j.celrep.2019.12.093" in BRENNAN_2020_REFERENCE.source
        assert "UNVERIFIED" in BRENNAN_2020_REFERENCE.source

    def test_criteria_dataclass_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            LRCriteria().rin_min_mohm = 1.0  # type: ignore[misc]


class TestEnrichmentDegenerateNull:
    def test_nan_fractions_give_nan_p(self, patching_df: pd.DataFrame, monkeypatch) -> None:
        """Defensive path: undefined LR fractions propagate as NaN, not an error."""
        classified = classify_lr_rs(patching_df)
        monkeypatch.setattr(lr_mod, "_lr_fraction", lambda is_lr, is_group: float("nan"))
        out = enrichment_test(classified, n_perms=10, rng=0)
        assert np.isnan(out["observed_diff"])
        assert np.isnan(out["p_perm"])
        assert np.isnan(out["null_median"])
