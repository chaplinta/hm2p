# Sync-pipeline diagnostics & report — Test Plan

_Status: design — not yet implemented._
_Companion to: `/workspace/docs/sync-pipeline-design.md` (the architect's spec)._
_Branch: `feat/soma-extraction-improvements` → recommended new branch `feat/sync-pipeline-diagnostics`._
_Audience: lead-developer implementing the 6-commit rollout in design §7._

This plan is precise enough that the lead-developer can implement tests
directly. It does **not** include the test code itself. Per
[CLAUDE.md](../../CLAUDE.md):

- Tests use small synthetic arrays only — never real data files. The single
  exception is the `@pytest.mark.golden` regression suite (§8) which is
  excluded from CI.
- `pytest` + `pytest-cov` + `hypothesis` for property tests +
  `pandera` for schema validation.
- Coverage target: ≥ 90 % across all new/modified modules in this work.
- Tests live under `tests/` mirroring `src/hm2p/`.

The test-engineer **does not write tests** — that is the lead-developer's
job, in commits 1–6 of the rollout.

---

## 1. Test scope per module

### 1.1 `src/hm2p/sync/diagnostics.py` (new) → `tests/sync/test_diagnostics.py` (new)

Pure-function module. No I/O. Target **100 % line coverage** — there are no
heavy/skip-conditional branches.

Functions to test (per design §1.1):
- `channel_scalars(times, fps_nominal, *, cfg) -> ChannelScalars`
- `cross_channel_scalars(cam, img, light_on, light_off, *, cfg) -> CrossChannelScalars`
- `drift_slope(times) -> tuple[float, float]`
- `classify(scalars, *, cfg) -> tuple[status, warnings, failures]`
- The internal `_CODE_LUT` (smoke-test it covers every code emitted by `classify`)

Unit-test scenarios (each its own `test_*` function):
- **Clean train**: 1000 pulses at exactly `fps`, jitter = 0 → CV ≈ 0,
  `n_isi_outliers == 0`, `drift_slope ≈ 0`, `min_isi == median_isi`.
- **Drifted train** (parametrize `drift_ppm ∈ {50, 100, 500, 1000, -200}`):
  recovered slope within ±5 % of input. R² > 0.999.
- **Single duplicated pulse**: `min_isi_ms ≈ 0`, `n_isi_outliers >= 1`.
- **Single missing pulse**: median unchanged, exactly 1 outlier with ISI ≈ 2 × median.
- **Empty array** (`times.shape == (0,)`): all int counts = 0, all floats = NaN
  (sentinel as per design §2.2). Function must not raise.
- **Single-element array**: `n_pulses == 1`, ISI undefined → NaN/-9999.
- **Two-element array**: median = MAD = single ISI, CV = 0.
- **Constant time array** (all zeros): all ISIs zero → CV = NaN (0/0 guarded).
- **Cross-channel temporal overlap**:
  - Identical span → `cross_overlap_s == max(duration)`.
  - Disjoint spans (cam ends before img starts) → `cross_overlap_s == 0`.
  - Partial overlap → exact analytical value.
- **Cross-channel start/end offsets**: cam zeroed to t=0; img starts at `δ` →
  `cross_start_offset_ms == δ × 1000`.
- **Light scalars**:
  - Regular 60-on / 60-off cycle → `light_period_median_s == 120`,
    `light_duty_cycle == 0.5`.
  - Empty light arrays → `light_first_state_at_t0 == -1`, period NaN.
  - Mismatched on/off counts (`|n_on − n_off| > 1`) → flagged.
  - First state inference: when `light_on[0] < light_off[0]`, infer that
    state at t=0 was `off` (lights came on at first edge). Symmetric inverse.

### 1.2 `src/hm2p/sync/align.py` (modified) → `tests/sync/test_align.py` (extend)

Existing tests stay (they exercise resample helpers + payload integration).
**Add** a new test class `TestStage5FailureClosedSemantics` covering:

- `run()` is given a missing `timestamps.h5` path → writes a stub `sync.h5`
  with `sync_status == "FAILED_NO_TIMESTAMPS"`, no resampled signals
  (`hd_deg`, `dff`, etc. absent), `sync_diag/` group present with sentinel
  scalars, `sync_warnings/sync_failures` JSON-encoded as string root attrs.
- `run()` with `cam_n_pulses == 0` (empty timestamps) → `FAILED_NO_PULSES`,
  stub written.
- `run()` with `pulse_count_diff_after_off_by_one > T_FRAME_COUNT_HARD` →
  `FAILED_FRAME_COUNT_MISMATCH`, stub written. Verify the failure reason
  string in `sync_failures` inlines the actual scalar.
- `run()` with cam/img streams disjoint → `FAILED_TEMPORAL_OVERLAP`, stub
  written.
- `run()` with cam_duration < 0.5 × img_duration → `FAILED_TRUNCATED_CAMERA`.
- `run()` happy path → status `OK`, **all** keys from the existing schema
  still written (regression: do not break existing payload).
- `run()` happy path with one warning condition (e.g. `cam_isi_cv = 0.025`)
  → status `OK_WITH_WARNINGS`, all payload keys still written, exactly one
  short code in `sync_warnings`, zero entries in `sync_failures`.
- `run()` writes `sync_status_version == "1.0"` regardless of outcome.
- `run()` reads `config/sync.yaml` if present; falls back to packaged
  defaults if missing (test by passing a non-existent config path).

### 1.3 `src/hm2p/sync/report.py` (new) → `tests/sync/test_report.py` (new)

Single function `build_report(sync_dir, output_path)`. Tests:

- **3-session tree**: write three synthetic `sync.h5` files (one `OK`, one
  `OK_WITH_WARNINGS`, one `FAILED_FRAME_COUNT_MISMATCH`); call
  `build_report`; assert parquet has 3 rows, correct `sync_status` per row,
  `sync_warnings`/`sync_failures` round-trip as JSON strings.
- **Empty input dir** (no `sync.h5` files): produces an empty parquet with
  the canonical column list and correct dtypes (validated via
  `validate_sync_report_parquet`).
- **One unreadable sync.h5** (truncated file): `read_error` column populated
  with the exception message, all other scalar columns NaN/sentinel; row
  not dropped.
- **`sync_diag/` missing** (legacy schema): treated as version `"0.0"` →
  `read_error == "schema version 0.0 — rebuild required"`; never silently
  succeed.
- **Column order is stable** — assert exact column list (regression
  protection for the parquet consumers).
- **Pandera schema** (`validate_sync_report_parquet`) catches: missing
  required column, wrong dtype on `sync_status`, non-string `sync_warnings`.
- **Aggregation invariants**:
  - `len(parquet) == len(glob('**/sync.h5'))`.
  - `parquet[parquet.sync_status.str.startswith('FAILED_')].shape[0] ==
    expected_failed_count` for the synthetic input.

### 1.4 `src/hm2p/ingest/daq.py` (modified) → `tests/ingest/test_daq.py` (extend)

Existing tests cover `parse_tdms()` and `run()`. **Add**:

- **`line_clock_times` written**: post-`run()`, the `timestamps.h5` contains
  a `line_clock_times` dataset of dtype `float64` with shape `(M,)` where
  `M = y_pix * n_imaging_frames + tdms_diag/sci_lines_truncated_n`.
- **`tdms_diag/` group populated**: assert all attrs from design §2.1 are
  present with correct dtypes and finite values for a synthetic TDMS-like
  input.
- **`sci_lines_truncated_n`**: when input has `len(line_idxs) % y_pix != 0`,
  the residual is recorded; when divisible, the attr is exactly 0.
- **`tdms_sample_rate_hz`**: matches `1 / wf_increment` from synthetic input.
- **Empty line clock raises**: `ValueError("No SciScan line-clock pulses")` —
  ingest **fails closed**, does not silently produce an empty file.
- **Empty cam-trigger raises**: existing behaviour preserved
  (`ValueError`).
- **Light-channel mismatch tolerated**: `len(light_on) != len(light_off)`
  no longer raises in ingest (sync stage classifies it instead). The
  resulting `timestamps.h5` is still written.
- **Non-saturated digital channel recorded**: `cam_min/cam_max` reflect the
  raw range of the channel; classification of "non-saturated" is the sync
  stage's job, ingest just records.

### 1.5 `src/hm2p/io/hdf5.py` (modified) → `tests/io/test_hdf5.py` (extend)

Add new validators / extend existing ones. Tests:

- `validate_timestamps_h5` accepts the new `line_clock_times` dataset and
  `tdms_diag/` group (positive case).
- `validate_timestamps_h5` raises `SchemaError("line_clock_times")` when
  the dataset is missing.
- `validate_timestamps_h5` raises when `line_clock_times` is not float64
  or not strictly increasing (consistent with existing `frame_times_*`
  rules).
- `validate_timestamps_h5` raises when `tdms_diag/` group is missing.
- `validate_sync_h5` accepts the new root attrs (`sync_status`,
  `sync_warnings`, `sync_failures`, `sync_status_version`) and the
  `sync_diag/` group.
- `validate_sync_h5` raises when `sync_status` is missing.
- `validate_sync_h5` raises when `sync_status` is not one of the 7 codes.
- `validate_sync_h5` raises when `sync_warnings` is not a string-encoded
  JSON list.
- `validate_sync_h5` accepts a stub (`FAILED_*`) sync.h5 missing the
  resampled-signal datasets — the new validator must permit missing
  `hd_deg`/`dff`/etc. when `sync_status.startswith("FAILED_")`.
- New `validate_sync_report_parquet`: positive test, missing-column raise,
  wrong-dtype raise.

### 1.6 `frontend/pages/sync_report_page.py` (new) → `tests/frontend/test_sync_report_page.py` (new)

Use `streamlit.testing.v1.AppTest` (the convention already used by every
file in `tests/frontend/`). Mock `frontend.data.load_sync_report`,
`load_session`, `is_sync_clean`, `get_dlc_champion`. Per-CLAUDE.md the
page must show a clear message when no data is available — no synthetic
fallback in the page itself.

Smoke tests:
- **Import**: `import frontend.pages.sync_report_page` raises nothing.
- **No parquet**: `load_sync_report` returns `None` → page shows the
  "Sync report not yet built" banner; no exception, no aggregate panels
  rendered.
- **Empty parquet**: empty DataFrame → same banner.
- **Mixed parquet** (3 rows: OK, WARN, FAILED): page renders the summary
  table, the 6 aggregate panels, and the deep-dive section. AppTest
  assertions on `at.dataframe`, `at.plotly_chart` (count == 6).
- **Failed-session deep-dive**: select the FAILED row → verdict panel
  renders the failure code, the resampled-signal panel is replaced with
  the "Resampled data was not written" caption.
- **Methods & References expander present**: assert it exists and contains
  a non-empty markdown body (sanity that the verbatim block from §4.5 is
  embedded).

If `pytest-streamlit` infra changes before implementation, the
lead-developer should follow whatever convention is in use for
`tests/frontend/test_signal_quality_page.py` and
`tests/frontend/test_pop_dynamics_page.py` (the most analogous pages).

---

## 2. The 18 failure modes — coverage matrix

The neuro-data-scientist's review enumerated 18 failure modes feeding the
diagnostic catalogue. Each row below maps a failure mode to the test it is
covered by, the synthetic-fixture construction, and the expected
`sync_status`. **No row is "untested" — if a mode cannot be triggered in a
unit test, the row says so explicitly and explains the alternative.**

| # | Failure mode | Module test(s) | Fixture construction | Expected `sync_status` | Hypothesis property test? |
|---|---|---|---|---|---|
| 1 | Missing TDMS / unreadable timestamps.h5 | `test_align.py::TestStage5FailureClosedSemantics::test_no_timestamps` | Pass non-existent path to `align.run`. | `FAILED_NO_TIMESTAMPS` | No — single discrete branch |
| 2 | Zero camera-trigger pulses | `test_daq.py::test_empty_cam_raises` + `test_align.py::test_no_pulses` | Synthetic TDMS-like dict with `cam_data` all-zero (no rising edges). | Stage 0 raises `ValueError`; if Stage 5 is given a degenerate timestamps.h5 with `cam_n_pulses == 0`, it classifies `FAILED_NO_PULSES`. | No |
| 3 | Zero SciScan line-clock pulses | `test_daq.py::test_empty_line_clock_raises` | TDMS dict with all-zero SciScan channel. | Stage 0 raises `ValueError("No SciScan line-clock pulses")`. | No |
| 4 | `len(line_idxs) % y_pix != 0` (truncated final frame) | `test_daq.py::test_sci_lines_truncated_n_recorded` | Build a line-clock train with `y_pix * 1000 + 5` rising edges. | Ingest succeeds; `tdms_diag/sci_lines_truncated_n == 5`. Sync may emit `frame_count_off_by_one` warning. | No |
| 5 | Frame-count mismatch (img vs TIFF) within off-by-one | `test_diagnostics.py::test_classify_frame_count_off_by_one_warning` | `n_tiff_frames = 1000`, `img_n_pulses = 1001`. | `OK_WITH_WARNINGS` with code `frame_count_off_by_one`. | No |
| 6 | Frame-count mismatch beyond off-by-one but within hard limit | `test_diagnostics.py::test_classify_frame_count_minor_mismatch` | Diff = ±3 (range 1..5). | `OK_WITH_WARNINGS` with `frame_count_minor_mismatch`. | Yes — boundary property test |
| 7 | Frame-count hard mismatch | `test_diagnostics.py::test_classify_frame_count_hard` | Diff = ±50. | `FAILED_FRAME_COUNT_MISMATCH` | Yes — boundary property test |
| 8 | High camera ISI jitter | `test_diagnostics.py::test_classify_high_cam_jitter` | `synthetic_clean_pulse_train(jitter_ms=3.0)` → CV ≈ 0.03 > 0.02. | `OK_WITH_WARNINGS` with `high_camera_jitter`. | Yes |
| 9 | High imaging ISI jitter | as #8 but img stream | imaging stream with elevated MAD. | `OK_WITH_WARNINGS` with `high_imaging_jitter`. | Yes |
| 10 | Linear drift | `test_diagnostics.py::test_classify_drift_*` | `synthetic_drifted_pulse_train(drift_ppm=200)`. | `OK_WITH_WARNINGS` with `linear_drift_camera` or `_imaging`. | Yes — drift recovery property |
| 11 | Duplicated pulses | `test_diagnostics.py::test_classify_duplicate_pulses` | `synthetic_corrupted_pulse_train(duplicate_idxs=[100])`. | `OK_WITH_WARNINGS` with `duplicate_pulses_camera`. | No |
| 12 | Non-saturated digital channel | `test_diagnostics.py::test_classify_non_saturated_digital` | `tdms_diag.cam_max = 0.7`. | `OK_WITH_WARNINGS` with `non_saturated_digital`. | No |
| 13 | Camera truncation (cam < 50 % of imaging) | `test_diagnostics.py::test_classify_truncated_camera_hard` | `cam_duration = 30 s`, `img_duration = 100 s`. | `FAILED_TRUNCATED_CAMERA` | No |
| 14 | Cross-stream temporal overlap low | `test_diagnostics.py::test_classify_overlap_*` | overlap fraction ∈ {0.92, 0.97, 0.999}. | 0.92 → FAILED, 0.97 → WARN (`temporal_overlap_low`), 0.999 → OK. | Yes — boundary at 0.95 / 0.99 |
| 15 | Cross-stream start-offset large | `test_diagnostics.py::test_classify_cross_start_offset_high` | `img[0] − cam[0] = 100 ms`. | `OK_WITH_WARNINGS` with `cross_start_offset_high`. | No |
| 16 | Light period drift | `test_diagnostics.py::test_classify_light_period_drift` | `diff(light_on)` = 130 s (10 s above 120 default). Boundary at threshold. | `OK_WITH_WARNINGS` with `light_period_drift`. | Yes — boundary on threshold |
| 17 | Light count mismatch (`|n_on − n_off| > 1`) | `test_diagnostics.py::test_classify_light_count_mismatch` | `n_on = 5`, `n_off = 7`. | `OK_WITH_WARNINGS` with `light_count_mismatch`. | No |
| 18 | Non-uniform DLC pose decimation | `test_diagnostics.py::test_classify_non_uniform_pose_decimation` + `test_align.py::test_kin_pose_decimation_warning` | Kinematics fixture with non-uniform decimation flag. | `OK_WITH_WARNINGS` with `non_uniform_pose_decimation` (per user confirmation: warning, not failure). | No |

**Cannot be triggered in unit tests:**

- **Hardware-level TDMS corruption** (truncated mid-channel, malformed
  metadata) — handled in `report.py` via the `read_error` column. Covered
  in §1.3 by writing a manually corrupted `sync.h5` (truncating the file at
  byte 100); the parquet row carries the exception message. No unit test
  can exercise an actually-corrupted TDMS file without checking in such a
  file, which violates the no-real-data rule.
- **AWS S3 unavailability for the report aggregator** — out of scope for
  unit tests; covered by integration tests in `tests/scripts/` if/when
  added.

---

## 3. `sync_status` classification tests

In `tests/sync/test_diagnostics.py::TestClassifyStatus`. All tests
operate on `SyncScalars` dataclasses constructed by hand (no I/O).

### 3.1 Tier-firing — one parametrised test per tier

`@pytest.mark.parametrize` over a list of `(scalars, expected_status)`
tuples, one row per tier. Each row constructs the **minimal** scalars dict
that satisfies that tier's predicate **and no earlier tier's predicate**:

1. `FAILED_NO_TIMESTAMPS` — `timestamps_present=False`.
2. `FAILED_NO_PULSES` — `cam_n_pulses=0` (timestamps present).
3. `FAILED_FRAME_COUNT_MISMATCH` — `pulse_count_diff_after_off_by_one=10`,
   pulses non-zero.
4. `FAILED_TEMPORAL_OVERLAP` — `cross_overlap_s/max_duration=0.90`.
5. `FAILED_TRUNCATED_CAMERA` — `cam_duration_s/img_duration_s=0.4`.
6. `OK_WITH_WARNINGS` — clean pulses but `cam_isi_cv=0.025`.
7. `OK` — every threshold within bounds, zero warning predicates.

### 3.2 Tier ordering (first-match-wins)

Critical because the absence of an upstream condition implies later
conditions. Tests:

- `FAILED_NO_TIMESTAMPS` precedes `FAILED_NO_PULSES`: with
  `timestamps_present=False` AND `cam_n_pulses=0`, status is
  `FAILED_NO_TIMESTAMPS` (the "no pulses" condition is implied; should not
  surface as a separate failure).
- `FAILED_NO_PULSES` precedes `FAILED_FRAME_COUNT_MISMATCH`: with
  `cam_n_pulses=0` AND `pulse_count_diff=999`, status is `FAILED_NO_PULSES`.
- `FAILED_FRAME_COUNT_MISMATCH` precedes `FAILED_TEMPORAL_OVERLAP`: with
  large frame diff AND low overlap, status is `FAILED_FRAME_COUNT_MISMATCH`.
- A `FAILED_*` always precedes `OK_WITH_WARNINGS`: even when warning
  conditions also fire, the failure tier wins; warnings/failures lists are
  populated independently.
- `OK_WITH_WARNINGS` precedes `OK`: any single warning predicate firing
  demotes from `OK` to `OK_WITH_WARNINGS`.

### 3.3 Boundary tests (hypothesis property)

Use `hypothesis` to verify monotonic behaviour around each threshold:

- `T_FRAME_COUNT_HARD = 5`. Strategy: `st.integers(min_value=-20, max_value=20)`.
  Property: `|diff| <= 5` ⇒ status starts with `OK*`; `|diff| > 5` ⇒
  `FAILED_FRAME_COUNT_MISMATCH` (assuming all other inputs clean).
- `T_OVERLAP_HARD = 0.95`. Strategy: `st.floats(0.5, 1.0)`.
  Property: `overlap_frac < 0.95` ⇒ `FAILED_TEMPORAL_OVERLAP`;
  `0.95 ≤ overlap_frac < 0.99` ⇒ `OK_WITH_WARNINGS` with
  `temporal_overlap_low`; `overlap_frac >= 0.99` ⇒ `OK`.
- `T_CV_CAM_WARN = 0.02`. Strategy: `st.floats(0.0, 0.05)`.
  Property: monotonic — increasing CV across the threshold flips status
  from `OK` to `OK_WITH_WARNINGS` exactly once.

### 3.4 Warnings list aggregation

- `OK_WITH_WARNINGS` with N triggered warnings → `len(sync_warnings) == N`,
  no duplicates, list deterministic in tier-table order.
- Empty warning predicates → `sync_warnings == []` and status is `OK`.
- All warning predicates fire simultaneously → `sync_warnings` contains
  exactly the canonical list of codes (regression protection).

### 3.5 Failures list is human-readable

- For each `FAILED_*` tier, `sync_failures` contains a string of the form
  `"frame_count_mismatch: pulse_count_diff_after_off_by_one=12 (threshold=5)"` —
  the failing scalar inlined.
- The short code is the first colon-separated token (machine-parseable);
  the rest is the human-readable explanation.
- JSON encoding round-trip: `json.loads(json.dumps(failures)) == failures`.

---

## 4. Schema validation tests

In `tests/io/test_hdf5.py` (extend) and `tests/sync/test_validate.py`
(extend the existing legacy-shim tests).

For each new HDF5 dataset / attr, three tests:
1. **Round-trip**: write → read → assert equality of every scalar.
2. **Positive validation**: a fixture dict matching the schema passes
   `validate_*`.
3. **Negative validation**: corrupt one field at a time → `pandera.SchemaError`
   is raised with a message naming the corrupted field.

Required negative tests (one per row in design §2.1 / §2.2):
- `line_clock_times` missing; wrong dtype; non-monotonic.
- `tdms_diag/` group missing.
- `tdms_diag/sci_lines_truncated_n` < 0 (must be ≥ 0).
- `sync_status` missing; not in the 7-code enum; wrong dtype.
- `sync_warnings`/`sync_failures` not a JSON-encoded list.
- `sync_diag/cam_n_pulses` negative (must be ≥ 0 or sentinel -9999).
- `sync_diag/cam_isi_cv` negative.
- `sync_diag/light_first_state_at_t0` not in `{-1, 0, 1}`.

Schema-version test:
- `sync_status_version == "0.0"` → validator marks file as legacy-rebuild-required.
- `sync_status_version == "1.0"` → validator runs the full new schema.
- Future-proofing: an unknown future version (e.g. `"2.0"`) raises a clear
  "unsupported version" message rather than silently accepting.

---

## 5. Failure-closed semantics tests

In `tests/sync/test_align.py::TestStage5FailureClosedSemantics` (already
sketched in §1.2) plus `tests/analysis/test_run.py` (new section).

### 5.1 Stage 5 stub-writing

For each `FAILED_*` precondition, assert:
- `sync.h5` exists.
- Root attr `sync_status` matches the expected code.
- `sync_diag/` group is populated with all scalars (sentinels where
  inputs were missing).
- `sync_warnings`/`sync_failures` JSON arrays are present and well-formed.
- **None** of `hd_deg`, `dff`, `frame_times`, `light_on`, `bad_frames`,
  `event_masks`, `spikes` are present in the stub. The test reads with
  `read_h5` and asserts `set(sync.keys()) == set(_DIAG_ONLY_KEYS)` (an
  empty set if all diag is in attrs).
- Stage 5 exits with code 0 (Snakemake rule succeeds).

### 5.2 Stage 6 entry guard (`src/hm2p/analysis/run.py`)

New file `tests/analysis/test_run_entry_guard.py`. Tests:

- `run_session(sync_h5_path)` with `FAILED_*` and no override → writes a
  sentinel `analysis.h5` containing only `session_id` and `skipped_reason`,
  returns early. No analysis arrays in output.
- `run_session(..., include_failed_sync=True)` → proceeds even when
  `sync_status` is `FAILED_*`. Verify by stubbing the analysis pipeline
  to record whether it was called.
- `run_session(..., include_failed_sync=False)` with `OK_WITH_WARNINGS`
  → proceeds (warnings are not failures).
- `run_session(..., include_failed_sync=False)` with `OK` → proceeds.
- `skipped_reason` string is human-readable and includes the
  `sync_status` code and the first entry of `sync_failures`.

### 5.3 Frontend banner / `st.stop()`

In `tests/frontend/test_sync_report_page.py` and a new helper test in
`tests/frontend/test_data.py::TestIsSyncClean`:

- `is_sync_clean(sync_attrs)` returns `(True, "")` for `OK` and
  `OK_WITH_WARNINGS`.
- `is_sync_clean(sync_attrs)` returns `(False, "<reason>")` for every
  `FAILED_*`.
- An analysis page (e.g. `hd_tuning_page`) given a mocked `FAILED_*`
  session calls `render_sync_failure_warning` and the AppTest sees
  `at.error[0].value` matching "failed sync verification".
- The same mock check confirms `st.stop()` halts rendering — the AppTest
  sees no plotly chart rendered after the banner.

---

## 6. Aggregation / report tests

Already sketched in §1.3. Concrete invariants the lead-dev must assert:

- `len(parquet) == len(sync.h5 files in sync_dir)` for any synthetic input
  (1, 3, 26 sessions).
- Per-status counts match expectation for a hand-crafted mix.
- Empty input → empty parquet with **all** columns present (defensive
  schema).
- Missing `sync_diag/` (legacy file) → row written with `read_error`,
  scalars sentinelled, never dropped.
- The aggregator does **not** open heavy datasets (`hd_deg`, `dff`).
  Verify by writing a sync.h5 with a deliberately-corrupt `dff` array but
  valid attrs — `build_report` must succeed.
- File order in the parquet is sorted by `exp_id` (deterministic for
  downstream diff-checks).

Hypothesis property test:
- For any list of N synthetic `(sync_status, sync_warnings_count)` tuples,
  the aggregator output has the same N rows, and per-status counts match
  the input distribution.

---

## 7. Frontend page tests

Frontend test infra **does** exist (`tests/frontend/` with 17 page
modules using `streamlit.testing.v1.AppTest`). The lead-developer should
follow the convention of `tests/frontend/test_signal_quality_page.py`
(closest analogue: page that loads a parquet + per-session HDF5).

Required smoke tests for `tests/frontend/test_sync_report_page.py`:

- **Imports and renders empty**: page imports, runs, shows the
  "Sync report not yet built" banner when `load_sync_report` returns `None`.
- **Renders mixed parquet**: 3-row mock parquet (OK, WARN, FAILED). Assert:
  - Summary table has 3 rows.
  - Status chip column has the expected colour codes (verified via
    `at.dataframe[0].column_config`).
  - 6 aggregate charts render (`len(at.plotly_chart) == 6`).
  - Default sort puts `FAILED_*` rows first.
- **Selectbox switching**: simulate selecting each session id; assert the
  deep-dive panel header text changes accordingly.
- **Failed session shows verdict, hides resampled signals**: select a
  `FAILED_*` row; assert the verdict block lists each failure code and
  the resampled-signal sub-panel is replaced by the documented caption.
- **Excluded session caption**: a row with `exclude=1` in the synthetic
  experiments fixture renders the `Notes` text from `experiments.csv` in
  the deep-dive caption.
- **Methods & References expander present**: assert `at.expander[0].label ==
  "Methods & references"` and the body contains the markdown headings.
- **Old `sync_page.py` deleted**: the package no longer imports it; a
  test in `tests/frontend/test_app_rendering.py` ensures `st.navigation()`
  no longer references the removed page.

---

## 8. Golden-fixture regression tests

Marker convention check: `pyproject.toml` does not currently register
custom markers, but the codebase uses inline markers like
`@pytest.mark.skipif(not _suite2p_available, ...)`. Register
`@pytest.mark.golden` in `pyproject.toml::[tool.pytest.ini_options].markers`
when adding these tests:

```toml
markers = [
  "golden: regression tests against three real sessions on S3 — not run in CI",
]
```

Run with `pytest -m golden`. Default CI invocation runs `pytest -m "not golden"`.

File: `tests/sync/test_golden_sessions.py` (new).

Three sessions (confirmed in `metadata/experiments.csv:14`, `:15`, `:22`):

| Role | exp_index | exp_id | Expected `sync_status` |
|---|---|---|---|
| Gold standard | 21 | `20221004_10_42_58_1118023` | `OK` or `OK_WITH_WARNINGS` (final tier to be filled in by implementer after running Stage 5 against this session) |
| Known bad #1 | 13 | `20220531_11_06_13_1117217` | `FAILED_*` (specific tier to be filled in by implementer — likely `FAILED_FRAME_COUNT_MISMATCH` or `FAILED_TEMPORAL_OVERLAP` based on the "Camera sync problem" annotation) |
| Known bad #2 | 14 | `20220601_13_53_18_1117217` | `FAILED_*` (same — fill in after running) |

Test pattern:
- Fetch each session's `timestamps.h5` + `kinematics.h5` + `ca.h5` from
  S3 (`s3://hm2p-derivatives/`).
- Skip the test (not fail) when `S3_REAL_DATA_AVAILABLE` env var is
  unset, the `boto3` credentials chain fails, or the file is missing.
- Run `align.run()` end-to-end into `tmp_path`.
- Assert the resulting `sync_status` matches the expected tier.
- Assert that for the gold standard, `len(sync_warnings) <= 2` (sanity
  cap; failure here means the "perfect" session is no longer perfect —
  that is itself a regression worth flagging).
- Assert that for both known-bad sessions, `len(sync_failures) >= 1`.

The lead-developer must:
1. Implement the test scaffold with the expected status as a
   `# TODO: confirm` placeholder.
2. Run the test once locally against S3 to determine the actual tier.
3. Replace the placeholder with the empirical tier.
4. Commit with a comment citing the session and the observed scalar that
   triggered the tier.

---

## 9. Property-based tests (hypothesis)

Six functions warrant `hypothesis` strategies. All in
`tests/sync/test_diagnostics.py` unless noted.

### 9.1 `drift_slope(times)`
- **Strategy**: monotonic float arrays, length 10..5000, increments
  drawn from `st.floats(min_value=1e-4, max_value=1.0, allow_nan=False)`.
- **Property**: returned slope is finite; for constant-ISI input, slope
  ≈ 0 within 1e-9; for input with cumulative multiplier `1 + α`, recovered
  slope ≈ α within 5 %.
- **R²**: must be in `[0, 1]` for any monotonic input.

### 9.2 `channel_scalars(times, fps_nominal)`
- **Strategy**: arrays from `synthetic_clean_pulse_train` over
  `fps ∈ st.floats(1.0, 200.0)`, `duration_s ∈ st.floats(1.0, 600.0)`,
  `jitter_ms ∈ st.floats(0.0, 5.0)`.
- **Property**: `cam_isi_median_ms ≈ 1000 / fps` within `5 × jitter_ms`.
- **Property**: `cam_isi_cv >= 0` always; equals 0 iff `jitter_ms == 0`.
- **Property**: `n_isi_outliers >= 0` always.

### 9.3 `cross_channel_scalars(cam, img, ...)`
- **Strategy**: two clean pulse trains with random offset
  `δ ∈ st.floats(-1.0, 1.0)`.
- **Property**: `cross_start_offset_ms ≈ -δ × 1000` within numerical
  tolerance.
- **Property**: `cross_overlap_s + max(0, -δ) + max(0, δ) ≈ max(durations)`.

### 9.4 `infer_light_polarity` / `light_first_state_at_t0`
- **Strategy**: `st.lists(st.floats(0, 1000), min_size=0, max_size=20)`
  for both on and off arrays (independent draws, then sorted within the
  function).
- **Property**: never raises for any input combination.
- **Property**: returns one of `{-1, 0, 1}`.
- **Property**: empty arrays → returns `-1`.

### 9.5 `classify(scalars, *, cfg)`
- **Strategy**: a Hypothesis composite generating valid `SyncScalars`
  with each scalar drawn from a sensible per-field range (e.g.
  `cam_isi_cv ∈ st.floats(0.0, 0.1)`, `pulse_count_diff ∈ st.integers(-100, 100)`).
- **Property**: returned status is always one of the 7 canonical codes.
- **Property**: `len(sync_warnings) > 0 ⇒ status == "OK_WITH_WARNINGS"`
  (warnings only co-exist with that status, not `OK`).
- **Property**: status starts with `FAILED_` ⇒ `len(sync_failures) >= 1`.
- **Property**: status `OK` ⇒ `sync_warnings == [] and sync_failures == []`.

### 9.6 `build_report` round-trip (`tests/sync/test_report.py`)
- **Strategy**: list of `(sync_status, n_warnings)` tuples, length 1..30.
- **Property**: aggregate per-status counts in the parquet match input
  counts exactly.

---

## 10. Non-test artefacts

### 10.1 `tests/sync/conftest.py` (new) — fixture builders

The lead-dev should add a `tests/sync/conftest.py` exporting:

- `synthetic_clean_pulse_train(rng, fps, duration_s, jitter_ms=0.5) -> np.ndarray`
- `synthetic_drifted_pulse_train(rng, fps, duration_s, drift_ppm) -> np.ndarray`
- `synthetic_corrupted_pulse_train(rng, fps, duration_s, *, missing_idxs=(),
  duplicate_idxs=()) -> np.ndarray`
- `write_synthetic_timestamps_h5(path, *, cam_kwargs, img_kwargs,
  light_kwargs, tdms_diag_kwargs) -> None` — minimal helper that mirrors
  the schema in design §2.1 so individual tests don't reinvent it.
- `write_synthetic_sync_h5(path, *, sync_status, sync_diag, warnings,
  failures, payload=None) -> None` — for `test_report.py` and the
  frontend smoke tests.
- `write_synthetic_kinematics_h5(path, n=600, *, bad_behav=None,
  decimation_uniform=True) -> None` — re-export the helper that
  `tests/sync/test_align.py` already inlines, to share with the new
  diagnostics tests. Move it from `test_align.py` to `conftest.py` in
  the same commit.
- `write_synthetic_ca_h5(path, t=180, n_rois=10, *, include_events=False,
  include_spikes=False, include_bad_imaging=False) -> None` — same.

These fixtures must be **documented** with a one-line docstring per
function and must use `np.random.default_rng(seed)` for reproducibility.

### 10.2 `pytest-cov` configuration

`pyproject.toml::[tool.coverage.run].source` already lists `src/hm2p`,
which transitively covers all new modules. **No change needed.** The CI
threshold (`fail_under = 90`) applies project-wide, so any drop in
coverage from the new modules will fail CI. The lead-dev should run:

```bash
pytest --cov=hm2p.sync --cov=hm2p.ingest.daq --cov=hm2p.io.hdf5 \
       --cov-report=term-missing tests/sync/ tests/ingest/test_daq.py \
       tests/io/test_hdf5.py tests/analysis/test_run_entry_guard.py
```

after each commit and confirm ≥ 90 % per file. Diagnostics module
should hit 100 %.

### 10.3 New marker registration

Add to `pyproject.toml::[tool.pytest.ini_options]`:

```toml
markers = [
  "golden: regression tests against three real sessions on S3 — not run in CI",
]
```

Default CI command should become `pytest -m "not golden"` to exclude
golden tests by default.

---

## 11. Open questions and design issues spotted

These are escalations the lead-dev should resolve **before** writing
tests, since they materially affect what to assert.

1. **Status code count: design says "9 codes" in §2.2 table, then
   collapses to 7 in §3.1.** The `validate_sync_h5` validator must accept
   exactly the 7 codes from §3.1; the "9 codes" reference in §2.2 is a
   stale leftover from the neuro-data-scientist's review. Confirm with
   the architect that 7 is final, then write the validator's enum
   accordingly.

2. **`sync_status_version` semantics for legacy files.** §2.2 says files
   with no `sync_status` attr are "treated as schema version `0.0` and
   rebuilt." The aggregator (§1.3) treats version `0.0` as a `read_error`.
   These are consistent only if the rebuild happens before the aggregator
   runs. The lead-dev should confirm the Snakemake DAG enforces this
   ordering (Stage 5 always runs before Stage 5b) and add a regression
   test that the aggregator never sees a `0.0` file in a fully-built
   tree.

3. **Light protocol phase at t=0** is design §8.1, still unresolved. Until
   the user decides between options (a)/(b)/(c), the
   `light_phase_unknown` warning should be **silenced** by the default
   config (`expected_first_state: unknown`), and the test plan should
   not assert that the warning fires for any synthetic session. When the
   decision lands, add a parametrised test over the three choices.

4. **`exclude=1` and `FAILED_*` interaction.** Design §8.3 says excluded
   sessions still get a sync.h5 and analysis.h5 (Stage 6 only skips on
   `FAILED_*`). The frontend page must render excluded `OK` sessions —
   confirm the test for "excluded session caption" in §7 actually
   verifies the deep-dive renders the data, not just the caption.

5. **`is_sync_clean` semantics with `OK_WITH_WARNINGS`.** Design §5.1
   says `OK_WITH_WARNINGS` is "considered clean for consumption." The
   test in §5.3 asserts `(True, "")` for that status — which means the
   warnings are **not** surfaced via this helper. Pages that need the
   warnings must read them separately. Confirm this is the intended
   contract before writing the test.

6. **Stub `sync.h5` schema validation.** A `FAILED_*` stub omits
   `hd_deg`, `dff`, etc. The validator must accept this. The lead-dev
   should confirm the architect's intent: should the validator have a
   single mode that conditionally requires payload datasets based on
   `sync_status`, or two separate validators (`validate_sync_h5_full`
   and `validate_sync_h5_stub`)? §1.5 above proposes the conditional
   single validator; flag if architect prefers the split.

---

_End of plan. Length: ~2300 words. Ready for the lead-developer to
implement tests directly per the rollout in design §7._
