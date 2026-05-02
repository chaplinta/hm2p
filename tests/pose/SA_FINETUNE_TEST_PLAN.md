# SuperAnimal Fine-Tune — Test Plan

_Status: design — not yet implemented._
_Companion to: `/workspace/docs/superanimal-fine-tune-design.md` (architect spec, commit `3adeb3c`)._
_Backing science spec: `/workspace/docs/superanimal-fine-tune-plan-v2.md` (v2 plan)._
_Branch: `feat/sync-pipeline-diagnostics`._
_Audience: lead-developer implementing the 6-commit rollout in design §5._

This plan is precise enough that the lead-developer can implement tests
directly. It does **not** include the test code itself. Per
[CLAUDE.md](../../CLAUDE.md):

- Tests use small synthetic numpy arrays only — never real data files.
- `pytest` + `pytest-cov` + `hypothesis` for property tests.
- `pandera` for HDF5 / DataFrame schema validation; `jsonschema` for the
  nested verdict-JSON schema (pandera is awkward for nested JSON).
- Coverage target: ≥ 90 % across new/modified modules; ≥ 95 % on
  `src/hm2p/pose/finetune.py` (pure compute).
- Statistical assertions are non-parametric only.

The test-engineer **does not write tests** — that is the lead-developer's
job, in commits 1–6 of the architect's rollout.

---

## 1. Test scope per commit

The architect's rollout (design §5) is six commits. Each commit must land
test-clean.

### 1.1 Commit 1 — `src/hm2p/pose/finetune.py` (pure compute)

**Files introduced.** `src/hm2p/pose/finetune.py`,
`tests/pose/test_finetune.py` (new), `tests/pose/conftest.py` (new — see §4).

**Module exports** (per design §1.1): `paired_wilcoxon_per_keypoint`,
`rank_biserial_paired`, `bootstrap_median_ci`, `bonferroni_alpha`,
`per_frame_euclidean_error`, `pck_at`, `hd_from_ear_vector`,
`circular_abs_error`, `probe_sa_detector_bbox_rate`, `KeypointVerdict`,
`GateConfig`, `Verdict`, `verdict_to_json`, `verdict_from_json`,
`evaluate_promotion_gate`.

**Tests required** (`tests/pose/test_finetune.py`):

| Group | Scenarios |
| --- | --- |
| `paired_wilcoxon_per_keypoint` | (a) baseline > candidate everywhere → all p < 1e-3; (b) baseline ≈ candidate → p ≥ 0.5; (c) baseline < candidate with `alternative="greater"` → p ≥ 0.5; (d) NaN handling: pairs with NaN dropped, NaN p when n_pairs < `min_pairs` (default 10); (e) shape mismatch → ValueError; (f) non-2D input → TypeError. |
| `rank_biserial_paired` | (a) hand-computed match against Kerby 2014 worked example; (b) range bound [-1, 1] for any input; (c) `r > 0` iff candidate < baseline on average; (d) all-zero diff → r = 0. |
| `bootstrap_median_ci` | (a) deterministic with seeded `np.random.default_rng(42)`; (b) coverage of true median ≥ 90 % over 200 trials at α = 0.05 (run as one slow test, marked `@pytest.mark.slow`); (c) `n_resamples=1` → degenerate CI (low == high == sample median); (d) NaN-only input raises `ValueError`; (e) ci=0.99 produces a wider interval than ci=0.95 on the same array. |
| `bonferroni_alpha` | trivial: `alpha / n_tests`; n_tests=0 raises. |
| `per_frame_euclidean_error` | (a) zero error on identical inputs; (b) NaN propagates only to the affected (frame, keypoint) cell; (c) shape mismatch → ValueError; (d) returns dtype float64 even from float32 inputs. |
| `pck_at` | (a) all-zero errors → 1.0; (b) all errors above threshold → 0.0; (c) NaN ignored (denominator excludes NaN); (d) empty array → NaN, no division-by-zero. |
| `hd_from_ear_vector` | (a) both ears at same y → angle 0; (b) right ear above left → angle π/2 (verify against `hm2p.kinematics.compute` if it exists; cite the convention in a comment); (c) returns wrapped to (-π, π]; (d) NaN ear coordinate propagates per-frame. |
| `circular_abs_error` | (a) zero on identical inputs; (b) wraps |Δ| at π so wrap-around is symmetric (input π+ε → returns π-ε); (c) NaN propagation. |
| `probe_sa_detector_bbox_rate` | helper takes `bbox_rate: float ∈ [0, 1]` (the architect's design §6 pitfall #3). Tests: (a) rate ≥ 0.90 returns `(True, "")`; (b) rate < 0.90 returns `(False, "<msg>")` with the rate inlined; (c) rate exactly 0.90 → pass (≥ not >); (d) n=0 frames probed → returns `(False, "no frames probed")`. |
| `Verdict` round-trip | `verdict_from_json(verdict_to_json(v)) == v` (dataclass equality); schema_version preserved; missing required field on parse → ValueError naming the field. |
| `evaluate_promotion_gate` | (a) clear winner → `overall_pass=True`, `fail_reasons=[]`; (b) candidate worse on `mid_back` → `overall_pass=False`, code `"regression_mid_back"` in `fail_reasons`; (c) candidate better on nose only → fails `tail_pct_reduction`; (d) HD field populated only when all three HD inputs are non-None; (e) `gate` echoed verbatim into the verdict; (f) custom `GateConfig` thresholds honoured. |

**Hypothesis property tests** (see §2 for full list).

**Coverage target.** ≥ 95 % per-line on `src/hm2p/pose/finetune.py`. The
module has no I/O, no heavy branches; this is achievable.

### 1.2 Commit 2 — `scripts/compare_models.py`

**Files introduced.** `scripts/compare_models.py`,
`tests/scripts/test_compare_models.py` (new).

**Tests required.**

| Group | Scenarios |
| --- | --- |
| Argparse | required args (`--baseline-id`, `--candidate-id`, `--labels-dir`); default `--alpha == 6.25e-3`; default `--seed == 42`; `--mode` defaults to `predict`; `--mode rmse-json` is accepted; `--upload-s3` is a flag (no value). |
| Mode `predict` end-to-end | mock `select_best_dlc_h5_s3` and h5 reads (using `_FakeS3` from `tests/scripts/test_declare_dlc_champion.py`); construct synthetic GT + two prediction sets in `tmp_path`; assert `verdict.json` is written with the expected `schema_version`, `baseline_id`, `candidate_id`, `n_frames_compared`. |
| Mode `rmse-json` triage | both JSONs supplied → script reports descriptive deltas only; explicit assertion that the verdict's per-keypoint `p_value_wilcoxon` is NaN (Wilcoxon cannot run without per-frame pairs); exit code 0 even if gate criteria are met for the descriptive part (this mode is non-authoritative). |
| Exit codes | `overall_pass=True` → 0; gate failed → 2; comparison cannot be performed (no overlapping sessions) → 3 with `verdict.json` carrying `meta.error`. |
| `--upload-s3` | with mock S3 client, assert `put_object` called with key `dlc-retrain/models/_compare_verdict.json` and `ContentType="application/json"`. |
| Fixture: clear winner | 50-frame test set where candidate beats baseline by 60 % on nose/tail, ties elsewhere → `overall_pass=True`. |
| Fixture: clear loser | candidate worse by 20 % on nose → `overall_pass=False`. |
| Fixture: mixed (regression on `mid_back`) | candidate better on nose/tail but worse on `mid_back` by 15 % → `overall_pass=False`, `fail_reasons` contains `"regression_mid_back"`. |
| Fixture: insufficient data | only 5 GT frames overlap (< `min_pairs`) → verdict's per-keypoint `p_value_wilcoxon` is NaN, exit code 3, `meta.error` populated. |
| `meta.skipped_sessions` | sessions present in `--labels-dir` but missing from one prefix are listed and the comparison is not aborted (continues with the remaining sessions). |
| Verdict content | the file written to disk validates against the `verdict.schema.json` (see §3). |

**Coverage target.** ≥ 90 % on `scripts/compare_models.py`.

### 1.3 Commit 3 — `scripts/run_dlc_retrain.py --sa-finetune` wiring

**Files modified.** `scripts/run_dlc_retrain.py`. **New test file**:
`tests/scripts/test_run_dlc_retrain.py`. **Modified**:
`tests/pose/test_select.py` (parametric architecture-token test, design §1.5).

DLC is heavy and not import-safe in CI; mock everything in
`deeplabcut.*`, `dlclibrary.*`, `scipy.io` config-yaml writers.

| Test | Setup | Assert |
| --- | --- | --- |
| `--sa-finetune` parses | argparse only | `args.sa_finetune is True` |
| `--sa-finetune` default epochs | argparse + branch | `epochs == 120` when `--sa-finetune` set without `--epochs` |
| ImageNet default epochs | argparse only | `epochs == 400` when `--sa-finetune` not set |
| Explicit `--epochs` honoured | argparse | `--sa-finetune --epochs 200` → 200 |
| `_train_sa_finetune` calls `build_weight_init` | mock `deeplabcut.modelzoo.weight_initialization.build_weight_init` | called with `super_animal="superanimal_topviewmouse"`, `model_name="hrnet_w32"`, `with_decoder=True`, `memory_replay=True` |
| `_train_sa_finetune` skips manual backbone rewrite | temp `pytorch_config.yaml` | the SA path does **not** mutate `model.backbone.*` keys (compare loaded YAML to expected snapshot) |
| `_train_sa_finetune` applies augmentation patch | temp `pytorch_config.yaml` | `data.train.affine.rotation == 30`, `affine.scaling == [0.7, 1.3]`, `gaussian_noise == 10` |
| `default_net_type` rewrite from `resnet_50` | temp `config.yaml` with `default_net_type: resnet_50` and `--sa-finetune` set | warning is emitted to stdout (`caplog` or `capsys`); the script writes `default_net_type: hrnet_w32` back to disk; **the original file's mtime changes** |
| `default_net_type` already `hrnet_w32` | temp `config.yaml` with correct net | no warning, no rewrite |
| Conversion-table assertion | `config.yaml` lacking conversion entry for one bodypart | raises `ValueError` naming the missing bodypart |
| Detector resolution `_v2` | mock `dlclibrary.list_available_detectors()` returning both | resolves to `fasterrcnn_resnet50_fpn_v2` (priority order) |
| Detector resolution fallback | only `fasterrcnn_resnet50_fpn` available | resolves to that one, no error |
| Detector resolution failure | neither available | raises with the available-list inlined in the error message |
| `dlclibrary.list_available_models()` missing SA | mock returns no `superanimal_topviewmouse_hrnet_w32` | raises `RuntimeError` |
| 256×256 input-size mismatch | mock pytorch_config with `data.train.input_size != [256, 256]` | warning emitted, **does not raise** (per design §6 pitfall #1); discrepancy recorded in stdout |
| `train_network` kwargs | mock `deeplabcut.train_network` | called with `epochs=120`, `save_epochs=10`, `batch_size=8`, and `pytorch_cfg_updates` containing `"train_settings.optimizer.params.lr": 5e-5`, `"model.backbone.freeze_bn_stats": True`, scheduler step-LR milestones `[90, 110]`, gamma `0.1` |
| `notes` string | inspect args passed to `declare_champion()` | `notes` contains `init: superanimal_topviewmouse_hrnet_w32 (memory replay)`, `conversion_array: [0,1,2,26,7,8,9,13]`, `detector: <resolved>`, `epochs: 120; lr: 5e-5; bs: 8; freeze_bn_stats: True` |
| `--maxiters` ignored under `--sa-finetune` | argparse | help text contains `(legacy; ignored under --sa-finetune)`; passing it does not raise; the value never reaches `train_network` |

**`tests/pose/test_select.py` additions** (parametric, design §4.5):

```text
("video_DLC_HrnetW32_hm2p-retrain_2026-03-20_shuffle1_snapshot-best-110.h5", "HrnetW32"),
("video_DLC_HrnetW32_hm2p-retrain_2026-03-20_shuffle2_snapshot-best-60.h5",  "HrnetW32"),
("video_DLC_Resnet50_hm2p-retrain_shuffle1_snapshot-50000.h5",               "Resnet50"),
```

Locks the design decision that init source is in `notes`, not in the
architecture token.

**Coverage target.** ≥ 80 % on the modifications (mocked-DLC paths).
The non-mockable DLC body itself is never executed in CI.

### 1.4 Commit 4 — `scripts/launch_dlc_finetune_ec2.py --sa-finetune` passthrough

**Files modified.** `scripts/launch_dlc_finetune_ec2.py`. **New test
file**: `tests/scripts/test_launch_dlc_finetune_ec2.py`.

Pure user-data string assembly. **No real boto3 calls.** Use `MagicMock`
for the `boto3.client(...)` return value, mirroring the `_FakeS3` pattern
in `tests/scripts/test_declare_dlc_champion.py`.

| Test | Assert |
| --- | --- |
| `--sa-finetune` propagates | `build_user_data(sa_finetune=True)` contains the literal substring `--sa-finetune` on the `python3 scripts/run_dlc_retrain.py` invocation line |
| Cost record tag | the cost-records dict (or `mode` field passed downstream) contains `mode == "sa-finetune"` (or `"<base>+sa"` per design §1.4 — confirm the lead-dev uses the format the architect chose) |
| Disk size | the `BlockDeviceMappings[0]["Ebs"]["VolumeSize"] == 120` when `sa_finetune=True` |
| Default arg | `sa_finetune=False` produces user-data identical to current main aside from any unrelated changes (snapshot-style assertion against an inline expected string) |
| Instance type | `INSTANCE_TYPE == "g4dn.xlarge"` regardless of `--sa-finetune` (design §1.4: no instance-type bump) |
| Compatibility with `--infer-only` | passing `--sa-finetune --infer-only` does not raise; user-data still contains `--sa-finetune` (re-running inference of an SA-finetuned model is a valid combination per architect note) |
| Env-var resolution | when `AWS_REGION` env var is set, the launched instance uses it; default region is `ap-southeast-2` (per project memory) |
| `--dry-run` | does not call `boto3.client(...).run_instances`; prints user-data to stdout |
| `--status` / `--terminate` short-circuits | passing these does not call `run_instances`; `describe_instances` / `terminate_instances` are called instead |

**Coverage target.** ≥ 80 % on the launcher (mocked-boto3 paths).

### 1.5 Commit 5 — docs only (+ Methods expander)

**Files modified.** `docs/dlc-retraining.md`, `docs/dlc-champion-model.md`,
`frontend/pages/tracking_quality_page.py` (Methods & References expander
only — no verdict-rendering logic yet, that is commit 6).

**Tests required.**

| Test | Assert |
| --- | --- |
| Page imports | `import frontend.pages.tracking_quality_page` raises nothing (regression: the new expander block does not break import) |
| Page renders empty | with all S3 loaders mocked to return `None`, `AppTest.from_file(...).run()` exits without exception and shows the existing "no data" banner |
| Methods expander present | `at.expander[0].label == "Methods & References"` (or the chosen label); body contains the substrings `"Ye"` and `"10.1038/s41467-024-48792-2"` (citation sanity) |
| Markdown body well-formed | the expander body is non-empty markdown (`len(...) > 100`) |

No coverage gate beyond existing for the docs-only pass.

### 1.6 Commit 6 (optional) — frontend display of `verdict.json`

**Files modified.** `frontend/pages/tracking_quality_page.py`,
`frontend/data.py` (cached loader for verdict.json).
**Modified test file**: `tests/frontend/test_tracking_quality_page.py`.

| Test | Assert |
| --- | --- |
| Page imports | unchanged from §1.5 |
| Mock verdict — pass | mock `frontend.data.load_verdict()` returns a clear-winner verdict (use the same fixture as the compare_models test); page renders a per-keypoint table; verdict banner shows green chip with "Promotion gate: PASS" |
| Mock verdict — fail | candidate-loses verdict; banner shows red chip with "Promotion gate: FAIL"; `fail_reasons` are listed |
| Mock verdict — mixed | regression on one keypoint; per-keypoint table shows red chip on that keypoint, green elsewhere; banner is FAIL |
| Missing verdict.json | `load_verdict()` returns `None` → page shows "Verdict not yet computed" banner (clear, not synthetic-fallback per CLAUDE.md); no exception |
| Stale verdict | mock verdict whose `candidate_id` does not match the current champion id (returned by `get_dlc_champion()`) → page shows a staleness warning via `render_champion_staleness_warning` |
| HD descriptive panel | when `verdict.hd` is non-empty, the page shows the HD median-circular-error comparison; when null, the panel is hidden |

**Coverage target.** Smoke-test only; no per-line gate beyond existing.
Follow the `streamlit.testing.v1.AppTest` convention used in
`tests/frontend/test_sync_report_page.py`.

---

## 2. Hypothesis property tests

Six functions warrant `hypothesis` strategies. All in
`tests/pose/test_finetune.py`.

### 2.1 `paired_wilcoxon_per_keypoint(e_baseline, e_candidate, *, alternative="greater")`

- **Strategy**: paired arrays of shape `(n, 8)` where
  `n ∈ st.integers(min_value=20, max_value=500)`, baseline drawn from
  `st.floats(min_value=0.0, max_value=200.0, allow_nan=False)`, candidate
  = baseline − `st.floats(min_value=0.5, max_value=10.0)` (always smaller).
- **Properties**:
  - All p-values are in [0, 1] or NaN (when n_pairs < `min_pairs`).
  - With strictly smaller candidate and n ≥ 50, all 8 p-values are < 0.001.
  - Output shape is `(n_keypoints,)`.
  - Symmetry: `paired_wilcoxon_per_keypoint(b, c, alternative="greater") + paired_wilcoxon_per_keypoint(c, b, alternative="greater")` ≈ 1 ± numerical-tol on each non-tied keypoint.

### 2.2 `rank_biserial_paired(e_baseline, e_candidate)`

- **Strategy**: paired arrays of shape `(n,)`, n ∈ [10, 500].
- **Properties**:
  - r ∈ [-1, 1] for every input.
  - Sign flip: `rank_biserial_paired(b, c) == -rank_biserial_paired(c, b)`.
  - Invariance under monotone-increasing transform: applying `np.log1p` to
    both arrays (after shifting non-negative) yields the same r within
    rounding (rank-based statistic).
  - Identical inputs → r = 0.

### 2.3 `bootstrap_median_ci(x, *, n_resamples, ci, rng)`

- **Strategy**: 1-D array, n ∈ [10, 200], values from
  `st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)`.
- **Properties**:
  - Returned tuple is `(median, low, high)`; `low ≤ median ≤ high`.
  - Reproducibility: identical seeded `rng` yields identical CI bytes.
  - Coverage: across 200 trials of n=50 from `np.random.normal(0, 1)`,
    the percentile CI contains the population median (0.0) ≥ 90 %
    (loose bound, percentile is biased on small samples).
  - `n_resamples=1` → CI collapses to `(median, median, median)`.

### 2.4 `evaluate_promotion_gate(...)`

- **Strategy**: composite — generate a `(n_frames, 8)` baseline with
  per-keypoint base errors drawn from `st.floats(0.5, 100.0)`, then build
  candidate by element-wise multiplying baseline by a per-keypoint scale
  drawn from `st.floats(0.2, 1.5)` (so candidate may be better or worse).
- **Properties**:
  - Returned `Verdict.overall_pass` is a Python bool.
  - When every per-keypoint scale is < 0.5 (candidate uniformly half the
    error), `overall_pass=True` for all gate configs at default thresholds.
  - When every per-keypoint scale is > 1.2 (candidate uniformly worse),
    `overall_pass=False` and `len(fail_reasons) >= 1`.
  - `Verdict.gate` always equals the input `GateConfig`.
  - `len(Verdict.keypoints) == 8`.

### 2.5 `verdict_to_json` ↔ `verdict_from_json` round-trip

- **Strategy**: build random `Verdict` instances via the above evaluator
  on synthetic inputs.
- **Properties**:
  - `verdict_from_json(verdict_to_json(v)) == v` (dataclass `__eq__`).
  - `json.loads(verdict_to_json(v))["schema_version"] == "1.0"`.
  - All float fields preserve to within 1e-12 (JSON float round-trip is
    exact at IEEE-754 precision via `json.dumps(..., allow_nan=True)`).

### 2.6 `evaluate_gate(per_keypoint_stats, threshold_p, threshold_r)` (boundary)

This is a unit-level helper exposed by `evaluate_promotion_gate` (the
predicate evaluator). Property tests around the v2 §4.6 thresholds:

- **Strategy**: synthesise per-keypoint stats with p ∈
  `st.floats(1e-12, 1.0)` and r ∈ `st.floats(-1.0, 1.0)`.
- **Properties**:
  - p exactly at α/8 (`6.25e-3`) → does **not** pass the
    `nose_significance` predicate (strict `<`, per design §3.1 table).
  - r exactly at 0.30 → **does** pass the `rank_biserial_min` predicate
    (`>=` per design §3.1).
  - For non-nose/non-tail keypoints, `pct_change_median == -0.10` exactly
    → does **not** trigger `no_regression` failure (`>` strict per
    design table — confirm with lead-dev, see Open Question §10.4).

---

## 3. Verdict-JSON schema validation

The architect pinned `schema_version: "1.0"`. The verdict has nested
fields (lists of dataclasses, dicts), so use `jsonschema` (not pandera).

### 3.1 Schema file

The lead-dev should add a new fixture:
`tests/pose/fixtures/verdict.schema.json` — a JSON-Schema-Draft-2020-12
document derived from the `Verdict` dataclass. Required top-level fields:
`schema_version`, `baseline_id`, `candidate_id`, `n_frames_compared`,
`keypoints` (array, length 8), `hd` (object), `gate` (object),
`gate_pass_per_keypoint` (object), `overall_pass` (boolean),
`fail_reasons` (array of strings), `generated_at` (RFC 3339 string),
`meta` (object — optional, see design §2.4).

### 3.2 Schema tests (`tests/pose/test_finetune.py::TestVerdictSchema`)

| Test | Assert |
| --- | --- |
| Round-trip | write → read → assert dataclass equality |
| Positive validation | `evaluate_promotion_gate(...)` output → `verdict_to_json` → `jsonschema.validate(..., schema)` passes |
| Missing field | drop `keypoints` from the JSON → raises `ValidationError` naming the missing field |
| Wrong type | `n_frames_compared = "71"` (string) → raises with clear message |
| Wrong enum | `gate_pass_per_keypoint["nose_tip"]["pass"] = "yes"` → raises (must be bool) |
| `schema_version: "0.0"` rejected | parser refuses; clear error citing the version |
| `schema_version: "2.0"` rejected | parser refuses (forward-compat: do not silently accept future versions) |
| `schema_version: "1.0"` accepted | parser succeeds |

### 3.3 Cross-verification

`verdict_from_json` must validate against the schema **before** parsing
into the dataclass. The validator is the single source of truth.

---

## 4. Test fixtures

Concrete fixtures the lead-dev should add. Cross-test fixtures live in
`tests/pose/conftest.py` (new); one-off fixtures stay per-test.

### 4.1 In `tests/pose/conftest.py` (new)

| Fixture | Construction |
| --- | --- |
| `synthetic_per_frame_errors_small(rng)` | shape `(20, 8)`, values from `rng.uniform(0, 50, (20, 8))`. For paired-Wilcoxon edge cases. |
| `synthetic_per_frame_errors_medium(rng)` | shape `(200, 8)`. Default for the gate end-to-end tests. |
| `synthetic_errors_with_nan(rng)` | shape `(50, 8)`, sprinkle NaN at rng-selected indices (~10 % missing). |
| `synthetic_errors_all_equal()` | shape `(50, 8)`, all values 5.0. Edge case: Wilcoxon must return NaN p (zero non-zero pairs). |
| `synthetic_errors_n1()` | shape `(1, 8)`. n_pairs < min_pairs → NaN p across the board. |
| `synthetic_clear_winner_pair(rng)` | tuple `(e_baseline, e_candidate)`: baseline median ~24 px on nose, ~59 px on tail; candidate is uniformly 0.4 × baseline on nose/tail and 1.0 × elsewhere. |
| `synthetic_clear_loser_pair(rng)` | candidate is 1.2 × baseline on nose. |
| `synthetic_mixed_pair(rng)` | candidate beats baseline on nose/tail but is 1.15 × on `mid_back`. |
| `synthetic_insufficient_pair(rng)` | 5 frames only — exercises `min_pairs` floor. |
| `mock_sa_detector_probe_pass()` | dict with `n_frames=10`, `n_with_bbox=10` → bbox_rate 1.0. |
| `mock_sa_detector_probe_partial()` | `n_frames=10`, `n_with_bbox=1` → 0.1 (fails ≥ 0.90 gate). |
| `mock_sa_detector_probe_zero()` | `n_frames=10`, `n_with_bbox=0` → 0.0. |
| `mock_dlc_project_dir(tmp_path)` | builds a NeuroBlueprint-style DLC project dir: `config.yaml` (with `default_net_type: hrnet_w32`, conversion table for all 8 bodyparts), `labeled-data/<sub>/CollectedData_<scorer>.h5` with 5 dummy rows, `dlc-models-pytorch/iteration-0/<task><date>-trainset95shuffle1/train/pytorch_config.yaml`. Enough surface to exercise the config-rewrite path without DLC training. |
| `mock_pytorch_config(tmp_path)` | minimal `pytorch_config.yaml` with `data.train.affine.{rotation,scaling}`, `data.train.gaussian_noise`, `data.train.input_size`, `model.backbone`. The augmentation-patch test mutates this and asserts. |
| `mock_boto3_ec2_client()` | `MagicMock` whose `run_instances`, `describe_instances`, `terminate_instances` record their call args. Pattern follows `_FakeS3` in `tests/scripts/test_declare_dlc_champion.py`. |
| `verdict_pass_fixture()` | full `Verdict` instance, all gate criteria met. JSON-serialised twin in `tests/pose/fixtures/verdict_pass.json`. |
| `verdict_fail_fixture()` | regression on `mid_back`. Twin: `verdict_fail.json`. |
| `verdict_mixed_fixture()` | nose passes, tail fails the 40 % threshold. Twin: `verdict_mixed.json`. |

### 4.2 Per-test fixtures

The clear-winner / clear-loser fixtures for `test_compare_models.py` may
reuse the conftest fixtures and only build the surrounding HDF5 files
(GT + prediction h5 in `tmp_path`) per-test.

---

## 5. Champion-promotion gate tests

Per design §3.1, the gate is six conjunctions. Tests live in
`tests/pose/test_finetune.py::TestPromotionGate` and
`tests/scripts/test_compare_models.py::TestGateOutcomes`.

### 5.1 Predicate-firing matrix

For each row in design §3.1, one parametrised test confirming the
predicate fires (or does not) at canonical inputs.

| Predicate | Pass case | Fail case | Boundary case |
| --- | --- | --- | --- |
| `nose_pct_reduction` (≥ 0.30) | 0.50 → True | 0.20 → False | 0.30 exactly → True (`>=` per design table) |
| `nose_significance` | p=1e-9, r=0.78 → True | p=0.01, r=0.78 → False (p ≥ α/8) | p exactly 6.25e-3 → False (strict `<`) |
| `tail_pct_reduction` (≥ 0.40) | 0.50 → True | 0.30 → False | 0.40 exactly → True |
| `tail_significance` | as nose | as nose | as nose |
| `head_p90_reduction` (≥ 0.20) | 0.30 → True | 0.10 → False | 0.20 exactly → True |
| `no_regression` (other 5 keypoints) | pct_change > -0.10 → True | pct_change = -0.20 → False | pct_change exactly -0.10 → True (`>` strict, but ≥ in design — clarify, see §10) |

### 5.2 Compound outcomes

| Scenario | Expected `overall_pass` | `fail_reasons` |
| --- | --- | --- |
| Clear winner | True | `[]` |
| Nose passes, tail fails 40 % | False | `["tail_pct_reduction"]` |
| Nose and tail pass, `mid_back` regresses 15 % | False | `["regression_mid_back"]` |
| Head_midpoint p90 only 10 % reduction (others pass) | False | `["head_p90_reduction"]` |
| Significant regression on `neck` (p < 0.05, r < 0) | False | `["regression_neck"]` |

### 5.3 `meta.skipped_sessions` propagation

| Scenario | Assert |
| --- | --- |
| 3 sessions, 1 missing in candidate prefix | `meta.skipped_sessions` lists the missing session id; `n_frames_compared` reflects only the 2 remaining. |
| All 3 sessions missing | exit 3, `meta.error` populated, no gate decision recorded. |
| Session has zero detected bboxes (probe failed) | session is added to `meta.skipped_sessions` with `reason="zero_bboxes"`. |

---

## 6. Rollback / champion-history tests

The architect's design §3.2 says rollback re-runs `declare_dlc_champion.py`
with the previous champion's identifiers. The prior manifest is archived
in `dlc-champion-history/{old_champion_id}.json`.

### 6.1 Reuse existing tests

`tests/scripts/test_declare_dlc_champion.py` already has the `_FakeS3`
fixture and tests for manifest writing and history archival. **No new
rollback-specific tests are required at the
`scripts/declare_dlc_champion.py` layer** — its behaviour is unchanged
under SA-finetune.

### 6.2 SA-finetune-aware additions

| Test | File | Assert |
| --- | --- | --- |
| `notes` field accepts SA annotation | `tests/scripts/test_declare_dlc_champion.py` (extend) | calling `declare_champion(notes="init: superanimal_topviewmouse_hrnet_w32 (memory replay) ...")` succeeds; the manifest written to `_FakeS3` has the full `notes` string un-truncated |
| Rollback sequence | `tests/scripts/test_declare_dlc_champion.py` (extend) | seed `_FakeS3` with a current SA manifest at `dlc-champion.json` and a prior ImageNet manifest at `dlc-champion-history/{old}.json`; call `declare_champion` with `--note "Rolled back; SA-finetune failed promotion gate"` and the prior champion's identifiers; assert (a) `dlc-champion.json` now matches the prior manifest, (b) the archived current SA manifest moved to `dlc-champion-history/`, (c) `notes` includes the rollback annotation |
| Staleness contract on rollback | new test in `tests/frontend/test_data.py::TestIsSessionCurrent` | given a derivative file with `dlc_champion_id == <SA_id>` and the current champion (after rollback) `<imagenet_id>`, `is_session_current()` returns `False` |
| Champion-history list | `tests/scripts/test_declare_dlc_champion.py` (extend) | after a rollback, `dlc-champion-history/` contains both the prior and the just-archived manifests; chronologically sortable by filename |

These tests reuse the `_FakeS3` pattern. No real S3 calls.

### 6.3 `is_session_current()` contract

`frontend/data.py::is_session_current(session, champion)` must return
`False` whenever the session's `dlc_champion_id` attr does not match
`champion.id`. Confirmed by `tests/frontend/test_data.py`. **Add a
parametric case** for the SA-finetune-id format
(`dlc-{YYYYMMDD}-hrnetw32-snap{N}` regardless of init source — design
§1.1 row for `select.py`).

---

## 7. Pitfall-coverage matrix

The architect's §6 enumerates seven pitfalls. Each row names the test
that exercises the mitigation, or marks the row as doc-only.

| Pitfall | Mitigation test | Type |
| --- | --- | --- |
| #1 — 256×256 input-size mismatch | `tests/scripts/test_run_dlc_retrain.py::test_input_size_mismatch_warns_not_raises` | unit + warning capture (`capsys` or `caplog`) |
| #2 — detector name fallback | `tests/scripts/test_run_dlc_retrain.py::TestDetectorResolution` (parametric) | parametric: only `_v2`; only base; neither |
| #3 — SA-detector bbox-rate probe | `tests/pose/test_finetune.py::TestProbeSaDetectorBboxRate` | pass/fail at the 90 % threshold; edge cases n=0, exactly 0.90 |
| #4 — 80/20 vs 95/5 split absolute-RMSE | none — doc-only | doc only |
| #5 — batch-size / VRAM | none — doc-only (operator-judged) | doc only |
| #6 — stale DLC example script | none — doc-only | doc only |
| #7 — DLC issue #2742 (`video_adapt=True`) | none — doc-only (this design does not call that path) | doc only |

Pitfalls 4–7 do not need code-level tests; they are operating-procedure
constraints documented in `docs/dlc-retraining.md` and the design.

---

## 8. CLI tests

CLI contracts for the new and modified scripts. Each script's test file
follows the convention of `tests/scripts/test_declare_dlc_champion.py`:
import the script via `sys.path.insert(0, ".../scripts")`, call its
`main()` with `argv` overridden, capture stdout via `capsys`.

### 8.1 `compare_models.py`

| Invocation | Expected exit code | Side effect |
| --- | --- | --- |
| `--mode predict --baseline-id ... --candidate-id ... --labels-dir ... --output verdict.json` (clear winner fixture) | 0 | `verdict.json` written; `overall_pass=True` |
| same, clear loser fixture | 2 | `verdict.json` written; `overall_pass=False` |
| missing `--labels-dir` | non-zero | argparse error |
| `--labels-dir` exists but empty | 3 | `verdict.json` carries `meta.error` |
| `--upload-s3` | 0 | `put_object` called once with the canonical key |
| schema-validating the written verdict | passes against `verdict.schema.json` |

### 8.2 `run_dlc_retrain.py --sa-finetune` (with mocked DLC)

| Invocation | Assert |
| --- | --- |
| `--sa-finetune` (defaults) | `train_network` called with `epochs=120`, `batch_size=8`, the v2 §4.3 `pytorch_cfg_updates` |
| `--sa-finetune --epochs 200` | `train_network` called with `epochs=200` |
| (no flag) | the legacy ImageNet HRNet path runs (mocked); `train_network` called with `epochs=400` |
| `--sa-finetune --infer-only` | training is **not** invoked; inference path runs |
| `--sa-finetune` with conversion table missing one bodypart | raises `ValueError` before any DLC call |
| Warning capture: `default_net_type: resnet_50` | warning emitted; config rewritten to `hrnet_w32` |

### 8.3 `launch_dlc_finetune_ec2.py --sa-finetune`

| Invocation | Assert |
| --- | --- |
| `--sa-finetune --dry-run` | user-data printed; contains `--sa-finetune`; `run_instances` not called |
| `--sa-finetune` | `run_instances` called with `BlockDeviceMappings[0]["Ebs"]["VolumeSize"] == 120`; `InstanceType == "g4dn.xlarge"`; user-data has `--sa-finetune` |
| `--sa-finetune --infer-only` | user-data has both flags |
| `--status` | `describe_instances` called; `run_instances` not called |
| `--terminate <instance-id>` | `terminate_instances` called once |

---

## 9. Frontend tests (commit 6)

Follow the `tests/frontend/test_sync_report_page.py` convention. Use
`streamlit.testing.v1.AppTest`. Mock all S3 loaders.

### 9.1 Required smoke tests

| Test | Mock | Assert |
| --- | --- | --- |
| Page imports clean | n/a | `import frontend.pages.tracking_quality_page` raises nothing |
| Empty state | `load_verdict()` → None | banner "Verdict not yet computed"; no exception; no per-keypoint table |
| Pass verdict | `verdict_pass_fixture` JSON | banner green "Promotion gate: PASS"; per-keypoint table has 8 rows; no `fail_reasons` listed |
| Fail verdict | `verdict_fail_fixture` JSON | banner red "Promotion gate: FAIL"; `fail_reasons` rendered as bullet list; offending keypoint has red chip |
| Mixed verdict | `verdict_mixed_fixture` JSON | banner red; tail row has red chip, nose row green |
| Stale verdict | verdict whose `candidate_id != current_champion.id` | `render_champion_staleness_warning` is invoked; banner mentions staleness |
| HD panel | verdict with non-empty `hd` field | HD table shows median circular error (rad) for both models, paired Wilcoxon p, rank-biserial r; no statistical-threshold chip (HD is descriptive only per v2 §4.5) |
| Methods expander | n/a | expander present; body contains DOI `10.1038/s41467-024-48792-2` |

### 9.2 Verdict schema enforcement at the frontend

The cached loader `frontend.data.load_verdict` should call
`verdict_from_json` (which in turn validates against
`verdict.schema.json`). Test:

| Test | Assert |
| --- | --- |
| Malformed verdict on S3 | `load_verdict()` returns `None` and logs the schema error; page falls back to "Verdict not yet computed" rather than crashing |
| Future-version verdict (`schema_version: "2.0"`) | `load_verdict()` returns `None`; banner says "Unsupported verdict schema version" (or simply "not yet computed"); no exception |

---

## 10. Open questions for the lead-developer

These are design ambiguities that materially affect the tests. Resolve
**before** writing tests.

1. **Bootstrap method.** Design §1.1 says percentile method; v2 §4.5
   confirms 10 K resamples, percentile. Pin this in the docstring of
   `bootstrap_median_ci` so future-dev does not silently switch to BCa.
   Tests must assert percentile-method behaviour (the test suite cannot
   detect the difference statistically — the docstring is the contract).

2. **Wilcoxon tie policy.** scipy's default is `zero_method="wilcox"`
   (Wilcoxon's original tie-handling: drop zero-diff pairs, halve count).
   v2 §4.5 does not specify. Recommend the lead-dev **pin** `zero_method`
   explicitly in `paired_wilcoxon_per_keypoint` (suggest `"wilcox"`) and
   add a test that all-zero-diff input returns NaN p (since all pairs are
   dropped) rather than 0.0 or 1.0.

3. **`evaluate_gate` "insufficient data" — fail or skip?** The design's
   table at §3.1 says n_pairs < `min_pairs` returns NaN p. What does the
   gate do with NaN p? Two options: (a) treat NaN as a fail
   (conservative), (b) treat NaN as a skip and require *the operator* to
   examine the verdict. Recommend (a) — fail closed — and document the
   choice. Tests assert NaN p → predicate fails → `fail_reasons` includes
   `"insufficient_data_<keypoint>"`.

4. **`no_regression` boundary semantics.** Design §3.1 table reads
   `pct_change_median > -0.10` (strict `>`). The architect's prose at the
   row "no regression" says "10 % regression band". Need to confirm
   whether `pct_change_median == -0.10` exactly is a pass (per "band")
   or a fail (per `>`). Recommend `>=` (a 10 % regression is the boundary
   and is permitted) for consistency with the `≥` thresholds elsewhere
   in the table. Tests must assert whichever the architect confirms.

5. **`gate.rank_biserial_min` floor.** Design §3.3 records the prompt
   asked for `r > 0.1` while v2 says `r > 0.30`. Architect followed v2
   (gate uses 0.30). The verdict reports a *secondary* boolean
   `rank_biserial_r > 0.10` per keypoint. **Test plan assumes the
   secondary diagnostic is computed and recorded but not gated on.** If
   the user later changes the gate threshold, only `GateConfig.rank_biserial_min`
   needs editing — tests parametrise on the threshold to stay robust.

6. **HD-error gate vs descriptive.** v2 §4.6 #5 is descriptive; design
   §3.1 ("HD circular-error check is descriptive only") confirms.
   `evaluate_promotion_gate` therefore must not factor HD into
   `overall_pass`. Tests must assert that even a verdict with a
   significantly *worse* HD on the candidate still passes the gate
   provided the per-keypoint criteria pass — this is intentional, the
   test makes the contract visible.

7. **Verdict `meta` block.** Design §2.4 carries `meta` outside the
   strict dataclass. Tests should assert `meta` round-trips through
   JSON unchanged but is **not** required by the schema (it is
   `additionalProperties: true` on the `meta` object, and the field
   itself is optional). Lead-dev to confirm whether `meta` is part of
   `Verdict` or only added at the I/O layer.

8. **`probe_sa_detector_bbox_rate` — pure helper or DLC call?** Design
   §6 pitfall #3 implementation is a "small helper in `finetune.py`" —
   the unit-testable surface is the rate predicate, not the DLC call.
   Recommend the helper take the rate as input (post-probe) so it stays
   pure compute. Tests then assert only the predicate; the actual DLC
   probing is integration and tested separately under
   `tests/scripts/test_run_dlc_retrain.py` with mocked DLC.

---

## 11. Coverage targets

Per-module coverage gates below. Project-wide
`fail_under = 90` in `pyproject.toml::[tool.coverage.report]` is
unchanged.

| Module / file | Target | Rationale |
| --- | --- | --- |
| `src/hm2p/pose/finetune.py` | ≥ 95 % | pure compute, no heavy branches, no I/O |
| `scripts/compare_models.py` | ≥ 90 % | CLI + IO, but I/O is mockable |
| `scripts/run_dlc_retrain.py` (modifications only) | ≥ 80 % | DLC body itself is mocked; uncovered lines are inside DLC calls |
| `scripts/launch_dlc_finetune_ec2.py` (modifications only) | ≥ 80 % | boto3 mocked; uncovered lines are inside SDK calls |
| `frontend/pages/tracking_quality_page.py` (commit 6) | smoke-test only — no per-line gate | Streamlit page; existing gate is "imports clean and renders without exception" |

Run after each commit:

```bash
pytest --cov=hm2p.pose.finetune --cov-report=term-missing tests/pose/test_finetune.py
pytest --cov-report=term-missing tests/scripts/test_compare_models.py
pytest --cov-report=term-missing tests/scripts/test_run_dlc_retrain.py
pytest --cov-report=term-missing tests/scripts/test_launch_dlc_finetune_ec2.py
```

---

_End of plan. Length target: ~2100 words. Ready for the lead-developer
to implement tests directly per the rollout in design §5._
