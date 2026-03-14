# Testing Strategy

Practical guide to how tests are written, run, and enforced in hm2p.

**Date:** 2026-03-14

---

## Unit Tests

### Framework

- **pytest** (>=8.2) with **pytest-cov** (>=5.0) for coverage
- **hypothesis** (>=6.108) for property-based testing of numerical functions
- **pandera** (>=0.20) for runtime DataFrame/HDF5 schema validation

### Conventions

- All tests live in `tests/`, mirroring the `src/hm2p/` module structure
- Every function (public and private) should have at least one test
- **Synthetic data only** -- tests never load real experimental data files. The sole
  exception is small synthetic arrays constructed inline or via fixtures
- Shared fixtures are in `tests/conftest.py`:
  - `penk_session` / `nonpenk_session` -- synthetic `Session` objects
  - `n_frames` (1000), `n_rois` (50), `fps_imaging` (29.97), `fps_camera` (100.0)
  - `synthetic_dff` -- `(n_rois, n_frames)` float32 array with sparse transients
  - `rng` -- seeded `np.random.default_rng(42)` for reproducibility
  - `tmp_h5` -- temporary HDF5 path via `tmp_path`

### Coverage

- **Target: 90%+** (hard requirement, enforced in CI via `--cov-fail-under=90`)
- Branch coverage enabled (`[tool.coverage.run] branch = true`)
- Excluded from coverage: `pragma: no cover`, `if TYPE_CHECKING:`, ellipsis stubs
- Current count: ~1486 tests across all modules

### Property-based testing with hypothesis

Numerical functions (dF/F, HD tuning, MVL, information theory) use hypothesis to
generate adversarial inputs and verify invariants. Example from
`tests/analysis/test_hypothesis_analysis.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

def hd_signal_strategy(min_frames=100, max_frames=500):
    return st.integers(min_value=min_frames, max_value=max_frames).flatmap(
        lambda n: st.tuples(
            arrays(np.float64, n, elements=st.floats(0, 10, allow_nan=False, allow_infinity=False)),
            arrays(np.float64, n, elements=st.floats(0, 360, allow_nan=False, allow_infinity=False, exclude_max=True)),
            st.just(np.ones(n, dtype=bool)),
        )
    )

class TestMVLInvariants:
    @given(hd_signal_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_mvl_bounded(self, data):
        signal, hd, mask = data
        tc, bc = compute_hd_tuning_curve(signal, hd, mask, n_bins=18)
        mvl = mean_vector_length(tc, bc)
        assert 0 <= mvl <= 1 + 1e-9
```

### Test module inventory

| Directory | Modules tested |
|---|---|
| `tests/analysis/` | activity, tuning, significance, comparison, decoder, stability, population, ahv, information, classify, gain, speed, anchoring, save, run, hypothesis invariants, cache |
| `tests/anatomy/` | register, injection |
| `tests/calcium/` | dff, events, event_dynamics, neuropil, spikes, run |
| `tests/extraction/` | suite2p, caiman, run_suite2p, zdrift |
| `tests/frontend/` | app_rendering, maze, hd_tuning, classify, decoder, pop_dynamics, ahv, anchoring, drift, light_analysis, signal_quality, summary, tracking_quality, patching_traces, cost_aws, data |
| `tests/ingest/` | daq, daq_parse, validate |
| `tests/io/` | hdf5, s3, nwb, aws_cost |
| `tests/kinematics/` | compute, compute_dataset, syllables |
| `tests/maze/` | topology, discretize, analysis, hypothesis, e2e |
| `tests/patching/` | config, io, ephys, protocols, spike_features, morphology, metrics, statistics, pca, run |
| `tests/pose/` | preprocess, quality, retrain, run |
| `tests/scripts/` | downstream_pipeline, stage3_helpers |
| `tests/sync/` | align, validate |
| `tests/` (root) | cli, config, session, plotting |

---

## Frontend / Streamlit Tests

### Framework

Streamlit's built-in **AppTest** framework for headless UI testing:

```python
from streamlit.testing.v1 import AppTest
```

### Pattern

Tests are in `tests/frontend/`. The core pattern from `test_app_rendering.py`:

```python
from unittest.mock import patch

MOCK_EXPERIMENTS = [{"exp_id": "20220804_13_52_02_1117646", ...}]
MOCK_ANIMALS = [{"animal_id": "1117646", "celltype": "penk", ...}]

class TestHomePageRendering:
    def test_home_renders(self):
        at = AppTest.from_file("frontend/pages/home_page.py", default_timeout=10)

        with patch("frontend.data.load_experiments", return_value=MOCK_EXPERIMENTS), \
             patch("frontend.data.load_animals", return_value=MOCK_ANIMALS), \
             patch("frontend.data.get_stage_summary", return_value=MOCK_STAGE_SUMMARY):
            at.run()

        err = _has_real_exception(at)
        assert err is None, f"Home page raised: {err}"
```

### Key practices

- **Mock all external calls**: S3, EC2, and data-loading functions are patched via
  `unittest.mock.patch` to avoid hitting real AWS resources
- **Shared mock data**: `MOCK_EXPERIMENTS`, `MOCK_ANIMALS`, `MOCK_PIPELINE_STATUS`,
  `MOCK_STAGE_SUMMARY` are defined once and reused across tests
- **Mock S3 client**: `_mock_s3_client()` returns a MagicMock that returns empty
  results for `list_objects_v2` and raises `NoSuchKey` for `get_object`
- **Exception filtering**: `_has_real_exception()` skips known benign errors like
  `st.page_link` raising `KeyError('url_pathname')` in headless mode
- **Interaction testing**: Use `at.selectbox[0].set_value(...)`, `at.button[0].click()`,
  `at.slider[0].set_value(...)` to simulate user interactions, then call `at.run()`
  again to process the interaction

### Frontend test files

| Test file | Pages covered |
|---|---|
| `test_app_rendering.py` | home, pipeline, sessions, animals, changelog |
| `test_maze_page.py` | maze |
| `test_hd_tuning_page.py` | hd_tuning |
| `test_classify_page.py` | classify |
| `test_decoder_page.py` | decoder |
| `test_pop_dynamics_page.py` | pop_dynamics |
| `test_ahv_page.py` | ahv |
| `test_anchoring_page.py` | anchoring |
| `test_drift_page.py` | drift |
| `test_light_analysis.py` | light, light_compare |
| `test_signal_quality_page.py` | signal_quality |
| `test_summary_page.py` | summary |
| `test_tracking_quality_page.py` | tracking_quality |
| `test_patching_traces_page.py` | patching_traces |
| `test_cost_aws_pages.py` | cost, aws |
| `test_data.py` | frontend/data.py module |

---

## Integration Tests

### Snakemake dry-run

CI runs a Snakemake dry-run (`snakemake -n`) to verify the DAG resolves correctly.
This creates mock directories and placeholder files for one session, then runs:

```bash
cd workflow && snakemake -n --profile profiles/local \
    --config data_root="../data" metadata_dir="../metadata"
```

This catches broken rules, missing inputs, and config errors without executing any
pipeline stages.

### Schema validation with pandera

The lint workflow runs pandera-related tests separately:

```bash
pytest tests/ -k "pandera or schema or validate" --no-header -q
```

This validates that DataFrame schemas (metadata CSVs) and HDF5 output schemas
conform to their declared pandera models.

---

## CI/CD

### Two GitHub Actions workflows

**`ci.yml`** -- Tests + coverage:
- Trigger: push/PR to `main`
- Matrix: Python 3.11 + 3.12
- Steps: checkout, install uv, `pip install -e ".[dev]"`, pytest with `--cov-fail-under=90`
- Coverage XML uploaded to **Codecov**
- Includes Snakemake dry-run job (Python 3.11 only)

**`lint.yml`** -- Static analysis:
- Trigger: push/PR to `main`
- Python 3.11
- Checks:
  - **ruff check** -- linting (pycodestyle, pyflakes, isort, bugbear, pyupgrade, naming, annotations, simplify)
  - **ruff format --check** -- formatting
  - **mypy** -- strict type checking
  - **bandit** -- security analysis (excludes tests/, old-pipeline/)
  - **checkov** -- Dockerfile security scanning
  - **pip-audit** -- dependency CVE scanning
  - **vulture** -- dead code detection (90% confidence)
  - **detect-secrets** -- secret scanning against `.secrets.baseline`
  - **pandera** schema validation tests

### Ruff configuration (from pyproject.toml)

```toml
line-length = 99
target-version = "py311"
select = ["E", "W", "F", "I", "B", "UP", "N", "ANN", "SIM"]
```

- Tests exempt from annotation checks (`ANN`)
- Neuroscience variable names (`F`, `F0`, `Fneu`, `T`) exempt from naming checks in
  calcium/extraction modules

---

## Running Tests Locally

### Quick test run

```bash
make test
```

Runs `python -m pytest tests/ -q --tb=short` (no coverage enforcement).

### Full coverage run

```bash
make test-cov
```

Runs `python -m pytest tests/ --cov=hm2p --cov-report=term-missing --cov-fail-under=90`.

### PYTHONPATH

The package is installed in editable mode (`pip install -e ".[dev]"`), so `src/hm2p` is
on the path. If running without installation, set:

```bash
PYTHONPATH=src:. pytest tests/
```

### Running specific tests

```bash
# Single file
pytest tests/calcium/test_dff.py

# Single class
pytest tests/analysis/test_tuning.py::TestHDTuningCurve

# Single test
pytest tests/analysis/test_tuning.py::TestHDTuningCurve::test_uniform_signal_flat_curve

# By keyword
pytest tests/ -k "hypothesis"

# Frontend only
pytest tests/frontend/

# Skip slow tests (hypothesis)
pytest tests/ -k "not hypothesis"
```

### Linting locally

```bash
make lint         # ruff check
make fmt          # ruff format + fix
make typecheck    # mypy
```

---

## Current Coverage Gaps

### Frontend pages without dedicated tests

The following 31 pages in `frontend/pages/` lack dedicated test files:

- `analysis_page.py`
- `anatomy_page.py`
- `batch_page.py`
- `calcium_page.py`
- `compare_page.py`
- `correlations_page.py`
- `dlc_page.py`
- `event_dynamics_page.py`
- `events_page.py`
- `explorer_page.py`
- `gain_page.py`
- `gallery_page.py`
- `info_theory_page.py`
- `moseq_explore_page.py`
- `moseq_page.py`
- `patching_page.py`
- `pop_dynamics_page.py` (has test but name mismatch -- verify)
- `population_page.py`
- `qc_report_page.py`
- `speed_page.py`
- `stability_page.py`
- `stats_page.py`
- `suite2p_page.py`
- `sync_page.py`
- `timeline_page.py`
- `trace_compare_page.py`
- `zdrift_page.py`

Note: Some pages (home, pipeline, sessions, animals, changelog) are tested via
`test_app_rendering.py` rather than having individual test files.

### Modules with potentially thin coverage

- `src/hm2p/analysis/cache.py` -- requires `duckdb` (not in standard dev deps), test collection fails
- `src/hm2p/io/nwb.py` -- NWB export (requires pynwb, neuroconv)
- `src/hm2p/kinematics/syllables.py` -- keypoint-MoSeq wrapper (requires kpms)
- `frontend/data.py` -- tested in `test_data.py` but many S3 code paths mocked out
- Plotting utilities (`src/hm2p/plotting.py`) -- visual output hard to assert on

### Recommended next steps

1. Add rendering tests for the 31 untested frontend pages following the `test_app_rendering.py` pattern
2. Add `duckdb` to dev dependencies or mark `test_cache.py` with `pytest.importorskip`
3. Add hypothesis tests for kinematics functions (`_windowed_gradient`, `_median_filter_1d`)
4. Add pandera schema tests for all HDF5 output files (ca.h5, kinematics.h5, sync.h5, analysis.h5)
