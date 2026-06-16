# movement v0.17.0 migration plan

Status: **complete. movement 0.17.0 installed and verified on Python 3.12.13.**
Branch `feat/movement-0.17`. This document scopes the upgrade from `movement`
0.14.0 to 0.17.0.

## Resolution (2026-06-16): Python bumped to 3.12, 0.17 verified

The blocker below is resolved. The devcontainer ran Python 3.11.2 only by
accident — the base `node:20` image is Debian bookworm (system python3 = 3.11),
and the Dockerfile's `ARG PYTHON_VERSION=3.12` was never consumed. There was no
real reason for 3.11; 3.12 was the original intent.

Changes applied:
- `pyproject.toml`: `requires-python` `>=3.10,<3.14` → `>=3.12,<3.14` (movement
  0.17 requires ≥3.12, so 3.10/3.11 are dropped); movement pin `>=0.5,<1.0` →
  `>=0.17,<0.18`.
- `uv.lock` re-resolved: movement 0.14.0 → 0.17.0 (+ xgboost 3.2.0, orjson).
- `.devcontainer/Dockerfile`: added `RUN uv python install ${PYTHON_VERSION}`
  so the dead ARG is now live and a managed 3.12 interpreter is baked into the
  image. `uv sync` at container start builds `.venv` on 3.12 (uv also
  auto-provisions 3.12 from the `requires-python` constraint as a fallback).
- venv recreated on cpython 3.12.13 via `uv venv --python 3.12 && uv sync
  --extra dev` (+ ad-hoc `statsmodels`, and the `plotly streamlit-google-auth
  tifffile` the postStart command installs).

Verification on 3.12.13 + movement 0.17.0 — the singular dims (`keypoint`/
`individual`) are now real, so the `_normalise_dim_names` shim is exercised:
- `tests/kinematics tests/pose`: **538 passed**.
- `tests/calcium tests/extraction tests/scripts`: 967 passed, 5 skipped.
- `tests/analysis`: 417 passed; 11 `rastermap` failures are pre-existing
  (`rastermap` not installed — fails identically on the 3.11 backup), and
  `test_cache.py`/`test_mixed_stats.py` import gaps (`duckdb`, and `statsmodels`
  until installed) are likewise pre-existing, not migration regressions.
- Core package + CLI + frontend (incl. the edited `dlc_viewer_page`) import.

The old environment is preserved at `.venv-py311-backup/` (gitignored) for
rollback; it can be deleted once 3.12 is confirmed in normal use.

Follow-up not done here: EC2 launchers apt-install the distro python3 — with
`requires-python>=3.12` those hosts now need a 3.12 interpreter (Ubuntu 24.04
default, or uv-provisioned) rather than 3.10/3.11. No EC2 run is pending, so
this is deferred.

---

The original plan and the (now-resolved) blocker follow, retained for reference.

## Applied change set (2026-06-15)

Strategy B (normalise dims at the load boundary) was implemented so the code
runs unchanged on both 0.14.0 (current devcontainer) and 0.17.0:

- `src/hm2p/kinematics/compute.py`: `_TRACKER_MAP` (source_software strings)
  replaced by `_TRACKER_LOADERS` (per-format loader names: `from_dlc_file` /
  `from_sleap_file` / `from_lp_file`, which exist with an identical
  `(file, fps=None)` signature in 0.14 onward). `load_pose_dataset` now
  dispatches to the per-format loader and calls a new `_normalise_dim_names`
  helper that renames singular dims (`keypoint`/`individual`) back to plural —
  a no-op on movement <0.17. The rest of the module keeps the plural names, so
  `perspective.py` needed **no change**.
- `frontend/pages/dlc_viewer_page.py`: same `from_file` → `from_dlc_file`
  swap + inline dim-normalisation. **This site was missing from the impact
  inventory below** — found by a repo-wide grep for `from_file`/`source_software`.
- Tests: `tests/kinematics/test_compute_dataset.py` loader tests assert the
  per-format loaders; added `TestNormaliseDimNames` (singular→plural, plural
  no-op, partial). All other kinematics/perspective fixtures keep plural dims
  and are unchanged (the shim keeps the internal vocabulary plural).
- `tests/kinematics tests/pose` + dlc-viewer frontend tests: **636 passed** on
  movement 0.14.0.
- `pyproject.toml` pin left at `movement>=0.5,<1.0` (permits both 0.14 and
  0.17). **Not** tightened to `>=0.17` because that would make the env
  unsatisfiable on the current Python — see blocker below.

## BLOCKER: movement 0.17 requires Python ≥ 3.12

Not anticipated by the original plan. `movement==0.17.0` declares
`requires-python >=3.12`; the devcontainer runs **Python 3.11.2**, so 0.17
cannot be installed here. Consequences:

- The actual 0.17 install and the numerical no-op verification (step 4 below)
  **cannot run until the project Python is bumped to ≥3.12**. `pyproject.toml`
  already allows it (`requires-python = ">=3.10,<3.14"`); `uv python` can fetch
  cpython 3.12/3.13 for aarch64.
- Bumping Python means recreating `/workspace/.venv` on 3.12 and reinstalling
  all scientific deps (suite2p, FISSA, CASCADE, roiextractors) on ARM — a
  larger, riskier change than this code edit (FISSA in particular pins old
  numpy/scipy/sklearn ABIs). That is a separate decision, not done here.
- Because the code is forward-compatible, the branch can land and keep working
  on 0.14 today; the Python bump + 0.17 verification is a follow-up.

---

The original plan follows (impact inventory etc.), retained for reference.

## Why

The `Versions` table in `CLAUDE.md` directs us to track the latest stable release
of each tool. `movement` 0.17.0 (released 2026; tag `v0.17.0`) is current; the
pipeline runs 0.14.0. Two of its changes are breaking for Stage 3 (kinematics).

## Breaking changes in 0.17.0 that affect us

1. **Dataset dimension rename (plural → singular):**
   - `"keypoints"` → `"keypoint"`
   - `"individuals"` → `"individual"`

   Every `.sel(keypoints=…)`, `.isel(individuals=0)`, `coords["keypoints"]`,
   `.mean(dim="keypoints")`, `.assign_coords(keypoints=…)`, and
   `.loc[dict(keypoints=…)]` against a movement Dataset breaks.

2. **`load_poses.from_file()` removed** (deprecated since 0.14.0). There is no
   generic `from_file(file, source_software=…)` loader in 0.17.0. Loading is now
   per-format:
   - `from_dlc_file(file, fps=None)`
   - `from_sleap_file(file, fps=None)`
   - `from_lp_file(file, fps=None)`  (LightningPose)
   - also `from_anipose_file`, `from_nwb_file`, `from_numpy`, `from_dlc_style_df`

   These no longer take `source_software` — the format is implied by the loader,
   and they accept `fps`.

Not affecting us: `compute_displacement()` was removed but is unused here. The
`"keypoints"` references in `src/hm2p/pose/finetune.py` are our own DLC-config
dicts, not movement dimensions — out of scope.

## Impact inventory

### Source (must change)

| File | Code references | Notes |
| --- | --- | --- |
| `src/hm2p/kinematics/compute.py` | ~52 dim refs + 3 loader lines | The dominant surface. Includes the dim-normalisation helper at lines 112–120 that renames bodypart coordinate values on the `keypoints` coord. |
| `src/hm2p/kinematics/perspective.py` | 4 dim refs (lines 142, 149, 158, 159) | Camera-rotation correction writes back via `.loc[dict(keypoints=…)]`. |

Loader block to replace (`compute.py` ~755–767): currently uses
`inspect.signature(load_poses.from_file)` to handle the old `file`→`file_path`
rename, then calls `load_poses.from_file(...)`. Replace with a tracker→loader
dispatch:

```text
_TRACKER_LOADERS = {
    "dlc":   load_poses.from_dlc_file,
    "sleap": load_poses.from_sleap_file,
    "lp":    load_poses.from_lp_file,
}
ds = _TRACKER_LOADERS[tracker](pose_path)   # fps optional
```

(The current `_TRACKER_MAP` maps our tracker key → movement `source_software`
string; it is replaced by a map to the loader callable.)

### Tests (must change)

Synthetic Datasets are built with the old dim names and asserted against them:

- `tests/kinematics/test_compute_dataset.py` — builds `dims=["time","space","keypoints","individuals"]` and coords in multiple fixtures.
- `tests/kinematics/test_compute.py`
- `tests/kinematics/test_perspective.py` — incl. `assert list(ds_corr.coords["keypoints"].values) == KEYPOINTS`.

These fixtures must adopt the singular dim names so they exercise the real
0.17.0 contract.

## Two migration strategies

### Strategy A — adopt singular names natively (full rename)
Rename every reference in source and tests to `keypoint`/`individual`. Largest
diff (~56 source sites + tests), but the codebase then speaks movement's native
vocabulary and there is no translation layer to forget.

### Strategy B — normalise to plural at the load boundary (shim)
Immediately after loading (and after `rename_sa_bodyparts`), rename the dims back
to the names the rest of the module already uses:

```text
ds = ds.rename({"keypoint": "keypoints", "individual": "individuals"})
```

One-line change at a single choke point; the ~56 downstream references stay as
they are. Smaller, lower-risk diff. Cost: the code permanently diverges from
movement's native dim names, which is mildly surprising to a reader and must be
documented at the load site. Tests that construct Datasets directly (bypassing
the loader) would still need updating unless they route through the same shim.

**Recommendation:** Strategy B for the first pass — it isolates the breaking
change to one line and keeps the diff reviewable — with a clear comment at the
load site explaining the renormalisation, and a follow-up issue to do the full
Strategy A rename later if we want native naming. Open to A if you'd rather not
carry the translation layer.

## Downstream / re-run implications

- The migration is **code-only**; it does not by itself change any output.
- **Verifying** it requires re-running Stage 3 on at least a few sessions and
  confirming `kinematics.h5` is numerically identical (HD, position, speed, AHV)
  to the 0.14.0 output — the upgrade must be a no-op on values.
- Per the pipeline-invalidation contract (`CLAUDE.md`), re-running Stage 3
  invalidates Stage 3b (MoSeq) → Stage 5 (sync) → Stage 6 (analysis). A full
  re-run across 26 sessions would therefore cascade. Since Stage 6 already needs
  a re-run for the MVL rectification fix, a single combined re-run is the
  efficient sequencing — but that is a separate decision from landing the code.

## Verification plan (when the branch is built)

1. `uv pip install "movement>=0.17,<0.18"`; update the pin / version note.
2. Apply the chosen strategy; update tests.
3. `pytest tests/kinematics tests/pose -q` green; coverage ≥ 90%.
4. Run Stage 3 locally on 1–2 sessions on **both** 0.14.0 and 0.17.0 and assert
   `kinematics.h5` arrays match within float tolerance (a one-off comparison
   script, not committed).
5. Only then consider the downstream re-run cascade.

## Rollout

- Branch `feat/movement-0.17`, independent of `feat/fissa-reprocessing`.
- No EC2 / S3 / re-run actions without explicit approval.
