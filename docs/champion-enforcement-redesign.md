# Champion Model Enforcement Redesign

_Date: 2026-05-14_
_Status: Design proposal (no code changes yet)_

---

## Executive Summary

The current champion model system has a critical flaw: multiple training runs
can leave pose H5 files from different models coexisting in the same S3
prefix (`pose/{sub}/{ses}/`). The `select_best_dlc_h5()` function picks by
highest snapshot number, which can select a file from an OLD model (e.g.
snapshot-best-110 from ImageNet beats snapshot-best-100 from a newer SA
model). The champion manifest exists but is not enforced at selection time.
Promotion copies new files to `pose/` without deleting old ones. The result:
the pipeline silently processed data from the wrong model.

This document describes a redesign where it is structurally impossible to
use the wrong model. The champion manifest becomes a hard gate at every
stage boundary, not just a label stamped after the fact.

---

## Root Cause Analysis

Five independent failures combined to produce the bug:

### Flaw 1: `select_best_dlc_h5` ignores the champion manifest

The pure function `select_best_dlc_h5()` (lines 34-69 of `pose/select.py`)
selects by highest `snapshot-best-N` number. It has no knowledge of which
model is the champion. The S3-aware wrapper `select_best_dlc_h5_s3()` checks
`promoted.json` first, but if `promoted.json` is absent or names a missing
file, it falls back to the heuristic — silently selecting the wrong file.

**The champion manifest is never consulted during file selection.**

### Flaw 2: Promotion does not clean old files

In `run_dlc_retrain.py` lines 1371-1383, the promotion loop copies files
from `pose-finetuned/{sub}/{ses}/` to `pose/{sub}/{ses}/` using
`s3.copy_object`. It never deletes old files. After two training runs,
`pose/{sub}/{ses}/` contains H5 files from both models. The heuristic then
picks whichever has the higher snapshot number, regardless of which model
produced it.

### Flaw 3: Champion declaration happens AFTER promotion

The champion is declared at line 1396, after promotion at line 1370. If
promotion succeeds but declaration fails (the `try/except` at line 1454
catches and logs the error), the new files are live in `pose/` but the
manifest still points to the old model — or to nothing. The next pipeline
run picks files by heuristic.

### Flaw 4: Stage 5 (sync) has zero champion awareness

`run_stage5_sync.py` never loads the champion manifest. It never checks
whether `kinematics.h5` was produced by the current champion. It reads
kinematics.h5 and ca.h5 and merges them, blindly trusting that kinematics
was produced correctly. The `dlc_champion_id` attribute in sync.h5 is
whatever kinematics.h5 carried — if that was `"unknown"`, sync.h5 inherits
`"unknown"`.

### Flaw 5: `resolve_champion_id` returns `"unknown"` as a soft label

When the triplet `(model_name, architecture, snapshot)` does not match the
manifest, `resolve_champion_id()` returns `"unknown"`. This is stored in
kinematics.h5 and propagated through to sync.h5. But the pipeline never
treats `"unknown"` as an error — it is just a string attribute. The pipeline
proceeds. Only the frontend shows a warning, which is easily missed if nobody
is looking at the frontend during a batch run.

---

## Design Principles

1. **The champion manifest is the single source of truth.** Every pipeline
   stage that consumes DLC-derived data must read it and enforce it.

2. **No silent fallbacks.** If a file does not match the champion, the
   pipeline errors. It does not fall back to heuristics, return `"unknown"`,
   or proceed with a warning.

3. **One model, one set of files.** Promotion deletes all old files before
   copying new ones. There is never more than one model's output in
   `pose/{sub}/{ses}/`.

4. **Provenance is verified, not just stamped.** Downstream stages verify
   that their inputs carry the current champion ID before proceeding.

5. **Declaration before promotion.** The manifest is written before files
   are copied to `pose/`, so the manifest is always ahead of or in sync
   with the live data — never behind it.

6. **Every decision is logged.** File selection, champion verification,
   and promotion all emit structured log messages naming the exact file
   and champion ID.

---

## Detailed Design

### 1. Champion-Aware File Selection

#### 1.1 New function: `select_champion_h5`

Replace the heuristic-based `select_best_dlc_h5` with a champion-first
selection function.

```python
# src/hm2p/pose/select.py

def select_champion_h5(
    h5_keys: list[str],
    champion_manifest: dict,
) -> str:
    """Select the H5 file that matches the current champion model.

    Matches by architecture + snapshot extracted from the filename
    against the champion manifest's architecture and snapshot fields.

    Parameters
    ----------
    h5_keys:
        S3 object keys of all .h5 files in the session's pose dir.
    champion_manifest:
        The parsed dlc-champion.json dict. Must not be None.

    Returns
    -------
    str
        The matching S3 key.

    Raises
    ------
    ChampionMismatchError
        If no file matches the champion manifest.
    """
```

Logic:
1. Filter out `_single` and `_filtered` variants (same as today).
2. For each remaining file, extract `(model_name, snapshot, architecture)`
   using `extract_dlc_provenance()` and `extract_architecture()`.
3. Match against `champion_manifest["architecture"]` and
   `champion_manifest["snapshot"]`.
4. If exactly one match: return it. Log the selection.
5. If zero matches: raise `ChampionMismatchError` with a message listing
   what was found vs what was expected.
6. If multiple matches (same model, same snapshot — e.g. duplicate uploads):
   log a warning and return the first one.

#### 1.2 New function: `select_champion_h5_s3`

S3-aware wrapper that lists keys, loads the champion manifest, and calls
`select_champion_h5`.

```python
def select_champion_h5_s3(
    s3_client: object,
    bucket: str,
    prefix: str,
    champion_manifest: dict,
) -> str:
    """List H5 files under an S3 prefix and select the champion match.

    Parameters
    ----------
    s3_client:
        boto3 S3 client.
    bucket:
        S3 bucket name.
    prefix:
        S3 key prefix for the session's pose directory.
    champion_manifest:
        The parsed dlc-champion.json. Caller must load this once
        and pass it to every session.

    Returns
    -------
    str
        The selected S3 key.

    Raises
    ------
    ChampionMismatchError
        If no file matching the champion exists under the prefix.
    NoPoseDataError
        If no .h5 files exist under the prefix at all.
    """
```

#### 1.3 New exception class: `ChampionMismatchError`

```python
class ChampionMismatchError(RuntimeError):
    """Raised when no pose file matches the current champion manifest."""

    def __init__(self, expected: dict, found: list[dict], prefix: str):
        self.expected = expected
        self.found = found
        self.prefix = prefix
        found_desc = ", ".join(
            f"{f['filename']} (arch={f['architecture']}, snap={f['snapshot']})"
            for f in found
        ) or "(none)"
        super().__init__(
            f"No pose file matches champion "
            f"{expected.get('champion_id', '?')} "
            f"(arch={expected.get('architecture')}, "
            f"snap={expected.get('snapshot')}) "
            f"under {prefix}. "
            f"Found: {found_desc}"
        )
```

#### 1.4 Deprecation of `select_best_dlc_h5`

The old `select_best_dlc_h5` and `select_best_dlc_h5_s3` are kept but
marked as deprecated with a `warnings.warn()` call. They must not be used
in any pipeline stage. Frontend pages that currently call them for display
purposes (e.g. `dlc_viewer_page.py`) are migrated to
`select_champion_h5_s3`. The old functions remain only for interactive
debugging.

The `promoted.json` per-session manifest is retired. It was a caching
mechanism that duplicated the champion manifest at session level. With
champion-first selection, it is no longer needed. The code that reads it
is removed. Existing `promoted.json` files on S3 are left in place but
ignored.

---

### 2. Clean Promotion

#### 2.1 Delete-before-copy in `run_dlc_retrain.py`

The promotion loop in `infer()` is replaced with a `promote_session()`
function that:

1. Lists ALL objects under `pose/{sub}/{ses}/` (H5, JSON, MP4, provenance
   sidecars — everything).
2. Deletes them all using `s3.delete_objects()` (batch delete, up to 1000
   keys per call).
3. Copies new files from `pose-finetuned/{sub}/{ses}/`.
4. Verifies the copy by listing the destination prefix and confirming the
   expected files exist.

```python
def _promote_session(
    s3, bucket: str, sub: str, ses: str,
    finetuned_prefix: str = "pose-finetuned",
) -> int:
    """Atomically promote finetuned pose output for one session.

    Deletes all existing files under pose/{sub}/{ses}/ before copying
    new files from {finetuned_prefix}/{sub}/{ses}/.

    Returns the number of files promoted.

    Raises
    ------
    RuntimeError
        If the copy verification fails (destination is empty after copy).
    """
```

#### 2.2 Verification after promotion

After all 26 sessions are promoted, the script:
1. Picks one session.
2. Lists `pose/{sub}/{ses}/`.
3. Runs `select_champion_h5` against the (already-declared) manifest.
4. If selection fails, raises an error that halts the pipeline and prints
   instructions for manual recovery.

---

### 3. Declaration Before Promotion

#### 3.1 Reorder: declare, then promote

The current order is: infer -> promote -> declare.

The new order is: infer -> declare -> promote -> verify.

Rationale: the manifest must exist before any file is written to `pose/`,
so that the moment a file appears in `pose/`, the manifest already describes
which file is expected. If declaration fails, promotion never starts.

```
infer() completes for all 26 sessions
  |
  v
declare_champion()         # writes dlc-champion.json
  |                        # if this fails, we stop — no files in pose/
  v
_promote_all_sessions()    # delete old + copy new for each session
  |
  v
_verify_all_promotions()   # confirm every session matches the manifest
```

#### 3.2 Failure handling

If `declare_champion()` raises: promotion is skipped. The `pose-finetuned/`
files remain as staging. The operator can fix the issue and re-run
`declare_dlc_champion.py` manually, then run a standalone promotion script.

If promotion fails for session N: the script continues to the remaining
sessions (best-effort), then reports which sessions failed. Failed sessions
retain old files (which were already deleted). The operator re-runs inference
for those sessions.

---

### 4. Provenance Enforcement at Every Stage Boundary

#### 4.1 Stage 3 (kinematics) — enforce champion on input

`run_stage3_kinematics.py` currently loads the manifest and stamps the
champion ID. But it does not refuse to proceed when the pose file does not
match the manifest.

Change: replace `find_dlc_h5()` (which delegates to
`select_best_dlc_h5_s3`) with `select_champion_h5_s3()`. If the function
raises `ChampionMismatchError`, the session is marked as `error` in the
summary, not silently skipped.

The manifest is required. If no manifest exists on S3, Stage 3 refuses to
run entirely:

```python
champion_manifest = get_champion_manifest(s3, DERIVATIVES_BUCKET)
if champion_manifest is None:
    print("FATAL: No champion manifest at s3://hm2p-derivatives/dlc-champion.json.")
    print("Declare a champion before running Stage 3.")
    sys.exit(1)
```

#### 4.2 Stage 5 (sync) — enforce champion on input

`run_stage5_sync.py` currently has zero champion awareness. Add:

1. Load champion manifest once at script start (same pattern as Stage 3).
2. Before processing each session, read `kinematics.h5` HDF5 attrs from
   S3 (download, open, read `dlc_champion_id`, close, delete temp file).
3. Compare `dlc_champion_id` against `champion_manifest["champion_id"]`.
4. If they do not match, refuse to run sync for that session:

```python
kin_champion_id = read_h5_attr(kin_local, "dlc_champion_id")
if kin_champion_id != champion_manifest["champion_id"]:
    print(f"  REFUSE: kinematics.h5 has dlc_champion_id='{kin_champion_id}', "
          f"but current champion is '{champion_manifest['champion_id']}'. "
          f"Re-run Stage 3 first.")
    return "error_stale_kinematics"
```

#### 4.3 Stage 6 (analysis) — enforce champion on input

Same pattern as Stage 5: read `dlc_champion_id` from `sync.h5`, compare
against the manifest, refuse to proceed if mismatched.

#### 4.4 Provenance written to every output

No change needed — the existing code already writes `dlc_champion_id` to
kinematics.h5 (line 2052 of `compute.py`), and sync.h5 copies it from
kinematics.h5 via `_KIN_PROVENANCE_KEYS` (line 97 of `align.py`). The
change is that the value can no longer be `"unknown"` in normal operation —
`"unknown"` means the pipeline was bypassed.

Each output also records a production timestamp:

```python
attrs["produced_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
```

And the exact DLC H5 filename used:

```python
attrs["dlc_h5_filename"] = Path(dlc_key).name
```

---

### 5. Elimination of `"unknown"` as a Valid State

#### 5.1 `resolve_champion_id` becomes `require_champion_id`

The current `resolve_champion_id()` returns `"unknown"` on mismatch. This
is replaced with a function that raises on mismatch:

```python
def require_champion_match(
    model_name: str,
    architecture: str | None,
    snapshot: str,
    champion_manifest: dict,
) -> str:
    """Return the champion_id if the triplet matches the manifest.

    Raises
    ------
    ChampionMismatchError
        If the triplet does not match the manifest.
    ValueError
        If architecture is None (SuperAnimal baseline — cannot be champion).
    """
```

The old `resolve_champion_id()` is kept as deprecated (for the frontend's
soft-check path where raising is not appropriate), but no pipeline script
calls it.

#### 5.2 What about pre-champion data?

Existing HDF5 files that carry `dlc_champion_id="unknown"` are treated as
stale by the frontend (unchanged from today). The pipeline re-run overwrites
them with files carrying the real champion ID.

---

### 6. Inference Pipeline Changes

#### 6.1 Session skip logic

`infer()` currently skips sessions that already have results in
`pose-finetuned/`:

```python
existing_resp = s3.list_objects_v2(
    Bucket=DERIVATIVES_BUCKET,
    Prefix=f"{FINETUNED_PREFIX}/{sub}/{ses_id}/",
    MaxKeys=1,
)
if existing_resp.get("Contents"):
    print("  Already has results, skipping")
```

This is dangerous: if a previous run wrote partial or wrong results, they
are kept. Change to: **always re-run inference when `--force` is passed;
when skipping, verify that the existing file matches the model being used
in this run** (compare the DLC project name and architecture in the
filename against what `config.yaml` would produce).

#### 6.2 Upload to staging, not pose/

Inference writes to `pose-finetuned/` (staging). This is correct and
unchanged. The key change is that promotion from staging to `pose/` is a
separate, verified step (Section 2).

---

### 7. Logging Contract

Every function that touches model selection must log:

| Function | Log message |
|---|---|
| `select_champion_h5` | `"Selected pose file: {filename} (champion={champion_id}, arch={arch}, snap={snap})"` |
| `select_champion_h5` (no match) | `"FATAL: No pose file matches champion {champion_id}. Found: {list}. Under: {prefix}"` |
| `_promote_session` (delete) | `"Deleted {n} old files from pose/{sub}/{ses}/"` |
| `_promote_session` (copy) | `"Copied {n} files to pose/{sub}/{ses}/ from pose-finetuned/"` |
| `declare_champion` | `"Declared champion: {champion_id} (arch={arch}, snap={snap})"` |
| Stage 3 (verify) | `"Champion check PASSED: kinematics input matches {champion_id}"` or `"REFUSED: ..."` |
| Stage 5 (verify) | `"Champion check PASSED: kinematics.h5 carries {champion_id}"` or `"REFUSED: ..."` |

All messages use the `logging` module at `INFO` level (pass/select) or
`ERROR` level (refuse/fatal). No silent fallbacks.

---

### 8. Frontend Changes

#### 8.1 `select_best_dlc_h5` calls in frontend

Three frontend pages call `select_best_dlc_h5` or `select_best_dlc_h5_s3`
directly:

1. `frontend/pages/training_fit_page.py` (line 198)
2. `frontend/pages/tracking_quality_page.py` (line 61)
3. `frontend/pages/dlc_viewer_page.py` (line 134)

These must be migrated to use a champion-aware selection that does not
raise on mismatch (the frontend should show stale data with a warning,
not crash). A new frontend-specific helper wraps the champion selection
with a soft fallback:

```python
def select_pose_h5_for_display(
    s3_client, bucket: str, prefix: str, champion: dict | None,
) -> tuple[str | None, bool]:
    """Select pose H5 for display, with staleness flag.

    Returns (h5_key, is_current). If no champion exists or no match
    is found, falls back to the old heuristic and returns
    is_current=False.
    """
```

#### 8.2 No other frontend changes needed

The existing `is_session_current()`, `render_champion_staleness_warning()`,
and `load_session()` staleness-check machinery in `frontend/data.py` is
correct and unchanged. The redesign ensures that the `dlc_champion_id`
attribute in HDF5 files is always correct (never `"unknown"` for
pipeline-produced data), which makes the frontend checks reliable.

---

## Implementation Order

The changes must be applied in this order to avoid breaking the pipeline
during the transition.

### Phase 1: Core selection + exception (no pipeline impact yet)

**Files changed:**
- `src/hm2p/pose/select.py`

**Changes:**
1. Add `ChampionMismatchError` exception class.
2. Add `NoPoseDataError` exception class.
3. Add `select_champion_h5(h5_keys, champion_manifest) -> str`.
4. Add `select_champion_h5_s3(s3_client, bucket, prefix, champion_manifest) -> str`.
5. Add `require_champion_match(model_name, architecture, snapshot, manifest) -> str`.
6. Mark `select_best_dlc_h5` and `select_best_dlc_h5_s3` as deprecated.
7. Mark `resolve_champion_id` as deprecated.

**Tests:**
- `tests/pose/test_select.py`: Add tests for `select_champion_h5` covering:
  exact match, no match (raises), multiple matches (warning + first),
  mixed old/new files, `_single`/`_filtered` exclusion.
- Test `require_champion_match` raises on mismatch, returns ID on match.

### Phase 2: Clean promotion

**Files changed:**
- `scripts/run_dlc_retrain.py`

**Changes:**
1. Extract promotion into `_promote_session(s3, bucket, sub, ses, finetuned_prefix)`.
2. `_promote_session` deletes all objects under `pose/{sub}/{ses}/` before
   copying from `pose-finetuned/`.
3. Reorder: `declare_champion()` is called BEFORE `_promote_all_sessions()`.
4. Add `_verify_all_promotions()` that runs `select_champion_h5_s3` for
   a sample of sessions after promotion.
5. Remove `promoted.json` writes from promotion (retire the mechanism).

**Tests:**
- Mock-based tests for `_promote_session` verifying delete-before-copy.
- Test that declaration failure prevents promotion.

### Phase 3: Stage 3 enforcement

**Files changed:**
- `scripts/run_stage3_kinematics.py`

**Changes:**
1. Replace `find_dlc_h5()` with `select_champion_h5_s3()`.
2. Require champion manifest at script start (exit 1 if absent).
3. Remove `resolve_champion_id()` call — the champion ID comes directly
   from the manifest (since `select_champion_h5_s3` already verified the
   file matches).
4. Pass `champion_manifest["champion_id"]` directly as `dlc_champion_id`.
5. Write `dlc_h5_filename` and `produced_at` to kinematics.h5 attrs.

### Phase 4: Stage 5 enforcement

**Files changed:**
- `scripts/run_stage5_sync.py`

**Changes:**
1. Load champion manifest at script start (exit 1 if absent).
2. After downloading `kinematics.h5`, read its `dlc_champion_id` attr.
3. Compare against `champion_manifest["champion_id"]`.
4. If mismatch: refuse to run, return `"error_stale_kinematics"`.
5. Log the champion check result for every session.

### Phase 5: Frontend migration

**Files changed:**
- `frontend/pages/training_fit_page.py`
- `frontend/pages/tracking_quality_page.py`
- `frontend/pages/dlc_viewer_page.py`

**Changes:**
1. Replace `select_best_dlc_h5` / `select_best_dlc_h5_s3` calls with
   `select_pose_h5_for_display()`.
2. Show staleness warning when `is_current=False`.

### Phase 6: Retire `promoted.json`

**Files changed:**
- `scripts/promote_dlc_model.py` — add deprecation notice to docstring;
  the script is kept for backward compatibility but is no longer called
  by the pipeline.
- `src/hm2p/pose/select.py` — remove `_load_promoted_json` and the
  `promoted.json` check from `select_best_dlc_h5_s3`.

---

## Data Flow After Redesign

```
Training completes
    |
    v
Inference: 26 sessions -> pose-finetuned/{sub}/{ses}/*.h5
    |
    v
declare_champion() -> writes dlc-champion.json to S3
    |                  [GATE: if this fails, STOP]
    v
_promote_all_sessions():
    for each session:
        1. DELETE all files in pose/{sub}/{ses}/
        2. COPY from pose-finetuned/{sub}/{ses}/
        3. VERIFY copy succeeded
    |
    v
_verify_all_promotions():
    pick 3 sample sessions
    select_champion_h5_s3() for each -> must succeed
    |
    v
Stage 3 (kinematics):
    1. REQUIRE champion manifest (exit if absent)
    2. select_champion_h5_s3() -> get pose H5 matching champion
       [GATE: if no match, ERROR for this session]
    3. Stamp champion_id, h5_filename, produced_at into kinematics.h5
    |
    v
Stage 5 (sync):
    1. REQUIRE champion manifest (exit if absent)
    2. Read kinematics.h5 dlc_champion_id attr
    3. COMPARE against manifest
       [GATE: if mismatch, REFUSE this session]
    4. Merge kinematics + calcium -> sync.h5
    5. Provenance propagated automatically via _KIN_PROVENANCE_KEYS
    |
    v
Stage 6 (analysis):
    1. REQUIRE champion manifest
    2. Read sync.h5 dlc_champion_id attr
    3. COMPARE against manifest [GATE]
    4. Produce analysis.h5
```

Every arrow with `[GATE]` is a hard stop on mismatch. There are no
fallbacks, no `"unknown"` values, no silent degradation.

---

## Migration Plan

1. **Implement Phase 1** on a feature branch. Merge after tests pass.
   No pipeline impact — old functions still work.

2. **Implement Phases 2-4** together on one feature branch. These are
   tightly coupled (reorder + enforce).

3. **Run a full training + inference cycle** using the new code on EC2.
   This is the first run that produces a clean `pose/` (one model per
   session) and a reliable champion manifest.

4. **Run Stage 3 -> 5 -> 6** using the new enforcement code. Every
   session gets a real `dlc_champion_id`.

5. **Implement Phase 5** (frontend migration). At this point all data
   on S3 is consistent.

6. **Implement Phase 6** (retire promoted.json). Low priority — the
   code simply ignores it.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| S3 delete + copy is not atomic — a reader during promotion sees partial data | Promotion writes a `_promotion_in_progress` marker file before deleting. The frontend checks for this marker and shows "promotion in progress" instead of loading data. Marker is deleted after copy + verify. |
| Champion manifest is a single point of failure — if it is deleted or corrupted, the entire pipeline halts | The manifest is archived to `dlc-champion-history/` before every overwrite. Recovery: copy the latest archive back to `dlc-champion.json`. The `declare_dlc_champion.py` script can re-declare from CLI args. |
| Frontend pages crash if `select_champion_h5_s3` raises | Frontend uses `select_pose_h5_for_display()` which catches the exception and falls back with `is_current=False`. Only pipeline scripts use the hard-raising version. |
| Existing `pose/` directories have files from multiple models | The first full training run under the new code will clean-promote all 26 sessions, deleting old files. Until then, Stage 3 uses the new champion-aware selection which ignores non-matching files (or errors if none match). |

---

## Appendix: Specific Code Changes by File

### `src/hm2p/pose/select.py`

| Change | Lines affected | Description |
|---|---|---|
| Add `ChampionMismatchError` | New class | Exception with structured fields (expected, found, prefix) |
| Add `NoPoseDataError` | New class | Raised when prefix has no H5 files at all |
| Add `select_champion_h5()` | New function | Pure function: list of keys + manifest -> matching key or raise |
| Add `select_champion_h5_s3()` | New function | S3 wrapper: list, filter, delegate to `select_champion_h5` |
| Add `require_champion_match()` | New function | Replaces `resolve_champion_id` with a raising version |
| Deprecate `select_best_dlc_h5()` | Line 34 | Add `warnings.warn("deprecated", DeprecationWarning)` |
| Deprecate `select_best_dlc_h5_s3()` | Line 72 | Same |
| Deprecate `resolve_champion_id()` | Line 201 | Same |
| Remove `_load_promoted_json()` | Lines 338-347 | Dead code after Phase 6 |

### `scripts/run_dlc_retrain.py`

| Change | Lines affected | Description |
|---|---|---|
| Extract `_promote_session()` | New function | Delete-all + copy + verify for one session |
| Reorder declare/promote | Lines 1370-1458 | declare_champion() BEFORE promote loop |
| Add `_verify_all_promotions()` | New function | Sample-check 3 sessions after promotion |
| Hard-fail on declare error | Line 1454 | Remove try/except — declaration failure stops the run |

### `scripts/run_stage3_kinematics.py`

| Change | Lines affected | Description |
|---|---|---|
| Require manifest | Line 369 | `sys.exit(1)` if `get_champion_manifest()` returns None |
| Replace `find_dlc_h5()` | Line 190 | Use `select_champion_h5_s3()` |
| Remove `resolve_champion_id()` call | Lines 221-231 | Use `champion_manifest["champion_id"]` directly |
| Add `dlc_h5_filename` to attrs | Line 298 | Pass `dlc_h5_filename=Path(dlc_key).name` to `run()` |

### `scripts/run_stage5_sync.py`

| Change | Lines affected | Description |
|---|---|---|
| Load manifest at start | After line 210 | Same pattern as Stage 3 |
| Read `dlc_champion_id` from kinematics.h5 | After line 96 | `h5py.File(kin_local)["attrs"]["dlc_champion_id"]` |
| Compare against manifest | New block | Refuse if mismatch |
| Log champion check | New log line | `"Champion check: kin={kin_cid}, manifest={manifest_cid}"` |

### `scripts/promote_dlc_model.py`

| Change | Description |
|---|---|
| Add deprecation notice | Docstring update only — script is kept but no longer called by pipeline |

### `src/hm2p/kinematics/compute.py`

| Change | Lines affected | Description |
|---|---|---|
| Add `dlc_h5_filename` parameter | Line 1808 | New param, default `"unknown"` |
| Add `produced_at` parameter | Line 1810 | New param, auto-set to UTC now |
| Write to attrs | Line 2052 | `"dlc_h5_filename": dlc_h5_filename, "produced_at": produced_at` |

### Frontend pages

| File | Change |
|---|---|
| `frontend/pages/training_fit_page.py` | Replace `select_best_dlc_h5_s3` with `select_pose_h5_for_display` |
| `frontend/pages/tracking_quality_page.py` | Replace `select_best_dlc_h5` with `select_pose_h5_for_display` |
| `frontend/pages/dlc_viewer_page.py` | Replace `select_best_dlc_h5` with `select_pose_h5_for_display` |
