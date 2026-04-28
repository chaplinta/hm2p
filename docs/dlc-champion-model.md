# DLC Champion Model — Design Specification

_Last updated: 2026-04-23 (decisions locked: Q1 auto-promotion, Q2 machine identity fields, Q3 single-PR migration, Q4 no retroactive champion)_

---

## Problem Statement

The pipeline produces per-session DLC pose outputs (`pose/{sub}/{ses}/*.h5`). As the
model is periodically retrained and improved, new inference runs overwrite or sit
alongside older outputs. Three problems follow:

1. No single authoritative record exists for "which model is currently the champion
   used for all analysis."
2. Rendered videos (`labelled_30fps.mp4`) carry no provenance — there is no way to
   tell whether a video was produced by the current champion or an older model.
3. The frontend has no mechanism to refuse to display analysis derived from
   a superseded model — it can only check whether an EC2 instance is running.

This document defines the champion-model concept, the provenance contract every
derivative must satisfy, the frontend enforcement mechanism, and the upgrade flow.

---

## 1. Champion Model Manifest

### 1.1 Location

One file acts as the single source of truth for the entire project:

```
s3://hm2p-derivatives/dlc-champion.json
```

This file is distinct from the per-session `pose/{sub}/{ses}/promoted.json` files,
which record which h5 was promoted for each session. The champion manifest records
what the current project-wide champion is. Per-session promoted.json files remain
as a session-level convenience cache.

### 1.2 Schema

```json
{
  "champion_id": "dlc-20260423-hrnetw32-snap50000",
  "model_name": "hm2p_hrnetw32_shuffle1",
  "architecture": "HrnetW32",
  "snapshot": "50000",
  "training_date": "2026-04-23",
  "training_run_id": "retrain-20260423T142500Z",
  "promoted_by_ec2_instance": "i-0abc123def456",
  "promoted_by_git_sha": "8f3a2b1",
  "promoted_at": "2026-04-23T14:30:00Z",
  "training_s3_prefix": "dlc-retrain/models/",
  "note": "",
  "notes": "Retrained after adding 183 labeled frames including head_midpoint."
}
```

**Field definitions:**

| Field | Type | Description |
| --- | --- | --- |
| `champion_id` | string | Stable, human-readable identifier. Format: `dlc-{YYYYMMDD}-{arch}-snap{N}`. Never reused. |
| `model_name` | string | DLC project/model name extracted from output filenames. |
| `architecture` | string | `"HrnetW32"` or `"Resnet50"`. |
| `snapshot` | string | Training iteration number as string. |
| `training_date` | string | ISO date the model training completed. |
| `training_run_id` | string | Identifier from the EC2 retrain run. Matches `run_id` in `dlc-retrain/_retrain_progress.json`. |
| `promoted_by_ec2_instance` | string | EC2 instance ID that ran `declare_dlc_champion.py`. Retrieved from the AWS Instance Metadata Service (IMDS) at promotion time. |
| `promoted_by_git_sha` | string | Output of `git rev-parse HEAD` from the hm2p repo clone on the EC2 instance at promotion time. Ties the promotion event to a specific codebase revision. |
| `promoted_at` | string | ISO 8601 UTC timestamp of promotion. |
| `training_s3_prefix` | string | S3 prefix where model weights live. |
| `note` | string | Optional free-text annotation. Empty string by default. Set by passing `--note "..."` when invoking `declare_dlc_champion.py` manually. |
| `notes` | string | Auto-generated description written by `run_dlc_retrain.py` summarising training parameters and labeled-frame count. |

### 1.3 Champion ID Format

The `champion_id` is the single string used to compare provenance across all
derivative files. Its format is:

```
dlc-{YYYYMMDD}-{arch_lower}-snap{N}
```

Examples:
- `dlc-20260423-hrnetw32-snap50000`
- `dlc-20260101-resnet50-snap30000`

The ID is constructed deterministically from the manifest fields so it can be
reconstructed from HDF5 attributes without fetching the manifest.

---

## 2. Provenance Contract

Every derivative that depends on DLC pose data must carry the same three identifiers
as attributes or sidecar metadata. These three fields form the **provenance triplet**:

| Field | Value |
| --- | --- |
| `dlc_model_name` | e.g. `"hm2p_hrnetw32_shuffle1"` |
| `dlc_snapshot` | e.g. `"50000"` |
| `dlc_champion_id` | e.g. `"dlc-20260423-hrnetw32-snap50000"` |

`dlc_champion_id` is new. The first two fields already exist in kinematics.h5 and
sync.h5; they are retained for backward compatibility with any code that reads them.

### 2.1 HDF5 Derivatives

`kinematics.h5`, `sync.h5`, and `analysis.h5` must all carry the provenance triplet
as HDF5 root attributes:

```python
# At write time (kinematics/compute.py, sync/align.py, analysis/save.py)
attrs["dlc_model_name"]   = model_name     # already written
attrs["dlc_snapshot"]     = snapshot       # already written
attrs["dlc_champion_id"]  = champion_id   # NEW — must be added
```

`analysis.h5` does not currently copy kinematics attrs. It must be updated to write
the provenance triplet sourced from its input `sync.h5`.

Reading back at verification time:
```python
with h5py.File(path, "r") as f:
    cid = f.attrs.get("dlc_champion_id", "")
```

### 2.2 Rendered Videos — Sidecar JSON

Rendered videos cannot carry embedded metadata without re-encoding. Instead each
rendered video gets a sidecar file at the same S3 path with `.provenance.json` suffix:

```
pose/{sub}/{ses}/labelled_30fps.provenance.json
pose/{sub}/{ses}/labelled_median_30fps.provenance.json
pose/{sub}/{ses}/labelled_pipeline_30fps.provenance.json
```

Schema:
```json
{
  "dlc_champion_id": "dlc-20260423-hrnetw32-snap50000",
  "dlc_model_name": "hm2p_hrnetw32_shuffle1",
  "dlc_snapshot": "50000",
  "source_h5_key": "pose/sub-1114353/ses-20210823T165950/video_hm2p_hrnetw32_shuffle1_snap50000.h5",
  "rendered_at": "2026-04-23T15:00:00Z",
  "render_mode": "raw"
}
```

This sidecar is written by `scripts/render_dlc_videos.py` immediately after the video
upload, in the same S3 `put_object` call batch.

### 2.3 Per-Session Promoted.json — Champion ID Added

The existing `pose/{sub}/{ses}/promoted.json` is extended with `dlc_champion_id`:

```json
{
  "h5_filename": "video_hm2p_hrnetw32_shuffle1_snap50000.h5",
  "h5_key": "pose/sub-1114353/ses-20210823T165950/video_hm2p_hrnetw32_shuffle1_snap50000.h5",
  "model_name": "hm2p_hrnetw32_shuffle1",
  "architecture": "HrnetW32",
  "snapshot": "50000",
  "dlc_champion_id": "dlc-20260423-hrnetw32-snap50000",
  "promoted_at": "2026-04-23T14:35:00Z"
}
```

`select_best_dlc_h5_s3` in `pose/select.py` is extended to return the
`dlc_champion_id` alongside the h5 key (or `None` if absent — for backward compat
with pre-champion promoted.json files). The caller passes this id into the kinematics
computation so it can be stamped into kinematics.h5.

### 2.4 What Does NOT Carry Provenance

The following are explicitly excluded because they do not depend on DLC pose data:

- `ca.h5` (Stage 4 — calcium-only, tracker-independent)
- `timestamps.h5` (Stage 0 — DAQ only)
- `ca_extraction/` (Stage 1 — Suite2p/CaImAn, tracker-independent)

---

## 3. Frontend Contract

### 3.1 Champion Loader

`frontend/data.py` acquires the champion manifest once per Streamlit session
(TTL 300 s cache — short enough to reflect a promotion within 5 minutes):

```python
@st.cache_data(ttl=300)
def get_dlc_champion() -> dict | None:
    """Load the current DLC champion manifest from S3.

    Returns the parsed dict, or None if the file does not exist or cannot
    be parsed. None means no champion has been declared (pre-champion state).
    """
    data = download_s3_bytes(DERIVATIVES_BUCKET, "dlc-champion.json")
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception:
        log.warning("dlc-champion.json is present but not valid JSON")
        return None
```

### 3.2 Session Currency Check

A helper function in `frontend/data.py` performs the staleness check for a single
session. All pages call this; no page reimplements the logic:

```python
def is_session_current(
    session_attrs: dict,
    champion: dict | None,
) -> tuple[bool, str]:
    """Check whether a session's derivatives match the current DLC champion.

    Parameters
    ----------
    session_attrs:
        Dict of HDF5 root attributes loaded from sync.h5 (or analysis.h5).
        Must contain key "dlc_champion_id" for a definitive check.
    champion:
        The current champion manifest from get_dlc_champion(), or None.

    Returns
    -------
    (is_current, reason)
        is_current: True if the session is definitely current, False if stale
        or if currency cannot be determined.
        reason: human-readable explanation, shown in the frontend warning.
    """
    if champion is None:
        # No champion declared yet — pre-champion state. Display with note.
        return True, "no champion declared"

    session_cid = session_attrs.get("dlc_champion_id", "")
    champion_cid = champion.get("champion_id", "")

    if not session_cid:
        # Derivative predates the champion system — treat as stale.
        return False, (
            f"Derivative predates the champion system. "
            f"Current champion: {champion_cid}. Re-run Stages 3–6."
        )

    if session_cid != champion_cid:
        return False, (
            f"Derivative produced by model '{session_cid}', "
            f"current champion is '{champion_cid}'. Re-run Stages 3–6."
        )

    return True, "current"
```

### 3.3 Staleness Banner

A shared UI function renders a prominent warning when a session is stale. This is
the only UI element for staleness — a banner at the top of the page content:

```python
def render_champion_staleness_warning(reason: str) -> None:
    """Render a staleness warning banner in the page body.

    Call after st.title() / st.caption() and before any analysis content.
    Only call when is_session_current() returns False.
    """
    st.warning(
        f"This session's data was produced by a superseded DLC model. "
        f"{reason} "
        f"Data is shown for reference only and should not be used for analysis.",
        icon="⚠",
    )
```

### 3.4 Per-Page Enforcement

Pages that display DLC-derived data (sync.h5 / analysis.h5) follow this pattern:

```python
# At the top of page rendering, after session selection:
champion = get_dlc_champion()
current, reason = is_session_current(session.get("attrs", {}), champion)
if not current:
    render_champion_staleness_warning(reason)
    # Do NOT return here — show data with warning so QC is still possible.
```

The choice is **show with warning, not refuse to show**. Rationale: refusing to
display stale sessions would block QC of those sessions, which is sometimes the
reason you are looking at them. The warning makes staleness unmistakable without
destroying utility.

The one exception is video currency checks in `dlc_viewer_page.py` (Section 3.5
below), where videos are compared directly against their sidecar provenance.

### 3.5 Video Currency Check in DLC Viewer

The DLC viewer loads videos by S3 key. Before showing a video, it checks the
sidecar `.provenance.json`:

```python
@st.cache_data(ttl=300)
def get_video_champion_id(sub: str, ses: str, mode: str) -> str | None:
    """Return the champion_id recorded in the video sidecar provenance file.

    Returns None if the sidecar does not exist (video predates champion system).
    """
    fname = VIDEO_FILENAMES.get(mode, "labelled_30fps.mp4")
    base = fname.rsplit(".", 1)[0]
    sidecar_key = f"pose/{sub}/{ses}/{base}.provenance.json"
    data = download_s3_bytes(DERIVATIVES_BUCKET, sidecar_key)
    if data is None:
        return None
    try:
        return json.loads(data).get("dlc_champion_id")
    except Exception:
        return None
```

The viewer then checks this against the current champion before syncing:

```python
def _video_is_current(sub: str, ses: str, mode: str, champion: dict | None) -> bool:
    if champion is None:
        return True  # pre-champion state, cannot determine
    video_cid = get_video_champion_id(sub, ses, mode)
    if video_cid is None:
        return False  # no provenance = predates champion system = stale
    return video_cid == champion.get("champion_id")
```

The "Sync all rendered videos to local cache" button guards against syncing stale
videos:

```python
if not _video_is_current(sub, ses, mode, champion):
    st.warning(
        f"Video for {ses} was rendered from a superseded model. "
        "Re-render before syncing.",
        icon="⚠",
    )
    # Skip this session in the sync loop.
    continue
```

### 3.6 Pages That Must Enforce the Champion Contract

The following pages display DLC-derived data and must call `is_session_current()`:

**Analysis / Science pages** (derive from sync.h5 or analysis.h5):
- `hd_tuning_page.py`
- `decoder_page.py`
- `stability_page.py`
- `rastermap_page.py`
- `maze_animation_page.py`
- `ahv_page.py`
- `gain_page.py`
- `anchoring_page.py`
- `light_page.py`, `light_compare_page.py`
- `correlations_page.py`
- `population_page.py`, `pop_dynamics_page.py`, `population_activity_page.py`
- `info_theory_page.py`
- `events_page.py`, `event_dynamics_page.py`
- `compare_page.py`, `trace_compare_page.py`
- `speed_page.py`
- `timeline_page.py`
- `explorer_page.py`
- `gallery_page.py`
- `summary_page.py`
- `analysis_page.py`
- `moseq_page.py`, `moseq_explore_page.py`, `moseq_exemplars_page.py`

**QC pages** (derive from pose/, kinematics.h5, or rendered videos):
- `dlc_viewer_page.py` (video currency check via sidecar, Section 3.5)
- `tracking_quality_page.py`
- `training_qc_page.py`
- `training_fit_page.py`
- `behaviour_page.py`
- `sync_page.py`

**Pages that do NOT need the check** (calcium-only or independent of pose):
- `calcium_page.py`
- `cascade_page.py`
- `signal_quality_page.py`
- `suite2p_page.py`
- `roi_viewer_page.py`
- `neuropil_analysis_page.py`
- `zdrift_page.py`
- `patching_*_page.py`
- `anatomy_page.py`

### 3.7 Pipeline Page

`pipeline_page.py` currently shows an overall stage table. It must be extended to
show the champion model identity in the DLC Training row:

```
Stage 2a — DLC Training    Complete (1/1)    Champion: dlc-20260423-hrnetw32-snap50000
```

And when a re-run is in progress, show:
```
Stage 2b — DLC Inference   Re-running (12/26)   Upcoming champion: <run_id>
```

---

## 4. Upgrade Flow — When a New Model Becomes Champion

### 4.1 Step-by-Step

1. **Train + infer + declare (automatic).** Run `scripts/launch_dlc_finetune_ec2.py`. The
   instance runs `scripts/run_dlc_retrain.py`, which:
   - Fine-tunes the model for up to `maxiters` iterations
   - Runs inference on all 26 sessions → `pose-finetuned/{sub}/{ses}/`
   - Promotes the best snapshot: calls `promote_dlc_model.py` to write
     `pose/{sub}/{ses}/promoted.json` for all 26 sessions
   - Declares the new champion: calls `declare_dlc_champion.py` (Section 4.2),
     which writes `dlc-champion.json` and clears `pipeline_rerun.json`
   - Self-terminates

   `declare_dlc_champion.py` is called automatically as the final step of a
   successful `run_dlc_retrain.py` run. It can also be invoked manually from the
   command line (see Section 4.2) if a re-declaration or correction is needed.

2. **Review inference quality.** Use `training_qc_page.py` and `dlc_viewer_page.py`
   to inspect outputs. The `dlc-champion.json` manifest is already written at this
   point, so the frontend shows the new champion identity.

3. **Rebuild downstream derivatives.** Run:
   ```bash
   uv run python scripts/run_downstream_pipeline.py
   ```
   This re-runs Stages 3 → 3b → 5 → 6 for all 26 sessions. Each output file
   now carries `dlc_champion_id` in its HDF5 root attrs.

4. **Re-render videos.** Run:
   ```bash
   uv run python scripts/render_dlc_videos.py --all
   ```
   The script writes `.provenance.json` sidecars alongside each video.

5. **Verify.** The pipeline page and all analysis pages now show the new champion.
   Sessions processed before step 3 completes will show the staleness warning
   until their derivatives are rebuilt.

### 4.2 New Script: `scripts/declare_dlc_champion.py`

This script performs the atomic promotion step. It is called automatically by
`run_dlc_retrain.py` at the end of a successful training+inference run. It can
also be invoked directly from the command line when a manual re-declaration is
needed (e.g. to correct metadata or add a note).

Steps:
1. Constructs `champion_id` from model name, architecture, and snapshot.
2. Reads EC2 instance ID from IMDS (`http://169.254.169.254/latest/meta-data/instance-id`).
3. Reads current git SHA (`git rev-parse HEAD`) from the hm2p repo clone on the instance.
4. Reads the current `dlc-champion.json` (if it exists) and archives it to
   `dlc-champion-history/{champion_id}.json` on S3 before overwriting.
5. Writes the new `dlc-champion.json` with `promoted_by_ec2_instance`,
   `promoted_by_git_sha`, and `promoted_at`.
6. Deletes `pipeline_rerun.json` (clearing the "pending re-run" state).
7. Logs the promotion event to `dlc-champion-history/promotions.log` on S3.

Invocation (automatic, from `run_dlc_retrain.py`):
```python
# Called at end of successful retrain run — no user interaction required
declare_champion(model_name=..., snapshot=..., training_run_id=..., notes=...)
```

Manual invocation (for corrections or re-declarations):
```bash
uv run python scripts/declare_dlc_champion.py \
    --model-name MODEL_NAME \
    --snapshot SNAPSHOT_ITER \
    --training-run-id RUN_ID \
    [--note "optional free-text annotation"] \
    [--dry-run]
```

The `--note` flag writes to the `note` field in the manifest (distinct from the
auto-generated `notes` field). Use it to annotate a re-declaration with a reason.

### 4.3 What Happens to Old Derivatives

Old derivatives (kinematics.h5, sync.h5, analysis.h5, videos) produced by the
previous champion are **kept in S3**. They are not deleted. They become stale in the
sense that `dlc_champion_id` no longer matches, and the frontend shows the warning.
Once re-run completes, the files are overwritten in place and the warning disappears.

Old rendered videos are overwritten in place by `render_dlc_videos.py` (no
`--skip-existing` flag when running a full re-render after a champion change).

### 4.4 Champion History

The archived `dlc-champion-history/{old_champion_id}.json` files provide a full audit
trail of every promoted model. This is append-only — nothing is deleted from the
history prefix. It is separate from `promoted.json` (per-session) and from the git
history of the trained DLC config in `sourcedata/trackers/dlc/`.

---

## 5. Codebase Touch Points

The following files require changes to implement the champion system. Changes are
ordered by dependency (earlier changes enable later ones).

### Phase 1 — Core Infrastructure (no UI changes yet)

| File | Change |
| --- | --- |
| `src/hm2p/pose/select.py` | Extend `select_best_dlc_h5_s3` to return `(h5_key, champion_id)`. Add `get_champion_manifest(s3, bucket)` function. Add `compute_champion_id(model_name, architecture, snapshot)` helper. |
| `scripts/promote_dlc_model.py` | Stamp `dlc_champion_id` into each `promoted.json` when writing it. Requires fetching `dlc-champion.json` once at script start. |
| `scripts/declare_dlc_champion.py` | New script (Section 4.2). Records `promoted_by_ec2_instance`, `promoted_by_git_sha`, `promoted_at`. Supports `--note` for optional manual annotation. |
| `scripts/run_dlc_retrain.py` | Add call to `declare_dlc_champion.py` (or its importable function) as the final step of a successful training+inference run. The script already self-terminates on completion — champion declaration must happen before that. |

### Phase 2 — Derivative Provenance

| File | Change |
| --- | --- |
| `src/hm2p/kinematics/compute.py` | Add `dlc_champion_id` parameter to `compute_kinematics(...)`. Write it to `attrs["dlc_champion_id"]` at line ~1844. |
| `scripts/run_stage3_kinematics.py` | Pass `champion_id` from `find_dlc_h5()` into `run_session()`. Fetch manifest once at script start; fall back to `None` if absent. |
| `src/hm2p/sync/align.py` | Copy `dlc_champion_id` from `kin_attrs` into `sync.h5` root attrs (add it to the list at line 153). |
| `src/hm2p/analysis/save.py` | Copy provenance triplet from input `sync.h5` attrs into `analysis.h5` root attrs. |
| `scripts/render_dlc_videos.py` | After uploading each video, write `.provenance.json` sidecar (Section 2.2). Pass `champion_id` through from `render_session()`. |

### Phase 3 — Frontend

| File | Change |
| --- | --- |
| `frontend/data.py` | Add `get_dlc_champion()`, `is_session_current()`, `render_champion_staleness_warning()`, `get_video_champion_id()`, `_video_is_current()`. |
| `frontend/pages/dlc_viewer_page.py` | Use `_video_is_current()` to guard the video sync button. Add champion identity display. |
| `frontend/pages/pipeline_page.py` | Show champion identity in the DLC Training row. |
| All analysis/QC pages listed in Section 3.6 | Call `is_session_current()` and `render_champion_staleness_warning()`. |

### Phase 4 — Migration of Existing Data

See Section 6.

---

## 6. Migration Path for Existing Data

At the time this design is adopted, 26 sessions of derivatives exist that were
produced before the champion system. They lack `dlc_champion_id` in their HDF5 attrs
and no sidecar provenance files exist for rendered videos.

### 6.1 Treatment of Pre-Champion Derivatives

These derivatives are treated as stale. `is_session_current()` returns `False` for
any session whose `dlc_champion_id` attribute is absent (Section 3.2). The frontend
will show a staleness warning on all 26 sessions until Phase 2 re-processing is
complete. This is intentional — the warnings motivate the re-run without blocking QC.

No retroactive champion is declared for pre-Phase-2 derivatives. The first
`dlc_champion_id` written into the system will be a real finetuned-snapshot ID
produced by the first full training run after Phase 1 is implemented.
Pre-existing derivatives from before Phase 2 are simply stale pending that re-run.

### 6.2 Migration Timeline

1. Implement Phase 1 (pose/select.py, declare_dlc_champion.py, run_dlc_retrain.py hook).
2. Run the next DLC training job via `launch_dlc_finetune_ec2.py`. The job
   auto-declares the first real champion on completion.
3. Implement Phase 2 (derivative provenance).
4. Re-run Stages 3 → 5 → 6 for all 26 sessions to stamp champion_id into HDF5.
5. Re-run `render_dlc_videos.py --all` to produce sidecar files.
6. Implement Phase 3 (frontend — single PR, all pages at once). By this point all
   26 sessions are current and the warnings only appear for future stale states.

### 6.3 Rendered Videos

Existing rendered videos have no sidecar. `_video_is_current()` returns `False`
for any video with no sidecar (no provenance = predates champion system = stale).
The DLC viewer warns about these videos and excludes them from bulk sync operations.
Individual playback still works — the warning is non-blocking. After
`render_dlc_videos.py --all` is re-run in step 5 above, sidecars are written and
the warnings disappear.

---

## 7. Relation to Existing Rerun System

The existing `pipeline_rerun.json` / `_get_rerun_status()` mechanism handles the
"EC2 is actively running" state — it marks downstream stages as "pending re-run"
during an active training or inference job. This mechanism is complementary to the
champion system, not replaced by it:

| Mechanism | Covers |
| --- | --- |
| `pipeline_rerun.json` | Active EC2 run in progress — stages show as "pending re-run" |
| `dlc-champion.json` + `dlc_champion_id` | Post-run staleness — data exists but was produced by a superseded model |

After a successful promotion via `declare_dlc_champion.py`, `pipeline_rerun.json`
is deleted. The pipeline page then shows Stage 2a as Complete and downstream stages
as Stale (because their derivatives lack the new champion_id) until re-run.

The `DOWNSTREAM_DEPS` dict in `frontend/data.py` does not need to change.

---

## 8. Schema Summary

### `s3://hm2p-derivatives/dlc-champion.json`

```json
{
  "champion_id": "dlc-{YYYYMMDD}-{arch}-snap{N}",
  "model_name": "<string>",
  "architecture": "<HrnetW32|Resnet50>",
  "snapshot": "<string>",
  "training_date": "<ISO date>",
  "training_run_id": "<string>",
  "promoted_by_ec2_instance": "<EC2 instance ID>",
  "promoted_by_git_sha": "<7-char git SHA>",
  "promoted_at": "<ISO 8601 UTC>",
  "training_s3_prefix": "<string>",
  "note": "<string — empty by default; set via --note flag on manual invocation>",
  "notes": "<string — auto-generated by run_dlc_retrain.py>"
}
```

### `pose/{sub}/{ses}/promoted.json` (extended)

```json
{
  "h5_filename": "<string>",
  "h5_key": "<string>",
  "model_name": "<string>",
  "architecture": "<string>",
  "snapshot": "<string>",
  "dlc_champion_id": "<string>",
  "promoted_at": "<ISO 8601 UTC>"
}
```

### `pose/{sub}/{ses}/labelled_30fps.provenance.json`

```json
{
  "dlc_champion_id": "<string>",
  "dlc_model_name": "<string>",
  "dlc_snapshot": "<string>",
  "source_h5_key": "<string>",
  "rendered_at": "<ISO 8601 UTC>",
  "render_mode": "<raw|median|pipeline>"
}
```

### HDF5 root attributes (kinematics.h5, sync.h5, analysis.h5)

New attribute added alongside existing `dlc_model_name` and `dlc_snapshot`:

```
dlc_champion_id  str  — e.g. "dlc-20260423-hrnetw32-snap50000"
```

---

## 9. Why This Design Is Fool-Proof

The design has one failure mode: a developer forgets to call `is_session_current()`
on a new page. To guard against this, the following convention is enforced:

Every page that calls `load_session()` (the function in `frontend/data.py` that
returns the dict of sync.h5 data) receives the champion check as part of that call:

`load_session()` is extended to accept a `champion` parameter and, if the session
data is stale, to attach a `"stale"` key to the returned dict:

```python
session = load_session(sub, ses, champion=get_dlc_champion())
if session.get("stale"):
    render_champion_staleness_warning(session["stale_reason"])
```

Because the champion check is embedded in `load_session()` — the one function every
data-loading page must call — any page that loads session data and omits the check
simply never surfaces the staleness flag, which is a visible omission during code
review. Pages that explicitly pass `champion=None` to opt out must document why.

This does not make it impossible to display stale data (the page can still ignore
`session["stale"]`), but it makes staleness visible in the data structure every
page receives, so omitting the check requires active effort rather than forgetting.
