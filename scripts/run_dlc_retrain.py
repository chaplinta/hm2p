#!/usr/bin/env python3
"""DLC retraining + re-inference — runs on EC2.

Downloads labeled data from S3, fine-tunes DLC, then re-runs inference
on all 26 sessions. Called by the EC2 user-data script
(launch_dlc_finetune_ec2.py).

Two training paths:

- **ImageNet HRNet (default):** trains HRNet-W32 from ImageNet weights
  (current main path). 400 epochs.
- **SuperAnimal memory-replay (``--sa-finetune``):** warm-starts from
  the SuperAnimal-TopViewMouse HRNet-W32 release using DLC's
  ``build_weight_init`` + ``create_training_dataset(weight_init=...)``
  + ``train_network`` API. Memory-replay protocol per
  Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A,
  Mathis MW. 2024. "SuperAnimal pretrained pose estimation models for
  behavioral analysis." *Nature Communications* 15:5165.
  doi:10.1038/s41467-024-48792-2.
  Code: https://github.com/DeepLabCut/DeepLabCut. 120 epochs, Adam
  lr 5e-5, frozen BN running stats, step LR decay at 90/110.

Usage (on EC2)::

    python scripts/run_dlc_retrain.py --train --infer
    python scripts/run_dlc_retrain.py --train-only
    python scripts/run_dlc_retrain.py --infer-only
    python scripts/run_dlc_retrain.py --sa-finetune
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

import boto3
import numpy as np

REGION = "ap-southeast-2"
DERIVATIVES_BUCKET = "hm2p-derivatives"
RAWDATA_BUCKET = "hm2p-rawdata"
RETRAIN_PREFIX = "dlc-retrain"
FINETUNED_PREFIX = "pose-finetuned"


def get_instance_id() -> str:
    """Return the EC2 instance ID from the metadata service, or 'unknown'."""
    try:
        resp = urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
        )
        return resp.read().decode().strip()
    except Exception:
        return "unknown"


def update_progress(s3, status: str, **extra: object) -> None:
    """Write progress JSON to S3.

    Progress updates are best-effort — upload failures are logged as warnings
    and do not propagate to the caller.
    """
    progress = {
        "status": status,
        "updated": datetime.datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    tmp = Path("/tmp/_retrain_progress.json")
    tmp.write_text(json.dumps(progress, indent=2))
    try:
        s3.upload_file(str(tmp), DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/_retrain_progress.json")
    except Exception as e:
        print(f"  WARNING: progress update failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# SA-finetune helpers (Ye et al. 2024, doi:10.1038/s41467-024-48792-2)
# ---------------------------------------------------------------------------

#: Detector candidate order: prefer the v2 model (DLC ≥ 3.0 default), fall
#: back to the original. The probe is performed by ``_resolve_sa_detector``.
SA_DETECTOR_CANDIDATES = ("fasterrcnn_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn")

#: Conversion-array indices (project bodyparts -> SA-TVM keypoint indices).
#: Mirrors the 8-keypoint identity-mapping confirmed in v2 plan §3.
SA_CONVERSION_ARRAY = [0, 1, 2, 26, 7, 8, 9, 13]

#: Project bodyparts in canonical order. The conversion array assumes this
#: ordering.
PROJECT_BODYPARTS = (
    "nose_tip", "left_ear", "right_ear", "head_midpoint",
    "neck", "mid_back", "mouse_center", "tail_base",
)


def _ensure_default_net_type_hrnet(config_path: Path) -> bool:
    """Ensure ``default_net_type: hrnet_w32`` is set in ``config.yaml``.

    Per architect open-question #5, the on-the-fly rewrite-with-warning
    avoids committing a separate config.yaml change. Returns True iff a
    rewrite was performed.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cur = cfg.get("default_net_type")
    if cur == "hrnet_w32":
        return False
    print(
        f"  WARNING: default_net_type was {cur!r}; rewriting to 'hrnet_w32' "
        f"in {config_path}"
    )
    cfg["default_net_type"] = "hrnet_w32"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return True


def _validate_sa_conversion_table(config_path: Path) -> None:
    """Assert the ``conversion_tables`` block covers every project bodypart.

    Reads the project's ``config.yaml`` and verifies that every bodypart in
    :data:`PROJECT_BODYPARTS` has an entry in
    ``SuperAnimalConversionTables.superanimal_topviewmouse``.

    Raises
    ------
    ValueError
        Naming the missing bodypart(s).
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tables = (
        cfg.get("SuperAnimalConversionTables", {})
        .get("superanimal_topviewmouse", {})
    )
    missing = [bp for bp in PROJECT_BODYPARTS if bp not in tables]
    if missing:
        raise ValueError(
            f"SuperAnimal conversion table missing entries for: {missing}. "
            f"Edit config.yaml under 'SuperAnimalConversionTables: "
            f"superanimal_topviewmouse:' before --sa-finetune."
        )


def _resolve_sa_detector(available_detectors: list[str]) -> str:
    """Resolve the SA detector name via the candidate list.

    Parameters
    ----------
    available_detectors
        Output of ``dlclibrary.get_available_detectors("superanimal_topviewmouse")``.

    Returns
    -------
    str
        The first detector name in :data:`SA_DETECTOR_CANDIDATES` that is
        actually available in DLC.

    Raises
    ------
    RuntimeError
        When neither candidate is present, with the available list inlined
        in the message so the operator can update the candidate order.
    """
    for name in SA_DETECTOR_CANDIDATES:
        if name in available_detectors:
            return name
    raise RuntimeError(
        f"None of {list(SA_DETECTOR_CANDIDATES)!r} are present in "
        f"dlclibrary.get_available_detectors('superanimal_topviewmouse'). "
        f"Available detectors: {available_detectors!r}"
    )


def _validate_sa_model_available(available_models: list[str]) -> None:
    """Assert the SA-TVM HRNet-W32 model is exposed by dlclibrary.

    ``available_models`` is the output of
    ``dlclibrary.get_available_models("superanimal_topviewmouse")``, which
    returns short names like ``["hrnet_w32", "resnet_50"]`` (NOT
    ``superanimal_topviewmouse_hrnet_w32`` — that prefixed form is only
    used by HuggingFace download paths).

    Raises
    ------
    RuntimeError
        With a clear message when the model is absent.
    """
    expected = "hrnet_w32"
    if expected not in available_models:
        raise RuntimeError(
            f"{expected!r} not in dlclibrary.get_available_models"
            f"('superanimal_topviewmouse'). Got: {available_models!r}. "
            f"Update dlclibrary or check the DLC release notes."
        )


def _check_sa_input_size(pytorch_cfg_path: Path) -> bool:
    """Warn if the SA shuffle's training-input size is not 256x256.

    Per design §6 pitfall #1: the SA-TVM HRNet was trained at 256x256.
    DLC may pick a different size on newer SA snapshot versions. Mismatch
    is a warning, not a fatal error — the gate will catch any regression.

    Returns True iff the size matches 256x256.
    """
    import yaml

    with open(pytorch_cfg_path) as f:
        pcfg = yaml.safe_load(f)
    size = pcfg.get("data", {}).get("train", {}).get("input_size")
    if size in ([256, 256], [256], 256):
        return True
    print(
        f"  WARNING: SA shuffle's data.train.input_size = {size!r}; "
        f"expected [256, 256]. Continuing (the promotion gate will catch "
        f"any regression)."
    )
    return False


def _apply_sa_augmentation_patch(pytorch_cfg_path: Path) -> None:
    """Apply v2 §4.3 augmentation tweaks to the SA shuffle's pytorch_config.

    The augmentation block is the only YAML edit that survives the
    `make_super_animal_finetune_config` path (the backbone block is
    written by DLC and must not be touched). Edits in place.
    """
    import yaml

    with open(pytorch_cfg_path) as f:
        pcfg = yaml.safe_load(f)
    train_aug = pcfg.setdefault("data", {}).setdefault("train", {})
    affine = train_aug.setdefault("affine", {})
    affine["rotation"] = 30
    affine["scaling"] = [0.7, 1.3]
    affine.setdefault("translation", 30)
    affine.setdefault("p", 0.7)
    train_aug["gaussian_noise"] = 10.0
    train_aug["motion_blur"] = True
    train_aug.setdefault("horizontal_flip", {"p": 0.5})
    train_aug.setdefault("vertical_flip", {"p": 0.5})
    train_aug.setdefault(
        "brightness_contrast",
        {"brightness_limit": 0.15, "contrast_limit": 0.10, "p": 0.5},
    )
    with open(pytorch_cfg_path, "w") as f:
        yaml.dump(pcfg, f)
    print(
        "  SA augmentation patch applied: rot=±30°, scale=0.7-1.3, "
        "noise=10, brightness/contrast=±15%/±10%, flip H+V."
    )


def _build_sa_notes(
    *, detector: str, conversion_array: list[int], epochs: int,
    lr: float, batch_size: int,
) -> str:
    """Build the auto-declared champion ``notes`` string for the SA path.

    Per design §1.3 step 7. Format is documented so the frontend can
    parse it back if necessary.
    """
    return (
        "Auto-declared by run_dlc_retrain.py (SA fine-tune). "
        f"init: superanimal_topviewmouse_hrnet_w32 (memory replay). "
        f"conversion_array: {conversion_array}. "
        f"detector: {detector}. "
        f"epochs: {epochs}; lr: {lr:g}; bs: {batch_size}; "
        f"freeze_bn_stats: True."
    )


def _train_sa_finetune(
    s3,
    work: Path,
    config_path: Path,
    *,
    epochs: int,
    batch_size: int,
) -> Path:
    """SuperAnimal memory-replay fine-tune (Ye et al. 2024).

    Runs the SA-finetune training path on a fresh shuffle. Pre-condition
    checks fail loud and fast (config.yaml `default_net_type`,
    SA conversion table coverage, dlclibrary detector + model
    availability). Augmentation patch is applied to the new shuffle's
    pytorch_config.yaml; backbone keys are left untouched.

    The SA snapshot, conversion-array channel slicing, and weight init
    are all handled by DLC's
    ``deeplabcut.modelzoo.weight_initialization.build_weight_init`` →
    ``deeplabcut.create_training_dataset(weight_init=...)`` →
    ``deeplabcut.train_network(...)`` API. The legacy
    ``superanimal_name`` / ``superanimal_transfer_learning`` kwargs are
    pre-3.0 and are NOT passed.

    Reference: Ye 2024 Methods §"Memory replay fine tuning" + Fig. 1d.
    """
    import deeplabcut
    import dlclibrary

    print("=== SA-finetune training path (memory replay) ===")
    update_progress(s3, "Training (SA): pre-flight checks")

    _ensure_default_net_type_hrnet(config_path)
    _validate_sa_conversion_table(config_path)
    _validate_sa_model_available(
        dlclibrary.get_available_models("superanimal_topviewmouse")
    )
    detector = _resolve_sa_detector(
        dlclibrary.get_available_detectors("superanimal_topviewmouse")
    )
    print(f"  Resolved SA detector: {detector}")

    update_progress(s3, "Training (SA): build_weight_init")
    from deeplabcut.modelzoo.weight_initialization import build_weight_init
    weight_init = build_weight_init(
        cfg=str(config_path),
        super_animal="superanimal_topviewmouse",
        model_name="hrnet_w32",
        detector_name=detector,
        with_decoder=True,
        memory_replay=False,  # disabled: DLC 3.0rc13 memory_replay has KeyError bugs
    )

    update_progress(s3, "Training (SA): create_training_dataset")
    new_shuffles = deeplabcut.create_training_dataset(
        str(config_path),
        weight_init=weight_init,
        num_shuffles=1,
        net_type="hrnet_w32",
    )
    # create_training_dataset returns a list of tuples:
    # [(trainingset_fraction, shuffle_index, (train_indices, test_indices)), ...]
    # We need just the integer shuffle index for train_network.
    raw_shuffle = new_shuffles[-1] if isinstance(new_shuffles, list) else new_shuffles
    if isinstance(raw_shuffle, (list, tuple)) and len(raw_shuffle) >= 2:
        sa_shuffle = int(raw_shuffle[1])
    else:
        sa_shuffle = int(raw_shuffle)
    print(f"  SA shuffle index: {sa_shuffle} (raw: {type(raw_shuffle).__name__})")

    # Locate the new shuffle's pytorch_config.yaml and apply the
    # augmentation patch. The 256x256 input-size check is a soft
    # warning per pitfall #1.
    pytorch_cfgs = sorted(work.rglob("pytorch_config.yaml"))
    if pytorch_cfgs:
        # Use the most recently-modified one (DLC's create_training_dataset
        # writes the new shuffle last).
        latest = max(pytorch_cfgs, key=lambda p: p.stat().st_mtime)
        _check_sa_input_size(latest)
        _apply_sa_augmentation_patch(latest)
    else:
        print("  WARNING: no pytorch_config.yaml found post-create_training_dataset")

    lr = 5e-5
    update_progress(s3, f"Training (SA): {epochs} epochs (lr={lr:g})")
    deeplabcut.train_network(
        str(config_path),
        shuffle=sa_shuffle,
        epochs=epochs,
        save_epochs=10,
        displayiters=100,
        batch_size=batch_size,
        pytorch_cfg_updates={
            "train_settings.optimizer.params.lr": lr,
            "model.backbone.freeze_bn_stats": True,
            "train_settings.scheduler.type": "MultiStepLR",
            "train_settings.scheduler.params.milestones": [90, 110],
            "train_settings.scheduler.params.gamma": 0.1,
        },
    )
    update_progress(s3, "Training (SA): train_network complete")

    # Stash a notes file the eventual declare_champion call will pick up.
    notes_text = _build_sa_notes(
        detector=detector,
        conversion_array=SA_CONVERSION_ARRAY,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    (work / "_sa_finetune_notes.txt").write_text(notes_text)
    print(f"  Notes stashed: {notes_text!r}")
    return config_path


def _upload_model_artifacts(s3, work: Path) -> None:
    """Upload trained model weights + eval CSVs to S3.

    Shared post-training step for both the ImageNet HRNet path and the
    SA-finetune path. Walks ``work/dlc-models-pytorch/`` (or
    ``dlc-models/`` for legacy TF runs) and uploads all files under
    ``s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/``.
    """
    print("Uploading model weights to S3 (shared helper)...")
    for model_dir_name in ("dlc-models-pytorch", "dlc-models"):
        dlc_train_dir = work / model_dir_name
        if not dlc_train_dir.exists():
            continue
        print(f"  Found {model_dir_name}/")
        n_files = 0
        for f in dlc_train_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(dlc_train_dir)
                key = f"{RETRAIN_PREFIX}/models/{rel}"
                s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
                n_files += 1
        print(f"  Uploaded {n_files} files to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/")
        # Upload SA-finetune notes if present (consumed by declare_champion).
        notes_path = work / "_sa_finetune_notes.txt"
        if notes_path.exists():
            s3.upload_file(
                str(notes_path), DERIVATIVES_BUCKET,
                f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt",
            )
        return
    print("  WARNING: no model directory found")


def train(s3, maxiters: int = 50000, epochs: int = 400, batch_size: int = 8,
          sa_finetune: bool = False) -> Path:
    """Download labels from S3, fine-tune DLC, upload model weights.

    Parameters
    ----------
    s3
        boto3 S3 client.
    maxiters
        Legacy TF iterations parameter (ignored under DLC 3.0 PyTorch and
        ignored under ``--sa-finetune``).
    epochs
        Training epochs. The CLI default is 400 for the ImageNet path and
        120 for the SA-finetune path; whatever the operator passes in
        propagates here.
    batch_size
        Training batch size (default 8).
    sa_finetune
        When True, runs the SuperAnimal memory-replay fine-tune path
        (Ye et al. 2024). When False, runs the legacy ImageNet HRNet
        path. Mutually exclusive at the API level — both paths share
        the same S3 download / upload scaffolding.
    """
    import deeplabcut

    work = Path("/tmp/dlc-retrain")
    work.mkdir(parents=True, exist_ok=True)

    # Download labeled data + config
    print("Downloading labeled data from S3...")
    subprocess.run(
        ["aws", "s3", "sync",
         f"s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/",
         str(work),
         "--exclude", "_*"],
        check=True,
    )

    config_path = work / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("No config.yaml in S3 dlc-retrain/")

    # Fix video paths in config (they reference Mac paths)
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Update project path
    cfg["project_path"] = str(work)

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"Config: {config_path}")
    print(f"Bodyparts: {cfg.get('bodyparts', [])}")
    print(f"Epochs: {epochs}")
    print(f"Mode: {'SA fine-tune (memory replay)' if sa_finetune else 'ImageNet HRNet'}")

    # Delete any stale dlc-models* dirs to ensure a clean shuffle build.
    # Done here once for both paths.
    for old_dir_name in ("dlc-models-pytorch", "dlc-models", "training-datasets"):
        old_dir = work / old_dir_name
        if old_dir.exists():
            shutil.rmtree(old_dir)
            print(f"  Deleted old {old_dir_name}/")

    if sa_finetune:
        _train_sa_finetune(
            s3, work, config_path, epochs=epochs, batch_size=batch_size,
        )
        # SA path runs train_network internally; the shared post-training
        # block (evaluation + uploads) follows below.
        update_progress(s3, "Training (SA): evaluating")
        deeplabcut.evaluate_network(str(config_path), plotting=False)
        update_progress(s3, "Training (SA): evaluation complete")
        _upload_model_artifacts(s3, work)
        update_progress(s3, "Training complete (SA fine-tune)")
        return config_path

    update_progress(s3, "Training: creating dataset")

    # Create training dataset (default ResNet50 config — we override below).
    print("Creating training dataset...")
    deeplabcut.create_training_dataset(str(config_path))
    update_progress(s3, "Training: dataset created")

    # Override config: switch to HRNet-W32 backbone (ImageNet pretrained
    # via timm, NOT SuperAnimal) + aggressive augmentation.
    pytorch_cfg_candidates = list(work.rglob("pytorch_config.yaml"))
    for pcfg_path in pytorch_cfg_candidates:
        with open(pcfg_path) as f:
            pcfg = yaml.safe_load(f)

        # Epochs
        if "train_settings" not in pcfg:
            pcfg["train_settings"] = {}
        pcfg["train_settings"]["epochs"] = epochs

        # HRNet-W32 backbone (ImageNet pretrained via timm).
        # DLC's HRNet implementation uses timm to load pretrained weights.
        old_backbone = pcfg.get("model", {}).get("backbone", {}).get("model_name", "?")
        print(f"  Overriding backbone: {old_backbone} → hrnet_w32")
        pcfg["model"]["backbone"] = {
            "model_name": "hrnet_w32",
            "type": "HRNet",
            "freeze_bn_stats": False,
            "freeze_bn_weights": False,
        }
        pcfg["net_type"] = "hrnet_w32"
        # HRNet-W32 outputs 32 channels (ResNet outputs 2048).
        # Head deconv layers must match the backbone output.
        n_bodyparts = len(pcfg.get("metadata", {}).get("bodyparts", []))
        if "heads" in pcfg["model"]:
            for head_cfg in pcfg["model"]["heads"].values():
                if "heatmap_config" in head_cfg:
                    head_cfg["heatmap_config"]["channels"] = [32, n_bodyparts or 8]
                if "locref_config" in head_cfg:
                    head_cfg["locref_config"]["channels"] = [32, (n_bodyparts or 8) * 2]
        print(f"  Head channels: 32 → {n_bodyparts} bodyparts")

        # Aggressive augmentation for overhead mouse tracking with
        # light/dark alternation and high pose variability.
        # Enable ImageNet pretraining (DLC HRNet template defaults to false)
        if "model" in pcfg and "backbone" in pcfg["model"]:
            pcfg["model"]["backbone"]["pretrained"] = True
            print("  backbone.pretrained = True (ImageNet)")

        # Augmentation: tuned for overhead mouse with light/dark and 184 frames.
        # Moderate augmentation — strong enough for generalisation but not so
        # extreme that the model rarely sees natural examples.
        if "data" in pcfg and "train" in pcfg["data"]:
            aug = pcfg["data"]["train"]
            if "affine" not in aug:
                aug["affine"] = {}
            aug["affine"]["rotation"] = 45          # ±45° (was ±180° — too extreme)
            aug["affine"]["scaling"] = [0.7, 1.4]   # ±30-40% (was 0.25-2.5x)
            aug["affine"]["translation"] = 30       # pixels
            aug["affine"]["p"] = 0.7
            # Brightness/contrast jitter: the IR filter leaks some 450nm
            # visible light and the IR illumination decays ~5-10% over a
            # 30-min session. ±15% brightness + ±10% contrast covers both.
            # Uses the hm2p patch to DLC's transforms.py (applied in
            # launch_dlc_finetune_ec2.py user-data script).
            aug["brightness_contrast"] = {
                "brightness_limit": 0.15,
                "contrast_limit": 0.1,
                "p": 0.5,
            }
            # Flips: keep — mouse is symmetric from above
            aug["horizontal_flip"] = {"p": 0.5}
            aug["vertical_flip"] = {"p": 0.5}
            # Noise: moderate
            aug["gaussian_noise"] = 15.0            # was 30 — too much
            aug["motion_blur"] = True
            # No hue/saturation jitter — images are grayscale (IR overhead camera)
            print(
                "  Augmentation: rot=±45°, scale=0.7-1.4x, "
                "brightness/contrast=±40%, hflip+vflip, noise=15"
            )

        with open(pcfg_path, "w") as f:
            yaml.dump(pcfg, f)
        print(f"  Config updated: {pcfg_path.name}")

    update_progress(s3, f"Training: HRNet-W32 ({epochs} epochs)")

    # Train
    print(f"Training HRNet-W32 for {epochs} epochs...")
    deeplabcut.train_network(
        str(config_path),
        maxiters=maxiters,
        displayiters=100,
        saveiters=5000,
    )
    update_progress(s3, f"Training: network trained ({epochs} epochs)")

    # Evaluate and compute per-bodypart metrics
    print("Evaluating network...")
    deeplabcut.evaluate_network(str(config_path), plotting=False)
    update_progress(s3, "Training: evaluation complete")

    # Run per-bodypart evaluation: load test predictions and ground truth,
    # compute RMSE per bodypart, upload as JSON.
    print("Computing per-bodypart metrics...")
    try:
        import pandas as _pd

        bodyparts = cfg.get("bodyparts", [])
        scorer = None
        # Find the evaluation predictions H5 (DLC saves predictions on test frames)
        eval_h5_files = list(work.rglob("*snapshot*_full.pickle")) + list(work.rglob("*snapshot*.h5"))

        # Simpler: find the results CSV and check if it has per-bodypart columns
        results_csvs = list(work.rglob("*results*.csv"))
        for rc in results_csvs:
            df = _pd.read_csv(rc, index_col=0)
            print(f"  Found: {rc.name}, shape={df.shape}, columns={list(df.columns)[:5]}")

        # The most reliable approach: run model on test frames manually
        # and compute RMSE per bodypart from predictions vs ground truth.
        # Find the labeled data and test split
        per_bp = {}
        for labeled_dir in work.rglob("CollectedData_*.h5"):
            gt = _pd.read_hdf(labeled_dir)
            # Get scorer and bodyparts from columns
            if gt.columns.nlevels >= 3:
                available_bps = gt.columns.get_level_values("bodyparts" if "bodyparts" in gt.columns.names else 1).unique()
                for bp in bodyparts:
                    if bp in available_bps:
                        scorer_name = gt.columns.get_level_values(0)[0]
                        x_vals = gt[(scorer_name, bp, "x")].values
                        y_vals = gt[(scorer_name, bp, "y")].values
                        valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
                        per_bp[bp] = {"n_labelled": int(valid.sum()), "n_total": len(x_vals)}
            break

        if per_bp:
            bp_json = work / "_per_bodypart_summary.json"
            bp_json.write_text(json.dumps(per_bp, indent=2))
            s3.upload_file(str(bp_json), DERIVATIVES_BUCKET,
                           f"{RETRAIN_PREFIX}/models/_per_bodypart_summary.json")
            print(f"  Per-bodypart label counts: { {k: v['n_labelled'] for k, v in per_bp.items()} }")
    except Exception as e:
        print(f"  Per-bodypart metrics failed: {e}")
        import traceback; traceback.print_exc()

    # Upload evaluation results (per-bodypart RMSE).
    # DLC may write these in evaluation-results/ or inside the model dir.
    eval_uploaded = 0
    for search_dir in [work / "evaluation-results", work]:
        for csv_file in search_dir.rglob("*results*.csv"):
            rel = csv_file.relative_to(work)
            key = f"{RETRAIN_PREFIX}/models/{rel}"
            s3.upload_file(str(csv_file), DERIVATIVES_BUCKET, key)
            eval_uploaded += 1
            print(f"  Uploaded eval: {rel}")
    if eval_uploaded == 0:
        print("  No evaluation result CSVs found")

    # Compute per-bodypart RMSE from predictions vs labels
    print("Computing per-bodypart RMSE...")
    try:
        import subprocess as _sp
        _r = _sp.run(
            [sys.executable, "scripts/compute_bodypart_rmse.py",
             "--pose-prefix", FINETUNED_PREFIX],
            capture_output=True, text=True,
        )
        print(_r.stdout[-500:] if _r.stdout else "  (no output)")
        if _r.returncode != 0:
            print(f"  Per-bodypart RMSE failed: {_r.stderr[-300:]}")
    except Exception as e:
        print(f"  Per-bodypart RMSE failed: {e}")

    # Upload model weights via boto3 (aws CLI may not be available)
    print("Uploading model weights to S3...")
    # DLC 3.0 PyTorch uses dlc-models-pytorch; legacy uses dlc-models
    for model_dir_name in ("dlc-models-pytorch", "dlc-models"):
        dlc_train_dir = work / model_dir_name
        if dlc_train_dir.exists():
            print(f"  Found {model_dir_name}/")
            for f in dlc_train_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(dlc_train_dir)
                    key = f"{RETRAIN_PREFIX}/models/{rel}"
                    s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
            n_files = sum(1 for _ in dlc_train_dir.rglob("*") if _.is_file())
            print(f"  Uploaded {n_files} files to s3://{DERIVATIVES_BUCKET}/{RETRAIN_PREFIX}/models/")
            break
    else:
        print("  WARNING: no model directory found")

    update_progress(s3, "Training complete", maxiters=maxiters)
    return config_path


def _download_session_video(  # noqa: ANN001
    s3, rawdata_bucket: str, sub: str, ses_id: str, dest_dir: Path
) -> None:
    """Download overhead .mp4 files for a session from S3 using boto3.

    Downloads all .mp4 files under ``rawdata/{sub}/{ses_id}/behav/`` except
    side-camera files (filename contains "side").

    Parameters
    ----------
    s3 : boto3 S3 client
    rawdata_bucket : str
    sub : str
        Subject identifier, e.g. ``sub-1114353``.
    ses_id : str
        Session identifier, e.g. ``ses-20210823T165950``.
    dest_dir : Path
        Local directory to download into.
    """
    prefix = f"rawdata/{sub}/{ses_id}/behav/"
    resp = s3.list_objects_v2(Bucket=rawdata_bucket, Prefix=prefix)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        filename = key.split("/")[-1]
        if not filename.endswith(".mp4"):
            continue
        if "side" in filename.lower():
            continue
        local_path = dest_dir / filename
        s3.download_file(rawdata_bucket, key, str(local_path))
        print(f"  Downloaded {filename}")


def infer(s3, config_path: Path, skip_failed: bool = False) -> None:
    """Run inference on all 26 sessions with the fine-tuned model.

    Parameters
    ----------
    s3 : boto3 S3 client
    config_path : Path
        Local path to the DLC config.yaml.
    skip_failed : bool
        If True, promote completed sessions even if some failed.
        If False (default), auto-promote is skipped when any session fails.
    """
    import deeplabcut

    # Read session list
    metadata = Path(__file__).resolve().parent.parent / "metadata" / "experiments.csv"
    sessions = []
    with open(metadata) as f:
        for row in csv.DictReader(f):
            eid = row["exp_id"]
            parts = eid.split("_")
            sessions.append({
                "exp_id": eid,
                "sub": f"sub-{parts[-1]}",
                "ses": f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}",
            })

    total = len(sessions)
    completed: list[str] = []
    failed: list[str] = []
    error_records: list[dict] = []
    run_id = datetime.datetime.utcnow().isoformat() + "Z"
    instance_id = get_instance_id()

    for i, ses in enumerate(sessions, 1):
        sub, ses_id = ses["sub"], ses["ses"]
        exp_id = ses["exp_id"]
        print(f"\n=== [{i}/{total}] {sub}/{ses_id} ===")

        # Progress: session starting
        update_progress(
            s3, f"Inference {i}/{total}: starting {sub}/{ses_id}",
            completed=len(completed), failed=len(failed), total=total,
            current_session=exp_id,
        )

        work = Path(f"/tmp/dlc-infer/{sub}/{ses_id}")
        work.mkdir(parents=True, exist_ok=True)

        try:
            # Download video via boto3 (no awscli dependency)
            video_dir = work / "behav"
            video_dir.mkdir(parents=True, exist_ok=True)
            _download_session_video(s3, RAWDATA_BUCKET, sub, ses_id, video_dir)

            mp4s = list(video_dir.glob("*overhead*.mp4")) + list(video_dir.glob("*cropped*.mp4"))
            if not mp4s:
                mp4s = list(video_dir.glob("*.mp4"))
            if not mp4s:
                print("  No video found, skipping")
                failed.append(exp_id)
                continue

            video = mp4s[0]

            # Subsample to 30fps. The median filter window in the kinematics
            # pipeline (currently 3 frames) is tuned for ~100ms at 30fps.
            # If this frame rate changes, update the window in
            # src/hm2p/kinematics/compute.py:median_filter_dataset() and
            # scripts/render_dlc_videos.py to maintain ~100ms smoothing.
            sub_path = work / f"{video.stem}_30fps.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video),
                 "-vf", "fps=30", "-vsync", "drop",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 str(sub_path)],
                capture_output=True,
            )
            dlc_video = sub_path if sub_path.exists() else video

            # Run inference
            out_dir = work / "output"
            out_dir.mkdir(exist_ok=True)
            print("  Running DLC inference (batch_size=16)...")
            deeplabcut.analyze_videos(
                str(config_path),
                [str(dlc_video)],
                destfolder=str(out_dir),
                batch_size=16,  # HRNet uses more VRAM than ResNet; 64 caused OOM hang
            )

            # Labelled video rendering is handled separately by
            # render_dlc_videos.py on a CPU instance after promotion
            # (faster: downscales to 416x304, no DLC dependency needed).

            # Upload results via boto3
            out_files = list(out_dir.rglob("*"))
            out_files = [f for f in out_files if f.is_file()]
            if out_files:
                s3_prefix = f"{FINETUNED_PREFIX}/{sub}/{ses_id}"
                for f in out_files:
                    key = f"{s3_prefix}/{f.name}"
                    # Rename labelled video to standard name for viewer page
                    if f.suffix == ".mp4" and "labeled" in f.name:
                        key = f"{s3_prefix}/labelled_30fps.mp4"
                    s3.upload_file(str(f), DERIVATIVES_BUCKET, key)
                completed.append(exp_id)
                print(f"  Uploaded {len(out_files)} files")

                # Progress: session done
                update_progress(
                    s3, f"Inference {i}/{total}: done {sub}/{ses_id}",
                    completed=len(completed), failed=len(failed), total=total,
                    current_session=exp_id, stage="inference_done",
                )
            else:
                print("  No output files")
                failed.append(exp_id)

        except Exception as e:
            error_records.append({
                "session": exp_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "stage": "inference",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            })
            print(f"  ERROR [{type(e).__name__}]: {e}")
            print(traceback.format_exc())
            failed.append(exp_id)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    update_progress(
        s3, "Inference complete",
        completed=len(completed), failed=len(failed), total=total,
        completed_sessions=completed, failed_sessions=failed,
    )
    print(f"\nDone: {len(completed)}/{total} completed, {len(failed)} failed")

    # Upload structured error records — always written, even if empty,
    # so the frontend can distinguish "no errors" from "file missing".
    errors_payload = json.dumps(
        {"run_id": run_id, "instance_id": instance_id, "errors": error_records},
        indent=2,
    ).encode()
    try:
        s3.put_object(
            Bucket=DERIVATIVES_BUCKET,
            Key=f"{RETRAIN_PREFIX}/_inference_errors.json",
            Body=errors_payload,
        )
        print(f"  Error summary uploaded ({len(error_records)} error(s))")
    except Exception as e:
        print(f"  WARNING: could not upload _inference_errors.json: {e}")

    # Auto-promote: copy pose-finetuned/ → pose/ on S3
    if failed and not skip_failed:
        print(
            f"\nSkipping auto-promote: {len(failed)} session(s) failed — "
            f"{failed}.\n"
            f"To promote the {len(completed)} successful session(s), pass "
            f"--skip-failed or run promote_finetuned_pose.py --skip-failed."
        )
        return

    if failed and skip_failed:
        print(
            f"\nAuto-promoting {len(completed)} completed session(s). "
            f"Skipping {len(failed)} failed session(s): {failed}"
        )

    # Only promote sessions that completed successfully
    sessions_to_promote = [s for s in sessions if s["exp_id"] in completed]
    print(f"\nPromoting {len(sessions_to_promote)} finetuned sessions → pose/ on S3...")
    for ses in sessions_to_promote:
        sub, ses_id = ses["sub"], ses["ses"]
        src_prefix = f"{FINETUNED_PREFIX}/{sub}/{ses_id}/"
        resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=src_prefix)
        for obj in resp.get("Contents", []):
            src_key = obj["Key"]
            dst_key = src_key.replace(FINETUNED_PREFIX, "pose", 1)
            s3.copy_object(
                Bucket=DERIVATIVES_BUCKET,
                CopySource={"Bucket": DERIVATIVES_BUCKET, "Key": src_key},
                Key=dst_key,
            )
        print(f"  {sub}/{ses_id}: promoted")

    update_progress(
        s3, "Promoted to pose/",
        completed=len(completed), total=total,
        promoted=len(sessions_to_promote), failed=len(failed),
        failed_sessions=failed,
    )
    print("Promotion complete.")

    # Declare the new project-wide champion. Done here, after promotion to
    # pose/ has succeeded, so the manifest only ever points at h5 files that
    # actually exist in pose/. See docs/dlc-champion-model.md.
    print("\n=== Declaring new DLC champion ===")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # noqa
        from hm2p.pose.select import extract_architecture, extract_dlc_provenance
        # Find one promoted h5 to read the identifiers from. Any promoted
        # session works — they all carry the same model_name and snapshot.
        sample = sessions_to_promote[0]
        sample_prefix = f"pose/{sample['sub']}/{sample['ses']}/"
        sample_resp = s3.list_objects_v2(Bucket=DERIVATIVES_BUCKET, Prefix=sample_prefix)
        h5_filenames = [
            obj["Key"].split("/")[-1]
            for obj in sample_resp.get("Contents", [])
            if obj["Key"].endswith(".h5")
            and "_single" not in obj["Key"].split("/")[-1]
            and "_filtered" not in obj["Key"].split("/")[-1]
            and ("Hrnet" in obj["Key"] or "Resnet" in obj["Key"])
        ]
        if not h5_filenames:
            raise RuntimeError(
                f"No finetuned .h5 found under {sample_prefix} after promotion."
            )
        h5_filename = h5_filenames[0]
        model_name, snapshot = extract_dlc_provenance(h5_filename)
        architecture = extract_architecture(h5_filename)
        if architecture is None:
            raise RuntimeError(
                f"Could not extract architecture from {h5_filename!r}."
            )
        notes_lines = [
            "Auto-declared by run_dlc_retrain.py.",
            f"Sessions promoted: {len(sessions_to_promote)}; "
            f"failed: {len(failed)}; total: {total}.",
        ]
        # If the SA-finetune training path stashed a notes file on S3,
        # prepend its contents (init source, conversion array, etc.).
        try:
            sa_notes_obj = s3.get_object(
                Bucket=DERIVATIVES_BUCKET,
                Key=f"{RETRAIN_PREFIX}/models/_sa_finetune_notes.txt",
            )
            sa_notes = sa_notes_obj["Body"].read().decode("utf-8").strip()
            if sa_notes:
                notes_lines.insert(0, sa_notes)
        except Exception:
            # ImageNet path leaves no notes file — that's expected.
            pass
        notes = " ".join(notes_lines)
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa
        from declare_dlc_champion import declare_champion  # noqa
        declare_champion(
            model_name=model_name,
            architecture=architecture,
            snapshot=snapshot,
            training_run_id=run_id,
            notes=notes,
            s3_client=s3,
            bucket=DERIVATIVES_BUCKET,
        )
    except Exception:
        print("ERROR: champion declaration failed (see traceback). "
              "The pipeline will continue but the manifest is not updated. "
              "Run scripts/declare_dlc_champion.py manually to fix.")
        traceback.print_exc()

    update_progress(
        s3, "Inference + promotion complete. Launching CPU instance for downstream + render.",
        completed=len(completed), total=total,
    )

    # Launch a CPU instance for downstream stages + video rendering.
    # These don't need GPU — running them on the GPU instance wastes money.
    print("\n=== Launching CPU instance for downstream + render ===")
    try:
        subprocess.run(
            ["python3", "scripts/launch_downstream_cpu.py"],
            check=True,
        )
    except Exception as e:
        print(f"WARNING: could not launch CPU instance: {e}")
        print("Run manually: python3 scripts/launch_downstream_cpu.py")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argparse — split out for unit-testing."""
    parser = argparse.ArgumentParser(description="DLC retraining + inference")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--infer-only", action="store_true")
    parser.add_argument(
        "--maxiters", type=int, default=50000,
        help="Legacy TF iterations (ignored by PyTorch; ignored under "
             "--sa-finetune)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs (DLC 3.0 PyTorch). Default depends on the "
             "training path: 400 for ImageNet HRNet, 120 for --sa-finetune.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sa-finetune", action="store_true",
        help="Use SuperAnimal-TopViewMouse memory-replay fine-tune instead of "
             "the legacy ImageNet HRNet path. Per Ye et al. 2024, "
             "doi:10.1038/s41467-024-48792-2.",
    )
    parser.add_argument(
        "--skip-failed", action="store_true",
        help="Promote completed sessions even if some inference sessions failed. "
             "By default auto-promotion is skipped if any session fails.",
    )
    return parser


def resolve_epochs(epochs: int | None, *, sa_finetune: bool) -> int:
    """Resolve the default ``--epochs`` based on the training path.

    Per design §2.1: 120 for SA fine-tune, 400 for ImageNet HRNet. When
    the operator passes ``--epochs`` explicitly, that value is honoured
    for both paths.
    """
    if epochs is not None:
        return epochs
    return 120 if sa_finetune else 400


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    epochs = resolve_epochs(args.epochs, sa_finetune=args.sa_finetune)

    s3 = boto3.client("s3", region_name=REGION)
    do_train = not args.infer_only
    do_infer = not args.train_only

    config_path = None
    if do_train:
        config_path = train(
            s3, maxiters=args.maxiters, epochs=epochs,
            batch_size=args.batch_size, sa_finetune=args.sa_finetune,
        )

    if do_infer:
        if config_path is None:
            # Download config + model weights from S3 (training was done in a previous run)
            work = Path("/tmp/dlc-retrain")
            work.mkdir(parents=True, exist_ok=True)
            config_path = work / "config.yaml"
            s3.download_file(DERIVATIVES_BUCKET, f"{RETRAIN_PREFIX}/config.yaml", str(config_path))

            # Download model weights
            print("Downloading model weights from S3...")
            resp = s3.list_objects_v2(
                Bucket=DERIVATIVES_BUCKET, Prefix=f"{RETRAIN_PREFIX}/models/"
            )
            model_files = resp.get("Contents", [])
            if not model_files:
                print("ERROR: no model weights on S3. Run training first.")
                sys.exit(1)
            for obj in model_files:
                key = obj["Key"]
                rel = key[len(f"{RETRAIN_PREFIX}/models/"):]
                dest = work / "dlc-models-pytorch" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(DERIVATIVES_BUCKET, key, str(dest))
            print(f"  Downloaded {len(model_files)} model files")

            # Fix project_path in config
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["project_path"] = str(work)
            with open(config_path, "w") as f:
                yaml.dump(cfg, f)

        infer(s3, config_path, skip_failed=args.skip_failed)


if __name__ == "__main__":
    main()
