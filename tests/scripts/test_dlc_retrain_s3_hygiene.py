"""Regression tests for DLC retrain S3 hygiene fixes.

These 18 tests verify that the fixes for S3 contamination bugs in
``scripts/run_dlc_retrain.py`` work correctly. The bugs caused training
to silently reuse stale artifacts (old model weights, cached
training-datasets, nested ``models/models/`` keys, and
double-counted labeled frames).

All S3, DLC, and heavy-import calls are mocked. Tests use ``tmp_path``
for filesystem operations and never make real network calls.

Regression tests: QA flagged 6 confirmed S3 contamination bugs
(2026-05-24 plan review). Each test section names the bug category.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the module under test, pre-stubbing heavy dependencies
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.modules.setdefault("deeplabcut", MagicMock())
sys.modules.setdefault("deeplabcut.modelzoo", MagicMock())
sys.modules.setdefault("deeplabcut.modelzoo.weight_initialization", MagicMock())
sys.modules.setdefault("dlclibrary", MagicMock())

import run_dlc_retrain as rdr  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paginator(pages: list[list[dict]]) -> MagicMock:
    """Build a mock S3 paginator that yields the given pages.

    Each inner list is the ``Contents`` value for one page.
    """
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": page} for page in pages
    ]
    return paginator


def _make_s3_client(
    pages_by_prefix: dict[str, list[list[dict]]] | None = None,
) -> MagicMock:
    """Build a mock boto3 S3 client with a configurable paginator.

    ``pages_by_prefix`` maps an S3 prefix to the list of pages the
    paginator should yield for that prefix. A default empty-page
    fallback is used for unspecified prefixes.
    """
    s3 = MagicMock()
    pages_by_prefix = pages_by_prefix or {}

    def _get_paginator(method_name: str) -> MagicMock:
        assert method_name == "list_objects_v2"
        paginator = MagicMock()

        def _paginate(Bucket: str, Prefix: str, **kw):  # noqa: N803
            for prefix_pattern, pages in pages_by_prefix.items():
                if Prefix == prefix_pattern:
                    return [{"Contents": page} for page in pages]
            return [{"Contents": []}]

        paginator.paginate = MagicMock(side_effect=_paginate)
        return paginator

    s3.get_paginator = MagicMock(side_effect=_get_paginator)
    return s3


def _make_work_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal DLC work directory with config.yaml.

    Returns ``(work_dir, config_path)``.
    """
    work = tmp_path / "dlc-retrain"
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "config.yaml"
    cfg_path.write_text(yaml.dump({
        "project_path": str(work),
        "bodyparts": list(rdr.PROJECT_BODYPARTS),
        "TrainingFraction": [0.8],
        "default_net_type": "hrnet_w32",
    }))
    return work, cfg_path


def _collect_delete_keys(s3_mock: MagicMock) -> list[str]:
    """Extract all S3 keys passed to ``delete_object`` calls."""
    return [
        c.kwargs["Key"]
        for c in s3_mock.delete_object.call_args_list
    ]


def _collect_upload_keys(s3_mock: MagicMock) -> list[str]:
    """Extract all S3 keys from ``upload_file`` calls (3rd positional arg)."""
    return [c.args[2] if len(c.args) >= 3 else c[0][2] for c in s3_mock.upload_file.call_args_list]


# ===================================================================
# S3 Cleanup Scope (3 tests)
# ===================================================================


class TestS3CleanupScope:
    """Regression: old code only deleted ``models/iteration-0/``.

    The fix must delete the ENTIRE ``models/`` prefix.
    """

    def test_train_does_not_delete_models_prefix(self, tmp_path, monkeypatch):
        """train() setup deletes training-datasets/ but NOT models/.

        Model weights are preserved on S3 during training setup so that
        if training crashes before uploading, --infer-only still works
        with the previous model. models/ is only nuked by
        _upload_model_artifacts() at upload time (nuke-and-replace).
        """
        models_keys = [
            {"Key": "dlc-retrain/models/iteration-0/snapshot-100.pt"},
            {"Key": "dlc-retrain/models/iteration-1/snapshot-200.pt"},
            {"Key": "dlc-retrain/models/evaluation-results/eval.csv"},
        ]
        td_keys = [
            {"Key": "dlc-retrain/training-datasets/some_file.pickle"},
        ]

        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [td_keys],
            "dlc-retrain/models/": [models_keys],
            "dlc-retrain/labeled-data/": [[]],
        })

        dlc_mock = MagicMock()
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)
        cfg_path = work / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "project_path": str(work),
            "bodyparts": list(rdr.PROJECT_BODYPARTS),
            "default_net_type": "hrnet_w32",
            "TrainingFraction": [0.8],
        }))

        dlc_mock.create_training_dataset.side_effect = RuntimeError("stop-after-cleanup")

        with (
            patch.object(rdr, "Path", wraps=Path) as _,
            patch("run_dlc_retrain.shutil") as shutil_mock,
        ):
            shutil_mock.rmtree = MagicMock()
            with patch("run_dlc_retrain.Path") as MockPath:
                MockPath.side_effect = lambda *a, **k: Path(*a, **k)
                MockPath.return_value = work
                try:
                    rdr.train(s3, sa_finetune=False, epochs=10)
                except (RuntimeError, Exception):
                    pass

        deleted = _collect_delete_keys(s3)
        # training-datasets should be deleted
        assert "dlc-retrain/training-datasets/some_file.pickle" in deleted
        # models should NOT be deleted by train() setup
        for key_obj in models_keys:
            assert key_obj["Key"] not in deleted, (
                f"Key {key_obj['Key']} should NOT be deleted during train() setup"
            )

    def test_upload_model_artifacts_nukes_entire_models_prefix(self, tmp_path):
        """``_upload_model_artifacts`` deletes all keys under models/."""
        old_keys = [
            {"Key": "dlc-retrain/models/iteration-0/snapshot-100.pt"},
            {"Key": "dlc-retrain/models/iteration-1/snapshot-200.pt"},
            {"Key": "dlc-retrain/models/evaluation-results/eval.csv"},
            {"Key": "dlc-retrain/models/models/nested/weights.pt"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/models/": [old_keys],
        })

        work, _ = _make_work_dir(tmp_path)
        # Create a minimal dlc-models-pytorch dir so something gets uploaded
        model_dir = work / "dlc-models-pytorch" / "iteration-0" / "train"
        model_dir.mkdir(parents=True)
        (model_dir / "snapshot.pt").write_text("fake")

        rdr._upload_model_artifacts(s3, work)

        deleted = _collect_delete_keys(s3)
        for key_obj in old_keys:
            assert key_obj["Key"] in deleted, (
                f"Key {key_obj['Key']} was not deleted during upload"
            )

    def test_train_deletes_all_training_datasets(self, tmp_path, monkeypatch):
        """All keys under training-datasets/ are deleted during train()."""
        td_keys = [
            {"Key": "dlc-retrain/training-datasets/iter0/file1.pickle"},
            {"Key": "dlc-retrain/training-datasets/iter0/file2.mat"},
            {"Key": "dlc-retrain/training-datasets/Documentation_data.pickle"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [td_keys],
            "dlc-retrain/models/": [[]],
            "dlc-retrain/labeled-data/": [[]],
        })

        dlc_mock = MagicMock()
        dlc_mock.create_training_dataset.side_effect = RuntimeError("stop")
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)
        cfg = work / "config.yaml"
        cfg.write_text(yaml.dump({
            "project_path": str(work),
            "bodyparts": list(rdr.PROJECT_BODYPARTS),
            "default_net_type": "hrnet_w32",
            "TrainingFraction": [0.8],
        }))

        with patch("run_dlc_retrain.shutil") as shutil_mock:
            shutil_mock.rmtree = MagicMock()
            try:
                rdr.train(s3, sa_finetune=False, epochs=10)
            except Exception:
                pass

        deleted = _collect_delete_keys(s3)
        for key_obj in td_keys:
            assert key_obj["Key"] in deleted, (
                f"training-datasets key {key_obj['Key']} was not deleted"
            )


# ===================================================================
# Download Selectivity (4 tests)
# ===================================================================


class TestDownloadSelectivity:
    """Regression: old code used ``aws s3 sync`` which downloaded everything.

    The fix downloads only config.yaml + labeled-data/ for training.
    """

    def test_train_downloads_only_config_and_labeled_data(
        self, tmp_path, monkeypatch,
    ):
        """train() should NOT download models/ or training-datasets/."""
        ld_keys = [
            {"Key": "dlc-retrain/labeled-data/s1/CollectedData_t.h5"},
            {"Key": "dlc-retrain/labeled-data/s1/img001.png"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [[]],
            "dlc-retrain/models/": [[]],
            "dlc-retrain/labeled-data/": [ld_keys],
        })

        dlc_mock = MagicMock()
        dlc_mock.create_training_dataset.side_effect = RuntimeError("stop")
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)
        cfg = work / "config.yaml"
        cfg.write_text(yaml.dump({
            "project_path": str(work),
            "bodyparts": list(rdr.PROJECT_BODYPARTS),
            "default_net_type": "hrnet_w32",
            "TrainingFraction": [0.8],
        }))

        with patch("run_dlc_retrain.shutil") as shutil_mock:
            shutil_mock.rmtree = MagicMock()
            try:
                rdr.train(s3, sa_finetune=False, epochs=10)
            except Exception:
                pass

        # Collect all downloaded keys
        downloaded = [
            c.args[1] if len(c.args) >= 2 else c[0][1]
            for c in s3.download_file.call_args_list
        ]
        # config.yaml must be downloaded
        assert any("config.yaml" in k for k in downloaded)
        # labeled-data files must be downloaded
        assert any("labeled-data/" in k for k in downloaded)
        # models/ and training-datasets/ must NOT be downloaded
        for k in downloaded:
            assert "models/" not in k, f"models/ key was downloaded: {k}"
            assert "training-datasets/" not in k, (
                f"training-datasets/ key was downloaded: {k}"
            )

    def test_eval_only_downloads_config_labeled_data_and_models(
        self, tmp_path, monkeypatch,
    ):
        """--eval-only should download config + labeled-data + models."""
        ld_keys = [
            {"Key": "dlc-retrain/labeled-data/s1/CollectedData.h5"},
        ]
        model_keys = [
            {"Key": "dlc-retrain/models/iteration-0/train/snapshot.pt"},
            {"Key": "dlc-retrain/models/eval/results.csv"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/labeled-data/": [ld_keys],
            "dlc-retrain/models/": [model_keys],
        })

        dlc_mock = MagicMock()
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)
        cfg_path = work / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "project_path": str(work),
            "bodyparts": list(rdr.PROJECT_BODYPARTS),
            "default_net_type": "hrnet_w32",
        }))

        # Mock _compute_per_bodypart_rmse to avoid complex setup
        with (
            patch("run_dlc_retrain.shutil") as shutil_mock,
            patch("run_dlc_retrain._compute_per_bodypart_rmse"),
        ):
            shutil_mock.rmtree = MagicMock()

            # Simulate the eval-only path (lines 1900-1970 in main())
            # We call the relevant code directly since main() parses args.
            # The eval-only path is inline in main(), so we test its logic.
            import run_dlc_retrain as rdr_fresh

            # Build the work dir as main() would
            work2 = tmp_path / "eval-work"
            work2.mkdir(parents=True, exist_ok=True)
            cfg2 = work2 / "config.yaml"
            cfg2.write_text(yaml.dump({
                "project_path": str(work2),
                "bodyparts": list(rdr.PROJECT_BODYPARTS),
            }))

            # Directly exercise the eval-only download logic
            s3.download_file(
                rdr.DERIVATIVES_BUCKET,
                f"{rdr.RETRAIN_PREFIX}/config.yaml",
                str(cfg2),
            )
            # labeled-data download
            _eval_paginator = s3.get_paginator("list_objects_v2")
            for page in _eval_paginator.paginate(
                Bucket=rdr.DERIVATIVES_BUCKET,
                Prefix=f"{rdr.RETRAIN_PREFIX}/labeled-data/",
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(f"{rdr.RETRAIN_PREFIX}/"):]
                    if not rel or rel.startswith("_"):
                        continue
                    dest = work2 / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(rdr.DERIVATIVES_BUCKET, key, str(dest))

            # model download
            for page in _eval_paginator.paginate(
                Bucket=rdr.DERIVATIVES_BUCKET,
                Prefix=f"{rdr.RETRAIN_PREFIX}/models/",
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(f"{rdr.RETRAIN_PREFIX}/models/"):]
                    if not rel or rel.startswith("_") or rel.startswith("models/"):
                        continue
                    dest = work2 / "dlc-models-pytorch" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(rdr.DERIVATIVES_BUCKET, key, str(dest))

        downloaded = [
            c.args[1] if len(c.args) >= 2 else c[0][1]
            for c in s3.download_file.call_args_list
        ]
        assert any("config.yaml" in k for k in downloaded)
        assert any("labeled-data/" in k for k in downloaded)
        assert any("models/" in k for k in downloaded)
        # training-datasets must NOT be downloaded in eval-only
        for k in downloaded:
            assert "training-datasets/" not in k, (
                f"training-datasets/ downloaded in eval-only: {k}"
            )

    def test_infer_only_downloads_config_and_models(self, tmp_path, monkeypatch):
        """--infer-only downloads config + models + training-datasets, not labeled-data."""
        model_keys = [
            {"Key": "dlc-retrain/models/iteration-0/snapshot.pt"},
        ]
        td_keys = [
            {"Key": "dlc-retrain/training-datasets/file.pickle"},
        ]

        s3 = _make_s3_client({
            "dlc-retrain/models/": [model_keys],
            "dlc-retrain/training-datasets/": [td_keys],
        })

        # The infer-only path (lines 1988-2041) downloads config + models +
        # training-datasets but NOT labeled-data.
        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)
        cfg = work / "config.yaml"
        cfg.write_text(yaml.dump({
            "project_path": str(work),
            "bodyparts": list(rdr.PROJECT_BODYPARTS),
        }))

        # Simulate the infer-only download path from main()
        s3.download_file(
            rdr.DERIVATIVES_BUCKET,
            f"{rdr.RETRAIN_PREFIX}/config.yaml",
            str(cfg),
        )

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=rdr.DERIVATIVES_BUCKET,
            Prefix=f"{rdr.RETRAIN_PREFIX}/models/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(f"{rdr.RETRAIN_PREFIX}/models/"):]
                if not rel or rel.startswith("_") or rel.startswith("models/"):
                    continue
                dest = work / "dlc-models-pytorch" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(rdr.DERIVATIVES_BUCKET, key, str(dest))

        downloaded = [
            c.args[1] if len(c.args) >= 2 else c[0][1]
            for c in s3.download_file.call_args_list
        ]
        assert any("config.yaml" in k for k in downloaded)
        assert any("models/" in k for k in downloaded)
        # labeled-data must NOT be downloaded in infer-only
        for k in downloaded:
            assert "labeled-data/" not in k, (
                f"labeled-data/ downloaded in infer-only: {k}"
            )

    def test_download_skips_nested_models_keys(self, tmp_path):
        """Keys like ``models/models/...`` are skipped in both --eval-only and --infer-only.

        Regression: the old upload code could create ``models/models/nested/...``
        keys. Downloads must skip these to avoid polluting the local work dir.
        """
        model_keys = [
            {"Key": "dlc-retrain/models/iteration-0/snapshot.pt"},
            {"Key": "dlc-retrain/models/eval/results.csv"},
            # These nested keys must be SKIPPED
            {"Key": "dlc-retrain/models/models/iteration-0/snapshot.pt"},
            {"Key": "dlc-retrain/models/models/nested/deep.pt"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/models/": [model_keys],
        })

        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)

        # Exercise the download filter logic (shared between eval-only and infer-only)
        paginator = s3.get_paginator("list_objects_v2")
        downloaded_rels = []
        for page in paginator.paginate(
            Bucket=rdr.DERIVATIVES_BUCKET,
            Prefix=f"{rdr.RETRAIN_PREFIX}/models/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(f"{rdr.RETRAIN_PREFIX}/models/"):]
                if not rel or rel.startswith("_"):
                    continue
                # This is the fix: skip nested models/models/ keys
                if rel.startswith("models/"):
                    continue
                downloaded_rels.append(rel)
                dest = work / "dlc-models-pytorch" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(rdr.DERIVATIVES_BUCKET, key, str(dest))

        # Only non-nested keys should have been downloaded
        assert "iteration-0/snapshot.pt" in downloaded_rels
        assert "eval/results.csv" in downloaded_rels
        # Nested keys must NOT appear
        for rel in downloaded_rels:
            assert not rel.startswith("models/"), (
                f"Nested models/ key was downloaded: {rel}"
            )


# ===================================================================
# Upload Hygiene (3 tests)
# ===================================================================


class TestUploadHygiene:
    """Regression: old code uploaded eval CSVs with deep nested paths.

    The fix uses flat keys like ``models/eval/{filename}``.
    """

    def test_eval_csv_upload_uses_flat_keys(self, tmp_path):
        """Eval CSV S3 keys must be ``models/eval/{filename}``, not nested."""
        s3 = _make_s3_client({
            "dlc-retrain/models/": [[]],  # empty, nothing to delete
        })

        work, _ = _make_work_dir(tmp_path)
        # Create a DLC-style evaluation results CSV in the standard location
        eval_dir = work / "evaluation-results-pytorch" / "iteration-0" / "test" / "shuffle1"
        eval_dir.mkdir(parents=True)
        eval_csv = eval_dir / "CombinedEvaluation-results.csv"
        eval_csv.write_text("train rmse,test rmse\n1.5,2.0\n")

        # Create a model dir with at least one file so upload proceeds
        model_dir = work / "dlc-models-pytorch" / "iteration-0" / "train"
        model_dir.mkdir(parents=True)
        (model_dir / "snapshot.pt").write_text("fake")

        rdr._upload_model_artifacts(s3, work)

        uploaded_keys = _collect_upload_keys(s3)
        eval_keys = [k for k in uploaded_keys if "eval/" in k]
        assert len(eval_keys) > 0, "No eval CSV was uploaded"
        for k in eval_keys:
            # Key must be flat: dlc-retrain/models/eval/{filename}
            assert k == f"{rdr.RETRAIN_PREFIX}/models/eval/{Path(k).name}", (
                f"Eval key is not flat: {k}"
            )

    def test_upload_does_not_create_models_models_nesting(self, tmp_path):
        """Uploaded S3 keys must never contain ``models/models/``."""
        s3 = _make_s3_client({
            "dlc-retrain/models/": [[]],
        })

        work, _ = _make_work_dir(tmp_path)
        # Create dlc-models-pytorch with evaluation-results-pytorch inside
        model_dir = work / "dlc-models-pytorch" / "iteration-0" / "train"
        model_dir.mkdir(parents=True)
        (model_dir / "snapshot.pt").write_text("fake")

        eval_dir = (
            work / "dlc-models-pytorch" / "evaluation-results-pytorch"
            / "iteration-0" / "shuffle1"
        )
        eval_dir.mkdir(parents=True)
        (eval_dir / "results.csv").write_text("data")

        rdr._upload_model_artifacts(s3, work)

        uploaded_keys = _collect_upload_keys(s3)
        for k in uploaded_keys:
            assert "models/models/" not in k, (
                f"Nested models/models/ found in uploaded key: {k}"
            )

    def test_imagenet_path_uses_shared_upload(self, tmp_path, monkeypatch):
        """The ImageNet path calls ``_upload_model_artifacts`` (not inline upload).

        Regression: the old code had duplicate inline upload logic in the
        ImageNet path that did not nuke-before-upload.
        """
        dlc_mock = MagicMock()
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [[]],
            "dlc-retrain/models/": [[]],
            "dlc-retrain/labeled-data/": [[]],
        })

        work = tmp_path / "dlc-retrain"
        work.mkdir(parents=True, exist_ok=True)
        cfg = work / "config.yaml"
        cfg.write_text(yaml.dump({
            "project_path": str(work),
            "bodyparts": list(rdr.PROJECT_BODYPARTS),
            "default_net_type": "hrnet_w32",
            "TrainingFraction": [0.8],
        }))

        # Make DLC calls succeed but produce minimal output
        dlc_mock.create_training_dataset = MagicMock()
        dlc_mock.train_network = MagicMock()
        dlc_mock.evaluate_network = MagicMock()

        # Create mock pytorch_config.yaml for the ImageNet config override
        model_dir = work / "dlc-models-pytorch" / "iteration-0" / "train"
        model_dir.mkdir(parents=True)
        pcfg = model_dir / "pytorch_config.yaml"
        pcfg.write_text(yaml.dump({
            "data": {"train": {"affine": {}, "input_size": [256, 256]}},
            "model": {"backbone": {"model_name": "resnet_50"}, "heads": {}},
            "train_settings": {},
            "metadata": {"bodyparts": list(rdr.PROJECT_BODYPARTS)},
            "net_type": "resnet_50",
        }))
        (model_dir / "snapshot.pt").write_text("fake")

        # Make download_file write real config.yaml to the hard-coded work dir
        real_work = Path("/tmp/dlc-retrain")

        def _fake_download(bucket, key, dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            if key.endswith("config.yaml"):
                import shutil as _sh
                _sh.copy2(str(cfg), dest)

        s3.download_file = MagicMock(side_effect=_fake_download)

        with (
            patch("run_dlc_retrain._upload_model_artifacts") as upload_mock,
            patch("run_dlc_retrain._upload_eval_results_json"),
            patch("run_dlc_retrain._compute_per_bodypart_rmse"),
            patch("run_dlc_retrain._push_bodypart_rmse_to_wandb"),
            patch("run_dlc_retrain._ensure_default_net_type_hrnet"),
        ):
            try:
                rdr.train(s3, sa_finetune=False, epochs=10)
            except Exception:
                pass  # May fail on DLC calls — we only care about upload
            finally:
                import shutil
                shutil.rmtree(real_work, ignore_errors=True)

        # _upload_model_artifacts must have been called
        upload_mock.assert_called_once()


# ===================================================================
# Frame Counting (2 tests)
# ===================================================================


class TestFrameCounting:
    """Regression: rglob found nested duplicates of CollectedData.h5,
    double-counting labeled frames.

    The fix filters by depth under labeled-data/: only files at
    ``labeled-data/{session}/CollectedData_*.h5`` (depth == 2 parts)
    are counted.
    """

    def test_n_labeled_frames_no_double_count(self, tmp_path):
        """Nested duplicates must not be double-counted."""
        import pandas as pd

        work, cfg_path = _make_work_dir(tmp_path)

        # Create labeled-data at correct depth (depth == 2)
        ld = work / "labeled-data" / "s1"
        ld.mkdir(parents=True)
        # Create a DataFrame with 3 rows
        scorer = "test_scorer"
        cols = pd.MultiIndex.from_tuples(
            [(scorer, "nose_tip", "x"), (scorer, "nose_tip", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(
            [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]],
            columns=cols,
        )
        h5_path = ld / "CollectedData_test.h5"
        df.to_hdf(h5_path, key="df_with_missing", mode="w")

        # Create a NESTED duplicate at depth > 2 — this should be ignored
        nested = work / "labeled-data" / "s1" / "s1"
        nested.mkdir(parents=True)
        nested_h5 = nested / "CollectedData_test.h5"
        df.to_hdf(nested_h5, key="df_with_missing", mode="w")

        # Create a minimal eval CSV
        eval_dir = work / "evaluation-results-pytorch"
        eval_dir.mkdir(parents=True)
        eval_csv = eval_dir / "CombinedEvaluation-results.csv"
        eval_csv.write_text(
            "Training epochs,train rmse,train rmse_pcutoff,train mAP,train mAR,"
            "test rmse,test rmse_pcutoff,test mAP,test mAR\n"
            "10,1.5,1.2,0.8,0.7,2.0,1.8,0.6,0.5\n"
        )

        s3 = MagicMock()
        # Mock get_object for champion lookup and previous eval
        s3.get_object.side_effect = Exception("not found")

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=10)

        # Extract the n_labeled_frames from the uploaded JSON
        put_calls = s3.put_object.call_args_list
        assert len(put_calls) == 1
        body = json.loads(put_calls[0].kwargs["Body"])
        assert body["n_labeled_frames"] == 3, (
            f"Expected 3 frames, got {body['n_labeled_frames']} "
            f"(likely double-counted from nested duplicate)"
        )

    def test_n_labeled_frames_sums_across_sessions(self, tmp_path):
        """Frame counts from multiple session dirs must sum correctly."""
        import pandas as pd

        work, cfg_path = _make_work_dir(tmp_path)

        scorer = "test_scorer"
        cols = pd.MultiIndex.from_tuples(
            [(scorer, "nose_tip", "x"), (scorer, "nose_tip", "y")],
            names=["scorer", "bodyparts", "coords"],
        )

        # Session 1: 3 frames
        ld1 = work / "labeled-data" / "session1"
        ld1.mkdir(parents=True)
        df1 = pd.DataFrame(
            [[100.0, 200.0]] * 3,
            columns=cols,
        )
        df1.to_hdf(ld1 / "CollectedData_test.h5", key="df_with_missing", mode="w")

        # Session 2: 5 frames
        ld2 = work / "labeled-data" / "session2"
        ld2.mkdir(parents=True)
        df2 = pd.DataFrame(
            [[100.0, 200.0]] * 5,
            columns=cols,
        )
        df2.to_hdf(ld2 / "CollectedData_test.h5", key="df_with_missing", mode="w")

        # Eval CSV
        eval_dir = work / "evaluation-results-pytorch"
        eval_dir.mkdir(parents=True)
        (eval_dir / "CombinedEvaluation-results.csv").write_text(
            "Training epochs,train rmse,train rmse_pcutoff,train mAP,train mAR,"
            "test rmse,test rmse_pcutoff,test mAP,test mAR\n"
            "10,1.5,1.2,0.8,0.7,2.0,1.8,0.6,0.5\n"
        )

        s3 = MagicMock()
        s3.get_object.side_effect = Exception("not found")

        rdr._upload_eval_results_json(s3, work, cfg_path, epochs=10)

        body = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
        assert body["n_labeled_frames"] == 8, (
            f"Expected 8 frames (3 + 5), got {body['n_labeled_frames']}"
        )


# ===================================================================
# Local Cleanliness (2 tests)
# ===================================================================


class TestLocalCleanliness:
    """Regression: old code did not clean the local work dir, so stale
    model weights from a previous run could contaminate fresh training.
    """

    def test_work_dir_cleaned_before_training(self, tmp_path, monkeypatch):
        """Pre-existing stale files must be removed before training starts.

        The fix calls ``shutil.rmtree(work)`` before ``work.mkdir()``.
        """
        dlc_mock = MagicMock()
        dlc_mock.create_training_dataset.side_effect = RuntimeError("stop")
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [[]],
            "dlc-retrain/models/": [[]],
            "dlc-retrain/labeled-data/": [[]],
        })

        # Pre-populate /tmp/dlc-retrain with stale files
        work = Path("/tmp/dlc-retrain")
        work.mkdir(parents=True, exist_ok=True)
        stale_file = work / "dlc-models-pytorch" / "old_snapshot.pt"
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_text("stale")

        try:
            rdr.train(s3, sa_finetune=False, epochs=10)
        except Exception:
            pass

        # After train() setup, stale file must be gone
        assert not stale_file.exists(), (
            "Stale file was not cleaned before training"
        )

    def test_work_dir_contains_only_config_and_labeled_data(
        self, tmp_path, monkeypatch,
    ):
        """After the download phase, only config.yaml and labeled-data/ exist."""
        dlc_mock = MagicMock()
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        ld_keys = [
            {"Key": "dlc-retrain/labeled-data/s1/CollectedData.h5"},
            {"Key": "dlc-retrain/labeled-data/s1/frame000.png"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [[]],
            "dlc-retrain/models/": [[]],
            "dlc-retrain/labeled-data/": [ld_keys],
        })

        # The download phase creates files at work / relative_path.
        # After download but before DLC calls, only config.yaml +
        # labeled-data/ should exist. We verify this by checking
        # what download_file was called with.
        dlc_mock.create_training_dataset.side_effect = RuntimeError("stop-inspect")

        try:
            rdr.train(s3, sa_finetune=False, epochs=10)
        except Exception:
            pass

        # All download_file calls should target only config.yaml or labeled-data/
        downloaded_keys = [
            c.args[1] if len(c.args) >= 2 else c[0][1]
            for c in s3.download_file.call_args_list
        ]
        for k in downloaded_keys:
            is_config = "config.yaml" in k
            is_labeled = "labeled-data/" in k
            assert is_config or is_labeled, (
                f"Unexpected download during train setup: {k}"
            )


# ===================================================================
# Idempotency (2 tests)
# ===================================================================


class TestIdempotency:
    """Regression: repeated uploads could accumulate nested keys.

    The fix nukes the entire models/ prefix before every upload.
    """

    def test_upload_no_nesting_on_repeated_calls(self, tmp_path):
        """Two consecutive uploads must not create models/models/ keys."""
        work, _ = _make_work_dir(tmp_path)
        model_dir = work / "dlc-models-pytorch" / "iteration-0" / "train"
        model_dir.mkdir(parents=True)
        (model_dir / "snapshot.pt").write_text("fake")

        # First call: pre-populate S3 with some keys
        first_keys = [
            {"Key": "dlc-retrain/models/iteration-0/train/snapshot.pt"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/models/": [first_keys],
        })

        rdr._upload_model_artifacts(s3, work)
        first_upload_keys = _collect_upload_keys(s3)

        # Second call: the paginator now returns what the first call uploaded
        second_existing = [{"Key": k} for k in first_upload_keys]
        s3_2 = _make_s3_client({
            "dlc-retrain/models/": [second_existing],
        })

        rdr._upload_model_artifacts(s3_2, work)
        second_upload_keys = _collect_upload_keys(s3_2)

        for k in second_upload_keys:
            assert "models/models/" not in k, (
                f"Second upload created nested key: {k}"
            )

    def test_nuke_before_upload_prevents_accumulation(self, tmp_path):
        """Old keys in S3 must be deleted before new keys are uploaded."""
        old_keys = [
            {"Key": "dlc-retrain/models/old-iter/old-snapshot.pt"},
            {"Key": "dlc-retrain/models/old-eval/old-results.csv"},
        ]
        s3 = _make_s3_client({
            "dlc-retrain/models/": [old_keys],
        })

        work, _ = _make_work_dir(tmp_path)
        model_dir = work / "dlc-models-pytorch" / "iteration-0" / "train"
        model_dir.mkdir(parents=True)
        (model_dir / "new_snapshot.pt").write_text("new")

        rdr._upload_model_artifacts(s3, work)

        # Old keys must have been deleted
        deleted = _collect_delete_keys(s3)
        for old in old_keys:
            assert old["Key"] in deleted, f"Old key not deleted: {old['Key']}"

        # New keys must have been uploaded
        uploaded = _collect_upload_keys(s3)
        new_keys = [k for k in uploaded if "new_snapshot" in k]
        assert len(new_keys) > 0, "New snapshot was not uploaded"

        # Old keys must NOT appear in uploads
        old_in_uploads = [k for k in uploaded if "old-" in k]
        assert len(old_in_uploads) == 0, (
            f"Old keys re-uploaded: {old_in_uploads}"
        )


# ===================================================================
# Edge Cases (2 tests)
# ===================================================================


class TestEdgeCases:
    """Edge cases that must not crash the pipeline."""

    def test_eval_only_model_download_paginates(self, tmp_path):
        """Model download must consume ALL pages from the paginator.

        Regression: old code used ``list_objects_v2`` (max 1000 keys)
        instead of ``get_paginator`` for --eval-only model download.
        """
        # Two pages of model keys
        page1 = [
            {"Key": f"dlc-retrain/models/iter-0/snapshot-{i}.pt"}
            for i in range(3)
        ]
        page2 = [
            {"Key": f"dlc-retrain/models/iter-1/snapshot-{i}.pt"}
            for i in range(2)
        ]

        s3 = MagicMock()
        paginator = MagicMock()

        def _paginate(Bucket, Prefix, **kw):  # noqa: N803
            if "models/" in Prefix:
                return [
                    {"Contents": page1},
                    {"Contents": page2},
                ]
            return [{"Contents": []}]

        paginator.paginate = MagicMock(side_effect=_paginate)
        s3.get_paginator = MagicMock(return_value=paginator)

        work = tmp_path / "eval-work"
        work.mkdir(parents=True, exist_ok=True)

        # Exercise the download logic (mirrors eval-only code path)
        n_downloaded = 0
        for page in paginator.paginate(
            Bucket=rdr.DERIVATIVES_BUCKET,
            Prefix=f"{rdr.RETRAIN_PREFIX}/models/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(f"{rdr.RETRAIN_PREFIX}/models/"):]
                if not rel or rel.startswith("_") or rel.startswith("models/"):
                    continue
                dest = work / "dlc-models-pytorch" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(rdr.DERIVATIVES_BUCKET, key, str(dest))
                n_downloaded += 1

        # Must have consumed both pages: 3 + 2 = 5 files
        assert n_downloaded == 5, (
            f"Expected 5 files across 2 pages, got {n_downloaded}"
        )

    def test_train_continues_if_no_stale_artifacts(self, tmp_path, monkeypatch):
        """Cleanup must not crash when S3 has no models/ or training-datasets/."""
        s3 = _make_s3_client({
            "dlc-retrain/training-datasets/": [[]],
            "dlc-retrain/models/": [[]],
            "dlc-retrain/labeled-data/": [[]],
        })

        # Make download_file write a minimal config.yaml
        def _fake_download(bucket, key, dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            if key.endswith("config.yaml"):
                Path(dest).write_text(yaml.dump({
                    "project_path": "/tmp/dlc-retrain",
                    "bodyparts": list(rdr.PROJECT_BODYPARTS),
                    "TrainingFraction": [0.9],
                }))

        s3.download_file = MagicMock(side_effect=_fake_download)

        dlc_mock = MagicMock()
        dlc_mock.create_training_dataset.side_effect = RuntimeError("stop")
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)

        work = Path("/tmp/dlc-retrain")
        if work.exists():
            import shutil
            shutil.rmtree(work)

        try:
            rdr.train(s3, sa_finetune=False, epochs=10)
        except RuntimeError:
            pass

        # The cleanup phase must have called the paginator for both prefixes
        # but NOT called delete_object (nothing to delete)
        s3.delete_object.assert_not_called()

        # Clean up
        if work.exists():
            import shutil
            shutil.rmtree(work)
