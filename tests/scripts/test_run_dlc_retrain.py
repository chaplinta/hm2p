"""Tests for ``scripts/run_dlc_retrain.py`` — SA-finetune wiring.

DLC and dlclibrary are heavy and not import-safe in CI, so all DLC
calls are mocked. These tests focus on:

- argparse plumbing (``--sa-finetune``, ``--epochs`` default resolution)
- pre-condition checks (``default_net_type`` rewrite, conversion table,
  detector fallback chain, model availability)
- pytorch_config.yaml augmentation patch
- ``train_network`` kwargs match v2 plan §4.3
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

# Pre-stub heavy imports so the module loads cleanly under test.
sys.modules.setdefault("deeplabcut", MagicMock())
sys.modules.setdefault("dlclibrary", MagicMock())

import run_dlc_retrain as rdr  # noqa: E402

# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_sa_finetune_flag_parses(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args(["--sa-finetune"])
        assert args.sa_finetune is True

    def test_sa_finetune_default_off(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args([])
        assert args.sa_finetune is False

    def test_epochs_default_none(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args([])
        assert args.epochs is None

    def test_explicit_epochs_honoured(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args(["--sa-finetune", "--epochs", "200"])
        assert args.epochs == 200

    def test_compatible_with_infer_only(self):
        parser = rdr._build_arg_parser()
        args = parser.parse_args(["--sa-finetune", "--infer-only"])
        assert args.sa_finetune is True
        assert args.infer_only is True

    def test_maxiters_help_mentions_legacy(self):
        parser = rdr._build_arg_parser()
        # maxiters has the legacy help text.
        for action in parser._actions:
            if action.dest == "maxiters":
                assert "Legacy" in action.help
                break
        else:
            pytest.fail("--maxiters argparse action not found")


# ---------------------------------------------------------------------------
# resolve_epochs
# ---------------------------------------------------------------------------


class TestResolveEpochs:
    def test_default_imagenet_400(self):
        assert rdr.resolve_epochs(None, sa_finetune=False) == 400

    def test_default_sa_120(self):
        assert rdr.resolve_epochs(None, sa_finetune=True) == 120

    def test_explicit_overrides_imagenet_default(self):
        assert rdr.resolve_epochs(50, sa_finetune=False) == 50

    def test_explicit_overrides_sa_default(self):
        assert rdr.resolve_epochs(75, sa_finetune=True) == 75


# ---------------------------------------------------------------------------
# _ensure_default_net_type_hrnet
# ---------------------------------------------------------------------------


class TestEnsureDefaultNetType:
    def test_no_rewrite_when_already_hrnet(self, tmp_path: Path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"default_net_type": "hrnet_w32"}))
        rewrote = rdr._ensure_default_net_type_hrnet(cfg_path)
        assert rewrote is False

    def test_rewrites_resnet_to_hrnet(self, tmp_path: Path, capsys):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"default_net_type": "resnet_50"}))
        rewrote = rdr._ensure_default_net_type_hrnet(cfg_path)
        assert rewrote is True
        out = capsys.readouterr().out
        assert "WARNING" in out
        # Disk content has been rewritten.
        new = yaml.safe_load(cfg_path.read_text())
        assert new["default_net_type"] == "hrnet_w32"

    def test_handles_missing_key(self, tmp_path: Path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"other_key": "x"}))
        rewrote = rdr._ensure_default_net_type_hrnet(cfg_path)
        assert rewrote is True


# ---------------------------------------------------------------------------
# _validate_sa_conversion_table
# ---------------------------------------------------------------------------


class TestValidateConversionTable:
    def _write_cfg(self, tmp_path: Path, table: dict) -> Path:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "SuperAnimalConversionTables": {
                "superanimal_topviewmouse": table,
            }
        }))
        return cfg_path

    def test_complete_table_passes(self, tmp_path: Path):
        full = {bp: i for i, bp in enumerate(rdr.PROJECT_BODYPARTS)}
        cfg = self._write_cfg(tmp_path, full)
        # Should not raise.
        rdr._validate_sa_conversion_table(cfg)

    def test_missing_bodypart_raises(self, tmp_path: Path):
        partial = {bp: i for i, bp in enumerate(rdr.PROJECT_BODYPARTS) if bp != "head_midpoint"}
        cfg = self._write_cfg(tmp_path, partial)
        with pytest.raises(ValueError, match="head_midpoint"):
            rdr._validate_sa_conversion_table(cfg)

    def test_empty_table_raises(self, tmp_path: Path):
        cfg = self._write_cfg(tmp_path, {})
        with pytest.raises(ValueError):
            rdr._validate_sa_conversion_table(cfg)


# ---------------------------------------------------------------------------
# _resolve_sa_detector
# ---------------------------------------------------------------------------


class TestResolveDetector:
    def test_v2_preferred_when_both_available(self):
        assert rdr._resolve_sa_detector(
            ["fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2", "other"]
        ) == "fasterrcnn_resnet50_fpn_v2"

    def test_falls_back_to_base(self):
        assert rdr._resolve_sa_detector(
            ["fasterrcnn_resnet50_fpn"]
        ) == "fasterrcnn_resnet50_fpn"

    def test_raises_when_neither(self):
        with pytest.raises(RuntimeError, match="Available detectors"):
            rdr._resolve_sa_detector(["yolov5", "ssd"])

    def test_raises_when_empty(self):
        with pytest.raises(RuntimeError):
            rdr._resolve_sa_detector([])


# ---------------------------------------------------------------------------
# _validate_sa_model_available
# ---------------------------------------------------------------------------


class TestValidateModelAvailable:
    def test_passes_when_available(self):
        rdr._validate_sa_model_available([
            "superanimal_topviewmouse_hrnet_w32",
            "other_model",
        ])

    def test_raises_when_missing(self):
        with pytest.raises(RuntimeError, match="superanimal_topviewmouse_hrnet_w32"):
            rdr._validate_sa_model_available(["other_model"])


# ---------------------------------------------------------------------------
# _check_sa_input_size
# ---------------------------------------------------------------------------


class TestCheckInputSize:
    def test_warns_only_on_mismatch(self, tmp_path: Path, capsys):
        path = tmp_path / "pytorch_config.yaml"
        path.write_text(yaml.dump({
            "data": {"train": {"input_size": [512, 512]}},
        }))
        ok = rdr._check_sa_input_size(path)
        assert ok is False
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "512" in out

    def test_passes_at_256(self, tmp_path: Path):
        path = tmp_path / "pytorch_config.yaml"
        path.write_text(yaml.dump({
            "data": {"train": {"input_size": [256, 256]}},
        }))
        assert rdr._check_sa_input_size(path) is True


# ---------------------------------------------------------------------------
# _apply_sa_augmentation_patch
# ---------------------------------------------------------------------------


class TestAugmentationPatch:
    def test_writes_v2_values(self, tmp_path: Path):
        path = tmp_path / "pytorch_config.yaml"
        path.write_text(yaml.dump({
            "data": {
                "train": {
                    "affine": {"rotation": 45, "scaling": [0.5, 2.0]},
                    "gaussian_noise": 30,
                },
            },
            # NB: model.backbone.* should NOT be touched.
            "model": {"backbone": {"model_name": "hrnet_w32_sa"}},
        }))
        rdr._apply_sa_augmentation_patch(path)
        new = yaml.safe_load(path.read_text())
        assert new["data"]["train"]["affine"]["rotation"] == 30
        assert new["data"]["train"]["affine"]["scaling"] == [0.7, 1.3]
        assert new["data"]["train"]["gaussian_noise"] == 10.0
        # Backbone block left alone.
        assert new["model"]["backbone"]["model_name"] == "hrnet_w32_sa"

    def test_creates_train_block_if_missing(self, tmp_path: Path):
        path = tmp_path / "pytorch_config.yaml"
        path.write_text(yaml.dump({"unrelated": "bits"}))
        rdr._apply_sa_augmentation_patch(path)
        new = yaml.safe_load(path.read_text())
        assert new["data"]["train"]["affine"]["rotation"] == 30


# ---------------------------------------------------------------------------
# _build_sa_notes
# ---------------------------------------------------------------------------


def test_build_sa_notes_format():
    notes = rdr._build_sa_notes(
        detector="fasterrcnn_resnet50_fpn_v2",
        conversion_array=[0, 1, 2, 26, 7, 8, 9, 13],
        epochs=120,
        lr=5e-5,
        batch_size=8,
    )
    assert "init: superanimal_topviewmouse_hrnet_w32 (memory replay)" in notes
    assert "fasterrcnn_resnet50_fpn_v2" in notes
    assert "epochs: 120" in notes
    assert "lr: 5e-05" in notes
    assert "bs: 8" in notes
    assert "freeze_bn_stats: True" in notes
    assert "[0, 1, 2, 26, 7, 8, 9, 13]" in notes


# ---------------------------------------------------------------------------
# _train_sa_finetune (heavy integration test with mocked DLC + dlclibrary)
# ---------------------------------------------------------------------------


def _make_minimal_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal NeuroBlueprint-style DLC project under tmp_path.

    Returns ``(work_dir, config_path, pytorch_config_path)``.
    """
    work = tmp_path / "dlc-retrain"
    work.mkdir()
    # config.yaml with hrnet_w32 default and full conversion table.
    cfg_path = work / "config.yaml"
    cfg_path.write_text(yaml.dump({
        "default_net_type": "hrnet_w32",
        "bodyparts": list(rdr.PROJECT_BODYPARTS),
        "SuperAnimalConversionTables": {
            "superanimal_topviewmouse": {
                bp: i for i, bp in enumerate(rdr.PROJECT_BODYPARTS)
            },
        },
        "project_path": str(work),
    }))
    # pytorch_config.yaml with 256x256 input + plausible affine block.
    shuffle_dir = work / "dlc-models-pytorch" / "iteration-0" / "shuffle2" / "train"
    shuffle_dir.mkdir(parents=True)
    pcfg = shuffle_dir / "pytorch_config.yaml"
    pcfg.write_text(yaml.dump({
        "data": {
            "train": {
                "input_size": [256, 256],
                "affine": {"rotation": 45, "scaling": [0.7, 1.4]},
            },
        },
        "model": {"backbone": {"model_name": "sa_hrnet_w32"}},
    }))
    return work, cfg_path, pcfg


class TestTrainSaFinetune:
    def test_calls_build_weight_init_with_correct_kwargs(
        self, tmp_path: Path, monkeypatch
    ):
        work, cfg, _ = _make_minimal_project(tmp_path)

        # Mock DLC primitives.
        build_weight_init = MagicMock(return_value="WI_OBJECT")
        create_dataset = MagicMock(return_value=2)
        train_network = MagicMock()
        list_models = MagicMock(return_value=["superanimal_topviewmouse_hrnet_w32"])
        list_detectors = MagicMock(return_value=[
            "fasterrcnn_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn",
        ])

        # Patch via sys.modules — `import deeplabcut` inside the function
        # will hit the mock.
        dlc_mock = MagicMock()
        dlc_mock.create_training_dataset = create_dataset
        dlc_mock.train_network = train_network
        dlc_mock.modelzoo.weight_initialization.build_weight_init = build_weight_init
        dlclib_mock = MagicMock()
        dlclib_mock.list_available_models = list_models
        dlclib_mock.list_available_detectors = list_detectors
        monkeypatch.setitem(sys.modules, "deeplabcut", dlc_mock)
        monkeypatch.setitem(sys.modules, "dlclibrary", dlclib_mock)
        # The "from deeplabcut.modelzoo.weight_initialization import ..."
        # import line goes through deeplabcut.modelzoo so we can use the
        # already-patched dlc_mock attribute path.
        monkeypatch.setitem(
            sys.modules, "deeplabcut.modelzoo",
            dlc_mock.modelzoo,
        )
        monkeypatch.setitem(
            sys.modules, "deeplabcut.modelzoo.weight_initialization",
            dlc_mock.modelzoo.weight_initialization,
        )

        s3 = MagicMock()
        rdr._train_sa_finetune(s3, work, cfg, epochs=120, batch_size=8)

        # build_weight_init called with the v2 §4.2 signature.
        build_weight_init.assert_called_once()
        kwargs = build_weight_init.call_args.kwargs
        assert kwargs["super_animal"] == "superanimal_topviewmouse"
        assert kwargs["model_name"] == "hrnet_w32"
        assert kwargs["with_decoder"] is True
        assert kwargs["memory_replay"] is True
        assert kwargs["detector_name"] == "fasterrcnn_resnet50_fpn_v2"
        assert kwargs["cfg"] == str(cfg)

        # create_training_dataset called with weight_init + net_type.
        create_dataset.assert_called_once()
        cd_kwargs = create_dataset.call_args.kwargs
        assert cd_kwargs["weight_init"] == "WI_OBJECT"
        assert cd_kwargs["net_type"] == "hrnet_w32"

        # train_network kwargs match v2 §4.3 exactly.
        train_network.assert_called_once()
        tn_kwargs = train_network.call_args.kwargs
        assert tn_kwargs["epochs"] == 120
        assert tn_kwargs["save_epochs"] == 10
        assert tn_kwargs["batch_size"] == 8
        upd = tn_kwargs["pytorch_cfg_updates"]
        assert upd["train_settings.optimizer.params.lr"] == pytest.approx(5e-5)
        assert upd["model.backbone.freeze_bn_stats"] is True
        assert upd["train_settings.scheduler.type"] == "MultiStepLR"
        assert upd["train_settings.scheduler.params.milestones"] == [90, 110]
        assert upd["train_settings.scheduler.params.gamma"] == pytest.approx(0.1)

    def test_skips_manual_backbone_rewrite(self, tmp_path: Path, monkeypatch):
        work, cfg, pcfg = _make_minimal_project(tmp_path)
        original_pcfg = yaml.safe_load(pcfg.read_text())

        dlc_mock = MagicMock()
        dlc_mock.modelzoo.weight_initialization.build_weight_init = MagicMock(
            return_value="WI"
        )
        dlc_mock.create_training_dataset = MagicMock(return_value=2)
        dlc_mock.train_network = MagicMock()
        dlclib_mock = MagicMock()
        dlclib_mock.list_available_models = MagicMock(
            return_value=["superanimal_topviewmouse_hrnet_w32"]
        )
        dlclib_mock.list_available_detectors = MagicMock(
            return_value=["fasterrcnn_resnet50_fpn_v2"]
        )
        for name, mod in [
            ("deeplabcut", dlc_mock),
            ("deeplabcut.modelzoo", dlc_mock.modelzoo),
            ("deeplabcut.modelzoo.weight_initialization",
             dlc_mock.modelzoo.weight_initialization),
            ("dlclibrary", dlclib_mock),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)

        rdr._train_sa_finetune(MagicMock(), work, cfg, epochs=120, batch_size=8)

        new_pcfg = yaml.safe_load(pcfg.read_text())
        # Backbone block must be unchanged.
        assert new_pcfg["model"]["backbone"] == original_pcfg["model"]["backbone"]

    def test_applies_augmentation_patch(self, tmp_path: Path, monkeypatch):
        work, cfg, pcfg = _make_minimal_project(tmp_path)

        dlc_mock = MagicMock()
        dlc_mock.modelzoo.weight_initialization.build_weight_init = MagicMock()
        dlc_mock.create_training_dataset = MagicMock(return_value=2)
        dlc_mock.train_network = MagicMock()
        dlclib_mock = MagicMock()
        dlclib_mock.list_available_models = MagicMock(
            return_value=["superanimal_topviewmouse_hrnet_w32"]
        )
        dlclib_mock.list_available_detectors = MagicMock(
            return_value=["fasterrcnn_resnet50_fpn_v2"]
        )
        for name, mod in [
            ("deeplabcut", dlc_mock),
            ("deeplabcut.modelzoo", dlc_mock.modelzoo),
            ("deeplabcut.modelzoo.weight_initialization",
             dlc_mock.modelzoo.weight_initialization),
            ("dlclibrary", dlclib_mock),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)

        rdr._train_sa_finetune(MagicMock(), work, cfg, epochs=120, batch_size=8)
        new = yaml.safe_load(pcfg.read_text())
        assert new["data"]["train"]["affine"]["rotation"] == 30
        assert new["data"]["train"]["affine"]["scaling"] == [0.7, 1.3]
        assert new["data"]["train"]["gaussian_noise"] == 10.0

    def test_raises_on_missing_conversion_table_entry(
        self, tmp_path: Path, monkeypatch
    ):
        work, cfg, _ = _make_minimal_project(tmp_path)
        # Drop one bodypart from the conversion table.
        cfg_data = yaml.safe_load(cfg.read_text())
        del cfg_data["SuperAnimalConversionTables"][
            "superanimal_topviewmouse"
        ]["head_midpoint"]
        cfg.write_text(yaml.dump(cfg_data))

        dlc_mock = MagicMock()
        dlclib_mock = MagicMock()
        dlclib_mock.list_available_models = MagicMock(return_value=[])
        for name, mod in [
            ("deeplabcut", dlc_mock),
            ("dlclibrary", dlclib_mock),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        with pytest.raises(ValueError, match="head_midpoint"):
            rdr._train_sa_finetune(MagicMock(), work, cfg, epochs=120, batch_size=8)

    def test_raises_when_sa_model_unavailable(self, tmp_path: Path, monkeypatch):
        work, cfg, _ = _make_minimal_project(tmp_path)
        dlc_mock = MagicMock()
        dlclib_mock = MagicMock()
        dlclib_mock.list_available_models = MagicMock(return_value=["something_else"])
        for name, mod in [
            ("deeplabcut", dlc_mock),
            ("dlclibrary", dlclib_mock),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        with pytest.raises(RuntimeError, match="superanimal_topviewmouse_hrnet_w32"):
            rdr._train_sa_finetune(MagicMock(), work, cfg, epochs=120, batch_size=8)

    def test_writes_notes_file(self, tmp_path: Path, monkeypatch):
        work, cfg, _ = _make_minimal_project(tmp_path)
        dlc_mock = MagicMock()
        dlc_mock.modelzoo.weight_initialization.build_weight_init = MagicMock()
        dlc_mock.create_training_dataset = MagicMock(return_value=2)
        dlc_mock.train_network = MagicMock()
        dlclib_mock = MagicMock()
        dlclib_mock.list_available_models = MagicMock(
            return_value=["superanimal_topviewmouse_hrnet_w32"]
        )
        dlclib_mock.list_available_detectors = MagicMock(
            return_value=["fasterrcnn_resnet50_fpn_v2"]
        )
        for name, mod in [
            ("deeplabcut", dlc_mock),
            ("deeplabcut.modelzoo", dlc_mock.modelzoo),
            ("deeplabcut.modelzoo.weight_initialization",
             dlc_mock.modelzoo.weight_initialization),
            ("dlclibrary", dlclib_mock),
        ]:
            monkeypatch.setitem(sys.modules, name, mod)
        rdr._train_sa_finetune(MagicMock(), work, cfg, epochs=120, batch_size=8)
        notes_path = work / "_sa_finetune_notes.txt"
        assert notes_path.exists()
        content = notes_path.read_text()
        assert "memory replay" in content
        assert "epochs: 120" in content


# ---------------------------------------------------------------------------
# Notes helper
# ---------------------------------------------------------------------------


def test_sa_conversion_array_constants():
    """Identity-mapping per v2 plan §3."""
    assert rdr.SA_CONVERSION_ARRAY == [0, 1, 2, 26, 7, 8, 9, 13]


def test_project_bodyparts_order():
    """Conversion array ordering depends on this tuple."""
    assert rdr.PROJECT_BODYPARTS == (
        "nose_tip", "left_ear", "right_ear", "head_midpoint",
        "neck", "mid_back", "mouse_center", "tail_base",
    )


def test_sa_detector_candidates_order():
    """v2 first, base fallback (per architect open-question #3)."""
    assert rdr.SA_DETECTOR_CANDIDATES == (
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_resnet50_fpn",
    )
