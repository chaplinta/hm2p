"""Tests for retraining helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hm2p.pose.retrain import (
    prepare_retraining_manifest,
    select_retraining_frames,
)


class TestSelectRetrainingFrames:
    def test_worst_method(self):
        lik = np.ones(200) * 0.95
        lik[10] = 0.1
        lik[50] = 0.2
        result = select_retraining_frames(lik, method="worst", n_frames=5)
        assert result["method"] == "worst"
        assert 10 in result["indices"]
        assert result["bins"] is None

    def test_stratified_method(self):
        rng = np.random.default_rng(42)
        lik = rng.uniform(0, 1, 500)
        result = select_retraining_frames(lik, method="stratified", n_frames=20)
        assert result["method"] == "stratified"
        assert result["bins"] is not None
        assert len(result["indices"]) > 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            select_retraining_frames(np.ones(100), method="invalid")


class TestPrepareRetrainingManifest:
    def test_basic_manifest(self, tmp_path):
        indices = np.array([10, 50, 80], dtype=np.intp)
        paths = [Path(f"/tmp/frame_{i:06d}.png") for i in indices]
        manifest = prepare_retraining_manifest(
            session_id="20220804_13_52_02_1117646",
            frame_indices=indices,
            frame_paths=paths,
        )
        assert manifest["session_id"] == "20220804_13_52_02_1117646"
        assert manifest["n_frames"] == 3
        assert manifest["frames"][0]["frame_index"] == 10

    def test_writes_json(self, tmp_path):
        indices = np.array([5, 15], dtype=np.intp)
        paths = [Path(f"/tmp/f_{i}.png") for i in indices]
        out_path = tmp_path / "manifest.json"
        prepare_retraining_manifest(
            session_id="test_session",
            frame_indices=indices,
            frame_paths=paths,
            output_path=out_path,
        )
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["n_frames"] == 2

    def test_with_quality_bins(self):
        indices = np.array([10, 50], dtype=np.intp)
        paths = [Path(f"/tmp/f_{i}.png") for i in indices]
        bins = [
            ("worst", np.array([10], dtype=np.intp)),
            ("good", np.array([50], dtype=np.intp)),
        ]
        manifest = prepare_retraining_manifest(
            session_id="test",
            frame_indices=indices,
            frame_paths=paths,
            quality_bins=bins,
        )
        assert manifest["frames"][0]["quality_bin"] == "worst"
        assert manifest["frames"][1]["quality_bin"] == "good"
