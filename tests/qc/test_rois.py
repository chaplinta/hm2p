"""Tests for hm2p.qc.rois (synthetic ROIs only)."""

from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd
import pytest

from hm2p.qc.rois import cv_reference, image_png_b64, overview_row, roi_contour, summarise_rois


def _disk(cy: int, cx: int, r: int) -> dict:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    m = yy**2 + xx**2 <= r**2
    return {"ypix": (yy[m] + cy), "xpix": (xx[m] + cx), "med": [cy, cx], "npix": int(m.sum())}


def test_roi_contour_encloses_disk():
    d = _disk(20, 30, 5)
    c = np.array(roi_contour(d["ypix"], d["xpix"]))
    assert c.shape[1] == 2 and len(c) >= 4
    assert c[:, 0].min() == pytest.approx(25, abs=1) and c[:, 0].max() == pytest.approx(35, abs=1)
    assert roi_contour([], []) == []


def test_image_png_b64_downscales():
    img = np.random.default_rng(0).normal(size=(512, 256))
    out = image_png_b64(img, max_side=128)
    assert out["width"] == 256 and out["height"] == 512
    assert base64.b64decode(out["png"])[:4] == b"\x89PNG"


def test_summary_counts_mismatch_and_recompute():
    stat = [_disk(10, 10, 3), _disk(30, 30, 4), _disk(50, 15, 2)]
    labels = np.array([1, 2, 0])
    probs = np.array([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.55, 0.25, 0.2]])
    feats = pd.DataFrame({"radius": [3.0, 4.0, 2.0], "skew": [1.0, np.nan, 0.5]})
    ca_types = np.array([0, 2, 2])  # soma, artefact(mismatch: label says dend), artefact
    rp = probs.copy()
    rp[1] = [0.2, 0.6, 0.2]
    s = summarise_rois(
        labels,
        probs,
        stat=stat,
        features=feats,
        mean_img=np.ones((64, 64)),
        ca_roi_types=ca_types,
        recomputed_probs=rp,
        roi_qc={"snr_event": np.array([4.0, 2.0, 1.0])},
    )
    json.dumps(s)
    assert s["counts"] == {"artefact": 1, "soma": 1, "dend": 1}
    assert s["argmax_matches_label"] is True
    assert s["n_ambiguous"] == 2
    assert s["ca_label_mismatch"] == 1
    assert s["recomputed_label_changes"] == 1
    assert s["recomputed_max_abs_diff"] == pytest.approx(0.3)
    assert s["features"][1][1] is None
    assert s["rois"][0]["med"] == [10.0, 10.0] and s["rois"][0]["outline"]
    assert s["roi_qc"]["snr_event"] == [4.0, 2.0, 1.0]
    assert overview_row(s)["n_soma"] == 1


def test_summary_length_checks():
    with pytest.raises(ValueError):
        summarise_rois([0, 1], np.zeros((3, 3)))
    s = summarise_rois([0], [[1.0, 0, 0]], ca_roi_types=[0, 1])
    assert s["ca_label_mismatch"] is None and s["ca_label_len"] == 2


def test_cv_reference_separable_classes():
    rng = np.random.default_rng(0)
    n = 90
    y = np.repeat([0, 1, 2], n // 3)
    X = pd.DataFrame({"a": y + rng.normal(0, 0.1, n), "b": rng.normal(size=n)})
    g = np.tile(["s1", "s2", "s3"], n // 3)
    X.iloc[0, 1] = np.nan  # NaN filled with training-fold medians
    ref = cv_reference(X, y, g, {"n_estimators": 10, "max_depth": 2})
    json.dumps(ref)
    assert ref["n_sessions"] == 3 and ref["macro_f1"] > 0.9
    assert np.array(ref["confusion"]).sum() == n
    assert set(ref["feature_quantiles"]["a"]) == {"artefact", "soma", "dend"}
    assert ref["reliability"]["soma"]
