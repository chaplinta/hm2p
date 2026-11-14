"""Synthetic report documents for testing the QC page builder (tests only).

Each document has the same structure ``scripts/make_qc_reports.py`` writes,
built by running the real summarisers on small synthetic arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hm2p.qc import movement, rois, spikes, syllables, tracking
from tests.qc.test_movement import _kin
from tests.qc.test_rois import _disk
from tests.qc.test_spikes import _ca
from tests.qc.test_syllables import _sequence
from tests.qc.test_tracking import _mouse

META = [
    {
        "exp_id": "20990101_10_00_00_0000001",
        "sub": "sub-0000001",
        "ses": "ses-20990101T100000",
        "animal": "0000001",
        "celltype": "penk",
        "exclude": False,
        "primary": True,
        "notes": None,
    },
    {
        "exp_id": "20990102_10_00_00_0000002",
        "sub": "sub-0000002",
        "ses": "ses-20990102T100000",
        "animal": "0000002",
        "celltype": "nonpenk",
        "exclude": True,
        "primary": False,
        "notes": None,
    },
    {
        "exp_id": "20990103_10_00_00_0000003",
        "sub": "sub-0000003",
        "ses": "ses-20990103T100000",
        "animal": "0000003",
        "celltype": "penk",
        "exclude": False,
        "primary": False,
        "notes": None,
    },
]


def _doc(name: str, entries: list[dict], **extra: object) -> dict:
    return {
        "report": name,
        "generated": "2099-01-01",
        "champion": {"champion_id": "dlc-test-snap1"},
        "sessions": entries,
        **extra,
    }


def _entries(make, overview) -> list[dict]:  # type: ignore[no-untyped-def]
    out = []
    for i, m in enumerate(META):
        if i == 2:  # one failed session
            out.append(
                {**m, "summary": None, "overview": None, "error": "FileNotFoundError: test"}
            )
            continue
        s = make(i)
        out.append({**m, "summary": s, "overview": overview(s), "error": None})
    return out


def tracking_doc() -> dict:
    def make(i: int) -> dict:
        s = tracking.summarise_tracking(
            _mouse(seed=i), fps=30.0, mm_per_px=0.5, light_on=np.arange(900) < 450
        )
        s.update(pose_file="test.h5", champion_id="dlc-test-snap1", light_from_kinematics=True)
        return s

    return _doc("tracking", _entries(make, tracking.overview_row))


def movement_doc() -> dict:
    def make(i: int) -> dict:
        s = movement.summarise_movement(
            _kin(seed=i), {"dlc_champion_id": "dlc-test-snap1", "scale_mm_per_px": 0.5}
        )
        s["champion_current"] = i == 0
        return s

    return _doc("movement", _entries(make, movement.overview_row))


def syllables_doc() -> dict:
    def make(i: int) -> dict:
        sid = _sequence(seed=i)
        n = sid.size
        rng = np.random.default_rng(i)
        s = syllables.summarise_syllables(
            sid,
            30.0,
            speed_cm_s=rng.uniform(0, 10, n),
            ahv_deg_s=rng.normal(0, 50, n),
            light_on=(np.arange(n) // 1800) % 2 == 0,
        )
        s.update(
            aligned_to_kinematics=True,
            provenance={"kappa": 1e6, "num_pcs": 4, "dlc_champion_id": "dlc-test-snap1"},
        )
        return s

    model = {
        "summary": {"n_sessions": 2, "n_unique_syllables": 100},
        "pca_variance": {"explained_variance_ratio": [0.6, 0.2, 0.1, 0.05]},
        "fit_info": None,
        "convergence": None,
    }
    return _doc("syllables", _entries(make, syllables.overview_row), model=model)


def rois_doc() -> dict:
    def make(i: int) -> dict:
        rng = np.random.default_rng(i)
        n = 12
        stat = [
            _disk(int(rng.integers(10, 110)), int(rng.integers(10, 110)), int(rng.integers(2, 6)))
            for _ in range(n)
        ]
        probs = rng.dirichlet([1, 1, 1], n)
        feats = pd.DataFrame({"radius": rng.uniform(2, 6, n), "skew": rng.normal(1, 0.5, n)})
        return rois.summarise_rois(
            probs.argmax(1),
            probs,
            stat=stat,
            features=feats,
            mean_img=rng.normal(size=(128, 128)),
            ca_roi_types=rng.integers(0, 3, n),
            recomputed_probs=probs,
            roi_qc={"snr_event": rng.uniform(1, 8, n)},
        )

    rng = np.random.default_rng(9)
    n = 60
    y = np.repeat([0, 1, 2], n // 3)
    X = pd.DataFrame({"radius": y + rng.normal(0, 0.5, n), "skew": rng.normal(size=n)})
    ref = rois.cv_reference(
        X, y, np.tile(["a", "b", "c"], n // 3), {"n_estimators": 5, "max_depth": 2}
    )
    classifier = {
        "model": {
            "n_training_rois": 60,
            "class_counts": {"artefact": 20, "soma": 20, "dend": 20},
            "test_f1_macro_from_split": 0.8,
            "test_f1_dend_from_split": 0.7,
            "feature_names": ["radius", "skew"],
        },
        "reference": ref,
    }
    return _doc("rois", _entries(make, rois.overview_row), classifier=classifier)


def spikes_doc() -> dict:
    def make(i: int) -> dict:
        return spikes.summarise_spikes(
            _ca(),
            {
                "fps_imaging": 10.0,
                "f0_method": "rolling",
                "spikes_model": "test",
                "neuropil_method": "fissa",
            },
            light_on=(np.arange(3000) // 600) % 2 == 0,
        )

    return _doc("spikes", _entries(make, spikes.overview_row))


DOCS = {
    "tracking": tracking_doc,
    "movement": movement_doc,
    "syllables": syllables_doc,
    "rois": rois_doc,
    "spikes": spikes_doc,
}
