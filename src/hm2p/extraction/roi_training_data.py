"""Manually labelled ROIs from the legacy dual Suite2p runs (classifier training data).

Each legacy session in ``/data/s2p/<session>/`` has two Suite2p runs on the
same ROI set: ``suite2p_soma`` (``iscell`` = ROIs accepted as soma) and
``suite2p_dend`` (``iscell`` = ROIs accepted as dendrites). Labels are
0 = artefact (accepted in neither), 1 = soma, 2 = dendrite. Features are the
26 classifier features from :func:`extract_soma_features`, at the frame rate
stored in the legacy ``ops.npy`` (9.6-9.8 Hz, the true imaging rate).

Used by ``scripts/train_roi_classifier.py`` and the classification QC report
(``scripts/make_qc_reports.py``); kept here so the QC report does not need the
training script's dependencies (optuna).

Pachitariu M, Stringer C, Dipoppa M, et al. 2017. "Suite2p: beyond 10,000
neurons with standard two-photon microscopy." bioRxiv. doi:10.1101/061507.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hm2p.extraction.soma_features import extract_soma_features

log = logging.getLogger(__name__)

S2P_ROOT = Path("/data/s2p")


def load_session_data(session_dir: Path) -> dict | None:
    """Features and 3-way labels for one legacy session, or None if unusable."""
    soma_dir = session_dir / "suite2p_soma" / "plane0"
    dend_dir = session_dir / "suite2p_dend" / "plane0"

    if not soma_dir.exists() or not dend_dir.exists():
        return None

    ic_soma = np.load(soma_dir / "iscell.npy")
    ic_dend = np.load(dend_dir / "iscell.npy")

    n_soma = int((ic_soma[:, 0] == 1).sum())
    n_dend = int((ic_dend[:, 0] == 1).sum())

    if n_soma + n_dend == 0:
        return None

    n_rois = len(ic_soma)
    labels = np.zeros(n_rois, dtype=np.int64)
    labels[ic_soma[:, 0] == 1] = 1
    labels[ic_dend[:, 0] == 1] = 2

    overlap = ((ic_soma[:, 0] == 1) & (ic_dend[:, 0] == 1)).sum()
    if overlap > 0:
        log.warning(
            "%s: %d ROIs labeled as both soma and dend — skipping", session_dir.name, overlap
        )
        return None

    stat = list(np.load(soma_dir / "stat.npy", allow_pickle=True))
    F = np.load(soma_dir / "F.npy").astype(np.float32)
    Fneu = np.load(soma_dir / "Fneu.npy").astype(np.float32)
    ops = np.load(soma_dir / "ops.npy", allow_pickle=True).item()
    fps = float(ops.get("fs", 9.6))

    features = extract_soma_features(stat, F, Fneu, fps=fps)

    return {
        "session_id": session_dir.name,
        "features": features,
        "labels": labels,
        "n_soma": n_soma,
        "n_dend": n_dend,
        "n_artefact": n_rois - n_soma - n_dend,
        "n_rois": n_rois,
    }


def load_all_sessions(root: Path = S2P_ROOT) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features, labels and session ids for every usable session under *root*."""
    feature_list = []
    label_list = []
    group_list = []

    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir():
            continue
        data = load_session_data(session_dir)
        if data is None:
            log.info("Skipping %s (no labeled cells)", session_dir.name)
            continue

        feature_list.append(data["features"])
        label_list.append(data["labels"])
        group_list.append(np.full(data["n_rois"], data["session_id"]))

        log.info(
            "  %s: %d ROIs (%d soma, %d dend, %d artefact)",
            data["session_id"],
            data["n_rois"],
            data["n_soma"],
            data["n_dend"],
            data["n_artefact"],
        )

    if not feature_list:
        raise FileNotFoundError(f"no labelled sessions under {root}")
    X = pd.concat(feature_list, ignore_index=True)
    y = np.concatenate(label_list)
    groups = np.concatenate(group_list)

    return X, y, groups
