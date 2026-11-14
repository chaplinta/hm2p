"""Tests for run_kpms.pca_variance_dict.

``scripts/run_kpms.py`` imports jax at module level (it runs in the kpms
container), so the function is extracted from the source and executed alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

SRC = Path(__file__).resolve().parents[2] / "scripts" / "run_kpms.py"


def _load():
    tree = ast.parse(SRC.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pca_variance_dict"
    )
    ns: dict = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
    return ns["pca_variance_dict"]


def test_sklearn_pca_object():
    f = _load()
    pca = PCA(n_components=3).fit(np.random.default_rng(0).normal(size=(50, 5)))
    out = f(pca)
    np.testing.assert_allclose(out["explained_variance_ratio"], pca.explained_variance_ratio_)


def test_dict_forms_and_empty():
    f = _load()
    pca = PCA(n_components=2).fit(np.random.default_rng(1).normal(size=(20, 4)))
    assert f({"variance_explained": np.array([0.5, 0.2])}) == {"variance_explained": [0.5, 0.2]}
    assert len(f({"model": pca})["explained_variance_ratio"]) == 2
    assert f(object()) == {}
