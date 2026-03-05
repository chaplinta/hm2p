# Manual-Install Packages

Several scientific packages have incompatible transitive dependency pins
(numpy, scikit-learn, tensorflow) that prevent them from coexisting in a
single `uv`/`pip` environment. They are **not** listed in `pyproject.toml`
and must be installed manually in separate conda environments.

The core pipeline (`uv sync --extra dev`) works without any of these.

---

## Packages

| Package | Issue | Install |
| --- | --- | --- |
| **cascade2p** | Pins `tensorflow==2.3` (Python 3.8 only) | `conda create -n cascade python=3.8 tensorflow==2.3 cascade2p -c conda-forge` |
| **CaImAn** | Not on PyPI (PyPI `caiman` is unrelated v0.1) | `conda install -c conda-forge caiman` |
| **FISSA** >=1.0 | Pins `scikit-learn<1.2` | `pip install fissa` in a dedicated env |
| **keypoint-MoSeq** >=0.6 | Pins `numpy<=1.26` | `pip install keypoint-moseq` in a dedicated env |
| **vame-py** >=0.12 | Requires `numpy>=2.2` (conflicts with kpms) | `pip install vame-py` in a dedicated env |
| **CEBRA** >=0.6 | Pins `numpy<2.0` on some platforms | `pip install cebra` in a dedicated env |

---

## Usage pattern

These packages are used in specific pipeline stages or analysis steps:

- **CASCADE** — Stage 4 spike inference. Run in its own conda env, read/write `ca.h5`.
- **CaImAn** — Stage 1 alternative extractor. Run in its own conda env.
- **FISSA** — Stage 4 optional neuropil subtraction. Run in its own env.
- **keypoint-MoSeq / VAME** — Stage 3b behavioural syllables (deferred). Run in their own envs.
- **CEBRA** — Analysis phase, population embeddings. Run in its own env.

Each tool reads/writes the same HDF5 files, so they interoperate via the filesystem
even though they can't share a Python environment.
