# Notepad — hm2p-v2

Working notes and decisions log. Not a design doc — use PLAN.md / ARCHITECTURE.md for that.

---

## Decisions Made

### 2026-02-28

- **Repo location**: `/Users/tristan/Neuro/hm2p-v2` → GitHub `chaplinta/hm2p` (private)
- **File rules**: read-only on `hm2p-analysis/` and Dropbox data; can copy INTO `hm2p-v2`
- **Calcium abstraction**: `roiextractors` as unified API (Suite2p default, CaImAn alt)
- **Pose abstraction**: `movement` as unified API (DLC default, SLEAP/LP alts)
- **Scope**: Stages 0–5 only for now — extraction + sync, no analysis
- **Primary behavioural outputs**: HD, position, speed (AHV also computed)
- **Future behaviour**: syllables/sequences (VAME, B-SOiD, MoSeq) — not in scope yet
- **Local processing**: all CPU stages can run locally; GPU stages need GPU (cloud or local)
- **Snakemake profiles**: `local`, `local-gpu`, `aws-batch`
- **Cloud provider**: AWS (S3 + EC2 Spot)
- **Design philosophy**: ground-up redesign, fully unit-tested, not a copy of old code
- **Versions**: always latest stable

### MD filename convention
- Root-level: ALL CAPS (PLAN, ARCHITECTURE, AGENTS, README, NOTEPAD)
- `docs/`: lowercase-hyphenated

---

## To Do

- [ ] Explore raw data + old code → write `docs/data-guide.md`
- [ ] Finalise HDF5 schemas (kinematics.h5, ca.h5, sync.h5)
- [ ] Set up `pyproject.toml` and project structure
- [ ] Set up GitHub Actions CI (pytest + ruff)
- [ ] Upload raw data to S3 following NeuroBlueprint layout

---

## Open Questions

- Which AWS region? (eu-west-2 London assumed — confirm)
- Will institutional HPC (UCL) be available as an alternative compute option?
- Any existing `experiments.csv` / `animals.csv` to copy in from legacy repo?
