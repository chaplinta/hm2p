# Research Landscape: AI-Built Neuroscience Pipelines (March 2026)

## Key Finding
No other project uses Claude Code (or any AI coding agent) to build an end-to-end
systems neuroscience pipeline. hm2p appears to be unique in this regard.

## Closest Comparable Projects

### photon-mosaic (neuroinformatics-unit)
- **URL**: https://github.com/neuroinformatics-unit/photon-mosaic
- **What**: Snakemake-based toolkit for multiphoton calcium imaging
- **Overlap with hm2p**: Suite2p + CaImAn integration, Snakemake orchestration,
  NeuroBlueprint data standard, NWB export, SLURM/HPC scaling
- **Differences**: No behaviour pipeline, no DLC/movement integration, no frontend,
  no cloud (AWS) support — HPC only. Adds Cellpose v3/v4 for anatomical ROI extraction.
- **Ideas for hm2p**:
  - Cellpose integration for anatomical ROI segmentation (complementary to Suite2p)
  - Their modular "step" architecture for swapping algorithms per stage
  - Consider contributing our AWS Batch executor back to Snakemake ecosystem

### UCLA 2P Miniscope (golshanilab)
- **URL**: https://github.com/golshanilab/UCLA_2P_Miniscope
- **What**: Combined 2P + miniscope pipeline (Suite2p + DLC + MATLAB)
- **Overlap**: Suite2p + DLC for same animals
- **Differences**: MATLAB-heavy, no cloud, no Snakemake, no automated pipeline
- **Ideas**: Their dual-modality (2P + miniscope) approach is interesting

### Mesmerize / mesmerize-core
- **URL**: https://github.com/kushalkolar/MESmerize (deprecated)
- **What**: GUI platform for calcium imaging with CaImAn backend, FAIR data
- **Status**: Original deprecated; mesmerize-core continues as CaImAn parameter
  optimization and visualization layer
- **Ideas**: Their parameter sweep / optimization approach for source extraction

### CIAtah (bahanonu)
- **URL**: https://github.com/bahanonu/ciatah
- **What**: MATLAB/Python calcium imaging analysis for 1P and 2P
- **Ideas**: Comprehensive imaging_tools table cataloguing all analysis tools

### OptiNiSt (Optical Neuroimage Studio)
- **URL**: PMC article PMC12124740
- **What**: Intuitive, scalable, extendable framework for optical neuroimage analysis
- **Ideas**: Their extensibility/plugin architecture

### NeuroWRAP
- **URL**: https://www.frontiersin.org/articles/10.3389/fninf.2023.1082111
- **What**: Framework for integrating, validating, and sharing neurodata workflows
- **Ideas**: Their validation and sharing approach

## AI/Agent Landscape

### Claude Code adoption (March 2026)
- 4% of GitHub commits now made by Claude Code (SemiAnalysis report)
- Claude Opus 4.6 (Feb 2026) — improved scientific reasoning
- Claude for Life Sciences launched Oct 2025
- everything-claude-code: agent harness with skills/memory (hackathon project)
- No scientific pipeline projects found using Claude Code specifically

### Agentic Science (emerging field)
- arXiv 2508.14111: "From AI for Science to Agentic Science" survey
- DeepAnalyze-8B: first agentic LLM for autonomous data science
- Agentic bioinformatics (Oxford Academic): end-to-end AI agents across research
- Trend: autonomous agents managing entire research pipelines, but all
  bioinformatics/genomics focused — no systems neuroscience examples

### Vibe Coding concerns
- arXiv 2601.15494: "Vibe Coding Kills Open Source" paper
- AI-generated "slop" contributions flooding open source
- Key risk: code that works but nobody understands
- Mitigation for hm2p: comprehensive tests, CLAUDE.md instructions, citations

## Ideas for hm2p Improvements (do not implement yet)

### From photon-mosaic
1. **Cellpose integration** for anatomical ROI segmentation
2. **Algorithm comparison mode** — run Suite2p AND CaImAn on same data, compare
3. **SLURM profile** for Snakemake (in addition to AWS Batch)

### From the broader landscape
4. **Parameter sweep/optimization** for extraction (mesmerize-core pattern)
5. **NWB export** as a first-class pipeline stage (not just archival)
6. **DataShuttle integration** for automated folder creation + S3 sync
7. **Interactive QC notebook** — Jupyter widget for manual ROI curation
8. **Provenance tracking** — log every parameter + software version per session
9. **Multi-session registration** — CaImAn's `register_multisession` for
   tracking same neurons across days
10. **Authentication** for Streamlit — critical before any public deployment

### From AI/agent trends
11. **Claude Code GitHub Action** — auto-implement issues, run tests, open PRs
12. **RAG on docs** — embed CLAUDE.md + ARCHITECTURE.md + method papers for
    better agent context
13. **Automated paper search** — agent checks for new methods in relevant fields
