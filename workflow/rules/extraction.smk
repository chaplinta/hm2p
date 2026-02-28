"""Stage 1 — 2P preprocessing and ROI extraction (Suite2p default, CaImAn alt).

GPU strongly recommended. Run with local-gpu or aws-batch profile.
"""


rule run_suite2p:
    """Run Suite2p on raw TIFF stack, producing native output folder."""
    input:
        tiffs=f"{DATA_ROOT}/rawdata/{{sub}}/{{ses}}/funcimg/",
    output:
        folder=directory(f"{DATA_ROOT}/derivatives/ca_extraction/{{sub}}/{{ses}}/suite2p/"),
    params:
        ops=config.get("suite2p_ops", {}),
    resources:
        mem_mb=32000,
        runtime=120,
        gpu=1,
    shell:
        """
        python -c "
import suite2p
from pathlib import Path
# TODO: implement Suite2p wrapper in extraction/suite2p.py
"
        """
