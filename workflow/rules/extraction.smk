"""Stage 1 — 2P preprocessing and ROI extraction (Suite2p default, CaImAn alt).

GPU strongly recommended. Run with local-gpu or aws-batch profile.
"""

import json as _json


rule run_suite2p:
    """Run Suite2p on raw TIFF stack, producing native output folder."""
    input:
        tiffs=f"{DATA_ROOT}/rawdata/{{sub}}/{{ses}}/funcimg/",
    output:
        folder=directory(f"{DATA_ROOT}/derivatives/ca_extraction/{{sub}}/{{ses}}/suite2p/"),
    params:
        fps=config.get("imaging_fps", 29.97),
        ops_json=lambda wc: _json.dumps(config.get("suite2p_ops", {})),
    resources:
        mem_mb=32000,
        runtime=120,
        gpu=1,
    shell:
        """
        python -c "
import json
from hm2p.extraction.run_suite2p import run_suite2p
from pathlib import Path

ops = json.loads('''{params.ops_json}''')
run_suite2p(
    tiff_dir=Path('{input.tiffs}'),
    output_dir=Path('{output.folder}').parent,
    ops_overrides=ops or None,
    fps={params.fps},
)
"
        """
