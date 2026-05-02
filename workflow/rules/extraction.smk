"""Stage 1 — 2P preprocessing and ROI extraction (Suite2p default, CaImAn alt).

GPU strongly recommended. Run with local-gpu or aws-batch profile.
"""

import json as _json

# Look up per-session indicator from experiments.csv.
# Falls back to "GCaMP6s" if the column is absent (assumption — verify per animal).
_indicator_default = "GCaMP6s"


def _session_indicator(wc) -> str:
    """Return GCaMP indicator for this session from experiments.csv."""
    session_id = wildcards_to_session_id(wc)
    row = EXPERIMENTS[EXPERIMENTS["exp_id"] == session_id]
    if row.empty or "indicator" not in row.columns:
        return _indicator_default
    val = str(row["indicator"].iloc[0]).strip()
    return val if val not in ("", "nan", "NaN") else _indicator_default


rule run_suite2p:
    """Run Suite2p on raw TIFF stack, producing native output folder.

    fps is read per-session from timestamps.h5 via fps_from_timestamps().
    tau is derived from the session indicator via tau_for_indicator().
    """
    input:
        tiffs=f"{DATA_ROOT}/rawdata/{{sub}}/{{ses}}/funcimg/",
        timestamps=f"{DATA_ROOT}/derivatives/timestamps/{{sub}}/{{ses}}/timestamps.h5",
    output:
        folder=directory(f"{DATA_ROOT}/derivatives/ca_extraction/{{sub}}/{{ses}}/suite2p/"),
    container:
        gpu_container()
    params:
        ops_json=lambda wc: _json.dumps(config.get("suite2p_ops", {})),
        indicator=_session_indicator,
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
    timestamps_h5=Path('{input.timestamps}'),
    indicator='{params.indicator}',
)
"
        """
