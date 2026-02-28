"""Stage 0 — data ingest, validation, DAQ parsing → timestamps.h5."""


rule parse_daq:
    """Parse TDMS DAQ file → timestamps.h5 (camera + imaging frame times, light pulses)."""
    input:
        tdms=f"{DATA_ROOT}/rawdata/{{sub}}/{{ses}}/behav/daq.tdms",
    output:
        h5=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/timestamps.h5",
    params:
        session_id=lambda wc: f"{wc.ses}_{wc.sub.replace('sub-', '')}",
    resources:
        mem_mb=2000,
        runtime=10,  # minutes
    shell:
        """
        python -c "
from hm2p.ingest.daq import run
from pathlib import Path
run(
    tdms_path=Path('{input.tdms}'),
    session_id='{params.session_id}',
    output_path=Path('{output.h5}'),
)
"
        """
