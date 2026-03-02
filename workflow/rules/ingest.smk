"""Stage 0 — data ingest, validation, DAQ parsing → timestamps.h5."""

from pathlib import Path


def _find_tdms(wildcards) -> str:
    """Locate the *-di.tdms file in funcimg/ for a given session."""
    funcimg = (
        Path(DATA_ROOT)
        / "rawdata"
        / wildcards.sub
        / wildcards.ses
        / "funcimg"
    )
    matches = sorted(funcimg.glob("*-di.tdms"))
    if not matches:
        raise FileNotFoundError(f"No *-di.tdms found in {funcimg}")
    return str(matches[0])


rule parse_daq:
    """Parse TDMS DAQ file → timestamps.h5 (camera + imaging frame times, light pulses)."""
    input:
        tdms=_find_tdms,
    output:
        h5=f"{DATA_ROOT}/derivatives/timestamps/{{sub}}/{{ses}}/timestamps.h5",
    container:
        cpu_container()
    params:
        session_id=wildcards_to_session_id,
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
