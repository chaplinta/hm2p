"""Stage 4 — calcium signal processing → ca.h5.

Neuropil subtraction → dF/F0 → CASCADE spike inference.
CPU only; fast enough on any machine.
"""


rule process_calcium:
    """Neuropil → dF/F0 → CASCADE spikes → ca.h5."""
    input:
        extraction=f"{DATA_ROOT}/derivatives/ca_extraction/{{sub}}/{{ses}}/suite2p/",
        timestamps=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/timestamps.h5",
    output:
        h5=f"{DATA_ROOT}/derivatives/calcium/{{sub}}/{{ses}}/ca.h5",
    resources:
        mem_mb=8000,
        runtime=30,
    shell:
        """
        python -c "
from hm2p.calcium import neuropil, dff, spikes
from pathlib import Path
# TODO: implement calcium processing pipeline
"
        """
