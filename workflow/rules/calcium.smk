"""Stage 4 — calcium signal processing → ca.h5.

Neuropil subtraction → dF/F0 → CASCADE spike inference.
CPU only; fast enough on any machine.
CASCADE is optional (run_cascade=False by default; enable via config).
"""


rule process_calcium:
    """Neuropil → dF/F0 → [CASCADE spikes] → ca.h5."""
    input:
        suite2p=f"{DATA_ROOT}/derivatives/ca_extraction/{{sub}}/{{ses}}/suite2p/",
        timestamps=f"{DATA_ROOT}/derivatives/timestamps/{{sub}}/{{ses}}/timestamps.h5",
    output:
        h5=f"{DATA_ROOT}/derivatives/calcium/{{sub}}/{{ses}}/ca.h5",
    params:
        session_id=wildcards_to_session_id,
        neuropil_coefficient=config.get("neuropil_coefficient", 0.7),
        dff_baseline_window_s=config.get("dff_baseline_window_s", 60.0),
        run_cascade=config.get("run_cascade", False),
        cascade_model=config.get("cascade_model", "Global_EXC_7.5Hz_smoothing200ms"),
    resources:
        mem_mb=8000,
        runtime=30,
    shell:
        """
        python -c "
from hm2p.calcium.run import run
from pathlib import Path
run(
    suite2p_dir=Path('{input.suite2p}'),
    timestamps_h5=Path('{input.timestamps}'),
    session_id='{params.session_id}',
    output_path=Path('{output.h5}'),
    neuropil_coefficient={params.neuropil_coefficient},
    dff_baseline_window_s={params.dff_baseline_window_s},
    run_cascade={params.run_cascade},
    cascade_model='{params.cascade_model}',
)
"
        """
