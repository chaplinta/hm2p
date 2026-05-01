"""Stage 5 — neural–behavioural synchronisation + sync diagnostics → sync.h5."""


rule sync_neural_behav:
    """Resample kinematics to imaging rate, merge with calcium traces, classify
    sync_status, write sync.h5. See docs/sync-pipeline-design.md §3.

    Inputs include timestamps.h5 (for diagnostics) plus config/sync.yaml
    (thresholds). Output is either a full sync.h5 (OK / OK_WITH_WARNINGS)
    or a stub containing only the sync_status + sync_diag/ attrs
    (FAILED_*). Snakemake treats the rule as successful in either case;
    Stage 6's entry guard refuses FAILED_* sessions by default.
    """
    input:
        kinematics=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/kinematics.h5",
        calcium=f"{DATA_ROOT}/derivatives/calcium/{{sub}}/{{ses}}/ca.h5",
        timestamps=f"{DATA_ROOT}/derivatives/timestamps/{{sub}}/{{ses}}/timestamps.h5",
    output:
        h5=f"{DATA_ROOT}/derivatives/sync/{{sub}}/{{ses}}/sync.h5",
    container:
        cpu_container()
    params:
        session_id=wildcards_to_session_id,
        config_path="config/sync.yaml",
    resources:
        mem_mb=4000,
        runtime=10,
    shell:
        """
        python -c "
from hm2p.sync.align import run
from pathlib import Path
run(
    kinematics_h5=Path('{input.kinematics}'),
    ca_h5=Path('{input.calcium}'),
    session_id='{params.session_id}',
    output_path=Path('{output.h5}'),
    timestamps_h5=Path('{input.timestamps}'),
    config_path=Path('{params.config_path}'),
)
"
        """
