"""Stage 5 — neural–behavioural synchronisation → sync.h5."""


rule sync_neural_behav:
    """Resample kinematics to imaging rate, merge with calcium traces → sync.h5."""
    input:
        kinematics=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/kinematics.h5",
        calcium=f"{DATA_ROOT}/derivatives/calcium/{{sub}}/{{ses}}/ca.h5",
    output:
        h5=f"{DATA_ROOT}/derivatives/sync/{{sub}}/{{ses}}/sync.h5",
    params:
        session_id=wildcards_to_session_id,
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
)
"
        """
