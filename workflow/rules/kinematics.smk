"""Stage 3 — behavioural kinematics via movement → kinematics.h5."""


rule compute_kinematics:
    """Load pose, apply orientation rotation, compute HD/position/speed/AHV, write kinematics.h5."""
    input:
        pose=f"{DATA_ROOT}/derivatives/pose/{{sub}}/{{ses}}/{{sub}}_{{ses}}_DLC_resnet50.h5",
        timestamps=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/timestamps.h5",
    output:
        h5=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/kinematics.h5",
    resources:
        mem_mb=4000,
        runtime=20,
    shell:
        """
        python -c "
from hm2p.kinematics.compute import run
from pathlib import Path
# TODO: load session metadata and call run()
"
        """
