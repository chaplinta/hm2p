"""Stage 2 — pose estimation (DLC default, SLEAP/LP alt).

GPU required. Run with local-gpu or aws-batch profile.
"""


rule run_dlc:
    """Run DeepLabCut inference on overhead video."""
    input:
        video=f"{DATA_ROOT}/rawdata/{{sub}}/{{ses}}/behav/{{sub}}_{{ses}}_overhead.mp4",
        model=f"{DATA_ROOT}/sourcedata/trackers/dlc/",
    output:
        pose=f"{DATA_ROOT}/derivatives/pose/{{sub}}/{{ses}}/{{sub}}_{{ses}}_DLC_resnet50.h5",
    resources:
        mem_mb=8000,
        runtime=60,
        gpu=1,
    shell:
        """
        python -c "
from hm2p.pose.run import run_tracker
from hm2p.session import get_session
from pathlib import Path
# TODO: load session and dispatch
"
        """
