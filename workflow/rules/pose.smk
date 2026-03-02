"""Stage 2 — pose estimation (DLC default, SLEAP/LP alt).

GPU required. Run with local-gpu or aws-batch profile.
"""


rule run_pose:
    """Run pose tracker on overhead video."""
    input:
        video=f"{DATA_ROOT}/rawdata/{{sub}}/{{ses}}/behav/{{sub}}_{{ses}}_overhead.mp4",
        model=f"{DATA_ROOT}/sourcedata/trackers/dlc/",
    output:
        folder=directory(f"{DATA_ROOT}/derivatives/pose/{{sub}}/{{ses}}/"),
    params:
        session_id=wildcards_to_session_id,
        metadata_dir="metadata",
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

session_id = '{params.session_id}'
ses = get_session(
    session_id,
    animals_csv=Path('{params.metadata_dir}') / 'animals.csv',
    experiments_csv=Path('{params.metadata_dir}') / 'experiments.csv',
)
run_tracker(
    session=ses,
    video_path=Path('{input.video}'),
    model_dir=Path('{input.model}'),
    output_dir=Path('{output.folder}'),
)
"
        """
