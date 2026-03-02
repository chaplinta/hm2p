"""Stage 2 — pose estimation (DLC default, SLEAP/LP alt).

GPU required. Run with local-gpu or aws-batch profile.
"""

from pathlib import Path


def _find_overhead_video(wildcards) -> str:
    """Locate the overhead video in behav/.

    Actual files on S3 are named ``*-cropped.mp4`` (legacy preprocessing).
    Falls back to ``*_overhead*.mp4`` for newer naming conventions.
    """
    behav = (
        Path(DATA_ROOT)
        / "rawdata"
        / wildcards.sub
        / wildcards.ses
        / "behav"
    )
    for pattern in ("*-cropped.mp4", "*_overhead*.mp4"):
        matches = sorted(behav.glob(pattern))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No overhead video found in {behav}")


rule run_pose:
    """Run pose tracker on overhead video."""
    input:
        video=_find_overhead_video,
        model=f"{DATA_ROOT}/sourcedata/trackers/dlc/",
    output:
        folder=directory(f"{DATA_ROOT}/derivatives/pose/{{sub}}/{{ses}}/"),
    container:
        gpu_container()
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
