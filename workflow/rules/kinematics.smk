"""Stage 3 — behavioural kinematics via movement → kinematics.h5."""

from pathlib import Path


def _find_pose_file(wildcards) -> str:
    """Locate the tracker-native pose output file in derivatives/pose/."""
    pose_dir = (
        Path(DATA_ROOT)
        / "derivatives"
        / "pose"
        / wildcards.sub
        / wildcards.ses
    )
    for pattern in ("*.h5", "*.csv", "*.slp"):
        matches = sorted(pose_dir.glob(pattern))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No pose file found in {pose_dir}")


def _find_behav_meta(wildcards) -> str:
    """Locate the behav/meta.txt (crop ROI, scale, maze corners)."""
    behav = (
        Path(DATA_ROOT)
        / "rawdata"
        / wildcards.sub
        / wildcards.ses
        / "behav"
    )
    matches = sorted(behav.glob("*.meta.txt"))
    if not matches:
        raise FileNotFoundError(f"No *.meta.txt found in {behav}")
    return str(matches[0])


rule compute_kinematics:
    """Load pose, apply orientation rotation, compute HD/position/speed/AHV, write kinematics.h5."""
    input:
        pose=_find_pose_file,
        timestamps=f"{DATA_ROOT}/derivatives/timestamps/{{sub}}/{{ses}}/timestamps.h5",
        meta_txt=_find_behav_meta,
    output:
        h5=f"{DATA_ROOT}/derivatives/movement/{{sub}}/{{ses}}/kinematics.h5",
    params:
        session_id=wildcards_to_session_id,
        metadata_dir="metadata",
    resources:
        mem_mb=4000,
        runtime=20,
    shell:
        """
        python -c "
from hm2p.kinematics.compute import run
from hm2p.pose.preprocess import load_meta
from hm2p.session import get_session, parse_bad_behav_times
from hm2p.io.hdf5 import read_h5
from pathlib import Path

session_id = '{params.session_id}'
ses = get_session(
    session_id,
    animals_csv=Path('{params.metadata_dir}') / 'animals.csv',
    experiments_csv=Path('{params.metadata_dir}') / 'experiments.csv',
)
meta = load_meta(Path('{input.meta_txt}'))
ts = read_h5(Path('{input.timestamps}'))
total_seconds = float(ts['frame_times_camera'][-1])
bad_behav_intervals = parse_bad_behav_times(ses.bad_behav_times, total_seconds)
run(
    pose_path=Path('{input.pose}'),
    timestamps_h5=Path('{input.timestamps}'),
    session_id=session_id,
    tracker=ses.tracker,
    orientation_deg=ses.orientation,
    scale_mm_per_px=meta['scale_mm_per_px'],
    maze_corners_px=meta['maze_corners'],
    bad_behav_intervals=bad_behav_intervals,
    output_path=Path('{output.h5}'),
)
"
        """
