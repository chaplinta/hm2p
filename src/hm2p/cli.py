"""hm2p command-line interface.

Entry point: hm2p <command> [options]
Commands are thin wrappers that delegate to pipeline stage functions.

Usage examples:
    hm2p validate --session 20220804_13_52_02_1117646
    hm2p ingest   --session 20220804_13_52_02_1117646
    hm2p kinematics --session 20220804_13_52_02_1117646
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="hm2p",
    help="hm2p cloud pipeline — two-photon calcium imaging in freely-moving mice.",
    no_args_is_help=True,
)


def _load_session(session_id: str) -> tuple[object, object]:
    """Load PipelineConfig and Session for a given session ID."""
    from hm2p.config import load_config
    from hm2p.session import get_session

    cfg = load_config()
    animals_csv = cfg.metadata_dir / "animals.csv"
    experiments_csv = cfg.metadata_dir / "experiments.csv"
    ses = get_session(session_id, animals_csv, experiments_csv)
    return cfg, ses


@app.command()
def validate(
    session: str = typer.Option(..., help="Session ID (YYYYMMDD_HH_MM_SS_<animal_id>)"),
) -> None:
    """Validate raw files for a session (Stage 0, step 1)."""
    from hm2p.ingest.validate import validate_session

    cfg, ses = _load_session(session)
    rawdata_root = cfg.data_root / "rawdata"
    result = validate_session(ses, rawdata_root)

    if result.ok:
        typer.echo(f"[OK] {session} — all required raw files present")
    else:
        typer.echo(f"[FAIL] {session} — missing files:")
        for item in result.missing:
            typer.echo(f"  • {item}")
        raise typer.Exit(code=1)


@app.command()
def ingest(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Parse DAQ TDMS file → timestamps.h5 (Stage 0, step 5)."""
    from hm2p.ingest.daq import run as daq_run

    cfg, ses = _load_session(session)

    funcimg_dir = (
        cfg.data_root
        / "rawdata"
        / ses.neurobluepint_sub
        / ses.neurobluepint_ses
        / "funcimg"
    )
    tdms_files = sorted(funcimg_dir.glob("*-di.tdms"))
    if not tdms_files:
        typer.echo(f"[FAIL] {session} — no *-di.tdms found in {funcimg_dir}")
        raise typer.Exit(code=1)

    out_dir = ses.derivatives_path("timestamps", cfg.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timestamps.h5"

    daq_run(tdms_files[0], session_id=session, output_path=out_path)
    typer.echo(f"[OK] {session} — timestamps.h5 written to {out_path}")


@app.command()
def kinematics(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Compute kinematics from pose output → kinematics.h5 (Stage 3)."""
    from hm2p.kinematics.compute import run as kin_run
    from hm2p.pose.preprocess import load_meta
    from hm2p.session import parse_bad_behav_times

    cfg, ses = _load_session(session)

    # --- Locate required inputs ---
    timestamps_h5 = ses.derivatives_path("timestamps", cfg.data_root) / "timestamps.h5"
    if not timestamps_h5.exists():
        typer.echo(
            f"[FAIL] {session} — timestamps.h5 not found at {timestamps_h5}. "
            "Run `hm2p ingest` first."
        )
        raise typer.Exit(code=1)

    pose_dir = ses.derivatives_path("pose", cfg.data_root)
    pose_candidates = sorted(pose_dir.glob("*.h5")) + sorted(pose_dir.glob("*.csv"))
    if not pose_candidates:
        typer.echo(
            f"[FAIL] {session} — no pose file found in {pose_dir}. "
            "Run Stage 2 (pose estimation) first."
        )
        raise typer.Exit(code=1)
    pose_path = pose_candidates[0]

    # --- Meta.txt (crop ROI, scale, maze corners) ---
    behav_dir = (
        cfg.data_root
        / "rawdata"
        / ses.neurobluepint_sub
        / ses.neurobluepint_ses
        / "behav"
    )
    meta_files = sorted(behav_dir.glob("*.meta.txt"))
    if not meta_files:
        typer.echo(f"[FAIL] {session} — no *.meta.txt found in {behav_dir}")
        raise typer.Exit(code=1)
    meta = load_meta(meta_files[0])

    # --- bad_behav intervals (total_seconds from timestamps) ---
    from hm2p.io.hdf5 import read_h5

    ts = read_h5(timestamps_h5)
    total_seconds = float(ts["frame_times_camera"][-1])
    bad_behav_intervals = parse_bad_behav_times(ses.bad_behav_times, total_seconds)

    # --- Output path ---
    out_dir = ses.derivatives_path("movement", cfg.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kinematics.h5"

    kin_run(
        pose_path=pose_path,
        timestamps_h5=timestamps_h5,
        session_id=session,
        tracker=ses.tracker,
        orientation_deg=ses.orientation,
        scale_mm_per_px=meta["scale_mm_per_px"],
        maze_corners_px=meta["maze_corners"],
        bad_behav_intervals=bad_behav_intervals,
        output_path=out_path,
        confidence_threshold=cfg.pose_confidence_threshold,
        gap_fill_frames=cfg.pose_gap_fill_frames,
        speed_active_threshold=cfg.speed_active_threshold_cm_s,
    )
    typer.echo(f"[OK] {session} — kinematics.h5 written to {out_path}")


@app.command()
def calcium(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Process calcium signals → ca.h5 (Stage 4).

    Stage 4 requires GPU for CASCADE spike inference and roiextractors
    integration with Suite2p/CaImAn outputs. Run this stage on the cloud
    (AWS EC2 g4dn Spot) or a local GPU machine via the Snakemake workflow:

        snakemake --profile profiles/aws-batch calcium --config session=<id>
    """
    typer.echo(
        "[INFO] Stage 4 (calcium processing) requires GPU for spike inference "
        "and is designed to run on the cloud via Snakemake.\n"
        "  snakemake --profile profiles/aws-batch calcium "
        f"--config session={session}\n"
        "Local roiextractors integration is not yet implemented in the CLI."
    )
    raise typer.Exit(code=0)


@app.command()
def sync(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Synchronise neural and behavioural data → sync.h5 (Stage 5)."""
    from hm2p.sync.align import run as sync_run

    cfg, ses = _load_session(session)

    kinematics_h5 = ses.derivatives_path("movement", cfg.data_root) / "kinematics.h5"
    ca_h5 = ses.derivatives_path("calcium", cfg.data_root) / "ca.h5"

    if not kinematics_h5.exists():
        typer.echo(
            f"[FAIL] {session} — kinematics.h5 not found at {kinematics_h5}. "
            "Run `hm2p kinematics` first."
        )
        raise typer.Exit(code=1)
    if not ca_h5.exists():
        typer.echo(
            f"[FAIL] {session} — ca.h5 not found at {ca_h5}. "
            "Run Stage 4 (calcium) first."
        )
        raise typer.Exit(code=1)

    out_dir = ses.derivatives_path("sync", cfg.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sync.h5"

    sync_run(
        kinematics_h5=kinematics_h5,
        ca_h5=ca_h5,
        session_id=session,
        output_path=out_path,
    )
    typer.echo(f"[OK] {session} — sync.h5 written to {out_path}")


if __name__ == "__main__":
    app()
