"""hm2p command-line interface.

Entry point: hm2p <command> [options]
Commands are thin wrappers that delegate to pipeline stage functions.

Usage examples:
    hm2p validate --session 20220804_13_52_02_1117646
    hm2p ingest   --session 20220804_13_52_02_1117646
    hm2p kinematics --session 20220804_13_52_02_1117646
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="hm2p",
    help="hm2p cloud pipeline — two-photon calcium imaging in freely-moving mice.",
    no_args_is_help=True,
)


@app.command()
def validate(
    session: str = typer.Option(..., help="Session ID (YYYYMMDD_HH_MM_SS_<animal_id>)"),
) -> None:
    """Validate raw files for a session (Stage 0, step 1)."""
    raise NotImplementedError


@app.command()
def ingest(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Parse DAQ TDMS file → timestamps.h5 (Stage 0, step 5)."""
    raise NotImplementedError


@app.command()
def kinematics(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Compute kinematics from pose output → kinematics.h5 (Stage 3)."""
    raise NotImplementedError


@app.command()
def calcium(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Process calcium signals → ca.h5 (Stage 4)."""
    raise NotImplementedError


@app.command()
def sync(
    session: str = typer.Option(..., help="Session ID"),
) -> None:
    """Synchronise neural and behavioural data → sync.h5 (Stage 5)."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
