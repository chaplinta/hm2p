"""NWB export via neuroconv — convert pipeline HDF5 outputs to NWB format.

Wraps neuroconv to convert roiextractors SegmentationExtractor objects and
kinematics arrays into NWB files suitable for archiving on DANDI.

Reference: neuroconv.readthedocs.io
"""

from __future__ import annotations

from pathlib import Path


def export_session_to_nwb(
    ca_h5: Path,
    kinematics_h5: Path,
    sync_h5: Path,
    session_id: str,
    output_path: Path,
) -> None:
    """Convert one session's pipeline outputs to an NWB file.

    Args:
        ca_h5: Stage 4 calcium output.
        kinematics_h5: Stage 3 kinematics output.
        sync_h5: Stage 5 sync output.
        session_id: Canonical session identifier stored as NWB metadata.
        output_path: Destination .nwb file path.
    """
    raise NotImplementedError
