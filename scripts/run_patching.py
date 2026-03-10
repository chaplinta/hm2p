#!/usr/bin/env python3
"""Run the patching electrophysiology pipeline.

Loads metadata, resolves the date-based ephys directory structure
(each animal's ephys data lives under ephys_root/<date_slice>/SW000N/),
processes all cells, and saves metrics to the analysis output directory.

Usage:
    python scripts/run_patching.py                    # run full pipeline
    python scripts/run_patching.py --dry-run           # show what would be done
    python scripts/run_patching.py --animal CAA-1116873  # single animal
    python scripts/run_patching.py --stats             # also run statistics + PCA
    python scripts/run_patching.py --config path.yaml  # custom config
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.patching.config import PatchConfig
from hm2p.patching.run import (
    load_metadata,
    process_cell,
    run_pca_analysis,
    run_statistics,
)
from hm2p.patching.metrics import build_metrics_table, compute_derived_metrics

logger = logging.getLogger(__name__)

# Default config path relative to repo root
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "patching.yaml"


def load_extended_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML config, which includes ephys_root_dir (non-standard field).

    PatchConfig expects ephys_dir (a single directory), but our data is
    organised as ephys_root_dir/<date_slice>/SW000N/.  We read the raw
    YAML to get ephys_root_dir, then build per-animal PatchConfig objects
    in the processing loop.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def build_animal_config(raw_config: dict[str, Any], date_slice: str) -> PatchConfig:
    """Build a PatchConfig with ephys_dir pointing to the date subdirectory.

    Parameters
    ----------
    raw_config : dict
        Raw YAML config with ephys_root_dir and other standard fields.
    date_slice : str
        Date directory name (e.g., '220307') for this animal.

    Returns
    -------
    PatchConfig
        Config with ephys_dir = ephys_root_dir / date_slice.
    """
    ephys_root = Path(raw_config["ephys_root_dir"])
    ephys_dir = ephys_root / date_slice

    return PatchConfig(
        metadata_dir=Path(raw_config["metadata_dir"]),
        morph_dir=Path(raw_config["morph_dir"]),
        ephys_dir=ephys_dir,
        processed_dir=Path(raw_config["processed_dir"]),
        analysis_dir=Path(raw_config["analysis_dir"]),
    )


def get_date_slice_map(metadata_dir: Path) -> dict[str, str]:
    """Read animals.csv and return {animal_id: date_slice} mapping."""
    animals_path = metadata_dir / "animals.csv"
    if not animals_path.exists():
        raise FileNotFoundError(f"animals.csv not found: {animals_path}")
    animals = pd.read_csv(animals_path)
    return dict(zip(animals["animal_id"], animals["date_slice"].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the patching electrophysiology pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to patching.yaml config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    parser.add_argument(
        "--animal",
        type=str,
        default=None,
        help="Process only this animal_id (e.g., CAA-1116873)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Also run summary statistics and PCA after processing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    raw_config = load_extended_config(args.config)
    metadata_dir = Path(raw_config["metadata_dir"])
    analysis_dir = Path(raw_config["analysis_dir"])

    # Copy metadata to a temp dir with UTF-8 encoding.
    # The source cells.csv uses Latin-1 (degree symbols in Orientation column).
    # load_metadata() uses pd.read_csv with default UTF-8 encoding, so we
    # re-encode the files to avoid UnicodeDecodeError.
    tmp_meta_dir = Path(tempfile.mkdtemp(prefix="hm2p-patching-meta-"))
    for csv_name in ("animals.csv", "cells.csv"):
        src = metadata_dir / csv_name
        dst = tmp_meta_dir / csv_name
        text = src.read_bytes().decode("latin-1")
        dst.write_text(text, encoding="utf-8")
    metadata_dir = tmp_meta_dir
    raw_config = {**raw_config, "metadata_dir": str(tmp_meta_dir)}
    print(f"Re-encoded metadata to UTF-8 in {tmp_meta_dir}")

    # Build animal -> date_slice mapping
    date_map = get_date_slice_map(metadata_dir)
    print(f"Animal -> date_slice mapping ({len(date_map)} animals):")
    for animal_id, date_slice in date_map.items():
        print(f"  {animal_id} -> {date_slice}")

    # Load metadata (merged cells + animals)
    # Use a temporary config just for loading metadata
    first_date = next(iter(date_map.values()))
    tmp_config = build_animal_config(raw_config, first_date)
    metadata = load_metadata(tmp_config)

    # Filter to single animal if requested
    if args.animal:
        metadata = metadata[metadata["animal_id"] == args.animal]
        if len(metadata) == 0:
            print(f"ERROR: No cells found for animal {args.animal}")
            sys.exit(1)

    print(f"\nTotal cells to process: {len(metadata)}")

    # Count cells with ephys data
    has_ephys = metadata["ephys_id"].notna() & (metadata["ephys_id"] != "")
    has_morph = metadata.get("good_morph", pd.Series(False, index=metadata.index))
    has_morph = has_morph.fillna(False).astype(bool)
    print(f"  With ephys data: {has_ephys.sum()}")
    print(f"  With good morphology: {has_morph.sum()}")
    print(f"  Morphology available in mount: NO (SWC files not in bind mount)")

    # Dry run: show what would be processed
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN — showing planned processing:")
        print(f"{'='*60}")
        for idx, row in metadata.iterrows():
            cell_id = row.get("cell_index", idx)
            animal_id = row["animal_id"]
            ephys_id = row.get("ephys_id", "")
            date_slice = date_map.get(animal_id, "???")
            cell_type = row.get("cell_type", "?")

            if pd.isna(ephys_id) or not str(ephys_id).strip():
                ephys_status = "NO EPHYS"
                ephys_path = ""
            else:
                ephys_id_str = str(ephys_id).strip()
                ephys_path = f"{raw_config['ephys_root_dir']}/{date_slice}/{ephys_id_str}"
                ephys_exists = Path(ephys_path).is_dir()
                ephys_status = "OK" if ephys_exists else "MISSING"

            morph_flag = bool(row.get("good_morph", False))
            morph_status = "SKIP (no SWC)" if morph_flag else "no morph"

            print(
                f"  Cell {cell_id:>2} | {animal_id} | {cell_type:>8} | "
                f"ephys: {ephys_status:>7} ({date_slice}/{ephys_id}) | "
                f"morph: {morph_status}"
            )

        print(f"\nOutput would go to: {analysis_dir}")
        return

    # Process cells grouped by animal (each animal has a different date_slice)
    all_metrics: list[dict[str, Any]] = []
    results: dict[str, str] = {}

    for animal_id, group in metadata.groupby("animal_id"):
        date_slice = date_map.get(str(animal_id))
        if date_slice is None:
            print(f"\nWARNING: No date_slice for animal {animal_id} — skipping")
            continue

        config = build_animal_config(raw_config, date_slice)
        ephys_dir = config.ephys_dir

        print(f"\n{'='*60}")
        print(f"Animal: {animal_id} | date_slice: {date_slice} | ephys_dir: {ephys_dir}")
        print(f"  Cells in group: {len(group)}")

        if not ephys_dir.is_dir():
            print(f"  WARNING: ephys_dir does not exist: {ephys_dir}")

        for idx, row in group.iterrows():
            cell_id = row.get("cell_index", idx)
            ephys_id = row.get("ephys_id", "")

            try:
                result = process_cell(row, config)
                if result is not None:
                    all_metrics.append(result)
                    results[f"{animal_id}_cell{cell_id}"] = "ok"
                    print(f"  Cell {cell_id} ({ephys_id}): OK")
                else:
                    results[f"{animal_id}_cell{cell_id}"] = "skip_no_data"
                    print(f"  Cell {cell_id} ({ephys_id}): skipped (no data)")
            except Exception as e:
                results[f"{animal_id}_cell{cell_id}"] = f"error: {e}"
                print(f"  Cell {cell_id} ({ephys_id}): ERROR — {e}")
                logger.exception("Failed to process cell %s", cell_id)

    # Build and save metrics table
    print(f"\n{'='*60}")
    print(f"Processing complete.")
    ok_count = sum(1 for v in results.values() if v == "ok")
    skip_count = sum(1 for v in results.values() if v.startswith("skip"))
    err_count = sum(1 for v in results.values() if v.startswith("error"))
    print(f"  OK: {ok_count}, Skipped: {skip_count}, Errors: {err_count}")

    if not all_metrics:
        print("No cells produced metrics — nothing to save.")
        return

    df = build_metrics_table(all_metrics)
    df = compute_derived_metrics(df)

    # Save output
    analysis_dir = Path(raw_config["analysis_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved metrics table to {output_path} ({len(df)} rows)")

    # Optionally run statistics and PCA
    if args.stats:
        print(f"\nRunning statistics...")
        stats_config = build_animal_config(raw_config, first_date)
        # Override analysis_dir to use the shared one
        stats_config = PatchConfig(
            metadata_dir=stats_config.metadata_dir,
            morph_dir=stats_config.morph_dir,
            ephys_dir=stats_config.ephys_dir,
            processed_dir=stats_config.processed_dir,
            analysis_dir=analysis_dir,
        )
        try:
            run_statistics(df, stats_config)
            print("  Statistics saved.")
        except Exception as e:
            print(f"  Statistics failed: {e}")
            logger.exception("Statistics failed")

        print("Running PCA...")
        try:
            run_pca_analysis(df, stats_config)
            print("  PCA saved.")
        except Exception as e:
            print(f"  PCA failed: {e}")
            logger.exception("PCA failed")

    # Print summary of errors
    if err_count > 0:
        print(f"\nFailed cells:")
        for key, status in results.items():
            if status.startswith("error"):
                print(f"  {key}: {status}")

    # Cleanup temp metadata dir
    shutil.rmtree(tmp_meta_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
