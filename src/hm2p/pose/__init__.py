"""Stage 2b — DLC inference / pose estimation (pluggable tracker dispatch).

Stage 2a (DLC Training) produces the fine-tuned model weights used here.
"""

from hm2p.pose.select import (
    ChampionMismatchError,
    extract_dlc_provenance,
    load_champion_manifest,
    select_best_dlc_h5,
    select_best_dlc_h5_s3,
    select_champion_h5,
    select_champion_h5_s3,
)

__all__ = [
    "ChampionMismatchError",
    "extract_dlc_provenance",
    "load_champion_manifest",
    "select_best_dlc_h5",
    "select_best_dlc_h5_s3",
    "select_champion_h5",
    "select_champion_h5_s3",
]
