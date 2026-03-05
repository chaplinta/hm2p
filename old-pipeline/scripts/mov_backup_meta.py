import os
from classes import Experiment
from utils import misc, behave as bu, metadata as mdutils, video as vidutils
from paths.config import M2PConfig
from classes.ProcPath import ProcPath

# Script to run through all video and load the metadata, thereby ensuring it's in the backup directory.
cfg = M2PConfig()

exp_ids = None

if exp_ids is None:
    exp_ids = mdutils.get_exp_ids(cfg)

for exp_id in exp_ids:

    print(exp_id)

    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    proc_data_path = os.path.join(cfg.video_path, exp_id)
    meta_data_path = os.path.join(proc_data_path, "meta")

    backup_meta_path = os.path.join(cfg.video_meta_bak_path, exp_id, "meta")
    misc.setup_dir(proc_data_path, clearout=False)

    print(meta_data_path)

    vid_orientation = mdutils.get_vid_orientation(cfg, exp_id)
    meta_data = bu.get_mov_meta(meta_data_path, backup_meta_path, vid_orientation)

print("Finished backing up movie meta data")