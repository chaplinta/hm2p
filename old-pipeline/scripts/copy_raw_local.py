import shutil

from paths import config
from utils import metadata as mdutils, misc as mutils
from paths import config
from classes import Experiment
from paths.config import M2PConfig
from pathlib import Path
import os

# Copies all data from remote server to somewhere local

cfg = config.M2PConfig()
raw_remote_path: Path = Path("/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze")
raw_local_path: Path = Path("/Users/tristan/Neuro/hm2p/raw")
exps_df = mdutils.get_exps(cfg)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]

    print(exp_id)

    remote_raw = mutils.get_exp_path(raw_remote_path, exp_id)
    local_raw = mutils.get_exp_path(raw_local_path, exp_id)

    if os.path.exists(local_raw):
        #raise Exception("Local raw path {} exists already!".format(local_raw))
        print("Local raw path {} exists already, skipping".format(local_raw))
        continue


    side_cam_ignore = "*side_left.camera.mp4"
    # Copy everything except the side camera (useless) and tifs if it's a raw recording
    if mutils.get_filetype(remote_raw, "*.raw", allow_missing=True):
        # Don't get tifs, the data is raw format, the tifs were created some other way and not needed.
        ignore_patter = shutil.ignore_patterns(side_cam_ignore, "*.tif")
    else:
        # The raw data was saved in tifs, so don't ignore it.
        print("Raw data is in tif format for {}".format(local_raw))
        ignore_patter = shutil.ignore_patterns(side_cam_ignore)

    shutil.copytree(remote_raw, local_raw, ignore=ignore_patter)