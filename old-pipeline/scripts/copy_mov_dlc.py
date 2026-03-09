from utils import metadata as mdutils
from paths import config
from classes.ProcPath import ProcPath
from classes.Experiment import Experiment
import os
import shutil

cfg = config.M2PConfig()
exps_df = mdutils.get_exps(cfg)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]
    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    exp = Experiment(m2p_paths.raw_data_path)

    mov_file = exp.tracking_video.cameras["overhead"].file

    file_name = os.path.split(mov_file)[1]
    (file_name_base, file_ext) = os.path.splitext(file_name)

    proc_mov_path = os.path.join(cfg.video_path, exp_id)
    crop_file = cropped_file = os.path.join(proc_mov_path, file_name_base + "-cropped" + file_ext)

    if not os.path.exists(crop_file):
        raise Exception("No crop file for {}".format(exp_id))

    shutil.copy2(crop_file, cfg.dlc_input_video_path)
