# Calculate behavioural metrics from DLC data
from utils import misc as m2putils, db
from classes import Experiment
import os
import pandas as pd
from utils import behave as bu
from paths.config import M2PConfig
from classes.ProcPath import ProcPath

def proc_single(cfg:M2PConfig, exp_id, rebuild_behav_data=False):

    # See https://github.com/DeepLabCut/DLCutils/blob/master/Demo_loadandanalyzeDLCdata.ipynb
    # and https://github.com/adamltyson/movement/blob/master/movement/io/dlc.py
    # Also https://github.com/adamltyson/spikey/blob/master/spikey/tuning/radial.py seems to be the analysis

    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    print("Loading experiment")
    exp = Experiment.Experiment(m2p_paths.raw_data_path)
    print("Loading experiment done.")

    print("Checking camera timings and frames")
    exp.check_cameras(ignore_cams="side_left")
    print("Experiment checks complete")

    raw_name = os.path.split(m2p_paths.raw_data_path)[1]
    video_path = os.path.join(cfg.video_path, raw_name)

    plot_sum_dir = os.path.join(video_path, "sum")
    m2putils.setup_dir(plot_sum_dir)

    # Get the DLC data
    cam_name = "overhead"
    cam = exp.tracking_video.cameras[cam_name]
    mov_fullfps_name = cam.file_name_base + "-cropped.mp4"
    mov_fullfps_name_base = os.path.splitext(mov_fullfps_name)[0]
    dlc_file = os.path.join(cfg.dlc_tracked_path, mov_fullfps_name_base + cfg.dlc_iter_name + ".h5")

    #mov_fullfps_path = os.path.join(video_path, mov_fullfps_name)
    # dlc_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".raw.h5")
    # behav_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".raw.metrics.h5")

    #behav_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".filtered.metrics.h5")

    meta_data_path = os.path.join(video_path, "meta")
    backup_meta_path = os.path.join(cfg.video_meta_bak_path, m2putils.path_leaf(exp.directory))

    print("Loading raw data")
    df = pd.read_hdf(os.path.join(dlc_file))

    # Load a previous save file instead if requested
    if not rebuild_behav_data and os.path.exists(m2p_paths.behave_file):
        df_behav = pd.read_hdf(m2p_paths.behave_file)
    else:
        print("Calculating HD, positions and velocity")
        df_behav = bu.calc_behav(df,
                                 exp_id,
                                 cfg,
                                 meta_data_path=meta_data_path,
                                 backup_meta_path=backup_meta_path,
                                 fps=exp.tracking_video.fps,
                                 frame_times=exp.cam_trigger_times,
                                 light_on_times=exp.Lighting.on_pulse_times,
                                 light_off_times=exp.Lighting.off_pulse_times)

        df_behav.to_hdf(m2p_paths.behave_file, key="df_move")

    db.add_exp_data(df_behav, cfg.db_behave_file, exp_id)

    print("Done")
