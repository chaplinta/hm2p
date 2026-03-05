from classes.Experiment import Experiment
from classes.ProcPath import ProcPath
from utils import metadata as mdutils
from paths import config
import os

# Checks if the pulses recorded in the experiment make sense.
# Saves lists of bad experiments in M2PConfig.exp_check_path and plots of putative bad frames in ProcPath.bad_2p_plots
expids = None

cfg = config.M2PConfig()

if expids is None:
    save_bad_exps = True
    expids = mdutils.get_exp_ids(cfg)
else:
    # If doing a subset don't overwrite the files.
    # Really do an append, but check if already exists in file etc todo
    save_bad_exps = False

bad_cam_sync_exps = []
bad_light_sync_exps = []
bad_2p_sync_exps = []
bad_2p_img_exps = []
for exp_id in expids:

    print(exp_id)

    m2p_paths = ProcPath(cfg, exp_id)

    print("Loading experiment data")
    # Remake tif incase anything is wrong with it
    exp = Experiment(m2p_paths.raw_data_path, remake_tif=True)

    print("Checking camera timings and frames")
    try:
        exp.check_cameras(ignore_cams=["side_left"])
    except Exception as e:
        print(str(e))
        bad_cam_sync_exps.append(exp_id)

    print("Checking sciscan timings and frames")
    try:
        exp.check_sci_frames()
    except Exception as e:
        print(str(e))
        bad_2p_sync_exps.append(exp_id)

    print("Check the light times")
    try:
        exp.check_light_times()
    except Exception as e:
        print(str(e))
        bad_light_sync_exps.append(exp_id)

    print("Checking for bad 2p frames")
    reg_tif = cfg.s2p_path / exp_id / "images" / "reg.tif"
    if not os.path.exists(reg_tif):
        #reg_tif = None
        raise Exception("No reg tif?")
    bad_frames = exp.get_bad_frames(m2p_paths.bad_2p_plots, reg_tif=reg_tif)
    if bad_frames and bad_frames.size >= 0:
        bad_2p_img_exps.append(bad_frames)

    print("Done")

print("Finished checking all experiments synchronization")

if bad_cam_sync_exps:
    print("Found experiments with bad camera timing!")
    print(bad_cam_sync_exps)
    if save_bad_exps:
        with open(cfg.exp_check_camsyncs_file, 'w') as f:
            for exp in bad_cam_sync_exps:
                f.write(f"{exp}\n")

if bad_light_sync_exps:
    print("Found experiments with bad camera timing!")
    print(bad_light_sync_exps)
    if save_bad_exps:
        with open(cfg.exp_check_lightsyncs_file, 'w') as f:
            for exp in bad_light_sync_exps:
                f.write(f"{exp}\n")

if bad_2p_sync_exps:
    print("Found experiments with bad 2p timing!")
    print(bad_2p_sync_exps)
    if save_bad_exps:
        with open(cfg.exp_check_2psyncs_file, 'w') as f:
            for exp in bad_2p_sync_exps:
                f.write(f"{exp}\n")

if bad_2p_img_exps:
    print("Found experiments with bad 2p images, probably ok just check them")
    print(bad_2p_img_exps)
    if save_bad_exps:
        with open(cfg.exp_check_2pimg_file, 'w') as f:
            for exp in bad_2p_img_exps:
                f.write(f"{exp}\n")

