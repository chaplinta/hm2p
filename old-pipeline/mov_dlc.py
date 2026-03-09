import shutil
import deeplabcut
import os
from classes import Experiment
from utils import misc
from paths.config import M2PConfig
from classes.ProcPath import ProcPath

# exp_id = "20210823_16_59_50_1114353"
# exp_id = "20210920_11_09_37_1114356"
#exp_id = "20211203_15_10_27_1115464"
#exp_id = "20211216_14_36_39_1115816"  # 8f
#exp_id = "20220408_15_01_57_1116663"
exp_id = "20220802_15_06_53_1117646"
#exp_id = "20220601_13_53_18_1117217"

cfg = M2PConfig()
m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

exp = Experiment.Experiment(m2p_paths.raw_data_path)

cam_name = "overhead"
proc_data_path = os.path.join(cfg.video_path, misc.path_leaf(exp.directory))
movie_file = misc.path_leaf(exp.tracking_video.cameras[cam_name].file)
movie_file_name = os.path.splitext(movie_file)[0]
movie_path = os.path.join(proc_data_path, movie_file_name + "-cropped.mp4")


dlc_path = os.path.join(proc_data_path, "dlc")
tracking_path = os.path.join(proc_data_path, "tracking")
misc.setup_dir(dlc_path, clearout=False)
misc.setup_dir(tracking_path, clearout=False)

print("Analyzing video")
print(dlc_path)
deeplabcut.analyze_videos(config=cfg.dlc_config_path,
                          videos=[movie_path],
                          shuffle=1,
                          destfolder=dlc_path,
                          gputouse=0)


print("Filtering")
# # # todo maybe this?
# # Arima model seems to fail, try median
# # deeplabcut.filterpredictions(config=dlc_config_path,
# #                              video=[movie_path],
# #                              shuffle=1,
# #                              destfolder=dlc_path,
# #                              filtertype='arima',
# #                              ARdegree=5,
# #                              MAdegree=2)
# deeplabcut.filterpredictions(config=cfg.dlc_config_path,
#                              video=[movie_path],
#                              shuffle=1,
#                              destfolder=dlc_path,
#                              filtertype='median',
#                              windowlength=3)
#
#
#
#
# print("Plotting raw trajectories")
# deeplabcut.plot_trajectories(config=cfg.dlc_config_path,
#                              videos=[movie_path],
#                              shuffle=1,
#                              destfolder=dlc_path,
#                              filtered=False)
#
# print("Plotting filtered trajectories")
# deeplabcut.plot_trajectories(config=cfg.dlc_config_path,
#                              videos=[movie_path],
#                              shuffle=1,
#                              destfolder=dlc_path,
#                              filtered=True)
#
# # Copy the output files of DLC to a tracking directory and clean the name up.
# h5_out_file = misc.get_filetype(dlc_path, "*0.h5", allow_multiple=True, get_first=True)
# h5_out_file = misc.get_filetype(dlc_path, "*filtered.h5", allow_multiple=True, get_first=True)
# pickle_out_file = misc.get_filetype(dlc_path, "*.pickle", allow_multiple=True, get_first=True)
#
# h5_raw_file = os.path.join(tracking_path, movie_file_name + "-cropped.raw.h5")
# h5_filt_file = os.path.join(tracking_path, movie_file_name + "-cropped.filtered.h5")
# pickle_raw_file = os.path.join(tracking_path, movie_file_name + "-cropped.raw.pickle")
#
# shutil.copy2(h5_out_file, h5_raw_file)
# shutil.copy2(h5_out_file, h5_filt_file)
# shutil.copy2(pickle_out_file, pickle_raw_file)


# # # This bit failing still
# # print("Creating labelled video")
# deeplabcut.create_labeled_video(config=cfg.dlc_config_path,
#                                 videos=[movie_path],
#                                 shuffle=1,
#                                 #videotype="mp4",
#                                 destfolder=tracking_path)

print("Done.")