import shutil

import pandas as pd
import os
from classes import Experiment
from utils import misc as m2putils, behave as bu
import numpy as np
from paths.config import M2PConfig
from classes.ProcPath import ProcPath


def sum_all(m2pcfg):
    # todo load all config
    # Iterate and call proc_s2p_single
    sum_single(m2pcfg, exp_id)

def sum_exp(cfg: M2PConfig, exp_id, create_bigspeed_videos=False):

    raw_data_path = m2putils.get_exp_path(cfg.raw_path, exp_id)

    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    print("Loading experiment")
    exp = Experiment.Experiment(m2p_paths.raw_data_path)
    print("Loading experiment done.")

    print("Checking camera timings and frames")
    exp.check_cameras(ignore_cams="side_left")
    print("Experiment checks complete")



    raw_name = os.path.split(m2p_paths.raw_data_path)[1]
    video_path = cfg.video_path / raw_name
    tracking_data_path = video_path / "tracking"
    plot_sum_dir = cfg.sum_behave_path / raw_name
    m2putils.setup_dir(plot_sum_dir)
    plot_sum_all_dir = cfg.sum_behave_path / "all"
    m2putils.setup_dir(plot_sum_all_dir)

    cam_name = "overhead"
    cam = exp.tracking_video.cameras[cam_name]

    mov_fullfps_name = cam.file_name_base + "-cropped.mp4"
    mov_fullfps_name_base = os.path.splitext(mov_fullfps_name)[0]
    mov_fullfps_path = video_path / mov_fullfps_name



    print("Loading data")
    df = pd.read_hdf(m2p_paths.behave_file)
    print("Done")

    print("Plotting summaries")

    bu.plot_summary(df=df,
                    fps=exp.tracking_video.fps,
                    plot_dir=plot_sum_dir)

    shutil.copy(os.path.join(plot_sum_dir, 'ear-distance.png'),
                os.path.join(plot_sum_all_dir, cam.file_name_base + 'ear-distance.png'))

    shutil.copy(os.path.join(plot_sum_dir, 'ear-distance-hist.png'),
                os.path.join(plot_sum_all_dir, cam.file_name_base + 'ear-distance-hist.png'))

    if create_bigspeed_videos:

        print("Creating big ear dist video")
        ear_positions_mm = np.empty((4, len(df)))
        ear_positions_mm[0, :] = df[bu.EAR_LEFT_MM_X]
        ear_positions_mm[1, :] = df[bu.EAR_LEFT_MM_Y]
        ear_positions_mm[2, :] = df[bu.EAR_RIGHT_MM_X]
        ear_positions_mm[3, :] = df[bu.EAR_RIGHT_MM_Y]
        ear_dist = bu.calc_ear_dist(ear_positions_mm)



        big_ear_dist = m2putils.get_crossings(ear_dist, 60)
        print(big_ear_dist)
        if big_ear_dist.size > 0:
            for i in range(big_ear_dist.size - 1):

                movie_arrow_path = os.path.join(plot_sum_dir, cam.file_name_base + "-bigeardist.{}.mp4".format(i))
                print(movie_arrow_path)
                start = big_ear_dist[i] - 100
                end = big_ear_dist[i] + 100
                if start < 0:
                    start = 0
                if end >= df[bu.EAR_LEFT_MM_X].values.size:
                    end = df[bu.EAR_LEFT_MM_X].values.size
                print(start, end)
                bu.create_arrow_mov(df,
                                    mov_fullfps_path,
                                    movie_arrow_path,
                                    start_frame=start,
                                    end_frame=end,
                                    speed_arrow=False,
                                    hd_arrow=False,
                                    ahv_arrow=False)


        print("Creating big speed video")
        big_speed = m2putils.get_crossings(np.abs(df[bu.SPEED_FILT_GRAD].values), 100)
        print(big_speed)
        if big_speed.size > 0:
            for i in range(big_speed.size - 1):
                movie_arrow_path = os.path.join(plot_sum_dir, cam.file_name_base + "-arrow-bigspeed.{}.mp4".format(i))
                start = big_speed[i] - 100
                end = big_speed[i] + 100
                if start < 0:
                    start = 0
                if end >= df[bu.AHV_FILT_GRAD].values.size:
                    end = df[bu.AHV_FILT_GRAD].values.size
                bu.create_arrow_mov(df,
                                    mov_fullfps_path,
                                    movie_arrow_path,
                                    start_frame=start,
                                    end_frame=end,
                                    ahv_arrow=False,
                                    hd_arrow=False)
                raise Exception("Stop")

        print("Creating big AHV video")
        big_ahv = m2putils.get_crossings(np.abs(df[bu.AHV_FILT_GRAD].values), 500)
        print(big_ahv)
        if big_ahv.size > 0:
            for i in range(big_ahv.size - 1):
                movie_arrow_path = os.path.join(plot_sum_dir, cam.file_name_base + "-arrow-bigahv.{}.mp4".format(i))
                start = big_ahv[i] - 100
                end = big_ahv[i] + 100
                if start < 0:
                    start = 0
                if end >= df[bu.AHV_FILT_GRAD].values.size:
                    end = df[bu.AHV_FILT_GRAD].values.size
                bu.create_arrow_mov(df,
                                    mov_fullfps_path,
                                    movie_arrow_path,
                                    start_frame=start,
                                    end_frame=end)



        pass

    print("Done")