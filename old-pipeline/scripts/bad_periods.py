import numpy as np

from paths import config
from utils import metadata as mdutils, img as imgutils, video as vutils
from paths import config
from classes import Experiment
from paths.config import M2PConfig
from classes.ProcPath import ProcPath
from pathlib import Path
import os
import ffmpeg


# Supply an experiment id or index list, or set both to none for all.
exp_ids = ["20210823_16_59_50_1114353"]

do_2p = False
do_behav = True

# https://trac.ffmpeg.org/wiki/Encode/H.264
h264_tune = 'film'
# Controls compression/speed trade off.
h264_preset = 'fast'
# H264 quality, apparently 17 is indistinguishable from lossless.
h264_crf = '17'

cfg = config.M2PConfig()
out_2p_path = Path("/Users/tristan/Neuro/hm2p/bad-2p")
out_parts_behav_path = Path("/Users/tristan/Neuro/hm2p/bad-behav-parts")
out_behav_path = Path("/Users/tristan/Neuro/hm2p/bad-behav")

out_2p_path.mkdir(exist_ok=True)
out_parts_behav_path.mkdir(exist_ok=True)
out_behav_path.mkdir(exist_ok=True)

exps_df = mdutils.get_exps(cfg, exp_ids)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]
    exp_index = exp_row["exp_index"]
    exp = None
    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)
    if do_2p:
        bad_2p_frames = mdutils.get_bad_2p(exp_row)
        print(bad_2p_frames)

        if bad_2p_frames:

            exp = Experiment.Experiment(m2p_paths.raw_data_path)
            img_file = exp.SciscanSettings.image_file
            img_data = imgutils.read_tif_vol(img_file)
            good_frames = np.arange(0, imgutils.count_tif_frames(exp.SciscanSettings.image_file))
            good_frames_bool = np.full(good_frames.shape, True)
            for i, badframes in enumerate(bad_2p_frames):
                bad_start_frame = badframes[0]
                bad_end_frame = badframes[1]
                bad_img_data = img_data[bad_start_frame:bad_end_frame, :, :]
                bad_indexes = np.logical_and(good_frames >= bad_start_frame, good_frames <= bad_end_frame)
                good_frames_bool[bad_indexes] = False

                bad_tif_path = os.path.join(out_2p_path,
                                            "{:02d}_bad_{:05d}-{:05d}.tif".format(exp_index,
                                                                                  bad_start_frame,
                                                                                  bad_end_frame))
                imgutils.write_tif_by_frames(bad_img_data, bad_tif_path)

            good_img_data = img_data[good_frames_bool, :, :]

            good_tif_path = os.path.join(out_2p_path,
                                        "{:02d}_good.tif".format(exp_index))
            imgutils.write_tif_by_frames(good_img_data, good_tif_path)

    if do_behav:

        if exp is None:
            exp = Experiment.Experiment(m2p_paths.raw_data_path)

        mov_file = exp.tracking_video.cameras["overhead"].file

        file_name = os.path.split(mov_file)[1]
        (file_name_base, file_ext) = os.path.splitext(file_name)

        proc_mov_path = os.path.join(cfg.video_path, exp_id)
        crop_file = os.path.join(proc_mov_path, file_name_base + "-cropped" + file_ext)

        if os.path.exists(crop_file):
            mov_file = crop_file

        n_frames = vutils.count_frames(mov_file)

        bad_behav_frames, bad_behav_times = mdutils.get_bad_behav(exp_row, exp.tracking_video.fps, n_frames)
        print(bad_behav_frames)

        if bad_behav_frames:

            start_good_frame = 0
            start_good_time = "0:00"
            bad_paths = []
            good_paths = []
            for i, badframes in enumerate(bad_behav_frames):
                start_bad_frame = badframes[0]
                end_bad_frame = badframes[1]

                start_bad_time = bad_behav_times[i][0]
                end_bad_time = bad_behav_times[i][1]



                bad_behav_path = os.path.join(out_parts_behav_path,
                                            "{:02d}_bad_{}-{}.mp4".format(exp_index,
                                                                          mdutils.mmss_lead_zeros(start_bad_time).replace(":", "_"),
                                                                          mdutils.mmss_lead_zeros(end_bad_time).replace(":", "_")))
                good_behav_path = os.path.join(out_parts_behav_path,
                                              "{:02d}_good_{}-{}.mp4".format(exp_index,
                                                                             mdutils.mmss_lead_zeros(start_good_time).replace(":", "_"),
                                                                             mdutils.mmss_lead_zeros(start_bad_time).replace(":", "_")))

                if end_bad_frame > n_frames:
                    end_bad_frame = n_frames

                # Plays fast for some reason
                print("Trim movie")
                (
                    ffmpeg
                    .input(mov_file)
                    .trim(start_frame=start_bad_frame, end_frame=end_bad_frame)
                    .setpts('PTS-STARTPTS') # Doesn't work without this
                    .filter('fps', '30')
                    .setpts('0.25*PTS') # 4x
                    #.filter('pp', 'al') # Don't remember what this does
                    .output(bad_behav_path)#, vcodec='h264', format='mp4',
                            #preset=h264_preset, tune=h264_tune, crf=h264_crf)
                    .overwrite_output()
                    .run()
                )

                (
                    ffmpeg
                    .input(mov_file)
                    .trim(start_frame=start_good_frame, end_frame=start_bad_frame)
                    .setpts('PTS-STARTPTS')  # Doesn't work without this
                    .filter('fps', '30')
                    .setpts('0.25*PTS')  # 4x
                    # .filter('pp', 'al') # Don't remember what this does
                    .output(good_behav_path)#, vcodec='h264', format='mp4',
                            #preset=h264_preset, tune=h264_tune, crf=h264_crf)
                    .overwrite_output()
                    .run()
                )

                start_good_frame = end_bad_frame
                start_good_time = end_bad_time
                bad_paths.append(bad_behav_path)
                good_paths.append(good_behav_path)

            # Merge all the bad behaviour into 1 file
            bad_behav_path = os.path.join(out_behav_path,
                                           "{:02d}_bad.mp4".format(exp_index))
            bad_text_path = os.path.join(out_behav_path, "concat.txt")
            if os.path.exists(bad_text_path):
                os.remove(bad_text_path)
            with open(bad_text_path, 'w') as f:
                for bad_path in bad_paths:
                    f.write(f"file {bad_path}\n")

            (
                ffmpeg
                .input(bad_text_path, format='concat', safe=0)
                .output(bad_behav_path, c='copy')
                .overwrite_output()
                .run()
            )

            os.remove(bad_text_path)

            # Merge all the good behaviour into 1 file
            good_behav_path = os.path.join(out_behav_path,
                                           "{:02d}_good.mp4".format(exp_index))
            good_text_path = os.path.join(out_behav_path, "concat.txt")
            if os.path.exists(good_text_path):
                os.remove(good_text_path)
            with open(good_text_path, 'w') as f:
                for good_path in good_paths:
                    f.write(f"file {good_path}\n")

            (
                ffmpeg
                .input(good_text_path, format='concat', safe=0)
                .output(good_behav_path, c='copy')
                .overwrite_output()
                .run()
            )

            os.remove(good_text_path)







