import os
import shutil
from classes import Experiment
from utils import misc, behave as bu, metadata as mdutils, video as vidutils
import ffmpeg
from paths.config import M2PConfig
from classes.ProcPath import ProcPath
from math import radians

cfg = M2PConfig()

exp_ids = None #["20211028_11_25_50_1115465", "20211029_13_50_08_1115465", "20220608_15_27_32_1117217"]

overwrite = False

if exp_ids is None:
    exp_ids = mdutils.get_exp_ids(cfg)

for exp_id in exp_ids:

    print(exp_id)

    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    exp = Experiment.Experiment(m2p_paths.raw_data_path)

    proc_data_path = os.path.join(cfg.video_path, misc.path_leaf(exp.directory))
    meta_data_path = os.path.join(proc_data_path, "meta")

    backup_meta_path = os.path.join(cfg.video_meta_bak_path, misc.path_leaf(exp.directory), "meta")
    misc.setup_dir(proc_data_path, clearout=False)

    print(meta_data_path)

    vid_orientation = mdutils.get_vid_orientation(cfg, exp_id)
    meta_data = bu.get_mov_meta(meta_data_path, backup_meta_path, vid_orientation)

    x = meta_data.crop_x
    y = meta_data.crop_y
    width = meta_data.crop_width
    height = meta_data.crop_height

    cam_name = "overhead"

    exp_meta = mdutils.get_exps(cfg, [exp_id])
    orientation = exp_meta['orientation'].values[0]

    file_name = os.path.split(exp.tracking_video.cameras[cam_name].file)[1]
    (file_name_base, file_ext) = os.path.splitext(file_name)

    input_file = os.path.join(proc_data_path, file_name_base + "-undistort" + file_ext)
    cropped_file = os.path.join(proc_data_path, file_name_base + "-cropped" + file_ext)

    # https://trac.ffmpeg.org/wiki/Encode/H.264
    # Controls compression/speed trade off.
    h264_preset = 'fast'
    # Tuning for type of movie, film is should be good for this image?
    h264_tune = 'film'
    # H264 quality, apparently 17 is indistinguishable from lossless.
    h264_crf = '17'

    crop_csv_path = os.path.join(meta_data_path, "crop.csv")
    config_file_name = "meta.txt"
    config_file_path = os.path.join(proc_data_path, config_file_name)

    if os.path.exists(cropped_file) and not overwrite:
        print("Cropped file exists, counting frames")
        in_frames = vidutils.count_frames(input_file)
        out_frames = vidutils.count_frames(cropped_file)
        print(in_frames, out_frames)
        if in_frames == out_frames:
            print("Croppped file already exists and has correct number of frames, skipping ...")
            continue
        else:
            print("Cropped file already exists but has the wrong number of frames, deleting and remaking...")
            os.remove(cropped_file)

    print("Cropping")
    (
        ffmpeg
        .input(input_file)
        #.trim(start_frame=0, end_frame=10000)
        .crop(x, y, width, height)
        .output(cropped_file) #, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
        .overwrite_output()
        .run()
    )

    rotated_file = os.path.join(proc_data_path, file_name_base + "-rotated" + file_ext)

    rotation = 0
    if rotation != 0:

        print("Rotating fine adjustment")
        (
            ffmpeg
            .input(cropped_file)
            .filter('rotate', radians(-rotation))
            .output(rotated_file)
            .overwrite_output()
            .run()
        )

        shutil.copyfile(rotated_file, cropped_file)
        os.remove(rotated_file)

    if orientation != 0:

        if orientation == 90:
            transpose = 2

            print("Transposing CCW")
            (
                ffmpeg
                .input(cropped_file)
                .filter('transpose', transpose)
                .output(rotated_file)
                .overwrite_output()
                .run()
            )

        elif orientation == 180:
            print("Flipping")
            (
                ffmpeg
                .input(cropped_file)
                .vflip()
                .output(rotated_file)
                .overwrite_output()
                .run()
            )


        else:
            raise Exception("Unrecognized orientation {}".format(orientation))

        shutil.copyfile(rotated_file, cropped_file)
        os.remove(rotated_file)

    # print("Downsample res by half")
    # half_res_file = os.path.join(proc_data_path, file_name_base + "-halfres" + file_ext)
    # (
    #     ffmpeg
    #     .input(cropped_file)
    #     .filter('scale', 'in_w*0.5', 'in_h*0.5')
    #     .output(half_res_file) #, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
    #     .overwrite_output()
    #     .run()
    # )
    #
    # print("Downsample res by quarter")
    # quarter_res_file = os.path.join(proc_data_path, file_name_base + "-quartres" + file_ext)
    # (
    #     ffmpeg
    #     .input(cropped_file)
    #     .filter('scale', 'in_w*0.25', 'in_h*0.25')
    #     .output(quarter_res_file) # vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
    #     .overwrite_output()
    #     .run()
    # )

    # print("Downsample fps to something more normal (30)")
    # low_fps_file = os.path.join(proc_data_path, file_name_base + "-lowfps" + file_ext)
    # (
    #     ffmpeg
    #     .input(cropped_file)
    #     .filter('fps', '30')
    #     .output(low_fps_file) # vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
    #     .overwrite_output()
    #     .run()
    # )

    # print("Downsample fps to 10")
    # sci_fps_file = os.path.join(proc_data_path, file_name_base + "-10fps" + file_ext)
    # (
    #     ffmpeg
    #     .input(cropped_file)
    #     .filter('fps', 10)
    #     .output(sci_fps_file) # vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
    #     .overwrite_output()
    #     .run()
    # )

    # print("Downsample res & fps to something more normal (30)")
    # low_fps_file = os.path.join(proc_data_path, file_name_base + "-halfres_lowfps" + file_ext)
    # (
    #     ffmpeg
    #     .input(half_res_file)
    #     .filter('fps', '30')
    #     .output(low_fps_file) # vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
    #     .overwrite_output()
    #     .run()
    # )
    #
    # print("Downsample res & fps to about whatever sciscan is")
    # sci_fps_file = os.path.join(proc_data_path, file_name_base + "-halfres_scifps" + file_ext)
    # (
    #     ffmpeg
    #     .input(half_res_file)
    #     .filter('fps', exp.SciscanSettings.frames_p_sec)
    #     .output(sci_fps_file) # vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
    #     .overwrite_output()
    #     .run()
    # )

    print("Done.")

print("All cropping complete.")


