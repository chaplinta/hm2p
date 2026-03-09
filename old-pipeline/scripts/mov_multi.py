import os.path
from paths import config
from utils import misc as hm2putils, metadata as mdutils
import ffmpeg
from pathlib import Path
from classes import Experiment
from classes.ProcPath import ProcPath

# Script to take a list of experiment ids and create movies in a separate directory for inspection.

# Specific ids
# exp_ids = ["20210823_16_59_50_1114353"]
exp_ids = None  # Just get all from meta

out_path = Path("/Users/tristan/Neuro/hm2p/inspect-mov")

# https://trac.ffmpeg.org/wiki/Encode/H.264
# Controls compression/speed trade off.
h264_preset = 'fast'
# H264 quality, apparently 17 is indistinguishable from lossless.
h264_crf = '17'
h264_tune = 'film'

cfg = config.M2PConfig()

out_path.mkdir(exist_ok=True)

exps_df = mdutils.get_exps(cfg, exp_ids)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]
    exp_index = exp_row["exp_index"]

    print(exp_id)

    exp_path = hm2putils.get_exp_path(cfg.raw_path, exp_id)

    # Get the raw file
    # mov_file = hm2putils.get_filetype(exp_path, "*overhead.camera.mp4")
    # mov_dir, mov_file_name = os.path.split(mov_file)
    # output_file = os.path.join(out_path, mov_file_name)

    # Get the cropped file
    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)
    exp = Experiment.Experiment(m2p_paths.raw_data_path)
    mov_file = exp.tracking_video.cameras["overhead"].file
    file_name = os.path.split(mov_file)[1]
    (file_name_base, file_ext) = os.path.splitext(file_name)
    proc_mov_path = os.path.join(cfg.video_path, exp_id)
    mov_file = os.path.join(proc_mov_path, file_name_base + "-cropped" + file_ext)
    output_file = os.path.join(out_path, file_name)

    print(output_file)

    if not os.path.exists(output_file):

        (
            ffmpeg
            .input(mov_file)
            .filter('fps', '30')
            .setpts('0.25*PTS')  # 4x
            .setpts('PTS-STARTPTS')  # Doesn't work without this
            #.filter('scale', 'in_w*0.5', 'in_h*0.5')
            .output(output_file) #, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune, crf=h264_crf)
            #.overwrite_output()
            .run()
        )
    else:
        print("File exists, skipping")
        continue


print("Done")

