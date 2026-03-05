import os
import cv2
import numpy as np
import imageio
from classes import Experiment
from utils import misc, metadata as mdutils, video as vidutils
from paths.config import M2PConfig
from classes.ProcPath import ProcPath
from math import isnan

# Currently takes a quite a while for large movies, it writes at about 60fps.
# See https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html

cfg = M2PConfig()

exp_ids = None #["20220804_11_21_59_1117646"]

exps_df = mdutils.get_exps(cfg, exp_ids)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]
    lens = exp_row["lens"]

    print(exp_id)

    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    exp = Experiment.Experiment(m2p_paths.raw_data_path)

    proc_data_path = os.path.join(cfg.video_path, misc.path_leaf(exp.directory))
    misc.setup_dir(proc_data_path, clearout=False)

    cam_name = "overhead"
    input_file = exp.tracking_video.cameras[cam_name].file

    file_name = os.path.split(input_file)[1]
    (file_name_base, file_ext) = os.path.splitext(file_name)

    output_file = os.path.join(proc_data_path, file_name_base + "-undistort" + file_ext)

    if os.path.exists(output_file):
        print("Undistorted file already exists, counting frames ...")
        in_frames = vidutils.count_frames(input_file)
        out_frames = vidutils.count_frames(output_file)
        print(in_frames, out_frames)
        if in_frames == out_frames:
            # raise Exception("Undistorted file already exists!")
            print("Undistorted file already exists and has correct number of frames, skipping ...")
            continue
        else:
            print("Undistorted file already exists but has the wrong number of frames, deleting and remaking...")
            os.remove(output_file)



    # Set max frames to none to do whole movie
    max_frames = None #1000

    # alpha = 0 means fewest extra pixels, but it may lose some.
    # alpha = 1 means it may have extra pixels, but it won't lose any.
    alpha = 1

    # Load calibration file.
    print('Loading calibration file ...')
    print(lens)
    # skip if lens is blank
    if not isinstance(lens, str):
        raise Exception("Lens is not set in metadata for experiment {}".format(exp_id))
    else:


        with np.load(cfg.cam_calibration_files[lens]) as npzfile:

            print(cfg.cam_calibration_files[lens])

            mtx = npzfile['mtx']
            dist = npzfile['dist']
            rvecs = npzfile['rvecs']
            tvecs = npzfile['tvecs']

            print('Loading movie and undistorting')
            print('Loading movie file ...')
            with imageio.get_reader(input_file) as mov_reader:
                meta_data = mov_reader.get_meta_data()
                fps = meta_data['fps']
                (width, height) = meta_data['source_size']
                n_frames = mov_reader.get_length()

                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), alpha, (width, height))

                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), cv2.CV_32FC1)

                print('Create output movie file ...')

                # https://trac.ffmpeg.org/wiki/Encode/H.264
                # Controls compression/speed trade off.
                h264_preset = 'fast'
                # Tuning for type of movie, film is should be good for this image?
                h264_tune = 'film'
                # H264 quality, apparently 17 is indistinguishable from lossless.
                h264_crf = '17'

                with imageio.get_writer(output_file,
                                        mode="I",
                                        fps=fps,
                                        pixelformat='gray',
                                        codec='h264',
                                        format='mp4',
                                        output_params=['-s', '{0}x{1}'.format(width, height),
                                                               '-preset', h264_preset,
                                                               '-tune', h264_tune,
                                                               '-crf', str(h264_crf)]) as mov_writer:



                    # For each image, apply calibration.
                    print('Reading and undistorting each frame...')

                    for i, img in enumerate(mov_reader):

                        if i % 100 == 0:
                            # For some reason n_frames can be inf?
                            #print("Frame #{:d}/{:d}".format(i, n_frames))
                            print("Frame #{:d}".format(i))

                        if max_frames and i > max_frames:
                            break

                        #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

                        mov_writer.append_data(dst)


        print("Undistort complete.")

print("All undistort complete.")