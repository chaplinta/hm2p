import glob
import os
import imageio
import numpy as np
import scipy
import time
from tifffile import TiffFile

def count_tif_frames(tif_path):

    # imagio method is too slow
    # start_time = time.time()
    # with imageio.get_reader(tif_path) as imreader:
    #     n_frames = len(imreader)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(n_frames)

    # Alternative method, similar speed
    # start_time = time.time()
    # tif_data = imageio.mimread(tif_path, memtest=False)
    # n_frames = len(tif_data)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(n_frames)

    # Use tifffile, it just gets the metadata instead of reading the whole thing
    # start_time = time.time()
    tif = TiffFile(tif_path)
    n_frames = len(tif.pages)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(n_frames)

    return n_frames


def normalize_img(data, prc_clip):
    min_val = np.percentile(data, prc_clip)
    max_val = np.percentile(data, (100 - prc_clip))

    # print("Pixel range: min={min} max={max} diff={d}".format(min=min_val, max=max_val,d=(max_val-min_val)))

    data = (data - min_val) / (max_val - min_val)  # normalize the data to 0 - 1
    data[data < 0] = 0
    data[data > 1] = 1

    return data


def normalize_img_8bit(data, prc_clip):
    data = normalize_img(data, prc_clip)
    data = ((2**8) - 1) * data  # Now scale
    data = data.astype('uint8')
    return data


def normalize_img_16bit(data, prc_clip):
    data = normalize_img(data, prc_clip)
    data = ((2**16) - 1) * data  # Now scale
    data = data.astype('uint16')
    return data


def load_tif_dir(tif_dir, file_filter="*.tif"):
    tif_files = glob.glob(os.path.join(tif_dir, file_filter))
    tif_files.sort()
    n_files = len(tif_files)

    n_frames = 0
    for tf in tif_files:
        single_tif_data = imageio.mimread(tf, memtest=False)

        n_frames += len(single_tif_data)
        width = single_tif_data[0].shape[0]
        height = single_tif_data[0].shape[1]

    tif_dtype = single_tif_data[0].dtype
    tif_data = np.zeros((n_frames, width, height), dtype=tif_dtype)

    comb_i_frame = 0
    for tf in tif_files:
        single_tif_data = imageio.mimread(tf, memtest=False)
        for i_frame in range(len(single_tif_data)):
            tif_data[comb_i_frame, :, :] = single_tif_data[i_frame].astype(tif_dtype)
            comb_i_frame += 1

    return tif_data


# Volread is broken for some reason.
# tif_data = imageio.volread(tif_file_path)
# Hack version for now
def read_tif_vol(tif_file_path):
    tif_data = imageio.mimread(tif_file_path, memtest=False)

    n_frames = len(tif_data)
    width = tif_data[0].shape[0]
    height = tif_data[0].shape[1]
    has_chans = len(tif_data[0].shape) > 2


    tif_dtype = tif_data[0].dtype
    if has_chans:
        n_chans = tif_data[0].shape[2]
        tif_matrix = np.zeros((n_frames, width, height, n_chans), dtype=tif_dtype)
    else:
        tif_matrix = np.zeros((n_frames, width, height), dtype=tif_dtype)
    for i_frame in range(n_frames):
        if has_chans:
            tif_matrix[i_frame, :, :, :] = tif_data[i_frame].astype(tif_dtype)
        else:
            tif_matrix[i_frame, :, :] = tif_data[i_frame].astype(tif_dtype)

    return tif_matrix

def read_tif_array(tif_file_path):
    tif_data = imageio.mimread(tif_file_path, memtest=False)

    n_frames = len(tif_data)
    width = tif_data[0].shape[0]
    height = tif_data[0].shape[1]
    tif_dtype = tif_data[0].dtype
    tif_array = np.zeros((n_frames, width * height), dtype=tif_dtype)
    for i_frame in range(n_frames):
        tif_array[i_frame, :] = tif_data[i_frame].flatten().astype(tif_dtype)

    return tif_array

def write_tif_by_frames(tif_data, tif_path):
    # volwrite can fail sometimes
    with imageio.get_writer(tif_path, format='TIFF', mode='v') as tif_writer:
        for i_frame in range(tif_data.shape[0]):
            tif_writer.append_data(tif_data[i_frame, :, :])

def tif_to_movie(tif_file_path, fps=None, video_file=None):
    # Volread is broken for some reason.
    # tif_data = imageio.volread(tif_file_path)
    # Hack version for now
    tif_data = read_tif_vol(tif_file_path)

    upscale = 2
    order = 0  # 0 is nearest neighbour#

    print("Normalizing pixel intensities")
    tif_data = tif_data.astype(np.float64)

    prc_clip = 0.1
    tif_data = normalize_img(tif_data, prc_clip)

    tif_data = 255 * tif_data  # Now scale by 255

    print("Converting TIF to 8bit")
    tif_data = tif_data.astype('uint8')

    n_frames = tif_data.shape[0]
    width = tif_data.shape[1]
    height = tif_data.shape[2]

    print("Image size {w}x{h} {n} frames.".format(w=width, h=height, n=n_frames))

    print("Upscaling image")
    tif_data = scipy.ndimage.zoom(tif_data, (1, upscale, upscale), order=order)
    width = tif_data.shape[1]
    height = tif_data.shape[2]
    print("Scaled image size {w}x{h} {n} frames.".format(w=width, h=height, n=n_frames))

    print("Check sizes are multiples of 16")

    pad_x = 0
    pad_y = 0
    block_size = 16
    if width % block_size != 0:
        pad_x = block_size - (width % block_size)
        if pad_x % 2 == 1:
            pad_x += 1
        pad_x /= 2
    if height % block_size != 0:
        pad_y = block_size - (height % block_size)
        if pad_y % 2 == 1:
            pad_y += 1
        pad_y /= 2

    pad_x = int(pad_x)
    pad_y = int(pad_y)
    print("Padding required = ({x},{y})".format(x=pad_x, y=pad_y))

    tif_data = np.pad(tif_data, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), 'constant')

    # Get new dimensions
    width = tif_data.shape[1]
    height = tif_data.shape[2]
    print("New image size {w}x{h} {n} frames.".format(w=width, h=height, n=n_frames))

    if video_file is not None:

        fps = round(fps, 0)
        print("Rounded FPS={f}".format(f=fps))

        # https://trac.ffmpeg.org/wiki/Encode/H.264
        # Controls compression/speed trade off.
        h264_preset = 'fast'
        # Tuning for type of movie, animation is should be good for this image?
        h264_tune = 'animation'
        # H264 quality, apparently 17 is indistinguishable from lossless.
        h264_crf = '17'

        mov_writer = imageio.get_writer(video_file,
                                        mode="I",
                                        fps=fps,
                                        pixelformat='gray',
                                        codec='h264',
                                        format='mp4',
                                        output_params=['-s', '{0}x{1}'.format(height,
                                                                              # height and width need to be flipped from matrix to img dims
                                                                              width),
                                                       '-preset', h264_preset,
                                                       '-tune', h264_tune,
                                                       '-crf', str(h264_crf)])

        print("Begin write loop")

        for i_frame in range(n_frames):
            if i_frame % 1000 == 0:
                print("Frame {fr}/{n}".format(fr=i_frame, n=n_frames))
            img_frame = tif_data[i_frame, :, :]

            mov_writer.append_data(img_frame)

        mov_writer.close()

        print("Done")
    else:
        print("Video file already exists {}".format(video_file))

    return tif_data