import ntpath
import os
import numpy as np
import shutil
import imageio
import glob
from paths import config
import pandas as pd

# Gets the file name on either windows or linux, wtf.
# Taken from https://stackoverflow.com/questions/44044932/how-can-a-function-optionally-return-one-or-more-values
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def setup_dir(directory, clearout=False):
    if os.path.exists(directory):
        if clearout:
            # This deletes the whole directory which often fails, e.g. if explorer has it open.
            # shutil.rmtree(directory)
            # os.makedirs(directory)

            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(directory)

def rmdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def copy2dir(source, target):
    """
    Copy a file from source to target, creating any directories needed in the target path.

    :param source: Path to the source file.
    :param target: Target file path, including the file name.
    """
    # Extract the target directory path

    target_dir = os.path.dirname(target)

    # Create the target directory if it doesn't exist
    setup_dir(target_dir)
    setup_dir(target)

    # Copy the file
    shutil.copy(source, target)


def get_crossings(data, trigger_val):
    # Can't find a less stupid way of doing this.
    # Blindly copied from:
    # https://stackoverflow.com/questions/50365310/python-rising-falling-edge-oscilloscope-like-trigger
    if data is None:
        return []
    else:
        return np.flatnonzero((data[:-1] <= trigger_val) & (data[1:] >= trigger_val)) + 1


def get_tdms_di_channel(tdms_file, group_name, chan_name):
    # The DI channels seem to be messed up and I have to do it this stupid way.

    # group = tdms_file[group_name]
    # channel = group[chan_name]
    # return channel[:]
    if group_name is None:
        group_name = chan_name
    else:
        group_name = group_name + " - " + chan_name

    if group_name not in tdms_file:
        return None

    g = tdms_file[group_name]
    for c_name in g:
        # Get the first one.s
        c = g[c_name]
        return c


def get_filetype(file_dir, file_type, allow_missing=False, allow_multiple=False, get_first=False):
    files = glob.glob(os.path.join(file_dir, file_type))
    if len(files) == 0 and allow_missing:
        file = None
    elif len(files) == 0 and not allow_missing:
        raise Exception("There is no {ft} in the directory {d}".format(ft=file_type, d=file_dir))
    elif len(files) == 1:
        file = files[0]
    else:
        if allow_multiple and get_first:
            files.sort(key=os.path.getmtime)
            file = files[0]
        else:
            raise Exception("There is more than one {ft} in the directory {d}".format(ft=file_type, d=file_dir))

    return file


def sci_raw_2_tif(imgfile, config_ini_reader=None, out_path=None, save_red=True, overwrite=True):
    print("Converting raw file to tiff")

    img_path, img_file_name = os.path.split(imgfile)
    img_name, ext = os.path.splitext(img_file_name)

    # This following line just tells me the number frames that it was supposed to acquire.
    # n_frames = int(float(config_ini['_']['no..of.frames.to.acquire']))
    # Instead, load this weird sciscan macro for imagej, it seems to be the only place that tells me
    # how many frames were acquired.
    # macro_file = base + "_IJmacro.txt"
    # with open(macro_file, "r") as f:
    #     macro_txt = f.read()
    # macro_chunks = macro_txt.split(" ")
    # for chunk in macro_chunks:
    #     if chunk.startswith("number="):
    #         n_frames = int(float(chunk.split("=")[1]))

    # Actually, this can overflow if more than 2^16!
    # So work out based on file size.

    file_size = os.path.getsize(imgfile)

    file_format = float(config_ini_reader['_']['file.format'])
    if file_format == 0:
        raw_dtype = np.dtype(np.uint16)
    else:
        raise Exception("Not sure what raw format this is")

    if out_path:
        img_path = out_path

    tif_path = os.path.join(img_path, img_name + ".tif")
    tif_red_path = os.path.join(img_path, img_name + ".red.tif")

    pix_x = int(float(config_ini_reader['_']['nomin.x.pixels']))
    pix_y = int(float(config_ini_reader['_']['y.pixels']))

    n_chans = int(float(config_ini_reader['_']['no.of.active.channels']))

    # bytes
    pix_depth = raw_dtype.itemsize

    bytes_per_frame = pix_x * pix_y * pix_depth

    n_frames = int(file_size / bytes_per_frame)
    print(n_frames)
    bytes_read = 0

    if overwrite or not os.path.exists(tif_path):
        with open(imgfile, "rb") as f, \
                imageio.get_writer(tif_path, format='TIFF', mode='v') as tif_writer, \
                imageio.get_writer(tif_red_path, format='TIFF', mode='v') as tif_red_writer:

            for i_frame in range(n_frames):
                data = f.read(bytes_per_frame)
                pix_array = np.frombuffer(data, dtype=raw_dtype)
                img = np.reshape(pix_array, (pix_y, pix_x)).astype(raw_dtype).byteswap()
                if n_chans == 1 or i_frame % 2 == 0:
                    tif_writer.append_data(img)
                else:
                    if save_red:
                        tif_red_writer.append_data(img)

                bytes_read += bytes_per_frame
        print(bytes_read)
    else:
        print(tif_path + " already exists, overwrite is false")
    path_list = [tif_path]
    if n_chans == 2 and save_red:
        path_list.append(tif_red_path)
    if not save_red or n_chans == 1:
        # delete the empty red file that was never used
        if os.path.exists(tif_red_path):
            os.remove(tif_red_path)
    return path_list



def get_exp_path(base_raw_path, exp_dir):
    datestr = exp_dir.split("_")[0]
    datestr = datestr[0:4] + "_" + datestr[4:6] + "_" + datestr[6:8]
    return os.path.join(base_raw_path, datestr, exp_dir)


# def copy_pair(is_soma, i_roi, soma_dend_pairs, file, path):
#     if is_soma:
#         do_copy = i_roi in soma_dend_pairs
#         pre_text = "pair-{}.".format(i_roi)
#     else:
#         do_copy = False
#         for i_soma in soma_dend_pairs:
#             if i_roi in soma_dend_pairs[i_soma]:
#                 do_copy = True
#                 pre_text = "pair-{}-{}.".format(i_soma, i_roi)
#                 break
#
#     if do_copy:
#         shutil.copy2(file, os.path.join(path, pre_text + path_leaf(file)))
#
# def copy_rsig(r, p, file, path, rpos=True, alpha=0.05):
#
#     if (r > 0 or not rpos) and p < alpha:
#         shutil.copy2(file, os.path.join(path, path_leaf(file)))


def get_somadend_pairs(cfg:config.M2PConfig, exp_id):

    dfcsv = pd.read_csv(cfg.meta_somadend_file)

    df = dfcsv.loc[dfcsv['exp_id'] == exp_id]

    pairs = {}
    for index, row in df.iterrows():
        soma_id = int(row["soma_id"])
        dend_id = int(row["dend_id"])
        if soma_id in pairs:
            pairs[soma_id].append(dend_id)
        else:
            pairs[soma_id] = [dend_id]

    return pairs

def get_expids():
    dfcsv = pd.read_csv("metadata/experiments.csv")

    return dfcsv["exp_id"]


