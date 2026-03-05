import numpy as np
from utils import misc
import os
from suite2p.run_s2p import run_s2p
from suite2p.registration import metrics, rigid
from suite2p import io
import shutil
from enum import Enum
class S2PRuntype(Enum):
    Standard = 1
    DeepInterpRedetect = 2
    DeepInterpSameROI = 3


def create_ops_std(op_name, fps, image_file, proc_data_path=None,
                   default_ops_file="s2p/ops_default.npy", classifier_file="s2p/classifier_soma.npy"):

    if default_ops_file is None:
        default_ops_file = "s2p/ops_default.npy"

    ops = np.load(default_ops_file, allow_pickle=True).item()

    # https://suite2p.readthedocs.io/en/latest/settings.html

    # Main settings
    # GCaMP7f tau is 0.7
    ops["tau"] = 0.7
    # Use the correct FPS
    ops["fs"] = fps
    # No need to try to fix bidiphase yet
    ops["do_bidiphase"] = False

    # Output settings
    # Can't remember what this does exactly. This is the description:
    # apply classifier before signal extraction with probability threshold of “preclassify”. If this is set to 0.0,
    # then all detected ROIs are kept and signals are computed.
    ops["preclassify"] = 0.0
    # Keep a log of how long stuff took
    ops["report_time"] = True

    # Registration
    # Make sure this is on
    ops['do_registration'] = True
    # Number of frames to use for reference image, default is 200 but try a larger version
    # ops["nimg_init"] = 1000 # Doesn't make any difference.
    # Batch of images to register, try a lot since I have the RAM and it will go faster.
    ops["batch_size"] = 10000
    # Allow a shift of 30%, why not. Probably should do it based on um but can't be bothered looking it up.
    ops["maxregshift"] = 0.3
    # Give this a whirl. Note  it causes an error if ops['do_registration'] = False
    ops["two_step_registration"] = True
    ops["keep_movie_raw"] = True  # this must be true for two_step_registration
    # Keep the registered tif
    ops["reg_tif"] = True

    # Non-rigid registration
    # Whether to do non rigid registration
    ops["nonrigid"] = True
    # The size of pixel blocks for non linear registration. I have fewer pixels (100x100) so smaller block seems like
    # the thing to do. HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient fft
    ops["block_size"] = [32, 32]
    # Default is 5.
    ops["maxregshiftNR"] = 5


    # ROI detection
    # sparse_mode = True means use the new algorithm, spatial_scale=0 means find it automatically.
    ops["sparse_mode"] = True
    ops["spatial_scale"] = 0
    # This denoise parameter is only in the gui, and says its uses some PCA based method to denoise. Sounds useful in
    # my data.
    ops["denoise"] = True
    # Soma masks should be connected
    ops["connected"] = True
    # Overlap greater than this will be discarded. Set to 1.0 for no discarded (weird).
    ops["max_overlap"] = 0.75 # 0.75 is default
    # Sounds useful for noisy data and soma
    ops["smooth_masks"] = True
    # Default is 20 but I can let it go higher
    ops["max_iterations"] = 100
    # Number of frames for detection. Default is 5000, I could go higher.
    ops["nbinned"] = 20000


    # Signal extraction
    # This is whether signal extraction from overlapping pixels is allowed. Doesn't sound useful.
    ops["allow_overlap"] = False
    # I generally don't have has many pixels (4x less) so use a smaller neuropil (default 350)
    ops["min_neuropil_pixels"] = 100

    # Use the classifer for soma
    ops["use_builtin_classifier"] = False
    ops["classifier_path"] = classifier_file


    s2p_dir = 'suite2p_{mode}'.format(mode=op_name)
    # For some annoying reason the binary file goes here, and it always makes a s2p sub dir,
    # so the binary always ends up in a different folder to the rest.
    if proc_data_path:
        fast_disk_dir = os.path.join(proc_data_path, "suite2p_temp")
        ops["move_bin"] = True  # This moves the bin file to the right spot after


    db = {'data_path': [proc_data_path],
          'look_one_level_down': False,
          'tiff_list': [image_file],
          'save_path0': proc_data_path,
          'save_folder': s2p_dir,
          'fast_disk': fast_disk_dir}

    if proc_data_path:
        # Need to delete any pre-existing files, otherwise s2p won't do the registration and goes looking for the binary
        # file that was moved.
        misc.rmdir(os.path.join(db['save_path0'], db['save_folder']))
        misc.rmdir(fast_disk_dir)

    return ops, db

def create_ops_dend(op_name, fps, image_file, proc_data_path,
                    default_ops_file="s2p/ops_default.npy", classifier_file="s2p/classifier_dend.npy"):

    ops, db = create_ops_std(op_name,
                             fps=fps,
                             image_file=image_file,
                             proc_data_path=proc_data_path,
                             default_ops_file=default_ops_file,
                             classifier_file=classifier_file)

    # Dendrite masks can be disconnected
    #ops["connected"] = False

    return ops, db


def process(ops, db):

    ops_proc = run_s2p(ops=ops, db=db)

    # Need to calculate reg metrics it seems.
    # First fix the reg_file, it moves to the temp dir for some reason.
    proc_data_path = db['data_path'][0]
    s2p_dir = db['save_folder']
    ops_proc['reg_file'] = format(os.path.join(proc_data_path, s2p_dir, "plane0", "data.bin"))

    # get_pc_metrics changed and needs a mov
    # I got the code bellow from suite2p source to see how to get the mov data
    Ly, Lx = ops_proc['Ly'], ops_proc['Lx']
    with io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=ops_proc['reg_file']) as f_reg:

        n_frames, Ly, Lx = f_reg.shape
        # Appears to use less than n_frames, not sure if that is ok
        nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames)
        inds = np.linspace(0, n_frames - 1, nsamp).astype('int')
        mov = f_reg[inds]
        mov = mov[:, ops_proc['yrange'][0]:ops_proc['yrange'][-1],
                     ops_proc['xrange'][0]:ops_proc['xrange'][-1]]
        ops_proc = metrics.get_pc_metrics(mov=mov, ops=ops_proc, use_red=False)

    np.save("{}.npy".format(os.path.join(proc_data_path, s2p_dir, "plane0", "ops")), ops_proc)

    shutil.rmtree(db['fast_disk'])

    return ops_proc

def img_reg(img, ref_img, smooth_sigma=1.15, maxregshift=0.4):
    # smooth_sigma is std in pixels of the gaussian used to smooth the phase correlation between the reference image and the
    # frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV).

    maskMul, maskOffset = rigid.compute_masks(refImg=ref_img, maskSlope=3 * smooth_sigma)
    #maskMul, maskOffset = rigid.compute_masks(refImg=img, maskSlope=3 * smooth_sigma)

    ref_phase_data = rigid.phasecorr_reference(refImg=ref_img,
                                               smooth_sigma=smooth_sigma,
                                               pad_fft=False)

    padx = int((ref_img.shape[0] - img.shape[0]) / 2)
    pady = int((ref_img.shape[1] - img.shape[1]) / 2)

    print(ref_img.shape, img.shape, padx, pady)

    if padx > 0 or pady > 0:
        img = np.pad(img, pad_width=((padx, padx), (pady, pady)), mode='constant', constant_values=0)

    print(ref_img.shape, img.shape)
    fake_vol_data = np.expand_dims(img, axis=0)

    ymax, xmax, cmax = rigid.phasecorr(data=rigid.apply_masks(data=fake_vol_data,
                                                              maskMul=maskMul, maskOffset=maskOffset),
                                       cfRefImg=ref_phase_data,
                                       maxregshift=maxregshift,
                                       smooth_sigma_time=0)

    return ymax, xmax, cmax







