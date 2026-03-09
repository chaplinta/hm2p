import configparser
import os
import shutil
import imageio
import numpy as np

import utils.img
from utils import misc, s2p, metadata as metautils
from classes import Experiment, S2PData
import matplotlib.pyplot as plt
from suite2p.registration import zalign
from paths.config import M2PConfig
from classes.ProcPath import ProcPath


def proc_single(cfg: M2PConfig,
                exp_id,
                zstack_id=None,
                run_suite2p=True,
                do_rigid_and_nonreg=True,
                plot_traces=True,
                plot_traces_bad=False,
                make_images=True,
                make_zproj=True,
                plot_reg_metrics=True,
                plot_pcas=True,
                make_movies=True,
                overwrite_movies=False,
                do_zstack=False,
                overwrite=True):


    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    s2p_class_base_path = cfg.s2p_class_path

    proc_data_path = m2p_paths.proc_s2p_path
    if not overwrite and os.listdir(proc_data_path) != []:
        print("Suite 2p data already exists for {}, skipping".format(exp_id))
        return

    exp = Experiment.Experiment(m2p_paths.raw_data_path)

    input_img_file = exp.SciscanSettings.image_file

    images_dir = os.path.join(proc_data_path, "images")
    movies_dir = os.path.join(proc_data_path, "movies")
    zpos_dir = os.path.join(proc_data_path, "zpos")
    soma_dir = os.path.join(proc_data_path, "soma")
    dend_dir = os.path.join(proc_data_path, "dend")
    soma_bad_dir = os.path.join(soma_dir, "bad")
    dend_bad_dir = os.path.join(dend_dir, "bad")

    config_ini = configparser.ConfigParser()
    config_ini.read(exp.SciscanSettings.ini_file)

    misc.setup_dir(proc_data_path, clearout=False)
    misc.setup_dir(images_dir, clearout=False)
    misc.setup_dir(movies_dir, clearout=(make_movies and overwrite_movies))
    misc.setup_dir(zpos_dir, clearout=False)
    misc.setup_dir(soma_dir, clearout=plot_traces)
    misc.setup_dir(dend_dir, clearout=plot_traces)
    misc.setup_dir(soma_bad_dir, clearout=plot_traces_bad)
    misc.setup_dir(dend_bad_dir, clearout=plot_traces_bad)

    regmetrics_dir = os.path.join(proc_data_path, "regmetrics")
    regmetrics_raw_plots_dir = os.path.join(regmetrics_dir, "raw-plots")
    regmetrics_raw_gifs_dir = os.path.join(regmetrics_dir, "raw-gifs")
    regmetrics_rigid_plots_dir = os.path.join(regmetrics_dir, "rigid-plots")
    regmetrics_rigid_gifs_dir = os.path.join(regmetrics_dir, "rigid-gifs")
    regmetrics_reg_plots_dir = os.path.join(regmetrics_dir, "reg-plots")
    regmetrics_reg_gifs_dir = os.path.join(regmetrics_dir, "reg-gifs")

    misc.setup_dir(regmetrics_dir, clearout=False)
    misc.setup_dir(regmetrics_raw_plots_dir, clearout=plot_pcas)
    misc.setup_dir(regmetrics_raw_gifs_dir, clearout=plot_pcas)
    misc.setup_dir(regmetrics_rigid_plots_dir, clearout=plot_pcas)
    misc.setup_dir(regmetrics_rigid_gifs_dir, clearout=plot_pcas)
    misc.setup_dir(regmetrics_reg_plots_dir, clearout=plot_pcas)
    misc.setup_dir(regmetrics_reg_gifs_dir, clearout=plot_pcas)

    # Write some important values out for quick reference
    sum_text_file = os.path.join(proc_data_path, "summary.txt")
    exp.write_s2p_sum_file(sum_text_file)

    # Run s2p
    if run_suite2p:

        bad_indexes = metautils.get_bad_2p_indexes(cfg, exp_id, exp.SciscanSettings.image_n_frames)
        bad_frames_file = os.path.join(proc_data_path, "bad_frames.npy")
        np.save(bad_frames_file, bad_indexes)



        print("Suite2p for soma")
        ops, db = s2p.create_ops_std("soma",
                                     fps=exp.SciscanSettings.frames_p_sec,
                                     image_file=input_img_file,
                                     proc_data_path=str(proc_data_path),
                                     classifier_file=os.path.join(s2p_class_base_path, "classifier_soma.npy"))

        ops['do_registration'] = True
        ops["two_step_registration"] = True

        s2p.process(ops, db)

        # Run a version for dendrites
        print("Suite2p for dendrites")
        ops, db = s2p.create_ops_dend("dend",
                                      fps=exp.SciscanSettings.frames_p_sec,
                                      image_file=input_img_file,
                                      proc_data_path=str(proc_data_path),
                                      classifier_file=os.path.join(s2p_class_base_path, "classifier_dend.npy"))

        shutil.copytree(os.path.join(proc_data_path, 'suite2p_soma'),
                        os.path.join(proc_data_path, 'suite2p_dend'))

        misc.setup_dir(os.path.join(proc_data_path, "suite2p_temp"))

        ops['do_registration'] = False
        # ops['roidetect'] = False
        # Think this removes dendrites when calculating compactness for classifier. Turn it off for dendrites.
        ops["crop_soma"] = False

        s2p.process(ops, db)

        if do_rigid_and_nonreg:
            # Do a rigid only registration for comparison
            print("Suite2p with rigid registration only")
            ops, db = s2p.create_ops_std("soma_rigid",
                                         fps=exp.SciscanSettings.frames_p_sec,
                                         image_file=exp.SciscanSettings.image_file,
                                         proc_data_path=str(proc_data_path))

            ops["nonrigid"] = False  # this seems to break the reg metrics for some reason.
            ops["roidetect"] = False  # Don't bother with roi detection, this is just for registration metrics.

            s2p.process(ops, db)

            # Do a dummy non registration for comparison
            print("Suite2p with no registration")
            ops, db = s2p.create_ops_std("soma_notreg",
                                         fps=exp.SciscanSettings.frames_p_sec,
                                         image_file=exp.SciscanSettings.image_file,
                                         proc_data_path=str(proc_data_path))
            # Turn registration off.
            ops['do_registration'] = False
            ops["roidetect"] = False  # Don't bother with roi detection, this is just for registration metrics
            # Seems this needs to be off for no reg or it gets confused.
            ops["two_step_registration"] = False
            ops["keep_movie_raw"] = False
            s2p.process(ops, db)

        print("Suite2p is finished processing")

    print("Reloading data")
    s2p_soma = S2PData.load_mode(proc_data_path, "soma")
    s2p_dend = S2PData.load_mode(proc_data_path, "dend")

    s2p_soma_rigid = None
    s2p_soma_notreg = None
    rigid_data_exists = os.path.exists(S2PData.get_mode_path(proc_data_path, "soma_rigid"))
    rigid_notreg_data_exists = rigid_data_exists and \
                               os.path.exists(S2PData.get_mode_path(proc_data_path, "soma_notreg"))
    if rigid_notreg_data_exists:
        s2p_soma_rigid = S2PData.load_mode(proc_data_path, "soma_rigid")
        s2p_soma_notreg = S2PData.load_mode(proc_data_path, "soma_notreg")

    if make_images:

        print("Saving tifs")
        s2p_soma.save_tif()
        # Save a nicer normalized one in the images dir
        prc_clip = 0.1
        s2p_soma.save_tif(tif_path=os.path.join(images_dir, "reg.tif"), prc_clip=prc_clip)

        # The tif for the unregistered, but suite2p doesn't create any images here so you have to load the source data.
        tif_raw_path = exp.SciscanSettings.image_file
        #shutil.copyfile(tif_raw_path, os.path.join(s2p_soma_notreg.dir, "reg.tif"))
        tif_data = utils.img.read_tif_vol(tif_raw_path)
        tif_data = utils.img.normalize_img_16bit(tif_data, prc_clip)
        imageio.volwrite(os.path.join(images_dir, "reg_notreg.tif"), tif_data, format="TIFF")
        del tif_data

        if rigid_data_exists:
            s2p_soma_rigid.save_tif()
            # Save a nicer normalized one in the images dir
            s2p_soma_rigid.save_tif(tif_path=os.path.join(images_dir, "reg_rigid.tif"), prc_clip=prc_clip)

        print("Creating ROI images")
        s2p_soma.create_roi_image(os.path.join(images_dir, "soma_good.png"), plot_good=True, plot_bad=False)
        # s2p_soma.create_roi_image(os.path.join(images_dir, "soma_bad.png"), plot_good=False, plot_bad=True)
        # s2p_soma.create_roi_image(os.path.join(images_dir, "soma_goodvsbad.png"), plot_good=True, plot_bad=True, good_vs_bad=True)
        s2p_dend.create_roi_image(os.path.join(images_dir, "dend_good.png"), plot_good=True, plot_bad=False)
        # s2p_dend.create_roi_image(os.path.join(images_dir, "dend_bad.png"), plot_good=False, plot_bad=True)
        # s2p_dend.create_roi_image(os.path.join(images_dir, "dend_goodvsbad.png"), plot_good=True, plot_bad=True, good_vs_bad=True)

    if make_zproj:
        print("Creating z-projections")
        s2p_soma.save_zproj(images_dir, "reg")

        # The tif for the unregistered, but suite2p doesn't create any images here so you have to load the source data.
        tif_raw_path = exp.SciscanSettings.image_file
        tif_data = utils.img.read_tif_vol(tif_raw_path)
        #tif_data = utils.img.normalize_img_16bit(tif_data, prc_clip)
        z_proj_mean = tif_data.mean(axis=0)
        z_proj_max = tif_data.max(axis=0)
        S2PData.save_zproj_img(z_proj_mean, z_proj_max, images_dir, "raw")
        del tif_data

        if rigid_data_exists:
            s2p_soma_rigid.save_zproj(images_dir, "rigid")
            #s2p_soma_notreg.save_zproj(images_dir, "notreg")

    if plot_reg_metrics:
        print("Save the reference image")
        s2p_soma.save_ref_img(regmetrics_dir)

        print("Plot the movement histograms")
        # todo maybe do this a bit better, close enough for now
        pix_size = exp.SciscanSettings.x_pixel_sz * 1000000
        s2p_soma.plot_movement_hists(regmetrics_dir, pix_size)

        print("Plot the shift for each PC high vs low")
        s2p_soma.plot_pca_shifts(s2p_soma_rigid,
                                 s2p_soma_notreg,
                                 plot_img_path=os.path.join(regmetrics_dir, "pca_shifts_pixels.png"))
        s2p_soma.plot_pca_shifts(s2p_soma_rigid,
                                 s2p_soma_notreg,
                                 plot_img_path=os.path.join(regmetrics_dir, "pca_shifts_um.png"),
                                 pix_size=pix_size)

    if plot_traces:
        print("Plotting traces")
        print("Plotting good soma traces")
        s2p_soma.plot_traces(soma_dir, "soma", plot_good=True, plot_bad=False)
        print("Plotting good dendrite traces")
        s2p_dend.plot_traces(dend_dir, "dend", plot_good=True, plot_bad=False)
        if plot_traces_bad:
            print("Plotting bad soma traces")
            s2p_soma.plot_traces(soma_bad_dir, "soma", plot_good=False, plot_bad=True)
            print("Plotting bad dendrite traces")
            s2p_dend.plot_traces(dend_bad_dir, "dend", plot_good=False, plot_bad=True)
        print("Done!")

    if plot_pcas:
        print("Plotting PCA for registered data")
        s2p_soma.plot_pcas(regmetrics_reg_plots_dir, regmetrics_reg_gifs_dir)

        if rigid_notreg_data_exists:
            print("Plotting PCA for rigid registered data")
            s2p_soma_rigid.plot_pcas(regmetrics_rigid_plots_dir, regmetrics_rigid_gifs_dir)

            print("Plotting PCA for non-registered data")
            s2p_soma_notreg.plot_pcas(regmetrics_raw_plots_dir, regmetrics_raw_gifs_dir)

    # Convert tifs to movies
    if make_movies:
        print("Creating movies ...")


        utils.img.tif_to_movie(input_img_file, s2p_soma.ops["fs"], os.path.join(movies_dir, "raw.mp4"))

        s2p_soma.create_movie(video_file=os.path.join(movies_dir, "reg.mp4"))

        if rigid_data_exists:
            s2p_soma_rigid.create_movie(video_file=os.path.join(movies_dir, "rigid.mp4"))
            #s2p_soma_notreg.create_movie(video_file=os.path.join(movies_dir, "raw.mp4"))

        print("Done!")

    if do_zstack and zstack_id:
        print("Z positions measurement")
        # Just use a sub sample to find the range. Using whole z-stack takes forever.
        init_frames = 10
        # todo maybe use z proj single image for init test
        zstack_img_path = os.path.join(cfg.zstack_path, zstack_id, "zstack-" + zstack_id + "_XYTZ.tif")

        print("Z pos - loading stack")
        zstack_data = utils.img.read_tif_vol(zstack_img_path)

        # Use last 100 only
        z_init_start = zstack_data.shape[0] - 100
        if z_init_start < 0:
            z_init_start = 0

        z_init_end = -1
        zstack_data_init = zstack_data[z_init_start:z_init_end, :, :]

        print("Z pos - loading data")
        tif_data = s2p_soma.load_tif_dir()

        padx = int((zstack_data.shape[1] - tif_data.shape[1]) / 2)
        pady = int((zstack_data.shape[2] - tif_data.shape[2]) / 2)

        if padx > 0 or pady > 0:
            print("Z pos - padding data")
            tif_data = np.pad(tif_data, pad_width=((0, 0), (padx, padx), (pady, pady)), mode='constant',
                              constant_values=0)

        print("Z pos - writing init padded img & bin file")
        # Save the image just for easy viewing
        tif_data_init = tif_data[0:init_frames, :, :]
        imageio.volwrite(os.path.join(zpos_dir, "reg-zstackres-init.tif"), tif_data_init)
        pad_bin_file = os.path.join(zpos_dir, "reg-zstackres-init.bin")
        with open(pad_bin_file, mode='wb') as write_file:
            write_file.write(bytearray(np.minimum(tif_data_init, 2 ** 15 - 2).astype('int16')))

        # Create a dummy ops just with the options required
        ops = {}
        ops['reg_file'] = pad_bin_file
        ops['batch_size'] = 1000  # default 200, can spare the ram to make it faster though.
        ops['Ly'] = zstack_data.shape[1]
        ops['Lx'] = zstack_data.shape[2]
        ops['1Preg'] = False
        # Standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the
        # frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV).
        ops['smooth_sigma'] = 1.15
        ops['smooth_sigma_time'] = 0
        ops['pad_fft'] = False
        ops['nframes'] = init_frames
        ops['maxregshift'] = 0.5

        # Call the suite 2p function.
        print("Z pos - calculating init positions")
        ops_orig, zcorr = zalign.compute_zpos(zstack_data_init, ops)
        zpos = np.argmax(zcorr, axis=0) + z_init_start
        zpos_csv = os.path.join(zpos_dir, "zpos-init.csv")
        np.savetxt(zpos_csv, zpos, fmt='%d', delimiter=',')

        f = plt.figure()
        plt.gca().invert_yaxis()
        plt.plot(zpos)
        plot_img_path = os.path.join(zpos_dir, "zpos-init.png")
        f.savefig(plot_img_path, dpi=300, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        print("Z pos - calculating all positions")
        z_est = int(np.round(np.mean(zpos)))
        z_max_move = 10
        z_start = z_est - z_max_move
        z_end = z_est + z_max_move
        if z_start < 0:
            z_start = 0
        if z_end >= tif_data.shape[0]:
            z_end = tif_data.shape[0]

        zstack_data = zstack_data[z_start:z_end, :, :]

        # Write the full bin file this time
        print("Z pos - writing full padded bin file")
        # Save the image just for easy viewing
        bigtiff  = (tif_data.size * tif_data.dtype.itemsize) >= 2 ^ 32
        imageio.volwrite(os.path.join(zpos_dir, "reg-zstackres.tif"), tif_data, bigtiff =bigtiff )
        pad_bin_file = os.path.join(zpos_dir, "reg-zstackres.bin")
        with open(pad_bin_file, mode='wb') as write_file:
            write_file.write(bytearray(np.minimum(tif_data, 2 ** 15 - 2).astype('int16')))

        ops['reg_file'] = pad_bin_file
        ops['nframes'] = tif_data.shape[0]

        ops_orig, zcorr = zalign.compute_zpos(zstack_data, ops)
        zpos = np.argmax(zcorr, axis=0) + z_init_start
        zpos_csv = os.path.join(zpos_dir, "zpos.csv")
        np.savetxt(zpos_csv, zpos, fmt='%d', delimiter=',')
        print("Z pos - done")

        # todo check z res, might not be 1um
        zpos_um = zpos - np.median(zpos[0:100])  # Centre around some initial positions

        f = plt.figure()
        plt.gca().invert_yaxis()
        plt.plot(s2p_soma.time, zpos_um)
        plt.xlabel('Time (s)')
        plt.xlabel('Z-positions (um)')
        plot_img_path = os.path.join(zpos_dir, "zpos.png")
        f.savefig(plot_img_path, dpi=300, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    print("Suite2p processing and plotting is complete!")
