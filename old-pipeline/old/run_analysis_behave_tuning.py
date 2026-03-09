import utils.ca
import utils.data
import utils.img
import utils.stats
import utils.video
from classes import Experiment, S2PData
import os
from utils import misc, behave as bu, tune_old as tu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import ffmpeg
import imageio
import scipy as sp
from matplotlib.ticker import FuncFormatter
import math
import cv2

# See https://github.com/DeepLabCut/DLCutils/blob/master/Demo_loadandanalyzeDLCdata.ipynb
# and https://github.com/adamltyson/movement/blob/master/movement/io/dlc.py
# Also https://github.com/adamltyson/spikey/blob/master/spikey/tuning/radial.py seems to be the analysis

video_base_path = "J:/Data/video/"
meta_bak_base_path = "J:/Data/video-meta-backup/"
proc_base_path = "J:/Data/behave-tuning/"
s2p_base_path = "J:/Data/s2p/"
base_raw_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze"

rebuild_behav_data = False
rebuild_resamp_data = False
soma_dend_pairs = {}

raw_data_path = os.path.join(base_raw_path, "2021_08_23", "20210823_16_59_50_1114353")
soma_dend_pairs = {1: [33, 92],
                   3: [60],
                   4: [174, 61, 209],
                   6: [88, 98],
                   35: [48]}

# raw_data_path = m2putils.get_exp_path(base_raw_path, "20210920_11_09_37_1114356")
# somadend_pairs = {7: [164, 154],
#                    8: [92, 121],
#                    18: [59],
#                    23: [132]}

print("Loading experiment")
exp = Experiment.Experiment(raw_data_path)
print("Loading experiment done.")

s2p_path = os.path.join(s2p_base_path, misc.path_leaf(exp.directory))

print("Checking camera timings and frames")
exp.check_cameras(ignore_cams=["side_left"])

print("Checking sciscan and frames")
exp.check_sci_frames()
# events_dir = os.path.join(proc_data_path, "gifs")
# m2putils.setup_dir(events_dir, clearout=True)

print("Check the light times")
exp.check_light_times()
print("Experiment checks complete")

raw_name = os.path.split(raw_data_path)[1]

proc_path = os.path.join(proc_base_path, raw_name)
misc.setup_dir(proc_path)

video_path = os.path.join(video_base_path, raw_name)
tracking_data_path = os.path.join(video_path, "tracking")
cam_name = "overhead"
cam = exp.tracking_video.cameras[cam_name]

mov_fullfps_name = cam.file_name_base + "-cropped.mp4"
mov_fullfps_path = os.path.join(video_path, mov_fullfps_name)
mov_fullfps_name_base = os.path.splitext(mov_fullfps_name)[0]
# dlc_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".raw.h5")
# behav_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".raw.metrics.h5")
dlc_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".filtered.h5")
behav_file = os.path.join(tracking_data_path, mov_fullfps_name_base + ".filtered.metrics.h5")
#behav_resample_file = os.path.join(proc_path, raw_name + ".raw.metrics-sciresamp.h5")
behav_resample_file = os.path.join(proc_path, raw_name + ".filtered.metrics-sciresamp.h5")
plot_ccorr_dir = os.path.join(proc_path, "ccorr")
plot_tune_dir = os.path.join(proc_path, "tuning")
plot_tune_pairs_dir = os.path.join(plot_tune_dir, "pairs")
plot_tune_hdmod_dir = os.path.join(plot_tune_dir, "hdmod")
plot_evoke_dir = os.path.join(proc_path, "evoked")
plot_zpos_dir = os.path.join(proc_path, "zpos")
plot_event_trace_dir = os.path.join(proc_path, "event-traces")
plot_spat_trace_dir = os.path.join(proc_path, "spatial-traces")
plot_pop_dir = os.path.join(proc_path, "pop")
plot_event_amps_dir = os.path.join(proc_path, "event-amps")


misc.setup_dir(plot_ccorr_dir, clearout=False)
misc.setup_dir(plot_tune_dir, clearout=False)
misc.setup_dir(plot_tune_pairs_dir, clearout=False)
misc.setup_dir(plot_tune_hdmod_dir, clearout=False)
misc.setup_dir(plot_evoke_dir, clearout=False)
misc.setup_dir(plot_zpos_dir, clearout=False)
misc.setup_dir(plot_event_trace_dir, clearout=False)
misc.setup_dir(plot_spat_trace_dir, clearout=False)
misc.setup_dir(plot_pop_dir, clearout=False)
misc.setup_dir(plot_event_amps_dir, clearout=False)




meta_data_path = os.path.join(video_path, "meta")
backup_meta_path = os.path.join(meta_bak_base_path, misc.path_leaf(exp.directory))

# print("Creating bad frames file")
# exp.save_bad_frames(overwrite=False,
#                     bad_frames_file=os.path.join(proc_path, "bad_frames.npy"))

df = pd.read_hdf(os.path.join(dlc_file))

scorer = df.columns.get_level_values(0)[0]
bodyparts = df.columns.get_level_values(1)

print("Calculating HD, positions and velocity")
df = bu.calc_behav(df,
                   meta_data_path=meta_data_path,
                   fps=exp.tracking_video.fps,
                   frame_times=exp.cam_trigger_times,
                   light_on_times=exp.Lighting.on_pulse_times,
                   light_off_times=exp.Lighting.off_pulse_times,
                   save_file=behav_file,
                   rebuild=rebuild_behav_data,
                   backup_meta_path=backup_meta_path)




print("Done")


print("Loading suite2p data")
soma_data = S2PData.load_mode(s2p_path, "soma")
dend_data = S2PData.load_mode(s2p_path, "dend")

print("Done")
# Probably a stupid way of doing but cbf with data frames fancy stuff.
if not rebuild_resamp_data and os.path.exists(behav_resample_file):

    print("Loading resampled behavioural data to imaging time scale")
    df_resampled = pd.read_hdf(behav_resample_file)
    print("Load done.")
else:

    # Probably a pandas way of doing this but this manual way is at least exact.
    print("Resampling behavioural data to imaging time scale")
    n_sci_frames = exp.SciscanSettings.sci_frame_times.size

    cols_for_resample = bu.ALL_SERIES
    data_resamp = np.zeros((n_sci_frames, len(cols_for_resample)))
    df_metrics = df[cols_for_resample]
    light_on = np.zeros(n_sci_frames, dtype='uint8')
    for i_sci_frame in range(n_sci_frames):

        sci_time = exp.SciscanSettings.sci_frame_times[i_sci_frame]
        if i_sci_frame == 0:
            sci_time_prev = sci_time - (1.0 / exp.SciscanSettings.frames_p_sec)
        else:
            sci_time_prev = exp.SciscanSettings.sci_frame_times[i_sci_frame - 1]

        cam_frames = np.where(np.logical_and(exp.cam_trigger_times >= sci_time_prev,
                                             exp.cam_trigger_times < sci_time))[0]

        sci_frame_data = df_metrics.iloc[cam_frames]
        sci_frame_data_mean = sci_frame_data.mean().values
        data_resamp[i_sci_frame, :] = sci_frame_data_mean

        # Look up to see if the light was on during this frame. Should I take a mean? It shouldn't be on for half a \
        # frame, but todo check.
        sci_time_index = exp.SciscanSettings.sci_frame_indexes[i_sci_frame]
        light_was_on = exp.Lighting.on_indexes[sci_time_index]
        light_on[i_sci_frame] = light_was_on


    df_resampled = pd.DataFrame(data=data_resamp, columns=cols_for_resample)

    # LIGHT_ON is already calculate in resampe sci time
    df_resampled[bu.LIGHT_ON] = light_on

    # HD absolute has to be the wrapped version of the resampled unwrapped angles.
    # Otherwise you average across the discontinuity and fuck it up.
    df_resampled[bu.HD_ABS] = bu.phase_wrap(df_resampled[bu.HD_UNWRAP])
    df_resampled[bu.HD_ABS_FILT] = bu.phase_wrap(df_resampled[bu.HD_UNWRAP_FILT])

    df_resampled[bu.HEADING_ALLO_ABS] = bu.phase_wrap(df_resampled[bu.HEADING_ALLO_UNWRAP])
    df_resampled[bu.HEADING_ALLO_ABS_FILT] = bu.phase_wrap(df_resampled[bu.HEADING_ALLO_UNWRAP_FILT])
    df_resampled[bu.HEADING_EGO_ABS] = bu.phase_wrap(df_resampled[bu.HEADING_EGO_UNWRAP])
    df_resampled[bu.HEADING_EGO_ABS_FILT] = bu.phase_wrap(df_resampled[bu.HEADING_EGO_UNWRAP_FILT])


    # Not sure but maybe the head pos goes out of maze bounds, fix just in case.
    maze_poly = bu.get_maze_poly()
    df_resampled = bu.fix_oob_df(df_resampled, maze_poly, bu.HEAD_X_RAW_MAZE, bu.HEAD_Y_RAW_MAZE)
    df_resampled = bu.fix_oob_df(df_resampled, maze_poly, bu.HEAD_X_FILT_MAZE, bu.HEAD_Y_FILT_MAZE)


    df_resampled.to_hdf(behav_resample_file, key="df_resampled")
    print("Resampling complete")

# # Set frames with dodgy AHV and bad imaging to Nan. Hopefully every thing handles this lol.
# bad_ahv_indexes = df_move[bu.AHV_FILT_GRAD].values > 500
# bad_img_indexes = exp.load_bad_frames(os.path.join(s2p_path, "bad_frames.npy"))
# bad_frames = np.concatenate((bad_ahv_indexes, bad_img_indexes))
# df_resampled[bad_frames] = np.nan


# print("Creating arrow video")
# movie_arrow_path = os.path.join(proc_path, cam.file_name_base + "-arrow.mp4")
# mov_scifps_name = cam.file_name_base + "-scifps.mp4"
# mov_scifps_path = os.path.join(video_path, mov_fullfps_name)
# bu.create_arrow_mov(df_resampled,
#                     mov_scifps_path,
#                     movie_arrow_path,
#                     start_frame=0,
#                     end_frame=500)


# tif_path = os.path.join(s2p_path, "images", "reg_notreg.tif")
# img_data = m2putils.read_tif_vol(tif_path)
# on_indexes = df_resampled[bu.LIGHT_ON].values == 1
# #on_indexes = df_resampled[bu.HD_ABS].values < 180
#
# off_indexes = np.logical_not(on_indexes)
# img_light_on = img_data[on_indexes, :, :].mean(axis=0)
# img_light_off = img_data[off_indexes, :, :].mean(axis=0)
#
# imageio.imwrite("~/on.png", img_light_on)
# imageio.imwrite("~/off.png", img_light_off)
# imageio.imwrite("~/diff.png", img_light_on - img_light_off)
#
# plt.imshow(img_data[10, :, 0:5])
# plt.show()
#
# edge_values = img_data[:, :, 0:5].mean(axis=1).mean(axis=1)
# edge_values = edge_values / np.max(edge_values)
# f = plt.figure()
# plt.plot(df_resampled[bu.LIGHT_ON].values)
# plt.plot(edge_values)
# plt.show()
# f.savefig("~/plot.png", dpi=500)
#
# raise Exception()

movie_head_arrow_path = os.path.join(proc_path, "head-arrow.mp4")
(movie_x, movie_y) = utils.video.get_mov_res(mov_fullfps_path)


#ca_data_set = [S2PData.CaDataType.DFONF0, S2PData.CaDataType.DECONV, S2PData.CaDataType.EVENTS_BIN, S2PData.CaDataType.EVENTS_AMP]
ca_data_set = [S2PData.CaDataType.DECONV, S2PData.CaDataType.EVENTS_BIN]
s2p_data_set = [soma_data, dend_data]
time_offset_set = [0]

do_crosscorr = True
add_peak_offset = False

plot_zpos = False

plot_tuning = True
plot_combined = True
plot_lightvsdark = True
plot_hd = False
plot_hd_cart = False
plot_hd_polar = False
plot_speed = True
plot_ahv = True
plot_place = False
plot_evoked = False

plot_event_dists = False

plot_spatial_traces = False
make_event_gifs = False

plot_event_traces = False

roi_label = "ROI"

light_indexes = df_resampled[bu.LIGHT_ON].values == 1
dark_indexes = np.logical_not(light_indexes)

# Sepi used 6 degree bins (60!)
bins_hd = np.linspace(0, 360, 40 + 1, endpoint=True)
bins_hdahv = np.linspace(0, 360, 12 + 1, endpoint=True)
bins_hdnsew = np.linspace(-45, 360-45, 4 + 1, endpoint=True)

speed_min = 0.5
speed_max = 40
bins_speed_trace = np.append([0], np.linspace(speed_min, speed_max, 14 + 1, endpoint=True))
#bins_speed_events = np.append([0], np.linspace(speed_min, speed_max, 7 + 1, endpoint=True))
bins_speed_events = np.append([0], np.linspace(speed_min, speed_max, 7 + 1, endpoint=True))
# bad_speeds = df_resampled[bu.SPEED_FILT_GRAD] > speed_max
# df_resampled[bu.SPEED_FILT_GRAD].iloc[bad_speeds] = speed_max



bins_ahv_pos = np.linspace(0, 90, 9+1, endpoint=True)
bins_ahv_pos = bins_ahv_pos[1:]
bins_ahv = np.hstack((np.flip(-bins_ahv_pos), -1, 1, bins_ahv_pos))
bins_ahv_abs = np.append([0, 1], bins_ahv_pos)


# Evoked responses
maze_width = 7
maze_height = 5
maze_bin_sub_big = 1
maze_bin_sub_small = 5
bins_x_big = np.linspace(0, maze_width, maze_width * maze_bin_sub_big + 1, endpoint=True)
bins_y_big = np.linspace(0, maze_height, maze_height * maze_bin_sub_big + 1, endpoint=True)
bins_x_small = np.linspace(0, maze_width, maze_width * maze_bin_sub_small + 1, endpoint=True)
bins_y_small = np.linspace(0, maze_height, maze_height * maze_bin_sub_small + 1, endpoint=True)

maze_poly = bu.get_maze_poly()

occmap_big, xedges, yedges = np.histogram2d(df[bu.HEAD_X_FILT_MAZE],
                                            df[bu.HEAD_Y_FILT_MAZE],
                                            bins=[bins_x_big, bins_y_big])
occmap_small, xedges, yedges = np.histogram2d(df[bu.HEAD_X_FILT_MAZE],
                                              df[bu.HEAD_Y_FILT_MAZE],
                                              bins=[bins_x_small, bins_y_small])

dpi = 300

font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

linewidth = 3

plt.rc('font', **font)
# plt.rcParams['axes.linewidth'] = 3 # set the value globally

# Z positions analysis
if plot_zpos:
    s2p_zpos_data_set = [soma_data, dend_data]
    zpos = None
    zpos_csv = os.path.join(s2p_path, "zpos", "zpos.csv")
    if os.path.exists(zpos_csv):
        zpos = np.loadtxt(zpos_csv, zpos, delimiter=',')
        speed_trace = df_resampled[bu.SPEED_FILT_GRAD].to_numpy()
        moving_indexes = speed_trace > speed_min
        #speed_trace = np.abs(df_resampled[bu.AHV_FILT_GRAD].to_numpy())



        all_roi_speed_r = []
        all_roi_z_r = []
        for s2p_data in s2p_zpos_data_set:
            (n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad) = s2p_data.get_n_roi_plot(plot_good=True, plot_bad=False)
            for i_roi in range(0, n_roi_cand):
                is_good = s2p_data.iscell[i_roi, 0] == 1
                if not is_good:
                    continue

                #roi_data = s2p_data.dFonF0[i_roi, :]
                roi_data = s2p_data.deconv[i_roi, :]

                (roi_speed_r, hd_p) = sp.stats.spearmanr(roi_data, speed_trace)
                (roi_z_r, hd_p) = sp.stats.spearmanr(roi_data, zpos)
                all_roi_speed_r.append(roi_speed_r)
                all_roi_z_r.append(roi_z_r)

        zpos_mid = np.median(zpos[0:100])
        zpos_centered = zpos - zpos_mid
        fig = plt.figure(tight_layout=True)
        plt.plot(s2p_data.time, zpos_centered)
        plt.xlabel('Time (s)')
        plt.ylabel('Z positions (um)')
        plt.ylim([- 10, 10])
        fig.savefig(os.path.join(plot_zpos_dir, "01-zpos.trace.png"), dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')



        fig = plt.figure(tight_layout=True)
        plt.plot(s2p_data.time[0:1000], zpos_centered[0:1000])
        plt.xlabel('Time (s)')
        plt.ylabel('Z positions (um)')
        plt.ylim([- 10, 10])
        fig.savefig(os.path.join(plot_zpos_dir, "02-zpos.trace-zoom.png"), dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.hist(zpos_centered)
        plt.xlabel('Z positions (um)')
        fig.savefig(os.path.join(plot_zpos_dir, "03-zpos.hist.png"), dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.hist(all_roi_speed_r)
        plt.xlabel('dF/F - speed correlation')
        fig.savefig(os.path.join(plot_zpos_dir, "04-dFonF.speed.hist.png"), dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.hist(all_roi_z_r)
        plt.xlabel('dF/F - ZPos correlation')
        fig.savefig(os.path.join(plot_zpos_dir, "05-dFonF.zpos.hist.png"), dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.scatter(all_roi_z_r, all_roi_speed_r)
        plt.xlabel('dF/F - ZPos correlation')
        plt.ylabel('dF/F - speed correlation')
        (hd_r, hd_p) = sp.stats.spearmanr(all_roi_z_r, all_roi_speed_r)
        plt.title("r={:.2f} p={:.3f}".format(hd_r, hd_p))
        fig.savefig(os.path.join(plot_zpos_dir, "06-dFonF.speed.zpos.scatter.png"), dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        # raise Exception()

if plot_tuning:
    print("Plot tuning curves")
    for s2p_data in s2p_data_set:
        roi_type = s2p_data.mode
        is_soma = roi_type == "soma"


        print("Plotting {} data".format(roi_type))

        (n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad) = s2p_data.get_n_roi_plot(plot_good=True, plot_bad=False)


        for i_roi in range(0, n_roi_cand):
            is_good = s2p_data.iscell[i_roi, 0] == 1

            if not is_good:
                continue

            print("Plotting {} data roi {}".format(roi_type, i_roi))

            for ca_data_type in ca_data_set:

                smooth_data = True
                plot_error = True
                if ca_data_type == S2PData.CaDataType.DFONF0:
                    # F_sub_neuropil = s2p_data.F[i_roi, :] - s2p_data.ops["neucoeff"] * s2p_data.Fneu[i_roi, :]
                    # f0 = np.percentile(F_sub_neuropil, 10)
                    # roi_data = (F_sub_neuropil - f0) / f0
                    roi_data = s2p_data.dFonF0[i_roi, :]
                    # plt.figure()
                    # plt.hist(roi_data, bins=20)
                    # plt.show()
                    ylabel_text = "dF/F0"
                    bins_speed = bins_speed_trace

                elif ca_data_type == S2PData.CaDataType.DECONV:
                    roi_data = s2p_data.deconv_norm[i_roi, :] * 100
                    ylabel_text = "Deconv. (% peak)"
                    bins_speed = bins_speed_trace
                elif ca_data_type == S2PData.CaDataType.EVENTS_BIN or ca_data_type == S2PData.CaDataType.EVENTS_AMP:

                    ca_trace = s2p_data.dFonF0[i_roi, :]
                    ca_events = utils.ca.get_ca_events(ca_trace,
                                                       smooth_sigma=3,
                                                       prc_mean=40,
                                                       prc_low=10,
                                                       prc_high=90,
                                                       prob_onset=0.2,
                                                       prob_offset=0.7)

                    ca_event_array = np.zeros(ca_trace.shape)
                    if ca_data_type == S2PData.CaDataType.EVENTS_BIN:
                        ca_event_array[ca_events.onsets] = 1 * 60 * exp.SciscanSettings.frames_p_sec # convert to events per min
                        roi_data = ca_event_array
                        ylabel_text = "Events/min"
                    elif ca_data_type == S2PData.CaDataType.EVENTS_AMP:
                        ca_event_array[ca_events.onsets] = ca_events.amps * 60 * exp.SciscanSettings.frames_p_sec # convert to events per min
                        roi_data = ca_event_array
                        ylabel_text = "Event dF/F0 per min"

                    print("Plotting {} data roi {} {} events".format(roi_type, i_roi, np.sum(ca_event_array > 0)))

                    smooth_data = True
                    plot_error = True
                    bins_speed = bins_speed_events
                else:
                    raise Exception()

                ca_data_label = S2PData.getCaDataTypeLabelShort(ca_data_type)
                time_offset_set_roi = time_offset_set


                # Cross correlation
                if do_crosscorr:
                    max_lags_time = 10
                    peak_max_range_time = 3
                    baseline_min_range = 30
                    ca_trace_for_corr = roi_data

                    fig = plt.figure(tight_layout=True)
                    max_lags = round(max_lags_time * exp.SciscanSettings.frames_p_sec)
                    lags, c, line, b = plt.xcorr(ca_trace_for_corr,
                                                 df_resampled[bu.SPEED_FILT_GRAD],
                                                 # df_resampled[bu.ACC_GRAD].abs(),
                                                 maxlags=max_lags)  # ,
                    # usevlines=False,
                    # marker='.')

                    # baseline_min_range = round(30 * exp.SciscanSettings.frames_p_sec)
                    # baseline_range = np.abs(lags) >= baseline_min_range
                    # baseline_mean = np.mean(c[baseline_range])
                    # baseline_prc = np.percentile(c[baseline_range], [1, 99])
                    # baseline_upp = baseline_prc[1]
                    # baseline_low = baseline_prc[0]

                    # Look for the leak anywhere.
                    # c_range = c
                    # lags_range = lags

                    # Look within a specific time window
                    peak_max_range = round(peak_max_range_time * exp.SciscanSettings.frames_p_sec)
                    peak_check_range = np.abs(lags) <= peak_max_range
                    c_range = c[peak_check_range]
                    lags_range = lags[peak_check_range]

                    # Get the peak by finding the largest
                    pos_cor = True  # c_mean > 0.3
                    if pos_cor:
                        i_peak = np.argmax(np.abs(c_range))
                    else:
                        i_peak = np.argmin(np.abs(c_range))
                    peak_lag = lags_range[i_peak]
                    peak_corr = c_range[i_peak]

                    # Get the peak by finding the first significant peak
                    # peaks, _ = find_peaks(c_range, height=baseline_upp)
                    # i_peak = np.argmin(np.abs(lags_range[peaks]))
                    #
                    # peak_lag = lags_range[peaks][i_peak]
                    # peak_corr = c_range[peaks][i_peak]

                    if peak_corr > 0:
                        peak_line_end = plt.ylim()[1]
                    else:
                        peak_line_end = plt.ylim()[0]

                    peak_time = peak_lag / exp.SciscanSettings.frames_p_sec

                    c_mean = np.mean(c)

                    plt.plot([peak_lag, peak_lag], [peak_corr, peak_line_end], 'k')
                    # plt.plot([np.min(lags), np.max(lags)], [baseline_mean, baseline_mean], 'k')
                    # plt.plot([np.min(lags), np.max(lags)], [baseline_upp, baseline_upp], 'k:')
                    # plt.plot([np.min(lags), np.max(lags)], [baseline_low, baseline_low], 'k:')
                    plt.title("Peak corr {:.3f} at {:.3f}s".format(peak_corr, peak_time))
                    plt.xlabel("Time lag (s)")
                    plt.ylabel("Correlation")
                    sci_fps = round(exp.SciscanSettings.frames_p_sec)
                    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x / sci_fps), ',')))
                    plot_img_path = os.path.join(plot_ccorr_dir, "{roi}.roi-{i}.speed-xcorr.{ca}.png"
                                                 .format(roi=roi_type,
                                                         i=i_roi,
                                                         ca="deconv"))

                    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

                    plt.cla()
                    plt.clf()
                    plt.close('all')

                if add_peak_offset:
                    time_offset_set_roi = time_offset_set_roi + [peak_lag]





                for time_offset in time_offset_set_roi:

                    roi_data_roll = np.roll(roi_data, time_offset)
                    # Smooth the trace?
                    if smooth_data:
                        #roi_data_roll = sp.ndimage.filters.convolve1d(roi_data_roll, [1, 1, 1], mode='nearest')
                        roi_data_roll = sp.ndimage.gaussian_filter1d(roi_data_roll, sigma=3, mode='nearest')

                    df_resampled[roi_label] = roi_data_roll

                    behave_trace = df_resampled[bu.SPEED_FILT_GRAD].to_numpy()
                    move_thresh = 0.5
                    move_indexes = np.abs(behave_trace) > move_thresh

                    df_moving = df_resampled.copy().iloc[move_indexes]

                    df_light = df_resampled.copy().iloc[light_indexes]
                    df_dark = df_resampled.copy().iloc[dark_indexes]

                    df_moving_light = df_resampled.copy().iloc[np.logical_and(move_indexes, light_indexes)]
                    df_moving_dark = df_resampled.copy().iloc[np.logical_and(move_indexes, dark_indexes)]

                    # HD/Heading generic function
                    if plot_hd:
                        tu.plot_hdheading(df_moving, df_moving_light, df_moving_dark, bu.HD_ABS_FILT, "HD",
                                          roi_label, bins_hd,
                                          roi_type, i_roi, ca_data_label, time_offset,
                                          plot_combined, plot_tune_dir, plot_lightvsdark, plot_hd_cart, plot_hd_polar,
                                          exp.SciscanSettings.frames_p_sec,
                                          soma_dend_pairs, plot_tune_pairs_dir, plot_tune_hdmod_dir,
                                          ylabel_text, linewidth, dpi)

                    # tu.plot_hdheading(df_moving, df_moving_light, df_moving_dark, bu.HEADING_EGO_ABS_FILT, "Heading-ego",
                    #                   roi_label, bins_hd,
                    #                   roi_type, i_roi, ca_data_label, time_offset,
                    #                   plot_combined, plot_tune_dir, plot_lightvsdark, plot_hd_cart, plot_hd_polar,
                    #                   exp.SciscanSettings.frames_p_sec,
                    #                   somadend_pairs, plot_tune_pairs_dir, plot_tune_hdmod_dir,
                    #                   ylabel_text, linewidth, dpi)
                    #
                    # tu.plot_hdheading(df_moving, df_moving_light, df_moving_dark, bu.HEADING_ALLO_ABS_FILT, "Heading-allo",
                    #                   roi_label, bins_hd,
                    #                   roi_type, i_roi, ca_data_label, time_offset,
                    #                   plot_combined, plot_tune_dir, plot_lightvsdark, plot_hd_cart, plot_hd_polar,
                    #                   exp.SciscanSettings.frames_p_sec,
                    #                   somadend_pairs, plot_tune_pairs_dir, plot_tune_hdmod_dir,
                    #                   ylabel_text, linewidth, dpi)



                    # AHV tuning
                    if plot_ahv:
                        df_grp = utils.data.df_grp_bin(df_resampled, bu.AHV_FILT_GRAD, roi_label, ["mean", "sem", "count"], bins_ahv)
                        df_grp_light = utils.data.df_grp_bin(df_light, bu.AHV_FILT_GRAD, roi_label, ["mean", "sem", "count"], bins_ahv)
                        df_grp_dark = utils.data.df_grp_bin(df_dark, bu.AHV_FILT_GRAD, roi_label, ["mean", "sem", "count"], bins_ahv)

                        # Combined
                        if plot_combined:
                            fig = plt.figure(tight_layout=True)
                            plt.errorbar(bins_ahv[:-1], df_grp["mean"], yerr=df_grp["sem"], linewidth=linewidth, color='0')
                            plt.xlabel("Angular head velocity (°/s)")
                            plt.ylabel(ylabel_text)
                            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, hd_p: format(int(x))))
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().yaxis.set_ticks_position('left')
                            plt.gca().xaxis.set_ticks_position('bottom')
                            plt.tight_layout()

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.AHV.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)

                            plt.cla()
                            plt.clf()
                            plt.close('all')

                        # Light vs dark
                        if plot_lightvsdark:
                            fig = plt.figure(tight_layout=True)
                            plt.errorbar(bins_ahv[:-1], df_grp_light["mean"], yerr=df_grp_light["sem"], label='light', linewidth=linewidth, color='0.5')
                            plt.errorbar(bins_ahv[:-1], df_grp_dark["mean"], yerr=df_grp_dark["sem"], label='dark', linewidth=linewidth, color='0')
                            plt.xlabel("Angular head velocity (°/s)")
                            plt.ylabel(ylabel_text)
                            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, hd_p: format(int(x))))
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().yaxis.set_ticks_position('left')
                            plt.gca().xaxis.set_ticks_position('bottom')
                            plt.tight_layout()

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.AHV.{ca}.offset_{off:+000}.ld.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)

                            plt.cla()
                            plt.clf()
                            plt.close('all')

                        # Abs AVH tuning
                        df_grp = utils.data.df_grp_bin(df_resampled, bu.AHV_FILT_GRAD, roi_label, ["mean", "sem", "count"],
                                                       bins_ahv_abs, absolute=True)
                        df_grp_light = utils.data.df_grp_bin(df_light, bu.AHV_FILT_GRAD, roi_label, ["mean", "sem", "count"],
                                                             bins_ahv_abs, absolute=True)
                        df_grp_dark = utils.data.df_grp_bin(df_dark, bu.AHV_FILT_GRAD, roi_label, ["mean", "sem", "count"],
                                                            bins_ahv_abs, absolute=True)

                        # Combined
                        if plot_combined:
                            fig = plt.figure(tight_layout=True)
                            plt.errorbar(bins_ahv_abs[:-1], df_grp["mean"], yerr=df_grp["sem"], label='light',
                                         linewidth=linewidth, color='0')
                            plt.xlabel("Absolute angular head velocity (°/s)")
                            plt.ylabel(ylabel_text)
                            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, hd_p: format(int(x))))
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().yaxis.set_ticks_position('left')
                            plt.gca().xaxis.set_ticks_position('bottom')
                            plt.tight_layout()

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.AHVABS.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)
                            plt.cla()
                            plt.clf()
                            plt.close('all')

                        # Light vs Dark
                        if plot_lightvsdark:
                            fig = plt.figure(tight_layout=True)
                            plt.errorbar(bins_ahv_abs[:-1], df_grp_light["mean"], yerr=df_grp_light["sem"], label='light', linewidth=linewidth, color='0.5')
                            plt.errorbar(bins_ahv_abs[:-1], df_grp_dark["mean"], yerr=df_grp_dark["sem"], label='dark', linewidth=linewidth, color='0')
                            plt.xlabel("Absolute angular head velocity (°/s)")
                            plt.ylabel(ylabel_text)
                            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, hd_p: format(int(x))))
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().yaxis.set_ticks_position('left')
                            plt.gca().xaxis.set_ticks_position('bottom')
                            plt.tight_layout()

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.AHVABS.{ca}.offset_{off:+000}.ld.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)
                            plt.cla()
                            plt.clf()
                            plt.close('all')

                    # Speed tuning
                    if plot_speed:
                        df_grp = utils.data.df_grp_bin(df_resampled, bu.SPEED_FILT_GRAD, roi_label, ["mean", "sem", "count"], bins_speed)
                        df_grp_light = utils.data.df_grp_bin(df_light, bu.SPEED_FILT_GRAD, roi_label, ["mean", "sem", "count"], bins_speed)
                        df_grp_dark = utils.data.df_grp_bin(df_dark, bu.SPEED_FILT_GRAD, roi_label, ["mean", "sem", "count"], bins_speed)

                        # Combined
                        if plot_combined:
                            fig = plt.figure(tight_layout=True)
                            if plot_error:
                                plt.errorbar(bins_speed[:-1], df_grp["mean"], yerr=df_grp["sem"], linewidth=linewidth, color='0')
                            else:
                                plt.plot(bins_speed[:-1], df_grp["mean"], linewidth=linewidth, color='0')
                            plt.xlabel("Head speed (cm/s)")
                            plt.ylabel(ylabel_text)
                            #plt.xscale('log')
                            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x))))
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().yaxis.set_ticks_position('left')
                            plt.gca().xaxis.set_ticks_position('bottom')
                            plt.tight_layout()

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.speed.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))

                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)

                            plt.cla()
                            plt.clf()
                            plt.close('all')

                        # Light vs dark
                        if plot_lightvsdark:
                            fig = plt.figure(tight_layout=True)
                            if plot_error:
                                plt.errorbar(bins_speed[:-1], df_grp_light["mean"], yerr=df_grp_light["sem"], linewidth=linewidth,
                                             color='0.5')
                                plt.errorbar(bins_speed[:-1], df_grp_dark["mean"], yerr=df_grp_dark["sem"], linewidth=linewidth,
                                             color='0')
                            else:
                                plt.plot(bins_speed[:-1], df_grp_light["mean"], linewidth=linewidth, color='0.5')
                                plt.plot(bins_speed[:-1], df_grp_dark["mean"], linewidth=linewidth, color='0')

                            plt.xlabel("Head speed (cm/s)")
                            plt.ylabel(ylabel_text)
                            #plt.xscale('log')
                            plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x))))
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().yaxis.set_ticks_position('left')
                            plt.gca().xaxis.set_ticks_position('bottom')
                            plt.tight_layout()

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.speed.{ca}.offset_{off:+000}.ld.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))

                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)

                            plt.cla()
                            plt.clf()
                            plt.close('all')





                    # # HD and AHV.
                    # df_hd = pd.cut(df_resampled[bu.HD_ABS_FILT], bins=bins_hdahv)
                    # df_hdahv = pd.cut(df_resampled[bu.AHV_FILT_GRAD], bins=bins_ahv)
                    # df_grp_hd_ahv = df_resampled.groupby([df_hd, df_hdahv])[roi_label].agg(["mean", "sem", "count"])
                    #
                    # map_hd_ahv_count = df_grp_hd_ahv["count"].values.reshape((bins_hdahv.size - 1, bins_ahv.size - 1))
                    # map_hd_ahv_mean = df_grp_hd_ahv["mean"].values.reshape((bins_hdahv.size - 1, bins_ahv.size - 1))
                    # map_hd_ahv_sem = df_grp_hd_ahv["sem"].values.reshape((bins_hdahv.size - 1, bins_ahv.size - 1))
                    # map_hd_ahv_mean = sp.ndimage.filters.gaussian_filter(map_hd_ahv_mean, sigma=1,
                    #                                                      mode=['wrap', 'nearest'])
                    # map_hd_ahv_mean[map_hd_ahv_count == 0] = np.NaN
                    #
                    # fig = plt.figure(tight_layout=True)
                    #
                    # plt.imshow(map_hd_ahv_mean.T, aspect='auto', origin='lower')
                    # cbar = plt.colorbar()
                    # cbar.ax.set_ylabel(ylabel_text, rotation=90)
                    # plt.xlabel("Head directrion (°)")
                    # plt.ylabel("Angular head velocity (°/s)")
                    # #plt.yscale('symlog')
                    #
                    # xticks = np.linspace(0, bins_hdahv.size - 1, 4 + 1)
                    # xlabels = np.linspace(0, 360, 4 + 1)
                    # plt.xticks(ticks=xticks, labels=xlabels)
                    # # yticks = np.linspace(0, bins_ahv.size - 1, 8 + 1)
                    # #ylabels = np.round(np.linspace(np.min(bins_ahv), np.max(bins_ahv), 8 + 1))
                    # #ylabels = np.round(np.logspace(np.log10(np.min(bins_ahv))), np.max(bins_ahv), 8 + 1)), 2)
                    # #plt.yticks(ticks=yticks, labels=ylabels)
                    # plt.tick_params(
                    #     axis='y',  # changes apply to the y-axis
                    #     which='both',  # both major and minor ticks are affected
                    #     bottom=False,  # ticks along the bottom edge are off
                    #     top=False,  # ticks along the top edge are off
                    #     labelleft=False)  # labels along the bottom edge are off
                    #
                    # plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.HD-AHV.{ca}.offset_{off:+000}.png"
                    #                              .format(roi=roi_type,
                    #                                      i=i_roi,
                    #                                      ca=ca_data_label,
                    #                                      off=time_offset))
                    # fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                    #
                    # plt.cla()
                    # plt.clf()
                    # plt.close('all')
                    #
                    #
                    #
                    # # HD and Abs AHV.
                    # df_hd = pd.cut(df_resampled[bu.HD_ABS_FILT], bins=bins_hdahv)
                    # df_hdahvabs = pd.cut(df_resampled[bu.AHV_FILT_GRAD], bins=bins_ahv_abs)
                    # df_grp_hd_ahvabs = df_resampled.groupby([df_hd, df_hdahvabs])[roi_label].agg(["mean", "sem", "count"])
                    #
                    # map_hd_ahvabs_count = df_grp_hd_ahvabs["count"].values.reshape((bins_hdahv.size - 1, bins_ahv_abs.size - 1))
                    # map_hd_ahvabs_mean = df_grp_hd_ahvabs["mean"].values.reshape((bins_hdahv.size - 1, bins_ahv_abs.size - 1))
                    # map_hd_ahvabs_sem = df_grp_hd_ahvabs["sem"].values.reshape((bins_hdahv.size - 1, bins_ahv_abs.size - 1))
                    # map_hd_ahvabs_mean = sp.ndimage.filters.gaussian_filter(map_hd_ahvabs_mean, sigma=[1, 0.5],
                    #                                                         mode=['wrap', 'nearest'])
                    # map_hd_ahvabs_mean[map_hd_ahvabs_count == 0] = np.NaN
                    #
                    # fig = plt.figure(tight_layout=True)
                    #
                    # plt.imshow(map_hd_ahvabs_mean.T, aspect='auto', origin='lower')
                    # cbar = plt.colorbar()
                    # cbar.ax.set_ylabel(ylabel_text, rotation=90)
                    # plt.xlabel("Head direction (°)")
                    # plt.ylabel("Absolute angular head velocity (°/s)")
                    # #plt.yscale('log')
                    # xticks = np.linspace(0, bins_hdahv.size - 1, 4 + 1)
                    # xlabels = np.linspace(0, 360, 4 + 1)
                    # plt.xticks(ticks=xticks, labels=xlabels)
                    # # yticks = np.linspace(0, bins_ahv_abs.size - 1, 8 + 1)
                    # # ylabels = np.round(np.linspace(np.min(bins_ahv_abs), np.max(bins_ahv_abs), 8 + 1))
                    # # plt.yticks(ticks=yticks, labels=ylabels)
                    # plt.tick_params(
                    #     axis='y',  # changes apply to the x-axis
                    #     which='both',  # both major and minor ticks are affected
                    #     bottom=False,  # ticks along the bottom edge are off
                    #     top=False,  # ticks along the top edge are off
                    #     labelleft=False)  # labels along the bottom edge are off
                    #
                    #
                    # plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.HD-AHVABS.{ca}.offset_{off:+000}.png"
                    #                              .format(roi=roi_type,
                    #                                      i=i_roi,
                    #                                      ca=ca_data_label,
                    #                                      off=time_offset))
                    # fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                    #
                    # plt.cla()
                    # plt.clf()
                    # plt.close('all')

                    # # # HD and speed.
                    # df_hd = pd.cut(df_resampled[bu.HD_ABS_FILT], bins=bins_hdahv)
                    # df_hdspeed = pd.cut(df_resampled[bu.SPEED_FILT_GRAD], bins=bins_speed)
                    # df_grp_hd_speed = df_resampled.groupby([df_hd, df_hdspeed])[roi_label].agg(["mean", "sem", "count"])
                    #
                    # map_hd_speed_count = df_grp_hd_speed["count"].values.reshape((bins_hdahv.size - 1, bins_speed.size - 1))
                    # map_hd_speed_mean = df_grp_hd_speed["mean"].values.reshape((bins_hdahv.size - 1, bins_speed.size - 1))
                    # map_hd_speed_sem = df_grp_hd_speed["sem"].values.reshape((bins_hdahv.size - 1, bins_speed.size - 1))
                    #
                    # map_hd_speed_mean = sp.ndimage.filters.gaussian_filter(map_hd_speed_mean, sigma=1,
                    #                                                        mode=['wrap', 'nearest'])
                    # map_hd_speed_mean[map_hd_speed_count == 0] = np.NaN
                    #
                    # fig = plt.figure(tight_layout=True)
                    #
                    # plt.imshow(map_hd_speed_mean.T, aspect='auto', origin='lower')
                    # cbar = plt.colorbar()
                    # cbar.ax.set_ylabel(ylabel_text, rotation=90)
                    # plt.xlabel("HD (°)")
                    # plt.ylabel("Speed (cm/s)")
                    # #plt.yscale('log')
                    #
                    # xticks = np.linspace(0, bins_hdahv.size - 1, 4 + 1)
                    # xlabels = np.linspace(0, 360, 4 + 1)
                    # plt.xticks(ticks=xticks, labels=xlabels)
                    # plt.tick_params(
                    #     axis='y',  # changes apply to the x-axis
                    #     which='both',  # both major and minor ticks are affected
                    #     bottom=False,  # ticks along the bottom edge are off
                    #     top=False,  # ticks along the top edge are off
                    #     labelleft=False)  # labels along the bottom edge are off
                    #
                    # plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.HD-speed.{ca}.offset_{off:+000}.png"
                    #                              .format(roi=roi_type,
                    #                                      i=i_roi,
                    #                                      ca=ca_data_label,
                    #                                      off=time_offset))
                    # fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
                    #
                    # plt.cla()
                    # plt.clf()
                    # plt.close('all')
                    #
                    # Place fields
                    if plot_place:

                        df_x = pd.cut(df_moving[bu.HEAD_X_FILT_MAZE], bins=bins_x_big)
                        df_y = pd.cut(df_moving[bu.HEAD_Y_FILT_MAZE], bins=bins_y_big)
                        df_grp = df_moving.groupby([df_x, df_y])[roi_label].add_exp_data(["mean", "sem", "count"])

                        map_mean = df_grp["mean"].values.reshape((bins_x_big.size - 1, bins_y_big.size - 1))

                        map_mean_nonan = np.copy(map_mean)
                        map_mean_nonan[np.isnan(map_mean)] = 0
                        map_mean_nonan_prob = map_mean_nonan / np.nansum(map_mean_nonan)
                        occmap_big_perc = occmap_big / np.nansum(occmap_big)
                        occmap_small_perc = occmap_small / np.nansum(occmap_small)
                        # KLD is still wrong
                        kld = utils.stats.kl_divergence(occmap_big_perc, map_mean_nonan_prob)
                        mi = utils.stats.mutual_inf(occmap_big_perc, map_mean_nonan)
                        print(mi, kld)

                        if plot_combined:
                            df_x = pd.cut(df_moving[bu.HEAD_X_FILT_MAZE], bins=bins_x_small)
                            df_y = pd.cut(df_moving[bu.HEAD_Y_FILT_MAZE], bins=bins_y_small)
                            df_grp = df_moving.groupby([df_x, df_y])[roi_label].add_exp_data(["mean", "sem", "count"])

                            map_count = df_grp["count"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))
                            map_mean = df_grp["mean"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))
                            map_sem = df_grp["sem"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))

                            map_mean = tu.filt_img_nan(map_mean, nan_indexes=map_count == 0)

                            fig = plt.figure(tight_layout=True)
                            plt.imshow(map_mean.T)
                            plt.tick_params(
                                axis='both',  # changes apply to the both axis
                                which='both',  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False)  # labels along the bottom edge are off
                            cbar = plt.colorbar()
                            cbar.ax.set_ylabel(ylabel_text, rotation=90)
                            maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
                            plt.plot(maze_bin_sub_small * np.array(maze_poly_x) - 0.5,
                                     maze_bin_sub_small * np.array(maze_poly_y) - 0.5,
                                     'k', linewidth=3)

                            plt.title("KLD={:0.2f} bits".format(kld))

                            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.place.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

                            plt.cla()
                            plt.clf()
                            plt.close('all')

                        # todo
                        if plot_lightvsdark:
                            df_x_light = pd.cut(df_moving_light[bu.HEAD_X_FILT_MAZE], bins=bins_x_small)
                            df_y_light = pd.cut(df_moving_light[bu.HEAD_Y_FILT_MAZE], bins=bins_y_small)
                            df_x_dark= pd.cut(df_moving_dark[bu.HEAD_X_FILT_MAZE], bins=bins_x_small)
                            df_y_dark = pd.cut(df_moving_dark[bu.HEAD_Y_FILT_MAZE], bins=bins_y_small)
                            df_grp_light = df_moving_light.groupby([df_x_light, df_y_light])[roi_label].add_exp_data(["mean", "count"])
                            df_grp_dark = df_moving_dark.groupby([df_x_dark, df_y_dark])[roi_label].add_exp_data(["mean", "count"])

                            map_count_light = df_grp_light["count"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))
                            map_mean_light = df_grp_light["mean"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))
                            map_count_dark = df_grp_dark["count"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))
                            map_mean_dark = df_grp_dark["mean"].values.reshape((bins_x_small.size - 1, bins_y_small.size - 1))

                            map_mean_light = tu.filt_img_nan(map_mean_light, nan_indexes=map_count_light == 0)
                            map_mean_dark = tu.filt_img_nan(map_mean_dark, nan_indexes=map_count_dark == 0)

                            maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy

                            resp_vmax = np.nanmax([np.nanmax(map_mean_light), np.nanmax(map_mean_dark)])
                            diff_vmax = np.nanmax(np.abs(map_mean_light - map_mean_dark))


                            fig = plt.figure(tight_layout=True)
                            plt.imshow((map_mean_light).T, vmin=0, vmax=resp_vmax)
                            plt.tick_params(
                                axis='both',  # changes apply to the both axis
                                which='both',  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False)  # labels along the bottom edge are off
                            plt.plot(maze_bin_sub_small * np.array(maze_poly_x) - 0.5,
                                     maze_bin_sub_small * np.array(maze_poly_y) - 0.5,
                                     'k', linewidth=3)

                            cbar = plt.colorbar()
                            cbar.ax.set_ylabel(ylabel_text, rotation=90)

                            plot_img_path = os.path.join(plot_tune_dir,
                                                         "{roi}.roi-{i}.place-01-light.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

                            plt.cla()
                            plt.clf()
                            plt.close('all')

                            fig = plt.figure(tight_layout=True)
                            plt.imshow((map_mean_dark).T, vmin=0, vmax=resp_vmax)
                            plt.tick_params(
                                axis='both',  # changes apply to the both axis
                                which='both',  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False)  # labels along the bottom edge are off

                            plt.plot(maze_bin_sub_small * np.array(maze_poly_x) - 0.5,
                                     maze_bin_sub_small * np.array(maze_poly_y) - 0.5,
                                     'k', linewidth=3)

                            cbar = plt.colorbar()
                            cbar.ax.set_ylabel(ylabel_text, rotation=90)

                            plot_img_path = os.path.join(plot_tune_dir,
                                                         "{roi}.roi-{i}.place-02-dark.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

                            plt.cla()
                            plt.clf()
                            plt.close('all')

                            fig = plt.figure(tight_layout=True)
                            plt.imshow((map_mean_light - map_mean_dark).T, cmap='seismic', vmin=-diff_vmax, vmax=diff_vmax)
                            plt.tick_params(
                                axis='both',  # changes apply to the both axis
                                which='both',  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                left=False,
                                labelleft=False,
                                labelbottom=False)  # labels along the bottom edge are off

                            plt.plot(maze_bin_sub_small * np.array(maze_poly_x) - 0.5,
                                     maze_bin_sub_small * np.array(maze_poly_y) - 0.5,
                                     'k', linewidth=3)

                            cbar = plt.colorbar()
                            cbar.ax.set_ylabel("Diff", rotation=90)

                            plot_img_path = os.path.join(plot_tune_dir,
                                                         "{roi}.roi-{i}.place-03-diff.{ca}.offset_{off:+000}.png"
                                                         .format(roi=roi_type,
                                                                 i=i_roi,
                                                                 ca=ca_data_label,
                                                                 off=time_offset))
                            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

                            plt.cla()
                            plt.clf()
                            plt.close('all')








        if plot_evoked:
            ca_data_type = S2PData.CaDataType.DFONF0
            print("Detecting and plotting {} {} movement evoked responses (speed)".format(roi_type,
                                                                                          S2PData.getCaDataTypeLabelShort(ca_data_type)))


            s2p_start_time = exp.SciscanSettings.sci_frame_times[0] #- (1.0 / exp.SciscanSettings.frames_p_sec)
            plot_label = "speed"

            # behave_trace = df_move[bu.SPEED_FILT_GRAD].to_numpy()
            # behave_trace_time = exp.cam_trigger_times
            # frame_interval = exp.tracking_video.fps

            behave_trace = df_resampled[bu.SPEED_FILT_GRAD].to_numpy()
            behave_trace_time = exp.SciscanSettings.sci_frame_times
            frame_interval = exp.SciscanSettings.frames_p_sec

            move_thresh = 0.5
            time_pre = 1
            time_post = 4
            move_indexes = np.abs(behave_trace) > move_thresh
            stat_indexes = np.logical_not(move_indexes)

            move_bouts_indexes = misc.get_crossings(move_indexes.astype('uint8'), 0.9)
            stat_bouts_indexes = misc.get_crossings(stat_indexes.astype('uint8'), 0.9)

            bouts = np.concatenate((stat_bouts_indexes, move_bouts_indexes))
            bouts.sort(kind='mergesort')

            bouts_intervals = np.diff(bouts)

            if stat_bouts_indexes[0] < move_bouts_indexes[0]:
                move_bouts_intervals = bouts_intervals[1::2]
                stat_bouts_intervals = bouts_intervals[::2]
            else:
                move_bouts_intervals = bouts_intervals[::2]
                stat_bouts_intervals = bouts_intervals[1::2]


            move_intervals_time = move_bouts_intervals / frame_interval
            stat_intervals_time = stat_bouts_intervals / frame_interval

            long_stat_bouts_indexes = np.where(stat_intervals_time >= time_pre)[0]

            long_stat_bouts = stat_bouts_indexes[long_stat_bouts_indexes]
            long_stat_bouts_end = move_bouts_indexes[long_stat_bouts_indexes]

            # Code does work, it finds bouts correctly.
            # bout_i = 1
            # start_i = long_stat_bouts_end[bout_i] - 100
            # end_i = long_stat_bouts_end[bout_i] + 200
            # plt.figure()
            # move_trace = behave_trace[start_i:end_i]
            # plt.plot(move_trace)
            # plt.show()
            # plt.close('all')
            #raise Exception()
            #
            # print(move_bouts_indexes.size, stat_bouts_indexes.size)


            trial_times = behave_trace_time[long_stat_bouts_end]

            s2p_data.plot_evoked(plot_dir=plot_evoke_dir,
                                 ca_data_type=ca_data_type,
                                 s2p_start_time=s2p_start_time,
                                 trial_times=trial_times,
                                 time_pre=time_pre,
                                 time_trial=0.1,
                                 time_post=time_post,
                                 plot_good=True,
                                 plot_bad=False,
                                 ignore_no_pre_time=True,
                                 ignore_no_post_time=True,
                                 plot_label=plot_label,
                                 pre_f0=True,
                                 plot_as_trace=True,
                                 behave_trace=behave_trace,
                                 behave_trace_time=behave_trace_time)

            # print("Detecting and plotting {} {} movement evoked responses (AHV)".format(roi_type,
            #                                                                             S2PData.getCaDataTypeLabelShort(
            #                                                                                 ca_data_type)))
            # move_indexes = m2putils.get_crossings(np.abs(df_move[bu.AHV_FILT_GRAD].to_numpy()), 2)
            # plot_label = "ahv"
            # s2p_data.plot_evoked(plot_dir=plot_evoke_dir,
            #                      ca_data_type=ca_data_type,
            #                      s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
            #                      trial_times=exp.cam_trigger_times[move_indexes],
            #                      time_pre=time_pre,
            #                      time_trial=time_trial,
            #                      time_post=time_post,
            #                      plot_good=True,
            #                      plot_bad=False,
            #                      ignore_no_pre_time=True,
            #                      plot_label=plot_label,
            #                      pre_stim_f0=False,
            #                      pre_stim_deconv=False)

if plot_event_dists:
    print("Plotting events distributions")
    n_events_stat = []
    n_events_move = []
    n_events_light = []
    n_events_dark = []

    amp_events_stat = []
    amp_events_move = []
    amp_events_light = []
    amp_events_dark = []
    stat_indexes = df_resampled[bu.SPEED_FILT_GRAD] < speed_min
    move_indexes = np.logical_not(stat_indexes)
    soma_indexes = []
    dend_indexes = []

    for s2p_data in s2p_data_set:
        roi_type = s2p_data.mode

        (n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad) = s2p_data.get_n_roi_plot(plot_good=True, plot_bad=False)

        for i_roi in range(0, n_roi_cand):
            is_good = s2p_data.iscell[i_roi, 0] == 1
            if not is_good:
                continue

            print("Plotting {} data roi {}".format(roi_type, i_roi))

            soma_indexes.append(roi_type == "soma")
            dend_indexes.append(roi_type == "dend")

            ca_trace = s2p_data.dFonF0[i_roi, :]

            # Get ca events
            ca_events = utils.ca.get_ca_events(ca_trace,
                                               smooth_sigma=3,
                                               prc_mean=40,
                                               prc_low=10,
                                               prc_high=90,
                                               prob_onset=0.2,
                                               prob_offset=0.7)



            event_trace = np.zeros(df_resampled.shape[0], dtype='bool')
            event_trace[ca_events.onsets] = True
            event_amp_trace = np.zeros(df_resampled.shape[0])
            event_amp_trace[ca_events.onsets] = ca_events.amps

            n_stat_events = np.sum(event_trace[stat_indexes])
            n_move_events = np.sum(event_trace[move_indexes])

            amp_stat_events = event_amp_trace[np.logical_and(event_trace, stat_indexes)]
            amp_move_events = event_amp_trace[np.logical_and(event_trace, move_indexes)]
            meanamp_stat_events = np.mean(amp_stat_events)
            meanamp_move_events = np.mean(amp_move_events)

            n_events_stat.append(n_stat_events)
            n_events_move.append(n_move_events)
            amp_events_stat.append(meanamp_stat_events)
            amp_events_move.append(meanamp_move_events)

            n_light_events = np.sum(event_trace[np.logical_and(light_indexes, move_indexes.values)])
            n_dark_events = np.sum(event_trace[np.logical_and(dark_indexes, move_indexes.values)])

            #amp_light_events = event_amp_trace[np.logical_and.reduce((event_trace, light_indexes, move_indexes))]
            #amp_dark_events = event_amp_trace[np.logical_and.reduce((event_trace, dark_indexes, move_indexes))]
            amp_light_events = event_amp_trace[np.logical_and(event_trace, light_indexes)]
            amp_dark_events = event_amp_trace[np.logical_and(event_trace, dark_indexes)]
            if amp_light_events.size > 0:
                meanamp_light_events = np.mean(amp_light_events)
            else:
                meanamp_light_events = 0
            if amp_dark_events.size > 0:
                meanamp_dark_events = np.mean(amp_dark_events)
            else:
                meanamp_dark_events = 0


            n_events_light.append(n_light_events)
            n_events_dark.append(n_dark_events)
            amp_events_light.append(meanamp_light_events)
            amp_events_dark.append(meanamp_dark_events)

            fig = plt.figure(tight_layout=True)
            plt.hist(amp_light_events, histtype=u'step', zorder=1)
            plt.hist(amp_dark_events, zorder=0)
            plt.xlabel("Event amplitude (df_move/F)")
            plt.ylabel("# events")
            plot_img_path = os.path.join(plot_event_amps_dir, "light-dark-event-amp-hist-{roi}-{i}.png".format(roi=roi_type,
                                                               i=i_roi))

            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
            plt.cla()
            plt.clf()
            plt.close('all')

    n_events_stat = np.array(n_events_stat)
    n_events_move = np.array(n_events_move)
    amp_events_stat = np.array(amp_events_stat)
    amp_events_move = np.array(amp_events_move)

    n_events_light = np.array(n_events_light)
    n_events_dark = np.array(n_events_dark)
    amp_events_light = np.array(amp_events_light)
    amp_events_dark = np.array(amp_events_dark)

    soma_indexes = np.array(soma_indexes)
    dend_indexes = np.array(dend_indexes)

    n_events_stat = 60 * (n_events_stat / (np.sum(stat_indexes) / exp.SciscanSettings.frames_p_sec))
    n_events_move = 60 * (n_events_move / (np.sum(move_indexes) / exp.SciscanSettings.frames_p_sec))
    n_events_light = 60 * (n_events_light / (np.sum(light_indexes) / exp.SciscanSettings.frames_p_sec))
    n_events_dark = 60 * (n_events_dark / (np.sum(dark_indexes) / exp.SciscanSettings.frames_p_sec))

    fig = plt.figure(tight_layout=True)
    plt.hist(n_events_move - n_events_stat)
    plt.xlabel("Events per minute")
    plt.ylabel("# soma/dendrite")
    plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "stat-move-event-diff-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(n_events_move)
    plt.xlabel("Events per minute")
    plt.ylabel("# soma/dendrite")
    plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "move-event-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(n_events_stat)
    plt.xlabel("Events per minute")
    plt.ylabel("# soma/dendrite")
    plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "stat-event-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(n_events_stat[soma_indexes], n_events_move[soma_indexes], color='blue')
    plt.scatter(n_events_stat[dend_indexes], n_events_move[dend_indexes], color='red')
    plt.xlabel("Stationary events per minute")
    plt.ylabel("Moving events per minute")
    max_axis = np.max([plt.xlim()[1], plt.ylim()[1]])
    plt.xlim(left=0, right=max_axis)
    plt.ylim(bottom=0, top=max_axis)
    plt.plot([0, max_axis], [0, max_axis], 'k')
    plot_img_path = os.path.join(plot_pop_dir, "state-move-event-scatter.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(amp_events_move[soma_indexes], amp_events_stat[soma_indexes], color='blue')
    plt.scatter(amp_events_move[dend_indexes], amp_events_stat[dend_indexes], color='red')
    plt.xlabel("Stationary mean event amplitude (df_move/F)")
    plt.ylabel("Moving mean event amplitude (df_move/F)")
    max_axis = 7 #np.max([plt.xlim()[1], plt.ylim()[1]])
    plt.xlim(left=0, right=max_axis)
    plt.ylim(bottom=0, top=max_axis)
    plt.plot([0, max_axis], [0, max_axis], 'k')
    plot_img_path = os.path.join(plot_pop_dir, "stat-move-event-amp-scatter.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(n_events_move[soma_indexes] - n_events_stat[soma_indexes],
                amp_events_move[soma_indexes] - amp_events_stat[soma_indexes], color='blue')
    plt.scatter(n_events_move[dend_indexes] - n_events_stat[dend_indexes],
                amp_events_move[dend_indexes] - amp_events_stat[dend_indexes], color='red')
    plt.xlabel("Light - dark (events/min))")
    plt.ylabel("Light - dark mean event amplitude (df_move/F)")
    plt.xlim(left=-3, right=3)
    plt.ylim(bottom=-3, top=3)
    plt.gca().set_aspect(1)
    plt.gca().axhline(y=0, color='k')
    plt.gca().axvline(x=0, color='k')
    plot_img_path = os.path.join(plot_pop_dir, "stat-move-event-amp-diff-scatter.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    #### Light vs dark


    fig = plt.figure(tight_layout=True)
    plt.hist(n_events_light - n_events_dark)
    plt.xlabel("Events per minute")
    plt.ylabel("# soma/dendrite")
    plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "light-dark-event-diff-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(n_events_light)
    plt.xlabel("Events per minute")
    plt.ylabel("# soma/dendrite")
    plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "light-event-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(n_events_dark)
    plt.xlabel("Events per minute")
    plt.ylabel("# soma/dendrite")
    plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "dark-event-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(n_events_dark[soma_indexes], n_events_light[soma_indexes], color='blue')
    plt.scatter(n_events_dark[dend_indexes], n_events_light[dend_indexes], color='red')
    plt.xlabel("Dark events per minute")
    plt.ylabel("Light events per minute")
    max_axis = np.max([plt.xlim()[1], plt.ylim()[1]])
    plt.xlim(left=0, right=max_axis)
    plt.ylim(bottom=0, top=max_axis)
    plt.plot([0, max_axis], [0, max_axis], 'k')
    plot_img_path = os.path.join(plot_pop_dir, "light-dark-event-scatter.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(amp_events_dark[soma_indexes], amp_events_light[soma_indexes], color='blue')
    plt.scatter(amp_events_dark[dend_indexes], amp_events_light[dend_indexes], color='red')
    plt.xlabel("Dark mean event amplitude (df_move/F)")
    plt.ylabel("Light mean event amplitude (df_move/F)")
    max_axis = 7 #np.max([plt.xlim()[1], plt.ylim()[1]])
    plt.xlim(left=0, right=max_axis)
    plt.ylim(bottom=0, top=max_axis)
    plt.plot([0, max_axis], [0, max_axis], 'k')
    plot_img_path = os.path.join(plot_pop_dir, "light-dark-event-amp-scatter.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(n_events_light[soma_indexes] - n_events_dark[soma_indexes],
                amp_events_light[soma_indexes] - amp_events_dark[soma_indexes], color='blue')
    plt.scatter(n_events_light[dend_indexes] - n_events_dark[dend_indexes],
                amp_events_light[dend_indexes] - amp_events_dark[dend_indexes], color='red')
    plt.xlabel("Light - dark (events/min))")
    plt.ylabel("Light - dark mean event amplitude (df_move/F)")
    plt.ylim(bottom=-3, top=3)
    plt.gca().axhline(y=0, color='k')
    plt.gca().axvline(x=0, color='k')
    plot_img_path = os.path.join(plot_pop_dir, "light-dark-event-amp-diff-scatter.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(amp_events_light - amp_events_dark)
    plt.xlabel("Mean event amplitude (df_move/F)")
    plt.ylabel("# soma/dendrite")
    #plt.xlim([-1, 4])
    plot_img_path = os.path.join(plot_pop_dir, "light-dark-event-amp-diff-hist.png")
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')


if plot_spatial_traces:
    print("Plotting spatial traces and events")
    def plot_spat_trace(event_colors, colorbar_label, file_label):
        # Plot events colored by amplitude.
        fig = plt.figure(tight_layout=True)

        plt.plot(df_resampled[bu.HEAD_X_FILT_MAZE].values,
                 df_resampled[bu.HEAD_Y_FILT_MAZE].values,
                 color='0.5', alpha=0.5, zorder=1)

        plt.scatter(df_resampled[bu.HEAD_X_FILT_MAZE].values[ca_events.onsets],
                    df_resampled[bu.HEAD_Y_FILT_MAZE].values[ca_events.onsets],
                    c=event_colors,
                    zorder=2)

        cbar = plt.colorbar()
        cbar.ax.set_ylabel(colorbar_label, rotation=90)

        plt.tick_params(
            axis='both',  # changes apply to the both axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the left edge are off
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off

        maze_poly_x, maze_poly_y = maze_poly.exterior.coords.xy
        plt.plot(np.array(maze_poly_x), np.array(maze_poly_y), 'k', linewidth=3)
        plt.gca().set_aspect(1)

        plot_img_path = os.path.join(plot_spat_trace_dir, "{roi}.roi-{i}.place-trace-{filelabel}.png"
                                     .format(roi=roi_type,
                                             i=i_roi,
                                             filelabel=file_label))
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')



    for s2p_data in s2p_data_set:
        roi_type = s2p_data.mode
        (n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad) = s2p_data.get_n_roi_plot(plot_good=True, plot_bad=False)
        for i_roi in range(0, n_roi_cand):
            is_good = s2p_data.iscell[i_roi, 0] == 1
            if not is_good:
                continue

            ca_trace = s2p_data.dFonF0[i_roi, :]

            # Get ca events
            ca_events = utils.ca.get_ca_events(ca_trace,
                                               smooth_sigma=3,
                                               prc_mean=40,
                                               prc_low=10,
                                               prc_high=90,
                                               prob_onset=0.2,
                                               prob_offset=0.7)



            plot_spat_trace(ca_events.amps, 'dF/F', "amp")
            # plot_spat_trace(df_resampled[bu.HD_ABS_FILT].values[ca_events.onsets], 'Head direction (°)', "hd")
            #
            # # ahv = df_resampled[bu.AHV_FILT_GRAD].values
            # # ahv[np.abs(ahv) > 500] = 500
            # # plot_spat_trace(ahv[ca_events.onsets], 'AHV (°/s)', "ahv")
            # plot_spat_trace(df_resampled[bu.SPEED_FILT_GRAD].values[ca_events.onsets], 'Speed (cm/s)', "speed")


if make_event_gifs:
    print("Plotting event gifs")
    mov_halfres_scifps_name = cam.file_name_base + "-halfres_scifps.mp4"
    mov_halfres_scifps_path = os.path.join(video_path, mov_halfres_scifps_name)

    for s2p_data in s2p_data_set:
        roi_type = s2p_data.mode

        (n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad) = s2p_data.get_n_roi_plot(plot_good=True, plot_bad=False)

        for i_roi in range(0, n_roi_cand):
            is_good = s2p_data.iscell[i_roi, 0] == 1
            if not is_good:
                continue

            ca_trace = s2p_data.dFonF0[i_roi, :]

            # Get ca events
            ca_events = utils.ca.get_ca_events(ca_trace,
                                               smooth_sigma=3,
                                               prc_mean=40,
                                               prc_low=10,
                                               prc_high=90,
                                               prob_onset=0.2,
                                               prob_offset=0.7)

            for i_event, event_index in enumerate(ca_events.onsets):
                pre_frames = 10
                post_frames = 20

                i_start = event_index - pre_frames
                i_end = event_index + post_frames

                ca_trace_event = ca_trace[i_start:i_end]
                light_on_trace = df_resampled[bu.LIGHT_ON].iloc[i_start:i_end].values
                trace_max = np.max(ca_trace_event)

                if trace_max < 2:
                    continue

                output_gif = os.path.join(plot_spat_trace_dir, "{}.{}.{}.gif".format(roi_type, i_roi, i_event))

                (
                    ffmpeg
                        .input(mov_halfres_scifps_path)
                        .trim(start_frame=i_start, end_frame=i_end)
                        .output(output_gif, loop=0)
                        .overwrite_output()
                        .run()
                )

                gif_data = utils.img.read_tif_vol(output_gif)

                (n_frames, gif_h, gif_w, gif_chans) = gif_data.shape

                min_div = 32.0
                pad_pix = int(math.ceil(float(gif_h * 0.2) / min_div) * min_div)

                gif_data_trace = np.pad(gif_data, ((0, 0), (pad_pix, 0), (0, 0), (0, 0)), 'constant', constant_values=0)



                gif_trace_canvas = np.zeros((pad_pix, gif_w, gif_chans))

                trace_min = np.min(ca_trace_event)
                ca_trace_event_pix = (ca_trace_event - trace_min) / (trace_max - trace_min)
                ca_trace_event_pix = (np.floor(pad_pix * (1 - ca_trace_event_pix))).astype(int)

                light_sqaure_pix = int(np.floor(pad_pix * 0.5))
                with imageio.get_writer(output_gif, mode='I') as writer:
                    for i in range(n_frames):

                        frame = gif_data_trace[i, :, :, :]

                        if light_on_trace[i]:
                            gif_trace_canvas = cv2.rectangle(gif_trace_canvas,
                                                             (0, 0),
                                                             (light_sqaure_pix, light_sqaure_pix),
                                                             color=(255, 255, 255, 0),
                                                             thickness=-1)
                        if i > 0:
                            time_pix1 = np.floor((i - 1) * gif_w / n_frames).astype(int)
                            time_pix2 = np.floor(i * gif_w / n_frames).astype(int)
                            gif_trace_canvas = cv2.line(gif_trace_canvas,
                                                        (time_pix1, ca_trace_event_pix[i - 1]),
                                                        (time_pix2, ca_trace_event_pix[i]),
                                                        color=(255, 0, 0, 0),
                                                        thickness=1)



                        if i == pre_frames + 1:
                            gif_trace_canvas = cv2.line(gif_trace_canvas,
                                                        (time_pix2, pad_pix),
                                                        (time_pix2, 0),
                                                        color=(255, 255, 255, 255),
                                                        thickness=1)
                            # Don't show anything just flash on this frame
                            frame[0:pad_pix, :, :] = 255
                        else:
                            frame[0:pad_pix, :, :] = gif_trace_canvas

                        writer.append_data(frame)

if plot_event_traces:
    print("Plotting event traces")
    for s2p_data in s2p_data_set:
        roi_type = s2p_data.mode
        (n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad) = s2p_data.get_n_roi_plot(plot_good=True, plot_bad=False)
        for i_roi in range(0, n_roi_cand):
            is_good = s2p_data.iscell[i_roi, 0] == 1
            if not is_good:
                continue

            dfOnF_trace = s2p_data.dFonF0[i_roi, :]
            deconv_trace = s2p_data.deconv_norm[i_roi, :]

            # Get ca events
            event_onset_p = 0.2
            #ca_trace_noise, ca_trace1_thresh = m2putils.calc_ca_noise(dfOnF_trace)
            #ca_event_times = m2putils.get_crossings(1 - ca_trace_noise, 1 - event_onset_p)
            ca_events = utils.ca.get_ca_events(dfOnF_trace)

            ca_event_times = ca_events.onsets

            time_pre = 1
            time_post = 4
            n_samp_pre = int(np.round(exp.SciscanSettings.frames_p_sec * time_pre))
            n_samp_post = int(np.round(exp.SciscanSettings.frames_p_sec * time_post))

            s2p_trial_data = S2PData.get_trial_indexes(s2p_data.time,
                                                       trial_times=s2p_data.time[ca_event_times],
                                                       time_trial=0, time_pre=time_pre, time_post=time_post,
                                                       ignore_no_pre_time=True, ignore_no_post_time=True)


            if len(s2p_trial_data.trial_indexes) == 0:
                continue
            ca_event_traces = S2PData.get_trace_trials(dfOnF_trace, s2p_trial_data.trial_indexes)

            # n_events = ca_event_traces.shape[1]
            # bad_events = []
            # for i_event in np.arange(1, n_events):
            #     prev_off_time = ca_events.offsets[i_event - 1]
            #     this_on_time = ca_events.onsets[i_event]
            #     if prev_off_time + n_samp_post > this_on_time:
            #         bad_events.append(i_event)
            #
            # ca_event_traces = np.delete(ca_event_traces, bad_events, axis=1)


            fig = plt.figure(tight_layout=True)
            plt.plot(s2p_trial_data.trial_time, ca_event_traces, linewidth=1, alpha=0.5, color='0.5')
            mean_ca_trace = np.mean(ca_event_traces, axis=1)
            plt.plot(s2p_trial_data.trial_time, mean_ca_trace, linewidth=2, color='black')

            plot_img_path = os.path.join(plot_event_trace_dir, "{roi}.roi-{i}.ca-events.png"
                                         .format(roi=roi_type,
                                                 i=i_roi))

            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            ca_event_traces = S2PData.get_trace_trials(deconv_trace, s2p_trial_data.trial_indexes)
            # ca_event_traces = np.delete(ca_event_traces, bad_events, axis=1)

            fig = plt.figure(tight_layout=True)
            plt.plot(s2p_trial_data.trial_time, ca_event_traces, linewidth=1, alpha=0.5, color='0.5')
            mean_ca_trace = np.mean(ca_event_traces, axis=1)
            plt.plot(s2p_trial_data.trial_time, mean_ca_trace, linewidth=2, color='black')

            plot_img_path = os.path.join(plot_event_trace_dir, "{roi}.roi-{i}.ca-deconv-events.png"
                                         .format(roi=roi_type,
                                                 i=i_roi))

            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            # Behavioural traces
            time_pre = 1
            time_post = 4
            behave_trace = df_resampled[bu.SPEED_FILT_GRAD].to_numpy()

            s2p_trial_data = S2PData.get_trial_indexes(s2p_data.time,
                                                       trial_times=s2p_data.time[ca_event_times],
                                                       time_trial=0, time_pre=time_pre, time_post=time_post,
                                                       ignore_no_pre_time=True, ignore_no_post_time=True)


            ca_events_behave = S2PData.get_trace_trials(behave_trace, s2p_trial_data.trial_indexes)
            ca_events_behave = np.abs(ca_events_behave - ca_events_behave[0]) #ca_events_behave[s2p_trial_data.n_pre_time_indexes]
            #ca_events_behave = bu.phase_unwrap(ca_events_behave, 360)
            #ca_events_behave = np.cumsum(ca_events_behave, axis=0)

            fig = plt.figure(tight_layout=True)
            plt.plot(s2p_trial_data.trial_time, ca_events_behave, linewidth=1, alpha=0.5, color='0.5')
            mean_ca_trace_behave = np.mean(ca_events_behave, axis=1)
            plt.plot(s2p_trial_data.trial_time, mean_ca_trace_behave, linewidth=2, color='black')
            # plt.yscale('log')

            plot_img_path = os.path.join(plot_event_trace_dir, "{roi}.roi-{i}.ca-events-behave.png"
                                         .format(roi=roi_type,
                                                 i=i_roi))

            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            time_pre = 1
            time_stim = 0
            time_post = 1


    plt.cla()
    plt.clf()
    plt.close('all')
    print("Analysis complete")



