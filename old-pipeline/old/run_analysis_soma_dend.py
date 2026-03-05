import utils.ca
import utils.img
from classes import Experiment, S2PData
import os
from utils import misc, behave as bu
import numpy as np
import scipy
import matplotlib.pyplot as plt
import shutil
from scipy.cluster.vq import vq
import pandas as pd

proc_base_path = "J:/Data/soma-dend/"
s2p_base_path = "J:/Data/s2p/"
behave_base_path = "J:/Data/behave-tuning/"
base_raw_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze"


id = "20210823_16_59_50_1114353"
soma_dend_pairs = {1: [33, 92],
                   3: [60],
                   4: [174, 61, 209],
                   6: [88, 98],
                   35: [48]}

# exp_id = "20210920_11_09_37_1114356"
# somadend_pairs = {7: [164, 154],
#                    8: [92, 121],
#                    18: [59],
#                    23: [132]}

#exp_id = "20210923_15_05_14_1114356"
#exp_id = "20210924_16_09_21_1114356"

raw_data_path = misc.get_exp_path(base_raw_path, id)

dpi = 300
plot_traces = False
corr_thresh = 0.5
per_same_thresh_dist = 0.9
per_same_thresh_prox = 0.8
pix_dist_thresh = 2.0
event_onset_p = 0.2
event_offset_p = 0.7

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
#
# plt.rc('font', **font)

exp = Experiment.Experiment(raw_data_path)
proc_data_path = os.path.join(proc_base_path, misc.path_leaf(exp.directory))

traces_path = os.path.join(proc_data_path, "traces")
hicorr_path = os.path.join(traces_path, "hi-corr")
hicorr_events_path = os.path.join(hicorr_path, "events")
misc.setup_dir(proc_data_path, clearout=False)
misc.setup_dir(traces_path, clearout=plot_traces)
misc.setup_dir(hicorr_path, clearout=plot_traces)
misc.setup_dir(hicorr_events_path, clearout=plot_traces)
#s2p_path = os.path.join(s2p_base_path, m2putils.path_leaf(exp.directory))
s2p_path = os.path.join(s2p_base_path, misc.path_leaf(exp.directory), "deepinterp-sameROI")

print("Checking sciscan and frames")
exp.check_sci_frames()

print("Loading suite2p data")
soma_data = S2PData.load_mode(s2p_path, "soma")
dend_data = S2PData.load_mode(s2p_path, "dend")

print("Loading resampled behavioural data to imaging time scale")
raw_name = os.path.split(raw_data_path)[1]
behave_path = os.path.join(behave_base_path, raw_name)
behave_resample_file = os.path.join(behave_path, raw_name + ".filtered.metrics-sciresamp.h5")
df_resampled = pd.read_hdf(behave_resample_file)
print("Load done.")

print("Processing pairs")


soma_indexes = np.where(soma_data.iscell == 1)[0]
dend_indexes = np.where(dend_data.iscell == 1)[0]

n_soma = soma_indexes.size
n_dend = dend_indexes.size
n_roi = soma_indexes.size + dend_indexes.size
n_samples = soma_data.dFonF0.shape[1]

n_soma_pairs = n_soma * n_soma
n_dend_pairs = n_dend * n_dend
n_soma_dend_pairs = n_soma * n_dend
n_pairs = n_soma_pairs + n_soma_dend_pairs + n_soma_dend_pairs

mat_ca = np.zeros((n_roi, n_samples))
mat_ca_pear_r = np.zeros((n_roi, n_roi))
mat_ca_pear_p = np.zeros((n_roi, n_roi))
mat_ca_act_pear_r = np.zeros((n_roi, n_roi))
mat_ca_act_pear_p = np.zeros((n_roi, n_roi))
mat_joint_events_per = np.zeros((n_roi, n_roi))


def get_roi(i):
    s2p_index = None
    if i < n_soma:
        is_soma = True
        s2p_index = soma_indexes[i]
        s2p_data = soma_data
        roi_type = "soma"
    else:
        is_soma = False
        s2p_index = dend_indexes[i - n_soma]
        s2p_data = dend_data
        roi_type = "dend"

    # dFonF0 = scipy.ndimage.filters.convolve1d(dFonF0, [1, 1, 1], mode='nearest')
    #dFonF0 = scipy.ndimage.gaussian_filter1d(dFonF0, sigma=10)

    return is_soma, roi_type, s2p_index, s2p_data



for i_roi1 in range(n_roi - 1):

    is_soma1, roi_type1, s2p_index1, s2p_data1 = get_roi(i_roi1)

    ca_trace1_raw = s2p_data1.dFonF0[s2p_index1, :]
    ca_trace1_smooth = scipy.ndimage.gaussian_filter1d(ca_trace1_raw, sigma=3)
    ca_trace1_norm_raw = s2p_data1.dFonF0_norm[s2p_index1, :]
    ca_trace1_deconv_raw = s2p_data1.deconv_norm[s2p_index1, :]
    ca_trace1 = scipy.ndimage.gaussian_filter1d(ca_trace1_norm_raw, sigma=3)
    ca_trace1 = ca_trace1 / np.max(ca_trace1)
    mat_ca[i_roi1, :] = ca_trace1

    ca_trace1_noise, ca_trace1_thresh, _ = utils.ca.calc_ca_noise(ca_trace1, smooth_sigma=None)

    ca_trace1_act = ca_trace1 > ca_trace1_thresh

    for i_roi2 in range(i_roi1, n_roi - 1):

        is_soma2, roi_type2, s2p_index2, s2p_data2 = get_roi(i_roi2)

        is_connected = s2p_index1 in soma_dend_pairs and s2p_index2 in soma_dend_pairs[s2p_index1] and \
                       roi_type1 == "soma" and roi_type2 == "dend"

        if not is_connected:
            continue

        ca_trace2_raw = s2p_data2.dFonF0[s2p_index2, :]
        ca_trace2_smooth = scipy.ndimage.gaussian_filter1d(ca_trace2_raw, sigma=3)
        ca_trace2_norm_raw = s2p_data2.dFonF0_norm[s2p_index2, :]
        ca_trace2_deconv_raw = s2p_data2.deconv_norm[s2p_index2, :]
        ca_trace2 = scipy.ndimage.gaussian_filter1d(ca_trace2_norm_raw, sigma=3)
        ca_trace2 = ca_trace2 / np.max(ca_trace2)

        ca_trace2_noise, ca_trace2_thresh, _ = utils.ca.calc_ca_noise(ca_trace2, smooth_sigma=None)

        ca_trace2_act = ca_trace2 > ca_trace2_thresh

        active_indexes = np.logical_or(ca_trace1_act, ca_trace2_act)

        r, p = scipy.stats.pearsonr(ca_trace1, ca_trace2)

        mat_ca_pear_r[i_roi1, i_roi2] = r
        mat_ca_pear_p[i_roi1, i_roi2] = p
        # mat_ca_pear_r[i_roi2, i_roi1] = hd_r
        # mat_ca_pear_p[i_roi2, i_roi1] = hd_p

        r_act, p_act = scipy.stats.pearsonr(ca_trace1[active_indexes], ca_trace2[active_indexes])

        mat_ca_act_pear_r[i_roi1, i_roi2] = r_act
        mat_ca_act_pear_p[i_roi1, i_roi2] = p_act
        # mat_ca_act_pear_r[i_roi2, i_roi1] = r_act
        # mat_ca_act_pear_p[i_roi2, i_roi1] = p_act

        ca_trace_pair_noise = ca_trace1_noise * ca_trace2_noise

        ca_trace_pair_onsets_cand = misc.get_crossings(1 - ca_trace_pair_noise, 1 - event_onset_p)
        ca_trace_pair_onsets = []
        ca_trace_pair_offsets = []
        ca_trace_pair_type = []
        pair_event_diff_prc = []
        pair_soma_max = []
        pair_dend_max = []
        for i, i_onset in enumerate(ca_trace_pair_onsets_cand):

            # Skip this onset if it's before the last offset.
            if len(ca_trace_pair_offsets) > 0 and i_onset < ca_trace_pair_offsets[-1]:
                continue
            sub_probs = ca_trace_pair_noise[i_onset:-1]
            # Too stupid to figure out how to do it without a loop
            i_offset = None
            for i_sub in range(sub_probs.size):
                found_offset = i_sub > 0 and sub_probs[i_sub] > sub_probs[i_sub - 1] and sub_probs[i_sub] > event_offset_p
                if found_offset:
                    # Make sure this was significant at some points
                    i_offset = i_sub + i_onset
                    if np.any(ca_trace_pair_noise[i_onset:i_offset] < 0.05):
                        ca_trace_pair_onsets.append(i_onset)
                        ca_trace_pair_offsets.append(i_offset)
                    else:
                        i_offset = None # not it wasn't, set this to null so the rest of it skips.
                    break

            if i_offset is None:
                continue

            dend_event_trace = ca_trace2[i_onset:i_offset]
            soma_event_trace = ca_trace1[i_onset:i_offset]
            if is_soma2 or not is_soma1:
                # It's not supposed to happen this way.
                raise Exception()

            dendsoma_diff = dend_event_trace - soma_event_trace
            dendsoma_diff_prc25 = np.percentile(dendsoma_diff, 25)

            soma_prc25 = np.percentile(soma_event_trace, 25)
            dend_prc25 = np.percentile(dend_event_trace, 25)

            event_alpha = 0.1
            trace1_sig = np.sum(ca_trace1_noise[i_onset:i_offset] < event_alpha) >= 1
            trace2_sig = np.sum(ca_trace2_noise[i_onset:i_offset] < event_alpha) >= 1

            # if trace1_sig and trace2_sig:
            #     event_type = "joint"
            #     ca_trace_pair_type.append(0)  # joint
            # elif trace1_sig:
            #     event_type= "somatic"
            #     ca_trace_pair_type.append(1)  # somatic
            # elif trace2_sig:
            #     event_type = "dendritic"
            #     ca_trace_pair_type.append(2)  # dendritic
            # else:
            #     # Not sure how this happens but probably joint
            #     event_type = "joint"
            #     ca_trace_pair_type.append(0)  # joint

            # if dendsoma_diff_prc25 > 0.01 and soma_prc25 < 0.05:
            #     event_type = "dendritic"
            # elif dendsoma_diff_prc25 < -0.02 and dend_prc25 < 0.02:
            #     event_type = "somatic"
            # else:
            #     event_type m= "joint"

            if dendsoma_diff_prc25 > 0.01 and soma_prc25 < 0.05:
                ca_trace_pair_type.append(2)   # dendritic
            elif dendsoma_diff_prc25 < -0.02 and dend_prc25 < 0.02:
                ca_trace_pair_type.append(1)   # somatic
            else:
                ca_trace_pair_type.append(0)  # joint

            # if dendsoma_diff_prc25 > 0.01 and not np.any(soma_event_trace > 0.05):
            #     ca_trace_pair_type.append(2)  # dendritic
            # elif dendsoma_diff_prc25 < -0.02 and not np.any(dend_event_trace > 0.02):
            #     ca_trace_pair_type.append(1)  # somatic
            # else:
            #     ca_trace_pair_type.append(0)  # joint

            # if np.any(dend_event_trace > 0.2) and not np.any(soma_event_trace > 0.05):
            #     ca_trace_pair_type.append(2)  # dendritic
            # elif np.any(soma_event_trace > 0.2) and not np.any(dend_event_trace > 0.05):
            #     ca_trace_pair_type.append(1)  # somatic
            # else:
            #     ca_trace_pair_type.append(0)  # joint


            pair_soma_max.append(np.max(ca_trace1[i_onset:i_offset]))
            pair_dend_max.append(np.max(ca_trace2[i_onset:i_offset]))

        ca_trace_pair_onsets = np.array(ca_trace_pair_onsets, dtype='int')
        ca_trace_pair_offsets = np.array(ca_trace_pair_offsets, dtype='int')
        ca_trace_pair_type = np.array(ca_trace_pair_type, dtype='int')


        # f = plt.figure(tight_layout=False)
        # plt.scatter(pair_soma_max, pair_event_diff_prc)
        # plot_name = "{}-{}.{}-{}.eventdiff25.png".format(roi_type1, s2p_index1, roi_type2, s2p_index2)
        # plot_path = os.path.join(hicorr_events_path, plot_name)
        # f.savefig(plot_path, dpi=dpi, facecolor='white')
        # plt.cla()
        # plt.clf()
        # plt.close('all')

        percent_same_events = np.sum(ca_trace_pair_type == 0) / ca_trace_pair_type.size
        percent_type1_events = np.sum(ca_trace_pair_type == 1) / ca_trace_pair_type.size
        percent_type2_events = np.sum(ca_trace_pair_type == 2) / ca_trace_pair_type.size
        mat_joint_events_per[i_roi1, i_roi2] = percent_same_events


        roi1_pix = np.vstack((s2p_data1.stat[s2p_index1]["xpix"], s2p_data1.stat[s2p_index1]["ypix"])).T
        roi2_pix = np.vstack((s2p_data2.stat[s2p_index2]["xpix"], s2p_data1.stat[s2p_index2]["ypix"])).T

        code, dist = vq(roi1_pix, roi2_pix)
        roi_dist = np.min(dist)

        if is_connected:
            print(s2p_index1,
                  s2p_index2,
                  ca_trace_pair_type.size,
                  np.sum(ca_trace_pair_type == 0),
                  np.sum(ca_trace_pair_type == 1),
                  np.sum(ca_trace_pair_type == 2),
                  percent_same_events,
                  percent_type1_events,
                  percent_type2_events)

        # if is_connected:
        #     pre_frames = int(np.round(5 * exp.SciscanSettings.frames_p_sec))
        #     post_frames = int(np.round(12 * exp.SciscanSettings.frames_p_sec))
        #     for i, i_onset in enumerate(ca_trace_pair_onsets):
        #         i_start = i_onset - pre_frames
        #         i_end = i_onset + post_frames
        #         i_offset = ca_trace_pair_offsets[i]
        #         i_offset_time = i_offset - i_start
        #         if i_start < 0 or i_end > ca_trace1_norm.size:
        #             continue
        #         f = plt.figure(tight_layout=False)
        #         event_time = soma_data.time[i_start:i_end] - soma_data.time[i_onset]
        #         plt.plot(event_time, ca_trace1_norm_raw[i_start:i_end], label="roi1", linewidth=0.5)
        #         plt.plot(event_time, ca_trace2_norm_raw[i_start:i_end], label="roi2", linewidth=0.5)
        #         plt.plot(event_time, ca_trace1_noise[i_start:i_end], label="roi1-noise")
        #         plt.plot(event_time, ca_trace2_noise[i_start:i_end], label="roi2-noise")
        #         plt.plot(event_time, ca_trace_pair_noise[i_start:i_end], label="joint")
        #         plt.plot(event_time, ca_trace_pair_noise[i_start:i_end], label="joint")
        #         #plt.plot([event_time[i_offset_time], event_time[i_offset_time]], [0, 1], 'k')
        #         plt.title(ca_trace_pair_type[i])
        #
        #         plt.cla()
        #         plt.clf()
        #         plt.close('all')


        if plot_traces and is_connected:

            trace_lower = ca_trace1_raw
            trace_upper = ca_trace2_raw
            trace_lower_color = 'b'
            trace_upper_color = 'r'
            # Always have soma blue on bottom now
            # if np.mean(s2p_data1.stat[s2p_index1]["ypix"]) < np.mean(s2p_data2.stat[s2p_index2]["ypix"]):
            #     trace_lower = ca_trace2_raw
            #     trace_upper = ca_trace1_raw
            #     trace_lower_color = 'r'
            #     trace_upper_color = 'b'

            scale_bar_time = 300
            scale_bar_ca = 1
            scale_bar_x_offset = scale_bar_time * 0.1

            trace_offset = np.max(trace_lower) * 1.1 - np.min([0, np.min(trace_lower)])
            scale_offset = (np.min(trace_upper)) * 1.5

            scale_bar_time_x0 = 0 - scale_bar_x_offset
            scale_bar_time_x1 = scale_bar_time_x0 + scale_bar_time
            scale_bar_time_y0 = scale_offset
            scale_bar_time_y1 = scale_bar_time_y0

            scale_bar_ca_x0 = 0 - scale_bar_x_offset
            scale_bar_ca_x1 = scale_bar_ca_x0
            scale_bar_ca_y0 = scale_offset
            scale_bar_ca_y1 = scale_offset + scale_bar_ca

            f = plt.figure(tight_layout=False)
            mean_img = s2p_data1.ops["max_proj"]
            mean_img = utils.img.normalize_img_8bit(mean_img, 0.3)
            plt.imshow(mean_img, cmap=plt.cm.gray, interpolation="none")

            cell_img1 = np.zeros((s2p_data1.ops['Ly'], s2p_data1.ops['Lx']))
            cell_img1[s2p_data1.stat[s2p_index1]["ypix"], s2p_data1.stat[s2p_index1]["xpix"]] = 1
            cell_img_mask1 = np.ma.masked_where(cell_img1 == 0, cell_img1)
            plt.imshow(cell_img_mask1, cmap = plt.cm.Blues_r, interpolation="none", alpha=1)

            cell_img2 = np.zeros((s2p_data2.ops['Ly'], s2p_data2.ops['Lx']))
            cell_img2[s2p_data2.stat[s2p_index2]["ypix"], s2p_data2.stat[s2p_index2]["xpix"]] = 1
            cell_img_mask2 = np.ma.masked_where(cell_img2 == 0, cell_img2)
            plt.imshow(cell_img_mask2, cmap = plt.cm.Reds_r, interpolation="none", alpha=1)


            #plt.title("Dist={} pix".format(np.round(roi_dist, 1)))

            plot_img_name = "{}-{}.{}-{}.01.img.png".format(roi_type1, s2p_index1, roi_type2, s2p_index2)
            plot_img_path = os.path.join(traces_path, plot_img_name)

            plt.gca().axis('off')
            f.savefig(plot_img_path, dpi=dpi, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            f = plt.figure(tight_layout=False)
            plt.plot(soma_data.time, trace_lower, label=roi_type1, linewidth=3, color=trace_lower_color)
            plt.plot(soma_data.time, trace_upper + trace_offset, label=roi_type2, linewidth=3, color=trace_upper_color)
            plt.plot([scale_bar_time_x0, scale_bar_time_x1], [scale_bar_time_y0, scale_bar_time_y1], 'k', linewidth=3)
            plt.plot([scale_bar_ca_x0, scale_bar_ca_x1], [scale_bar_ca_y0, scale_bar_ca_y1], 'k', linewidth=3)
            plt.gca().axis('off')

            # plt.title("{} {} {} {} n={} %={}".format(roi_type1, s2p_index1, roi_type2, s2p_index2, ca_trace_pair_type.size,
            #                                          np.round(percent_same_events, 2)))

            plot_trc_name = "{}-{}.{}-{}.02.trace.png".format(roi_type1, s2p_index1, roi_type2, s2p_index2)
            plot_trc_path = os.path.join(traces_path, plot_trc_name)
            f.savefig(plot_trc_path, dpi=dpi, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

             # Plot zoomed trace
            zoom_start = 900
            zoom_end = zoom_start + 60

            scale_bar_time = 30
            scale_bar_ca = 1
            scale_bar_x_offset = 2

            scale_offset = (np.min(ca_trace1_raw)) * 1.5

            scale_bar_time_x0 = zoom_start - scale_bar_x_offset
            scale_bar_time_x1 = scale_bar_time_x0 + scale_bar_time
            scale_bar_time_y0 = scale_offset
            scale_bar_time_y1 = scale_bar_time_y0

            scale_bar_ca_x0 = zoom_start - scale_bar_x_offset
            scale_bar_ca_x1 = scale_bar_ca_x0
            scale_bar_ca_y0 = scale_offset
            scale_bar_ca_y1 = scale_offset + scale_bar_ca

            f = plt.figure(tight_layout=False)

            i_zoom_start = np.argmax(soma_data.time > zoom_start)
            i_zoom_end = np.argmax(soma_data.time > zoom_end)
            # plt.plot(soma_data.time[i_zoom_start:i_zoom_end], trace_lower[i_zoom_start:i_zoom_end], label=roi_type1, linewidth=3, color='b')
            # plt.plot(soma_data.time[i_zoom_start:i_zoom_end], trace_upper[i_zoom_start:i_zoom_end], label=roi_type2, linewidth=3, color='r')
            # plt.plot([scale_bar_time_x0, scale_bar_time_x1], [scale_bar_time_y0, scale_bar_time_y1], 'k', linewidth=3)
            # plt.plot([scale_bar_ca_x0, scale_bar_ca_x1], [scale_bar_ca_y0, scale_bar_ca_y1], 'k', linewidth=3)
            # plt.gca().axis('off')

            f = plt.figure(tight_layout=False)
            plt.plot(soma_data.time[i_zoom_start:i_zoom_end], trace_lower[i_zoom_start:i_zoom_end], label=roi_type1, linewidth=3, color=trace_lower_color)
            plt.plot(soma_data.time[i_zoom_start:i_zoom_end], trace_upper[i_zoom_start:i_zoom_end] + trace_offset, label=roi_type2, linewidth=3, color=trace_upper_color)
            plt.plot([scale_bar_time_x0, scale_bar_time_x1], [scale_bar_time_y0, scale_bar_time_y1], 'k', linewidth=3)
            plt.plot([scale_bar_ca_x0, scale_bar_ca_x1], [scale_bar_ca_y0, scale_bar_ca_y1], 'k', linewidth=3)
            plt.gca().axis('off')

            # plt.title(
            #     "{} {} {} {} n={} %={}".format(roi_type1, s2p_index1, roi_type2, s2p_index2, ca_trace_pair_type.size,
            #                                    np.round(percent_same_events, 2)))

            plot_trc_zoom_name = "{}-{}.{}-{}.02.trace-zoom.png".format(roi_type1, s2p_index1, roi_type2, s2p_index2)
            plot_trc_zoom_path = os.path.join(traces_path, plot_trc_zoom_name)
            f.savefig(plot_trc_zoom_path, dpi=dpi, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            if is_connected:
                shutil.copy2(plot_trc_path, os.path.join(hicorr_path, plot_trc_name))
                shutil.copy2(plot_trc_zoom_path, os.path.join(hicorr_path, plot_trc_zoom_name))
                shutil.copy2(plot_img_path, os.path.join(hicorr_path, plot_img_name))

                pre_frames = int(np.round(5 * exp.SciscanSettings.frames_p_sec))
                post_frames = int(np.round(5 * exp.SciscanSettings.frames_p_sec))

                soma_amps = np.zeros(ca_trace_pair_onsets.shape)
                dend_amps = np.zeros(ca_trace_pair_onsets.shape)

                shutil.copy2(plot_trc_path, os.path.join(hicorr_events_path, plot_trc_name))
                shutil.copy2(plot_trc_zoom_path, os.path.join(hicorr_events_path, plot_trc_zoom_name))
                shutil.copy2(plot_img_path, os.path.join(hicorr_events_path, plot_img_name))

                n_same_events_to_plot = None #3
                i_same_event_both = 0
                i_same_event_diff = 0
                for i, i_onset in enumerate(ca_trace_pair_onsets):

                    # Not sure why this happens but it does.
                    if ca_trace_pair_offsets[i] == 0:
                        continue
                    amp1 = np.max(ca_trace1_smooth[i_onset:ca_trace_pair_offsets[i]])
                    amp2 = np.max(ca_trace2_smooth[i_onset:ca_trace_pair_offsets[i]])
                    if roi_type1 == "soma" and roi_type2 == "dend":
                        soma_amps[i] = amp1
                        dend_amps[i] = amp2
                    elif roi_type1 == "dend" and roi_type2 == "soma":
                        soma_amps[i] = amp2
                        dend_amps[i] = amp1
                    else:
                        raise Exception()

                    if n_same_events_to_plot is not None:
                        if ca_trace_pair_type[i] == 0:
                            if i_same_event_both <= n_same_events_to_plot and amp1 > 0.5 and amp2 > 0.5:
                                i_same_event_both += 1
                            elif i_same_event_diff <= n_same_events_to_plot and np.abs(amp1 - amp2) > 0.5 \
                                    and (amp1 > 0.5 or amp2 > 0.5):
                                i_same_event_diff += 1
                            else:
                                continue
                    i_start = i_onset - pre_frames
                    i_end = i_onset + post_frames
                    i_offset = ca_trace_pair_offsets[i]
                    i_offset_time = i_offset - i_start

                    if i_start < 0 or i_end > ca_trace1.size:
                        continue

                    event_time = soma_data.time[i_start:i_end] - soma_data.time[i_onset]
                    offset_time = event_time[-1]
                    if i_offset_time < event_time.size:
                        offset_time = event_time[i_offset_time]


                    if ca_trace_pair_type[i] == 0:
                        ca_trace_pair_type_name = "Somatic-dendritic"
                    elif ca_trace_pair_type[i] == 1:
                        ca_trace_pair_type_name = "Somatic"
                    else:
                        ca_trace_pair_type_name = "Dendritic"

                    f = plt.figure(tight_layout=False)
                    plt.plot(event_time, ca_trace1[i_start:i_end], label=roi_type1, linewidth=3, color='b')
                    plt.plot(event_time, ca_trace2[i_start:i_end], label=roi_type2, linewidth=3, color='r')
                    # plt.plot(event_time, ca_trace2_norm[i_start:i_end] - ca_trace1_norm[i_start:i_end], label=roi_type1,
                    #          linewidth=3, color='purple')

                    plt.plot(event_time, ca_trace_pair_noise[i_start:i_end], label="joint", color='k', linewidth=1)
                    plt.plot(event_time, ca_trace1_noise[i_start:i_end], label="joint", color='r', linewidth=1)
                    plt.plot(event_time, ca_trace2_noise[i_start:i_end], label="joint", color='b', linewidth=1)
                    plt.title(ca_trace_pair_type_name)
                    plt.ylim(bottom=-0.1, top=1)
                    min_yaxis = plt.ylim()[0]
                    max_yaxis = plt.ylim()[1]
                    plt.plot([0, 0], [min_yaxis, max_yaxis], 'k')
                    plt.plot([offset_time, offset_time], [min_yaxis, max_yaxis], 'k')
                    diff25 = np.percentile(ca_trace2[i_onset:i_offset] - ca_trace1[i_onset:i_offset], 25)
                    plt.plot([0, offset_time], [diff25, diff25], 'k')
                    plot_evt_name = "{}-{}.{}-{}.04.event.{}.png".format(roi_type1, s2p_index1, roi_type2, s2p_index2, i)
                    plot_evt_path = os.path.join(hicorr_events_path, plot_evt_name)
                    f.savefig(plot_evt_path, dpi=dpi, facecolor='white')
                    plt.cla()
                    plt.clf()
                    plt.close('all')

                    f = plt.figure(tight_layout=False)
                    plt.plot(event_time, ca_trace1_raw[i_start:i_end], label=roi_type1, linewidth=3, color='b')
                    plt.plot(event_time, ca_trace2_raw[i_start:i_end], label=roi_type2, linewidth=3, color='r')
                    plt.plot(event_time, ca_trace_pair_noise[i_start:i_end], label="joint", color='k', linewidth=1)
                    plt.plot(event_time, ca_trace1_noise[i_start:i_end], label="joint", color='r', linewidth=1)
                    plt.plot(event_time, ca_trace2_noise[i_start:i_end], label="joint", color='b', linewidth=1)
                    plt.ylim(bottom=-0.5)
                    plt.title(ca_trace_pair_type_name)
                    min_yaxis = plt.ylim()[0]
                    max_yaxis = plt.ylim()[1]
                    plt.plot([0, 0], [min_yaxis, max_yaxis], 'k')
                    plt.plot([offset_time, offset_time], [min_yaxis, max_yaxis], 'k')
                    plot_evt_raw_name = "{}-{}.{}-{}.04.event.{}.raw.png".format(roi_type1, s2p_index1, roi_type2,
                                                                                 s2p_index2, i)
                    plot_evt_raw_path = os.path.join(hicorr_events_path, plot_evt_raw_name)
                    f.savefig(plot_evt_raw_path, dpi=dpi, facecolor='white')

                    plt.cla()
                    plt.clf()
                    plt.close('all')


                    # f = plt.figure(tight_layout=False)
                    # plt.plot(event_time, ca_trace1_deconv_raw[i_start:i_end], label=roi_type1, linewidth=3, color='b')
                    # plt.plot(event_time, ca_trace2_deconv_raw[i_start:i_end], label=roi_type2, linewidth=3, color='r')
                    # # plt.plot(event_time, ca_trace_pair_noise[i_start:i_end], label="joint", color='k', linewidth=3)
                    # plt.title(ca_trace_pair_type_name)
                    # min_yaxis = plt.ylim()[0]
                    # max_yaxis = plt.ylim()[1]
                    # plt.plot([0, 0], [min_yaxis, max_yaxis], 'k')
                    # plt.plot([offset_time, offset_time], [min_yaxis, max_yaxis], 'k')
                    # plot_evt_raw_name = "{}-{}.{}-{}.04.event.{}.raw.deconv.png".format(roi_type1, s2p_index1, roi_type2,
                    #                                                              s2p_index2, i)
                    # plot_evt_raw_path = os.path.join(hicorr_events_path, plot_evt_raw_name)
                    # f.savefig(plot_evt_raw_path, dpi=dpi, facecolor='white')
                    #
                    # plt.cla()
                    # plt.clf()
                    # plt.close('all')


                f = plt.figure(tight_layout=False)
                sparse_indexes = np.round(np.linspace(0, ca_trace1_raw.size - 1, 1000)).astype('int32')
                #plt.scatter(ca_trace1_raw[sparse_indexes], ca_trace2_raw[sparse_indexes], color='0.5', alpha=0.5)
                plt.scatter(soma_amps[ca_trace_pair_type == 0], dend_amps[ca_trace_pair_type == 0], color='k')
                plt.scatter(soma_amps[ca_trace_pair_type == 1], dend_amps[ca_trace_pair_type == 1], color='b')
                plt.scatter(soma_amps[ca_trace_pair_type == 2], dend_amps[ca_trace_pair_type == 2], color='r')

                plt.xlabel("Soma amplitude (dF/F)")
                plt.ylabel("Dend amplitude (dF/F)")

                max_axis = np.max([plt.xlim()[1], plt.ylim()[1]])
                plt.xlim(left=0, right=max_axis)
                plt.ylim(bottom=0, top=max_axis)
                plt.plot([0, max_axis], [0, max_axis], 'k')

                plot_trc_zoom_name = "{}-{}.{}-{}.03.amp-scatter.png".format(roi_type1, s2p_index1, roi_type2,
                                                                             s2p_index2)
                plot_trc_zoom_path = os.path.join(traces_path, plot_trc_zoom_name)
                f.savefig(plot_trc_zoom_path, dpi=dpi, facecolor='white')
                plt.cla()
                plt.clf()
                plt.close('all')

                f = plt.figure(tight_layout=False)
                light_indexes = df_resampled[bu.LIGHT_ON].to_numpy()
                dark_indexes = np.logical_not(light_indexes)

                # Get the bool array of onsets in time
                ca_trace_pair_onsets_bool = np.zeros(light_indexes.shape, dtype='bool')
                ca_trace_pair_onsets_bool[ca_trace_pair_onsets] = True
                ca_trace_pair_soma_amps_array = np.zeros(light_indexes.shape)
                ca_trace_pair_soma_amps_array[ca_trace_pair_onsets] = soma_amps
                ca_trace_pair_dend_amps_array = np.zeros(light_indexes.shape)
                ca_trace_pair_dend_amps_array[ca_trace_pair_onsets] = dend_amps

                # Match events to light/dark
                onsets_light_bool = np.logical_and(ca_trace_pair_onsets_bool, light_indexes)
                onsets_dark_bool = np.logical_and(ca_trace_pair_onsets_bool, dark_indexes)

                # Convert bool to indexes, in time
                onsets_light_indexes = np.where(onsets_light_bool)[0]
                onsets_dark_indexes = np.where(onsets_dark_bool)[0]

                plt.scatter(ca_trace_pair_soma_amps_array[onsets_light_indexes], ca_trace_pair_dend_amps_array[onsets_light_indexes], color='0', alpha=0.5)
                plt.scatter(ca_trace_pair_soma_amps_array[onsets_dark_indexes], ca_trace_pair_dend_amps_array[onsets_dark_indexes], color='0.5', alpha=0.5)

                plt.xlabel("Soma amplitude (dF/F)")
                plt.ylabel("Dend amplitude (dF/F)")

                max_axis = np.max([plt.xlim()[1], plt.ylim()[1]])
                plt.xlim(left=0, right=max_axis)
                plt.ylim(bottom=0, top=max_axis)
                plt.plot([0, max_axis], [0, max_axis], 'k')

                plot_trc_zoom_name = "{}-{}.{}-{}.03.amp-scatter.lightdark.png".format(roi_type1, s2p_index1, roi_type2,
                                                                             s2p_index2)
                plot_trc_zoom_path = os.path.join(traces_path, plot_trc_zoom_name)
                f.savefig(plot_trc_zoom_path, dpi=dpi, facecolor='white')
                plt.cla()
                plt.clf()
                plt.close('all')

                f = plt.figure(tight_layout=False)
                event_hd = df_resampled[bu.HD_ABS_FILT].iloc[ca_trace_pair_onsets].to_numpy()
                plt.scatter(soma_amps, dend_amps, c=event_hd, cmap='hsv', alpha=0.5)

                plt.xlabel("Soma amplitude (dF/F)")
                plt.ylabel("Dend amplitude (dF/F)")

                max_axis = np.max([plt.xlim()[1], plt.ylim()[1]])
                plt.xlim(left=0, right=max_axis)
                plt.ylim(bottom=0, top=max_axis)
                plt.plot([0, max_axis], [0, max_axis], 'k')
                plt.colorbar()
                plot_trc_zoom_name = "{}-{}.{}-{}.03.amp-scatter.HD.png".format(roi_type1, s2p_index1, roi_type2,
                                                                                       s2p_index2)
                plot_trc_zoom_path = os.path.join(traces_path, plot_trc_zoom_name)
                f.savefig(plot_trc_zoom_path, dpi=dpi, facecolor='white')
                plt.cla()
                plt.clf()
                plt.close('all')

                f = plt.figure(tight_layout=False)
                event_speed = df_resampled[bu.SPEED_FILT_GRAD].iloc[ca_trace_pair_onsets].to_numpy()
                plt.scatter(soma_amps, dend_amps, c=event_speed, alpha=0.5)

                plt.xlabel("Soma amplitude (dF/F)")
                plt.ylabel("Dend amplitude (dF/F)")

                max_axis = np.max([plt.xlim()[1], plt.ylim()[1]])
                plt.xlim(left=0, right=max_axis)
                plt.ylim(bottom=0, top=max_axis)
                plt.plot([0, max_axis], [0, max_axis], 'k')
                plt.colorbar()
                plot_trc_zoom_name = "{}-{}.{}-{}.03.amp-scatter.speed.png".format(roi_type1, s2p_index1, roi_type2,
                                                                                s2p_index2)
                plot_trc_zoom_path = os.path.join(traces_path, plot_trc_zoom_name)
                f.savefig(plot_trc_zoom_path, dpi=dpi, facecolor='white')
                plt.cla()
                plt.clf()
                plt.close('all')

                f = plt.figure(tight_layout=False)
                plt.hist(pair_event_diff_prc)
                plot_name = "{}-{}.{}-{}.eventdiff25.png".format(roi_type1, s2p_index1, roi_type2, s2p_index2)
                plot_path = os.path.join(hicorr_events_path, plot_name)
                f.savefig(plot_path, dpi=dpi, facecolor='white')
                plt.cla()
                plt.clf()
                plt.close('all')









print("Done")