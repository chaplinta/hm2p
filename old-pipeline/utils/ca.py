import numpy as np
import scipy
from dataclasses import dataclass
from utils.misc import get_crossings
import matplotlib.pyplot as plt
import os
import shutil
import warnings

CA_DFONF0 = "dFonF0"
CA_DFONF0_CLEAN = "dFonF0_clean"
CA_DECONV_NORM = "deconv_norm"
CA_DECONV_NORM_CLEAN = "deconv_norm_clean"
CA_EVENTS_MASK = "event_masks"
CA_EVENTS_ONSET = "event_onset"
CA_EVENTS_OFFSET = "event_offset"
CA_EVENTS_AMP = "event_amp"
CA_EVENTS_NOISE = "event_noise"

CA_TYPES = [CA_DFONF0, CA_DECONV_NORM, CA_EVENTS_ONSET, CA_EVENTS_OFFSET, CA_EVENTS_AMP, CA_EVENTS_NOISE]

EVT_DET_SMOOTH_SIGMA = 3
EVT_DET_PRC_MEAN = 40
EVT_DET_PRC_LOW = 10
EVT_DET_PRC_HIGH = 90
EVT_DET_PROB_ONSET = 0.2
EVT_DET_PROB_OFFSET = 0.7
EVT_DET_ALPHA = 1

def get_ca_label(ca_type):
    if ca_type in [CA_DFONF0, CA_DFONF0_CLEAN]:
        return "dF/F0"
    elif ca_type in [CA_DECONV_NORM, CA_DECONV_NORM_CLEAN]:
        return "Deconvolved"
    elif ca_type == CA_EVENTS_MASK:
        return "Event occuring"
    elif ca_type == CA_EVENTS_AMP:
        return "Event amplitude"
    elif ca_type in ["event_joint", "event_somatic", "event_dendritic"]:
        return "Event rate"
    else:
        raise Exception()

def get_ca_unit(ca_type):

    if ca_type in [CA_DFONF0, CA_DFONF0_CLEAN]:
        return "dF/F0"
    elif ca_type in [CA_DECONV_NORM, CA_DECONV_NORM_CLEAN]:
        return "Deconvolved dF/F0"
    elif ca_type == CA_EVENTS_ONSET:
        return "Event rate (events/min)"
    elif ca_type == CA_EVENTS_AMP:
        return "dF/F0"
    elif ca_type in ["event_joint", "event_somatic", "event_dendritic"]:
        return "Event rate (events/min)"
    else:
        raise Exception()


def get_ca_unit_rate(ca_type):
    if ca_type in [CA_DFONF0, CA_DFONF0_CLEAN]:
        return "dF/F0/s"
    elif ca_type in [CA_DECONV_NORM, CA_DECONV_NORM_CLEAN]:
        return "Deconvolved (norm./s)"
    elif ca_type == CA_EVENTS_ONSET:
        return "Event rate (events/min)"
    elif ca_type == CA_EVENTS_MASK:
        return "Event occurring (%)"
    elif ca_type == CA_EVENTS_AMP:
        return "dF/F0/s"
    elif ca_type in ["event_joint", "event_somatic", "event_dendritic"]:
        return "Event rate (events/min)"
    else:
        raise Exception()
def get_ca_axis_label_occ(ca_type):

    if ca_type in [CA_DFONF0, CA_DFONF0_CLEAN]:
        return get_ca_label(ca_type)
    else:
        return "{} ({}/s)".format(get_ca_label(ca_type), get_ca_unit(ca_type))

@dataclass
class CaEvents:

    onsets: np.array = None
    offsets: np.array = None
    masks: np.array = None
    amps: np.array = None
    noise_probs: np.array = None
    ca_trace_norm: np.array = None

def calc_ca_noise(ca_trace, smooth_sigma=None, prc_mean=40, prc_low=10, prc_high=90, prob_onset=0.2):
    # From Voigts and Harnett. Estimate noise by fitting a gaussian based on percentiles.
    # In the paper, smooth sigma was 3 frames, mean was 40th percentile, std was 10 90 percentiles.

    # Can't have negatives.
    ca_trace_rect = np.copy(ca_trace)
    ca_trace_rect[ca_trace_rect < 0] = 0

    if smooth_sigma:
        # Might already be smoothed
        ca_trace_smooth = scipy.ndimage.gaussian_filter1d(ca_trace_rect, sigma=smooth_sigma)
        # Has to be normalised again.
        ca_trace_smooth = ca_trace_smooth / np.max(ca_trace_smooth)
        ca_trace_smooth[ca_trace_smooth < 0] = 0
        min_smooth = np.min(ca_trace_smooth)
        max_smooth = np.max(ca_trace_smooth)
        ca_trace_smooth = (ca_trace_smooth - min_smooth) / (max_smooth - min_smooth)
    else:
        ca_trace_smooth = ca_trace_rect


    ca_trace_mean = np.percentile(ca_trace_smooth, prc_mean)
    ca_trace_std = np.percentile(ca_trace_smooth, prc_high) - np.percentile(ca_trace_smooth, prc_low)
    ca_trace_thresh = scipy.stats.norm.ppf(q=1-prob_onset, loc=ca_trace_mean, scale=ca_trace_std)

    # Fold, everything bellow mean is noise.
    ca_trace_smooth[ca_trace_smooth < ca_trace_mean] = ca_trace_mean
    ca_trace_noise = 1 - scipy.stats.norm.cdf(x=ca_trace_smooth, loc=ca_trace_mean, scale=ca_trace_std)
    ca_trace_noise *= 2

    # Fuck knows why but this happens sometimes.
    if np.min(ca_trace_noise) < 0:
        raise Exception("Noise should be between 0 and 1")
    if np.max(ca_trace_noise) > 1:
        raise Exception("Noise should be between 0 and 1")

    return ca_trace_noise, ca_trace_thresh, ca_trace_smooth


def get_ca_events(ca_trace_raw,
                  smooth_sigma=3, prc_mean=40, prc_low=10, prc_high=90, prob_onset=0.2,
                  prob_offset=0.7, alpha=0.05, plot_path=None, exp_id="", roi_id=np.nan):

    ca_trace_rect = np.copy(ca_trace_raw)
    ca_trace_rect[ca_trace_rect < 0] = 0

    ca_trace_noise, ca_trace_thresh, ca_trace_norm = calc_ca_noise(ca_trace_rect,
                                                                   smooth_sigma,
                                                                   prc_mean,
                                                                   prc_low,
                                                                   prc_high,
                                                                   prob_onset)

    ca_event_onsets_cand = get_crossings(1 - ca_trace_noise, 1 - prob_onset)
    ca_event_onsets = []
    ca_event_offsets = []
    ca_event_masks = np.zeros(ca_trace_raw.shape, dtype='int')
    ca_event_amps = []

    for i, i_onset in enumerate(ca_event_onsets_cand):

        # Skip this onset if it's before the last offset (i.e. overlapping offset)
        if len(ca_event_onsets) > 0 and i_onset < ca_event_onsets[-1]:
            continue
        sub_probs = ca_trace_noise[i_onset:-1]
        # Too stupid to figure out how to do it without a loop
        i_offset = None
        for i_sub in range(sub_probs.size):
            found_offset = i_sub > 0 and sub_probs[i_sub] > sub_probs[i_sub - 1] and sub_probs[i_sub] > prob_offset
            if found_offset:
                # Make sure this was significant at some point
                i_offset = i_sub + i_onset
                if np.any(ca_trace_noise[i_onset:i_offset] < alpha):
                    ca_event_onsets.append(i_onset)
                    ca_event_offsets.append(i_offset)
                    ca_event_amps.append(np.max(ca_trace_raw[i_onset:i_offset]))
                    ca_event_masks[i_onset:i_offset] = 1
                else:
                    # There was an offset but the event was not overall significant.
                    pass

                break

        # Didn't find an offset, it's probably at the end.
        if i_offset == None:
            i_offset = ca_trace_raw.shape[0] - 1

        if plot_path:

            frame_int = 0.1
            time_pre = 2
            time_post = 10
            samples_pre = int(np.round(time_pre / frame_int))
            samples_post = int(np.round(time_post / frame_int))

            i_onset_plot = i_onset - samples_pre
            i_offset_plot = i_onset + samples_post
            if i_onset_plot < 0:
                # Reduce time pre if it hits the start of the recording.
                time_pre = time_pre + i_onset_plot * frame_int
                i_onset_plot = 0

            if i_offset_plot > ca_trace_raw.shape[0] - 1:
                i_offset_plot = ca_trace_raw.shape[0] - 1

            time_event_post = (i_offset_plot - i_onset_plot) * frame_int - time_pre

            time = np.linspace(-time_pre, time_event_post, i_offset_plot - i_onset_plot)

            trace_norm = ca_trace_norm[i_onset_plot:i_offset_plot]
            trace_noise = ca_trace_noise[i_onset_plot:i_offset_plot]
            trace_raw = ca_trace_raw[i_onset_plot:i_offset_plot]

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            ax1.plot(time, trace_noise, 'k')
            ax1.plot(time, trace_norm, 'b')
            ax1.set_ylabel("Norm dF/F0")
            #ax1.set_ylim(bottom=-0.1, top=1)

            ax2.plot(time, trace_raw, 'b')
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("dF/F0")
            #ax2.set_ylim(bottom=-0.1)

            plot_img_path = os.path.join(plot_path,
                                         "event-detect-{}-{:03d}-{:04d}.png".format(exp_id, roi_id, i))
            fig.savefig(plot_img_path, dpi=75, facecolor='white')
            plt.cla()
            plt.clf()
            plt.close('all')

    ca_event_onsets = np.array(ca_event_onsets, dtype='int')
    ca_event_offsets = np.array(ca_event_offsets, dtype='int')
    ca_event_amps = np.array(ca_event_amps, dtype='float')

    return CaEvents(onsets=ca_event_onsets,
                    offsets=ca_event_offsets,
                    masks=ca_event_masks,
                    amps=ca_event_amps,
                    noise_probs=ca_trace_noise,
                    ca_trace_norm=ca_trace_norm)

def get_ca_events_std(ca_trace,
                      std_thresh=2,
                      min_duration=0.75):

    # just like zong 2022.
    # find traces more than 2std that last for some duration (he used 0.75s but I think for GCaMP6s).
    trace_std = np.std(ca_trace)

    event_cand = ca_trace > (trace_std * std_thresh)

    # Find event onsets
    ca_event_onsets_cand = get_crossings(event_cand, 0.9)
    ca_event_onsets = []
    ca_event_offsets = []
    ca_event_indexes = []
    ca_event_amps = []

    for i, i_onset in enumerate(ca_event_onsets_cand):

        # Skip this onset if it's before the last offset (i.e. overlapping offset)
        if len(ca_event_onsets) > 0 and i_onset < ca_event_onsets[-1]:
            continue
        sub_probs = ca_trace_noise[i_onset:-1]
        # Too stupid to figure out how to do it without a loop
        i_offset = None
        for i_sub in range(sub_probs.size):
            found_offset = i_sub > 0 and sub_probs[i_sub] > sub_probs[i_sub - 1] and sub_probs[i_sub] > prob_offset
            if found_offset:
                # Make sure this was significant at some point
                i_offset = i_sub + i_onset
                if np.any(ca_trace_noise[i_onset:i_offset] < alpha):
                    ca_event_onsets.append(i_onset)
                    ca_event_offsets.append(i_offset)
                    ca_event_amps.append(np.max(ca_trace[i_onset:i_offset]))
                else:
                    # There was an offset but the event was not overall significant.
                    pass

                break



    ca_event_onsets = np.array(ca_event_onsets, dtype='int')
    ca_event_offsets = np.array(ca_event_offsets, dtype='int')
    ca_event_indexes = np.array(ca_event_indexes, dtype='int')
    ca_event_amps = np.array(ca_event_amps, dtype='float')

    return CaEvents(onsets=ca_event_onsets,
                    offsets=ca_event_offsets,
                    amps=ca_event_amps,
                    noise_probs=ca_trace_noise)

@dataclass
class CaJointEvents:

    onsets: np.array = None
    offsets: np.array = None
    masks1: np.array = None
    masks2: np.array = None
    event_ids_joint: np.array = None
    event_ids_soma: np.array = None
    event_ids_dend: np.array = None
    prc25_diff: np.array = None
    prc25_roi_1: np.array = None
    prc25_roi_2: np.array = None
    amps1_norm: np.array = None
    amps2_norm: np.array = None
    amps1_raw: np.array = None
    amps2_raw: np.array = None
    means1_deconv: np.array = None
    means2_deconv: np.array = None
    noise_probs_joint: np.array = None
    noise_probs1: np.array = None
    noise_probs2: np.array = None
    event_type: np.array = None
    event_corr_r: np.array = None
    event_corr_p: np.array = None
    noise_corr_r: np.array = None
    noise_corr_p: np.array = None

def get_joint_ca_events(ca_trace1_norm, ca_trace2_norm,
                        ca_trace1_raw, ca_trace2_raw,
                        ca_trace1_deconv, ca_trace2_deconv,
                        smooth_sigma, event_onset_p, event_offset_p, frame_int, plot_path, pair_id, noise_alpha,
                        roi_type1, roi_type2):

    # Can't have negatives.
    ca_trace1_norm[ca_trace1_norm < 0] = 0
    ca_trace2_norm[ca_trace2_norm < 0] = 0

    ca_trace1_noise, ca_trace1_thresh, _ = calc_ca_noise(ca_trace1_norm, smooth_sigma)
    ca_trace2_noise, ca_trace2_thresh, _ = calc_ca_noise(ca_trace2_norm, smooth_sigma)

    ca_trace_pair_noise = ca_trace1_noise * ca_trace2_noise

    ca_trace_pair_onsets_cand = get_crossings(1 - ca_trace_pair_noise, 1 - event_onset_p)
    ca_trace_pair_onsets = []
    ca_trace_pair_offsets = []
    ca_trace_pair_type = []
    prc25_diff = []
    prc25_roi_1 = []
    prc25_roi_2 = []
    pair_max_1_norm = []
    pair_max_2_norm = []
    pair_max_1_raw = []
    pair_max_2_raw = []
    pair_mean_1_deconv = []
    pair_mean_2_deconv = []
    event_corr_r = []
    event_corr_p = []
    noise_corr_r = []
    noise_corr_p = []
    n_joint_frames_pre = 1
    n_joint_frames_post = 2
    event_corr_thresh = 0.7

    masks1 = np.zeros(ca_trace1_norm.shape, dtype='int')
    masks2 = np.zeros(ca_trace1_norm.shape, dtype='int')
    event_ids_joint = np.zeros(ca_trace1_norm.shape, dtype='int')
    event_ids_soma = np.zeros(ca_trace1_norm.shape, dtype='int')
    event_ids_dend = np.zeros(ca_trace1_norm.shape, dtype='int')

    dend_event_id = 1
    soma_event_id = 1
    joint_event_id = 1
    for i, i_onset in enumerate(ca_trace_pair_onsets_cand):

        # Skip this onset if it's before the last offset.
        if len(ca_trace_pair_offsets) > 0 and i_onset < ca_trace_pair_offsets[-1]:
            continue

        max_event_samples = 100
        sub_probs = ca_trace_pair_noise[i_onset:i_onset+max_event_samples]
        # Too stupid to figure out how to do it without a loop
        i_offset = None
        found_offset = False
        for i_sub in range(sub_probs.size):
            prob_increasing = sub_probs[i_sub] > sub_probs[i_sub - 1]
            #found_offset = i_sub > 0 and prob_increasing and sub_probs[i_sub] > event_offset_p
            found_offset = i_sub > 0 and sub_probs[i_sub] > event_offset_p
            i_offset = i_onset + i_sub

            if found_offset:
                ca_trace_pair_onsets.append(i_onset)
                ca_trace_pair_offsets.append(i_offset)
                break

                # # Make sure this was significant at some point
                # roi1_p = np.amin(ca_trace1_noise[i_onset-n_joint_frames_pre:i_onset+n_joint_frames_post])
                # roi2_p = np.amin(ca_trace2_noise[i_onset-n_joint_frames_pre:i_onset+n_joint_frames_post])
                #
                # one_trace_sig = roi1_p < noise_alpha or roi2_p < noise_alpha
                # if not one_trace_sig:
                #     found_offset = False
                #     i_offset = None
                #     break
                # else:
                #     ca_trace_pair_onsets.append(i_onset)
                #     ca_trace_pair_offsets.append(i_offset)
                #     break

        if not found_offset:
            continue

        soma_event_trace = ca_trace1_norm[i_onset:i_offset]
        dend_event_trace = ca_trace2_norm[i_onset:i_offset]
        soma_event_raw_trace = ca_trace1_raw[i_onset:i_offset]
        dend_event_raw_trace = ca_trace2_raw[i_onset:i_offset]
        soma_noise_trace = ca_trace1_noise[i_onset:i_offset]
        dend_noise_trace = ca_trace2_noise[i_onset:i_offset]

        # soma_diff_thresh = 0.01
        # soma_thresh = 0.05
        # dend_diff_thresh = 0.02
        # dend_thresh = 0.02
        # perc_thresh = 25
        soma_diff_thresh = 0.05
        soma_thresh = 0.5
        dend_diff_thresh = 0.05
        dend_thresh = 0.5
        perc_thresh = 25

        n_base_frames = 10
        #soma_baseline = np.mean(ca_trace1_norm[i_onset - n_base_frames:i_onset])
        #dend_baseline = np.mean(ca_trace2_norm[i_onset - n_base_frames:i_onset])

        dendsoma_diff = dend_event_trace - soma_event_trace
        #dendsoma_diff = (dend_event_trace - dend_baseline) - (soma_event_trace - soma_baseline)
        #dendsoma_diff = dend_noise_trace - soma_noise_trace
        dendsoma_diff_prc25 = np.percentile(dendsoma_diff, perc_thresh)

        soma_prc25 = np.percentile(soma_event_trace, perc_thresh)
        dend_prc25 = np.percentile(dend_event_trace, perc_thresh)
        # soma_prc25 = np.percentile(soma_event_trace - soma_baseline, perc_thresh)
        # dend_prc25 = np.percentile(dend_event_trace - dend_baseline, perc_thresh)



        # soma_p = np.min(ca_trace1_noise[i_onset-n_joint_frames_pre:i_onset+n_joint_frames_post])
        # dend_p = np.min(ca_trace2_noise[i_onset-n_joint_frames_pre:i_onset+n_joint_frames_post])
        soma_p = np.min(soma_noise_trace)
        dend_p = np.min(dend_noise_trace)

        soma_sig = soma_p <= noise_alpha
        dend_sig = dend_p <= noise_alpha

        soma_maybe_sig = soma_p < 0.5
        dend_maybe_sig = dend_p < 0.5

        is_dend_event = dend_sig and not soma_maybe_sig
        is_soma_event = soma_sig and not dend_maybe_sig
        # Disregard if both soma and dend only maybe sig
        is_joint_event = (soma_sig and dend_maybe_sig) or (dend_sig and soma_maybe_sig)

        # if dendsoma_diff_prc25 > soma_diff_thresh and soma_prc25 < soma_thresh:
        #     pair_event_type = 2  # dendritic
        # elif dendsoma_diff_prc25 < -dend_diff_thresh and dend_prc25 < dend_thresh:
        #     pair_event_type = 1  # somatic
        # else:
        #     pair_event_type = 0  # joint

        if is_dend_event:
            pair_event_type = 2  # dendritic
            masks1[i_onset:i_offset] = -1
            masks2[i_onset:i_offset] = 1

            event_ids_dend[i_onset:i_offset] = dend_event_id
            dend_event_id += 1
        elif is_soma_event:
            pair_event_type = 1  # somatic
            masks1[i_onset:i_offset] = 1
            masks2[i_onset:i_offset] = -1

            event_ids_soma[i_onset:i_offset] = soma_event_id
            soma_event_id += 1
        elif is_joint_event:
            pair_event_type = 0  # joint
            masks1[i_onset:i_offset] = 2
            masks2[i_onset:i_offset] = 2

            event_ids_joint[i_onset:i_offset] = joint_event_id
            joint_event_id += 1
        else:
            # Turns out neither are sig, just ignore. Continue with loop.
            # Remove last onset and offset.
            ca_trace_pair_onsets.pop()
            ca_trace_pair_offsets.pop()
            continue

        prc25_diff.append(dendsoma_diff_prc25)
        prc25_roi_1.append(soma_prc25)
        prc25_roi_2.append(dend_prc25)

        warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
        event_corr = scipy.stats.spearmanr(soma_event_raw_trace, dend_event_raw_trace)
        noise_corr = scipy.stats.pearsonr(soma_noise_trace, dend_noise_trace)
        warnings.resetwarnings()

        event_corr_r.append(event_corr.statistic)
        event_corr_p.append(event_corr.pvalue)

        noise_corr_r.append(noise_corr.statistic)
        noise_corr_p.append(noise_corr.pvalue)

        event_correlated = event_corr.statistic > event_corr_thresh and event_corr.pvalue < 0.05







        # if not soma_sig and dend_sig:
        #     pair_event_type = 2  # dendritic
        # elif soma_sig and not dend_sig:
        #     pair_event_type = 1  # somatic
        # elif soma_sig and dend_sig:
        #     pair_event_type = 0  # joint
        # else:
        #     raise Exception("Found an event that was neither somatic nor dendritic.")

        ca_trace_pair_type.append(pair_event_type)

        pair_max_1_norm.append(np.max(ca_trace1_norm[i_onset:i_offset]))
        pair_max_1_raw.append(np.max(ca_trace1_raw[i_onset:i_offset]))

        pair_max_2_norm.append(np.max(ca_trace2_norm[i_onset:i_offset]))
        pair_max_2_raw.append(np.max(ca_trace2_raw[i_onset:i_offset]))

        pair_mean_1_deconv.append(np.mean(ca_trace1_deconv[i_onset:i_offset]))
        pair_mean_2_deconv.append(np.mean(ca_trace2_deconv[i_onset:i_offset]))

        raw_corr = np.corrcoef(ca_trace1_raw, ca_trace2_raw)[0, 1]
        putative_pair = raw_corr >= 0.3
        is_soma_dend_pair = roi_type1 == "soma" and roi_type2=="dend"

        if plot_path and putative_pair and is_soma_dend_pair:

            time_pre_plot = 2
            time_post_plot = 10
            time_pre = time_pre_plot
            time_post = time_post_plot
            samples_pre = int(np.round(time_pre / frame_int))
            samples_post = int(np.round(time_post / frame_int))

            i_onset_plot = i_onset - samples_pre
            i_offset_plot = i_onset + samples_post
            if i_onset_plot < 0:
                # Reduce time pre if it hits the start of the recording.
                time_pre = time_pre + i_onset_plot * frame_int
                i_onset_plot = 0

            i_time_onset = samples_pre
            i_time_offset = samples_pre + (i_offset - i_onset)

            if i_offset_plot > ca_trace1_raw.shape[0] - 1:
                i_offset_plot = ca_trace1_raw.shape[0] - 1

            time_event_post = (i_offset_plot - i_onset_plot) * frame_int - time_pre

            time = np.linspace(-time_pre, time_event_post, i_offset_plot - i_onset_plot)
            event_type_name = get_eventtype_name(pair_event_type)

            trace1 = ca_trace1_norm[i_onset_plot:i_offset_plot]
            trace2 = ca_trace2_norm[i_onset_plot:i_offset_plot]
            noise_joint = ca_trace_pair_noise[i_onset_plot:i_offset_plot]
            trace1_raw = ca_trace1_raw[i_onset_plot:i_offset_plot]
            trace2_raw = ca_trace2_raw[i_onset_plot:i_offset_plot]
            trace1_noise = ca_trace1_noise[i_onset_plot:i_offset_plot]
            trace2_noise = ca_trace2_noise[i_onset_plot:i_offset_plot]
            dendsoma_diff = trace2 - trace1
            dendsoma_diff_raw = trace2_raw - trace1_raw
            dendsoma_diff_noise = trace2_noise - trace1_noise

            #fig = plt.figure(tight_layout=False)

            fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True)

            fig.set_figwidth(5)
            fig.set_figheight(12)


            ax1.plot(time, noise_joint, 'k')
            ax1.plot(time, trace1, 'b')
            ax1.plot(time, trace2, 'r')
            ylim = ax1.get_ylim()
            ax1.set_ylabel("Norm dF/F0")
            ax1.set_xlim(left=-time_pre_plot, right=time_post_plot)
            xlim = ax1.get_xlim()
            ax1.set_ylim(bottom=-0.1, top=1.1)
            ax1.plot([xlim[0], xlim[1]], [event_onset_p, event_onset_p], 'k')
            ax1.plot([time[i_time_onset], time[i_time_onset]], [ylim[0], ylim[1]], 'k')
            if i_time_offset < len(time):
                ax1.plot([time[i_time_offset], time[i_time_offset]], [ylim[0], ylim[1]], 'k')

            ax1.set_title("{} diff25={:.2f} soma25={:.2f} dend25={:.2f}".format(
                          event_type_name,
                          dendsoma_diff_prc25,
                          soma_prc25,
                          dend_prc25,
                          fontsize=10))

            # Plot raw traces
            ax2.plot(time, trace1_raw, 'b')
            ax2.plot(time, trace2_raw, 'r')
            #ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("dF/F0")
            #ax2.set_ylim(bottom=-0.1)
            ylim = ax2.get_ylim()
            ax2.plot([time[i_time_onset], time[i_time_onset]], [ylim[0], ylim[1]], 'k')
            if i_time_offset < len(time):
                ax2.plot([time[i_time_offset], time[i_time_offset]], [ylim[0], ylim[1]], 'k')

            ax2.set_title("Onset index={:.0f} offfset index={:.0f}".format(i_onset, i_offset))

            # Plot noise traces
            ax3.plot(time, trace1_noise, 'b')
            ax3.plot(time, trace2_noise, 'r')
            #ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Noise")
            ax3.set_ylim(bottom=-0.1, top=1.1)
            ylim = ax3.get_ylim()
            ax3.plot([xlim[0], xlim[1]], [noise_alpha, noise_alpha], 'k')
            ax3.plot([time[i_time_onset], time[i_time_onset]], [ylim[0], ylim[1]], 'k')
            if i_time_offset < len(time):
                ax3.plot([time[i_time_offset], time[i_time_offset]], [ylim[0], ylim[1]], 'k')
            ax3.set_title("soma_p={:.3f} dend_p={:.3f} noise_r={:.3f}".format(
                soma_p,
                dend_p,
                noise_corr.statistic),
                fontsize=10)

            ax4.plot(time, dendsoma_diff, 'k')
            #ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Diff norm")
            ylim = ax4.get_ylim()
            ax4.plot([time[i_time_onset], time[i_time_onset]], [ylim[0], ylim[1]], 'k')
            ax4.plot([xlim[0], xlim[1]], [0, 0], 'k')
            if i_time_offset < len(time):
                ax4.plot([time[i_time_offset], time[i_time_offset]], [ylim[0], ylim[1]], 'k')

            ax5.plot(time, dendsoma_diff_raw, 'k')
            #ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Diff raw")
            ylim = ax5.get_ylim()
            ax5.plot([time[i_time_onset], time[i_time_onset]], [ylim[0], ylim[1]], 'k')
            ax5.plot([xlim[0], xlim[1]], [0, 0], 'k')
            if i_time_offset < len(time):
                ax5.plot([time[i_time_offset], time[i_time_offset]], [ylim[0], ylim[1]], 'k')

            ax6.plot(time, dendsoma_diff_noise, 'k')
            ax6.set_xlabel("Time (s)")
            ax6.set_ylabel("Diff noise")
            ylim = ax6.get_ylim()
            ax6.plot([time[i_time_onset], time[i_time_onset]], [ylim[0], ylim[1]], 'k')
            ax6.plot([xlim[0], xlim[1]], [0, 0], 'k')
            if i_time_offset < len(time):
                ax6.plot([time[i_time_offset], time[i_time_offset]], [ylim[0], ylim[1]], 'k')


            plot_img_path = os.path.join(plot_path,
                                         "event-detect-{}-{:04d}.png".format(pair_id, len(ca_trace_pair_type)))
            fig.savefig(plot_img_path, dpi=75, facecolor='white')

            if pair_event_type == 0:
                plot_sub_path = os.path.join(plot_path, "01-joint")
            elif pair_event_type == 1:
                plot_sub_path = os.path.join(plot_path, "02-soma")
            elif pair_event_type == 2:
                plot_sub_path = os.path.join(plot_path, "03-dend")

            # Copy the plot file to the sub folder.
            plot_img_path_sub = os.path.join(plot_sub_path,
                                             "event-detect-{}-{:04d}.png".format(pair_id, len(ca_trace_pair_type)))
            if not os.path.exists(plot_sub_path):
                os.makedirs(plot_sub_path)
            shutil.copyfile(plot_img_path, plot_img_path_sub)

            plt.cla()
            plt.clf()
            plt.close('all')

    ca_trace_pair_onsets = np.array(ca_trace_pair_onsets, dtype='int')
    ca_trace_pair_offsets = np.array(ca_trace_pair_offsets, dtype='int')
    ca_trace_pair_type = np.array(ca_trace_pair_type, dtype='int')
    prc25_diff = np.array(prc25_diff, dtype='float')
    prc25_roi_1 = np.array(prc25_roi_1, dtype='float')
    prc25_roi_2 = np.array(prc25_roi_2, dtype='float')
    pair_max_1_norm = np.array(pair_max_1_norm, dtype='float')
    pair_max_2_norm = np.array(pair_max_2_norm, dtype='float')
    pair_max_1_norm = np.array(pair_max_1_norm, dtype='float')
    pair_max_1_raw = np.array(pair_max_1_raw, dtype='float')
    pair_max_2_raw = np.array(pair_max_2_raw, dtype='float')
    pair_mean_1_deconv = np.array(pair_mean_1_deconv, dtype='float')
    pair_mean_2_deconv = np.array(pair_mean_2_deconv, dtype='float')
    ca_trace_pair_noise = np.array(ca_trace_pair_noise, dtype='float')
    event_corr_r = np.array(event_corr_r, dtype='float')
    event_corr_p = np.array(event_corr_p, dtype='float')
    noise_corr_r = np.array(noise_corr_r, dtype='float')
    noise_corr_p = np.array(noise_corr_p, dtype='float')

    return CaJointEvents(onsets=ca_trace_pair_onsets,
                         offsets=ca_trace_pair_offsets,
                         event_type=ca_trace_pair_type,
                         prc25_diff=prc25_diff,
                         prc25_roi_1=prc25_roi_1,
                         prc25_roi_2=prc25_roi_2,
                         amps1_norm=pair_max_1_norm,
                         amps2_norm=pair_max_2_norm,
                         amps1_raw=pair_max_1_raw,
                         amps2_raw=pair_max_2_raw,
                         means1_deconv=pair_mean_1_deconv,
                         means2_deconv=pair_mean_2_deconv,
                         noise_probs_joint=ca_trace_pair_noise,
                         noise_probs1=ca_trace1_noise,
                         noise_probs2=ca_trace2_noise,
                         event_corr_r=event_corr_r,
                         event_corr_p=event_corr_p,
                         noise_corr_r=noise_corr_r,
                         noise_corr_p=noise_corr_p,
                         masks1=masks1,
                         masks2=masks2,
                         event_ids_joint=event_ids_joint,
                         event_ids_soma=event_ids_soma,
                         event_ids_dend=event_ids_dend)

def get_eventtype_name(event_type):

    if event_type == 0:
        return "JOINT"
    elif event_type == 1:
        return "SOMATIC"
    elif event_type == 2:
        return "DENDRITIC"

