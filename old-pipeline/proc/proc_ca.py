# Calculate CA traces, events etc from S2p data.
from classes import Experiment, S2PData
from utils import ca as cautils, db, metadata as metautils
import pandas as pd
from paths.config import M2PConfig
from classes.ProcPath import ProcPath
import numpy as np
from utils.db import get_roi_df, get_ca_df

def proc_ca(cfg:M2PConfig, exp_id, plot_events=False):

    m2p_paths = ProcPath(cfg, exp_id)

    print("Loading experiment")
    exp = Experiment.Experiment(m2p_paths.raw_data_path)
    print("Loading experiment done.")

    print("Checking sciscan and frames")
    exp.check_sci_frames()
    print("Done checking sciscan and frames")

    print("Loading suite2p data")
    soma_data = S2PData.load_mode(m2p_paths.proc_s2p_path, "soma")
    dend_data = S2PData.load_mode(m2p_paths.proc_s2p_path, "dend")
    print("Done")

    # Check if any somas also ended up as dendrites too
    doubled_up = np.logical_and(soma_data.iscell[:,0], dend_data.iscell[:,0])
    if np.any(doubled_up):
        print(doubled_up)
        print(soma_data.iscell[:,0])
        print(dend_data.iscell[:,0])
        raise Exception("Some somas are also dendrites: {} {}".format(exp_id, np.where(doubled_up)))

    bad_s2p_indexes = metautils.get_bad_2p_indexes(cfg, exp_id, exp.SciscanSettings.image_n_frames)

    print("Processing soma data")
    df_roi, df_ca = proc_s2p_data(cfg, exp_id, soma_data, exp.SciscanSettings.frames_p_sec, bad_s2p_indexes, plot_events)
    print("Processing dend data")
    df_roi, df_ca = proc_s2p_data(cfg, exp_id, dend_data, exp.SciscanSettings.frames_p_sec, bad_s2p_indexes, plot_events, df_roi=df_roi, df_ca=df_ca)
    print("Done")

    print("Saving to database")
    # Aggregate with existing data if any.
    db.add_exp_data(df_roi, cfg.db_ca_roi_file, exp_id)
    db.add_exp_data(df_ca, cfg.db_ca_file, exp_id)
    print("Done")

def proc_s2p_data(cfg:M2PConfig, exp_id, s2p_data, fps,
                  bad_s2p_indexes=None, plot_events=False, df_roi=None, df_ca=None):

    if df_roi is None:
        df_roi = get_roi_df()
    if df_ca is None:
        df_ca = get_ca_df()

    plot_events_path = None
    if plot_events:
        plot_events_path = cfg.proc_raw_event_plots_path

    roi_type = s2p_data.mode

    n_roi_cand = s2p_data.iscell.shape[0]

    for i_roi in range(n_roi_cand):
        is_good = s2p_data.iscell[i_roi, 0] == 1

        if not is_good:
            continue

        dFonF0 = s2p_data.dFonF0[i_roi, :]
        deconv_norm = s2p_data.deconv_norm[i_roi, :]

        ca_trace = np.copy(dFonF0)
        ca_events = cautils.get_ca_events(ca_trace,
                                          smooth_sigma=cautils.EVT_DET_SMOOTH_SIGMA,
                                          prc_mean=cautils.EVT_DET_PRC_MEAN,
                                          prc_low=cautils.EVT_DET_PRC_LOW,
                                          prc_high=cautils.EVT_DET_PRC_HIGH,
                                          prob_onset=cautils.EVT_DET_PROB_ONSET,
                                          prob_offset=cautils.EVT_DET_PROB_OFFSET,
                                          alpha=cautils.EVT_DET_ALPHA,
                                          plot_path=plot_events_path,
                                          exp_id=exp_id,
                                          roi_id=i_roi)

        event_onsets = np.zeros(ca_trace.shape)
        event_onsets[ca_events.onsets] = 1

        n_events_good = np.sum(event_onsets[~bad_s2p_indexes])
        n_frames_good = ca_trace.shape[0] - np.sum(bad_s2p_indexes)
        events_per_min = n_events_good / (n_frames_good / fps / 60)

        event_amp = np.zeros(ca_trace.shape)
        event_amp[ca_events.onsets] = ca_events.amps

        dFonF0_clean = dFonF0 * ca_events.masks
        deconv_norm_clean = deconv_norm * ca_events.masks

        # Try to use SNR calculation from Zong et al. 2022 but it's totally confusing.
        # "We next calculated each cell's signal to noise ratio (SNR, signal/noise), where “signal” is defined as the
        # mean amplitude over all 90th percentiles of ΔF/F(t) in significant transients and “noise” denotes the noise
        # level of ΔF/F(t), calculated as the mean of differences of ΔF/F(t) in periods outside significant transients.
        # A threshold of 3 was set for SNR, and only cells passing this threshold were selected for spatial tuning
        # analysis."
        # What is "mean of differences" supposed to be?
        # The code shows something different:
        # https://github.com/kavli-ntnu/MINI2P_toolbox/blob/main/Analysis/%2Bpreprocessing/CalculateSNR.m
        # This code says:
        # Signal=median(Peaks)-median(Baseline);
        # noise = std(RawTrancient);
        # SNR=Signal/noise;
        # So signal is difference in peak to baseline, noise is just std of non events.

        # I am going to make signal = mean(event_amplitudes)
        # Noise will be std of the non-event periods.
        dFonF0_nonevent_good = dFonF0[np.logical_and(np.logical_not(bad_s2p_indexes), np.logical_not(ca_events.masks))]
        mean_event_dFonF0_amp = np.mean(ca_events.amps)
        std_nonevent_dFonF0 = np.std(dFonF0_nonevent_good)
        signal = mean_event_dFonF0_amp
        noise = std_nonevent_dFonF0
        snr = signal / noise

        df_roi_new = pd.DataFrame({"exp_id": exp_id,
                                    "roi_id": i_roi,
                                    "roi_type": roi_type,
                                    "n_events": n_events_good,
                                    "events_per_min": events_per_min,
                                    "mean_event_dFonF0_amp": mean_event_dFonF0_amp,
                                    "std_nonevent_dFonF0": std_nonevent_dFonF0,
                                    "snr": snr}, index=[0])

        df_roi = pd.concat([df_roi, df_roi_new], ignore_index=True, verify_integrity=True)

        n_frames = len(dFonF0)
        time = np.arange(0, n_frames / fps, 1/fps, dtype=np.float64)
        df_ca_new = pd.DataFrame({"exp_id": np.array([exp_id] * n_frames, dtype=np.str),
                                  "roi_id": np.array([int(i_roi)] * n_frames, dtype=np.int),
                                  "frame_id": np.array(list(range(n_frames)), dtype=np.int),
                                  #"bad": np.array(bad_s2p_indexes, dtype=np.bool),
                                  "time": time,
                                  "dFonF0": dFonF0,
                                  "dFonF0_norm_smooth": ca_events.ca_trace_norm,
                                  "dFonF0_clean": dFonF0_clean,
                                  "event_masks": ca_events.masks,
                                  "event_noise": ca_events.noise_probs,
                                  "deconv_norm": deconv_norm,
                                  "deconv_norm_clean": deconv_norm_clean,
                                  "event_onset": event_onsets,
                                  "event_amp": event_amp})

        df_ca = pd.concat([df_ca, df_ca_new], ignore_index=True, verify_integrity=True)


    return df_roi, df_ca


