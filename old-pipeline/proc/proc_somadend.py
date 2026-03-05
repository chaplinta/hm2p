import os
import scipy
import numpy as np
from paths.config import M2PConfig
from classes.ProcPath import ProcPath
from scipy.cluster.vq import vq
from utils import misc, db
from classes import Experiment, S2PData
from utils import misc as m2putils, ca as cautils, behave as beutils
import pandas as pd

from utils.db import get_pair_df, get_event_df, get_pair_ca_df


def proc_single(cfg:M2PConfig,
                exp_id,
                plot_events=False,
                event_onset_p=0.2,
                event_offset_p=0.5,
                smooth_sigma=3,
                noise_alpha=0.5):



    proc_data_path = cfg.proc_path / exp_id
    proc_data_path.mkdir(exist_ok=True)

    raw_data_path = misc.get_exp_path(cfg.raw_path, exp_id)

    exp = Experiment.Experiment(raw_data_path)
    s2p_path = os.path.join(cfg.s2p_path, m2putils.path_leaf(exp.directory))

    print("Checking sciscan and frames")
    exp.check_sci_frames()

    print("Loading suite2p data")
    soma_data = S2PData.load_mode(s2p_path, "soma")
    dend_data = S2PData.load_mode(s2p_path, "dend")

    # print("Loading resampled behavioural data to imaging time scale")
    # raw_name = os.path.split(raw_data_path)[1]
    # behave_path = os.path.join(cfg.behave_path, raw_name)
    # behave_resample_file = os.path.join(behave_path, raw_name + ".filtered.metrics-sciresamp.h5")
    # df_resampled = pd.read_hdf(behave_resample_file)
    # print("Load done.")

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

    # Load known pairs.
    # Actually just process every thing and then filter out the pairs that are not connected.
    #somadend_pairs = m2putils.get_somadend_pairs(cfg, exp_id)

    df_pair = get_pair_df()
    df_event = get_event_df()
    df_ca_pair = get_pair_ca_df()


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
        # dFonF0 = scipy.ndimage.gaussian_filter1d(dFonF0, sigma=10)

        s2p_index = int(s2p_index)

        return is_soma, roi_type, s2p_index, s2p_data

    for i_roi1 in range(n_roi - 1):

        is_soma1, roi_type1, s2p_index1, s2p_data1 = get_roi(i_roi1)

        ca_trace1_raw = s2p_data1.dFonF0[s2p_index1, :]
        ca_trace1_deconv = s2p_data1.deconv_norm[s2p_index1, :]
        ca_trace1_norm_raw = s2p_data1.dFonF0_norm[s2p_index1, :]
        #ca_trace1_raw[ca_trace1_raw < 0] = 0

        ca_trace1 = scipy.ndimage.gaussian_filter1d(ca_trace1_raw, sigma=smooth_sigma)
        min1 = np.min(ca_trace1)
        max1 = np.max(ca_trace1)
        #ca_trace1 = (ca_trace1 - min1) / (max1 - min1)
        ca_trace1 = ca_trace1 / max1

        for i_roi2 in range(i_roi1 + 1, n_roi - 1):

            is_soma2, roi_type2, s2p_index2, s2p_data2 = get_roi(i_roi2)

            # is_connected = s2p_index1 in somadend_pairs and s2p_index2 in somadend_pairs[s2p_index1] and \
            #                roi_type1 == "soma" and roi_type2 == "dend"
            #
            # if not is_connected:
            #     continue

            ca_trace2_raw = s2p_data2.dFonF0[s2p_index2, :]
            ca_trace2_deconv = s2p_data2.deconv_norm[s2p_index2, :]
            ca_trace2_norm_raw = s2p_data2.dFonF0_norm[s2p_index2, :]
            #ca_trace2_raw[ca_trace2_raw < 0] = 0

            ca_trace2 = scipy.ndimage.gaussian_filter1d(ca_trace2_raw, sigma=smooth_sigma)
            min2 = np.min(ca_trace2)
            max2 = np.max(ca_trace2)
            #ca_trace2 = (ca_trace2 - min2) / (max2 - min2)
            ca_trace2 = ca_trace2 / max2

            pair_id = "{}-{}-{}".format(exp_id, s2p_index1, s2p_index2)

            plot_path = None
            if plot_events:
                plot_path = cfg.proc_raw_dendplots_path

            ca_joint = cautils.get_joint_ca_events(ca_trace1,
                                                   ca_trace2,
                                                   ca_trace1_raw=ca_trace1_raw,
                                                   ca_trace2_raw=ca_trace2_raw,
                                                   ca_trace1_deconv=ca_trace1_deconv,
                                                   ca_trace2_deconv=ca_trace2_deconv,
                                                   smooth_sigma=None,
                                                   event_onset_p=event_onset_p,
                                                   event_offset_p=event_offset_p,
                                                   frame_int=1/exp.SciscanSettings.frames_p_sec,
                                                   plot_path=plot_path,
                                                   pair_id=pair_id,
                                                   noise_alpha=noise_alpha,
                                                   roi_type1=roi_type1,
                                                   roi_type2=roi_type2)

            # Calculate correlations between the two traces.
            corr_r, corr_p = scipy.stats.pearsonr(ca_trace1_raw, ca_trace2_raw)
            corr_deconv_r, corr_deconv_p = scipy.stats.pearsonr(ca_trace1_deconv, ca_trace2_deconv)
            corr_noise_r, corr_noise_p = scipy.stats.pearsonr(ca_joint.noise_probs1, ca_joint.noise_probs2)

            # todo Phi coefficient, Matthews correlation coefficient, Jaccard Index,
            # Hamming Distance, Tanimoto coefficient
            corr_event_r = 0
            corr_event_p = 1


            # More summary stats of pair.
            n_events = ca_joint.onsets.size
            n_events_joint = np.sum(ca_joint.event_type == 0)
            n_events_roi_1 = np.sum(ca_joint.event_type == 1)
            n_events_roi_2 = np.sum(ca_joint.event_type == 2)

            # Calculate roi distance.
            roi1_pix = np.vstack((s2p_data1.stat[s2p_index1]["xpix"], s2p_data1.stat[s2p_index1]["ypix"])).T
            roi2_pix = np.vstack((s2p_data2.stat[s2p_index2]["xpix"], s2p_data1.stat[s2p_index2]["ypix"])).T
            code, dist = vq(roi1_pix, roi2_pix)
            roi_dist = np.min(dist)

            df_pair = pd.concat([df_pair, pd.DataFrame([{"exp_id": str(exp_id),
                                                          "pair_id": str(pair_id),
                                                          "roi_index_1": int(s2p_index1),
                                                          "roi_type_1": str(roi_type1),
                                                          "roi_index_2": int(s2p_index2),
                                                          "roi_type_2": str(roi_type2),
                                                          "corr_r": corr_r,
                                                          "corr_p": corr_p,
                                                          "corr_deconv_r": corr_deconv_r,
                                                          "corr_deconv_p": corr_deconv_p,
                                                          "corr_noise_r": corr_noise_r,
                                                          "corr_noise_p": corr_noise_p,
                                                          "corr_event_r": corr_event_r,
                                                          "corr_event_p": corr_event_p,
                                                          "n_events": n_events,
                                                          "n_events_joint": n_events_joint,
                                                          "n_events_roi_1": n_events_roi_1,
                                                          "n_events_roi_2": n_events_roi_2,
                                                          "dist": roi_dist}])])


            n_frames = ca_trace1_raw.size

            indexes_joint = ca_joint.onsets[ca_joint.event_type == 0]
            indexes_somatic = ca_joint.onsets[ca_joint.event_type == 1]
            indexes_dendritic = ca_joint.onsets[ca_joint.event_type == 2]

            event_joint = np.zeros(n_frames)
            event_somatic = np.zeros(n_frames)
            event_dendritic = np.zeros(n_frames)

            event_joint[indexes_joint] = 1
            event_somatic[indexes_somatic] = 1
            event_dendritic[indexes_dendritic] = 1

            data_dict = {"exp_id": [str(exp_id)] * n_frames,
                         "pair_id": [str(pair_id)] * n_frames,
                         "roi_index_1": [int(s2p_index1)] * n_frames,
                         "roi_type_1": [str(roi_type1)] * n_frames,
                         "roi_index_2": [int(s2p_index2)] * n_frames,
                         "roi_type_2": [str(roi_type2)] * n_frames,
                         "frame_id": list(range(n_frames)),
                         "event_joint": event_joint,
                         "event_somatic": event_somatic,
                         "event_dendritic": event_dendritic,
                         "masks1": ca_joint.masks1,
                         "masks2": ca_joint.masks2,
                         "event_ids_joint": ca_joint.event_ids_joint,
                         "event_ids_soma": ca_joint.event_ids_soma,
                         "event_ids_dend": ca_joint.event_ids_dend,
                         }

            data_df = pd.DataFrame(data_dict)
            df_ca_pair = pd.concat([df_ca_pair, data_df])


            for i_event in range(n_events):
                df_event = pd.concat([df_event, pd.DataFrame([{"exp_id": str(exp_id),
                                                                "pair_id": str(pair_id),
                                                                "roi_index_1": int(s2p_index1),
                                                                "roi_type_1": str(roi_type1),
                                                                "roi_index_2": int(s2p_index2),
                                                                "roi_type_2": str(roi_type2),
                                                                "onset_index": ca_joint.onsets[i_event],
                                                                "offset_index": ca_joint.offsets[i_event],
                                                                "type": ca_joint.event_type[i_event],
                                                                "prc25_diff": ca_joint.prc25_diff[i_event],
                                                                "prc25_roi1": ca_joint.prc25_roi_1[i_event],
                                                                "prc25_roi2": ca_joint.prc25_roi_2[i_event],
                                                                "amp1_norm": ca_joint.amps1_norm[i_event],
                                                                "amp2_norm": ca_joint.amps2_norm[i_event],
                                                                "amp1_raw": ca_joint.amps1_raw[i_event],
                                                                "amp2_raw": ca_joint.amps2_raw[i_event],
                                                                "mean1_deconv": ca_joint.amps1_raw[i_event],
                                                                "mean2_deconv": ca_joint.amps2_raw[i_event],
                                                                "event_corr_r": ca_joint.event_corr_r[i_event],
                                                                "event_corr_p": ca_joint.event_corr_p[i_event],
                                                                "noise_corr_r": ca_joint.noise_corr_r[i_event],
                                                                "noise_corr_p": ca_joint.noise_corr_p[i_event]
                                                               }])])




    # Aggregate with existing data if any.
    db.add_exp_data(df_pair, cfg.db_somadend_pairs_file, exp_id)
    db.add_exp_data(df_event, cfg.db_somadend_events_file, exp_id)
    db.add_exp_data(df_ca_pair, cfg.db_somadend_ca_file, exp_id)

    print("Done")


def proc_resample_single(cfg:M2PConfig, exp_id):

    m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

    print("Loading experiment")
    exp = Experiment.Experiment(m2p_paths.raw_data_path)
    print("Loading experiment done.")

    print("Checking sciscan and frames")
    exp.check_sci_frames()

    df_events = pd.read_hdf(cfg.db_somadend_events_file)
    df_behave = pd.read_hdf(cfg.db_behave_file)

    event_exp_indexes = df_events[df_events["exp_id"] == exp_id].index
    behave_exp_indexes = df_behave[df_behave["exp_id"] == exp_id].index
    df_events = df_events[df_events.index.isin(event_exp_indexes)]
    df_behave = df_behave[df_behave.index.isin(behave_exp_indexes)]

    df_resampled = beutils.resample_to_events(exp, df_events, df_behave)

    df_resampled.to_hdf(m2p_paths.behave_events_file, key="df_resampled")

    db.add_exp_data(df_resampled, cfg.db_behave_events_file, exp_id)

