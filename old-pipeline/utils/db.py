import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from paths import config
from utils import metadata as metautils, misc as misc, behave as bu
from classes.Experiment import Experiment
from classes.ProcPath import ProcPath
import configparser
import os

# Aggregate various individual datasets into one.
def add_exp_data(df, db_file, exp_id):
    if db_file.exists():
        print("Loading database from disk")
        df_db = pd.read_hdf(db_file)

        drop_indexes = df_db[df_db["exp_id"] == exp_id].index

        print("Dropping previous data")
        # df_db = df_db.drop(drop_indexes) # This is super slow for some reason.
        df_db = df_db[~df_db.index.isin(drop_indexes)]

        print("Adding new data")
        df_db = pd.concat([df_db, df], ignore_index=True, verify_integrity=True)
    else:
        df_db = df

    print("Saving database back to disk")
    df_db.to_hdf(str(db_file), key="key", mode='w')


def get_tune_df():
    raise Exception("redo like the ca tables")
    df_tune = pd.DataFrame(columns={"exp_id": pd.Series(dtype='string'),
                                    "roi_id": pd.Series(dtype='int'),
                                    "tune_type": pd.Series(dtype='string'),
                                    "tune_param": pd.Series(dtype='string'),
                                    "resp_type": pd.Series(dtype='string'),
                                    "resp_mean": pd.Series(dtype='float'),
                                    "resp_sem": pd.Series(dtype='float'),
                                    "resp_n": pd.Series(dtype='int')})

    #df_tune = df_tune.set_index(["exp_id", "roi_id", "tune_type", "tune_param", "resp_type"])

    return df_tune


def get_tune_pair_df():
    raise Exception("redo like the ca tables")
    # Basically the same as the roi version, consider integrating.
    df_tune = pd.DataFrame(columns={"exp_id": pd.Series(dtype='string'),
                                    "pair_id": pd.Series(dtype='int'),
                                    "tune_type": pd.Series(dtype='string'),
                                    "tune_param": pd.Series(dtype='string'),
                                    "resp_type": pd.Series(dtype='string'),
                                    "resp_mean": pd.Series(dtype='float'),
                                    "resp_sem": pd.Series(dtype='float'),
                                    "resp_n": pd.Series(dtype='int')})

    #df_tune = df_tune.set_index(["exp_id", "pair_id", "tune_type", "tune_param", "resp_type"])

    return df_tune


def get_pairstat_df():
    raise Exception("redo like the ca tables")
    df_roistat = pd.DataFrame(columns={"exp_id": pd.Series(dtype='string'),
                                       "roi_id": pd.Series(dtype='int'),
                                       "resp_type": pd.Series(dtype='string'),
                                       "stat_name": pd.Series(dtype='string'),
                                       "stat": pd.Series(dtype='float')})

    #df_roistat = df_roistat.set_index(["exp_id", "roi_id", "resp_type", "stat_name"])

    return df_roistat


def get_roistat_df():
    raise Exception("redo like the ca tables")
    # Basically the same as the roi version, consider integrating.
    df_roistat = pd.DataFrame(columns={"exp_id": pd.Series(dtype='string'),
                                       "pair_id": pd.Series(dtype='int'),
                                       "resp_type": pd.Series(dtype='string'),
                                       "stat_name": pd.Series(dtype='string'),
                                       "stat": pd.Series(dtype='float')})

    #df_roistat = df_roistat.set_index(["exp_id", "pair_id", "resp_type", "stat_name"])

    return df_roistat


def add_roi_stat(df, exp_id, id_col, roi_id, resp_type, stat_name, stat):
    raise Exception("redo like the ca tables")
    dict_stat = {"exp_id": exp_id,
                 id_col: roi_id,
                 "resp_type": resp_type,
                 "stat_name": stat_name,
                 "stat": stat}

    df_stat = pd.DataFrame([dict_stat])

    df = pd.concat([df, df_stat])

    return df


def get_roi_df():
    df_roi = pd.DataFrame(np.empty(0, dtype=[('exp_id', str),
                                             ('roi_id', int),
                                             ('roi_type', str),
                                             ('n_events', int),
                                             ('events_per_min', float),
                                             ('mean_event_dFonF0_amp', float),
                                             ('std_nonevent_dFonF0', float),
                                             ('snr', float)]))

    return df_roi


def get_ca_df():
    df_ca = pd.DataFrame(np.empty(0, dtype=[("exp_id", str),
                                            ("roi_id", int),
                                            ("frame_id", int),
                                            #("bad", bool),
                                            ("time", float),
                                            ("dFonF0", float),
                                            ("dFonF0_clean", float),
                                            ("dFonF0_norm_smooth", float),
                                            ("deconv_norm", float),
                                            ("deconv_norm_clean", float),
                                            ("event_masks", int),
                                            ("event_noise", float),
                                            ("event_permin", float),
                                            ("event_amp", float)]))

    return df_ca

def get_roi_events_df():
    df_roi_events = pd.DataFrame(np.empty(0, dtype=[("exp_id", str),
                                                               ("roi_id", int),
                                                               ("onset_index", int),
                                                               ("offset_index", int),
                                                                ]))

    return df_roi_events

def get_pair_df():

    df_pair = pd.DataFrame(np.empty(0, dtype=[("exp_id", str),
                                              ("pair_id", str),
                                              ("roi_index_1", int),
                                              ("roi_type_1", str),
                                              ("roi_index_2", int),
                                              ("roi_type_2", str),
                                              ("corr_r", float),
                                              ("corr_p", float),
                                              ("corr_deconv_r", float),
                                              ("corr_deconv_p", float),
                                              ("corr_noise_r", float),
                                              ("corr_noise_p", float),
                                              ("corr_event_r", float),
                                              ("corr_event_p", float),
                                              ("n_events", int),
                                              ("n_events_joint", int),
                                              ("n_events_roi_1", int),
                                              ("n_events_roi_2", int),
                                              ("dist", float)]))

    # df_pair = pd.DataFrame({"exp_id": pd.Series(dtype='string'),
    #                         "pair_id": pd.Series(dtype='string'),
    #                         "roi_index_1": pd.Series(dtype='int'),
    #                         "roi_type_1": pd.Series(dtype='string'),
    #                         "roi_index_2": pd.Series(dtype='int'),
    #                         "roi_type_2": pd.Series(dtype='string'),
    #                         "corr_r": pd.Series(dtype='float'),
    #                         "corr_p": pd.Series(dtype='float'),
    #                         "corr_evnt_r": pd.Series(dtype='float'),
    #                         "corr_evnt_p": pd.Series(dtype='float'),
    #                         "n_events": pd.Series(dtype='int'),
    #                         "n_events_joint": pd.Series(dtype='int'),
    #                         "n_events_roi_1": pd.Series(dtype='int'),
    #                         "n_events_roi_2": pd.Series(dtype='int'),
    #                         "dist": pd.Series(dtype='float'),
    #                         })

    # For some reason I made pair_id = exp_id + roi1 + ro2
    #df_pair = df_pair.set_index('pair_id')

    return df_pair


def get_pair_ca_df():

    df_pair_ca = pd.DataFrame(np.empty(0, dtype=[("exp_id", str),
                                                        ("pair_id", str),
                                                        ("roi_index_1", int),
                                                        ("roi_type_1", str),
                                                        ("roi_index_2", int),
                                                        ("roi_type_2", str),
                                                        ("frame_id", int),
                                                        ("event_joint", float),
                                                        ("event_somatic", float),
                                                        ("event_dendritic", float),
                                                        ("masks1", int),
                                                        ("masks2", int),
                                                        ("event_ids_joint", int),
                                                        ("event_ids_soma", int),
                                                        ("event_ids_dend", int),]))

    # df_pair_ca = pd.DataFrame(columns={"exp_id": pd.Series(dtype='string'),
    #                                    "pair_id": pd.Series(dtype='string'),
    #                                    "roi_index_1": pd.Series(dtype='int'),
    #                                    "roi_type_1": pd.Series(dtype='string'),
    #                                    "roi_index_2": pd.Series(dtype='int'),
    #                                    "roi_type_2": pd.Series(dtype='string'),
    #                                    "frame_id": pd.Series(dtype='int'),
    #                                    "event_joint": pd.Series(dtype='float'),
    #                                    "event_somatic": pd.Series(dtype='float'),
    #                                    "event_dendritic": pd.Series(dtype='float')})

    # df_pair_ca = pd.DataFrame(columns={"exp_id": pd.Series(dtype='string'),
    #                                    "pair_id": pd.Series(dtype='string'),
    #                                    "roi_index_1": pd.Series(dtype='int'),
    #                                    "roi_type_1": pd.Series(dtype='string'),
    #                                    "roi_index_2": pd.Series(dtype='int'),
    #                                    "roi_type_2": pd.Series(dtype='string'),
    #                                    "frame_id": pd.Series(dtype='int'),
    #                                    "dFonF01": pd.Series(dtype='float'),
    #                                    "dFonF02": pd.Series(dtype='float'),
    #                                    "dFonF0_clean1": pd.Series(dtype='float'),
    #                                    "dFonF0_clean2": pd.Series(dtype='float'),
    #                                    "deconv_norm1": pd.Series(dtype='float'),
    #                                    "deconv_norm2": pd.Series(dtype='float'),
    #                                    "deconv_norm_clean1": pd.Series(dtype='float'),
    #                                    "deconv_norm_clean2": pd.Series(dtype='float'),
    #                                    "event_masks1": pd.Series(dtype='int'),
    #                                    "event_masks2": pd.Series(dtype='int'),
    #                                    "event_onset1": pd.Series(dtype='float'),
    #                                    "event_onset2": pd.Series(dtype='float'),
    #                                    "event_amp1": pd.Series(dtype='float'),
    #                                    "event_amp2": pd.Series(dtype='float')})

    # For some reason I made pair_id = exp_id + roi1 + ro2
    #df_pair_ca = df_pair_ca.set_index(["exp_id", "pair_id", "frame_id"])

    return df_pair_ca


def get_event_df():
    df_pair_events = pd.DataFrame(np.empty(0, dtype=[("exp_id", str),
                                                       ("pair_id", str),
                                                       ("roi_index_1", int),
                                                       ("roi_type_1", str),
                                                       ("roi_index_2", int),
                                                       ("roi_type_2", str),
                                                       ("onset_index", int),
                                                       ("offset_index", int),
                                                       ("type", int),
                                                       ("amp1_norm", float),
                                                       ("amp2_norm", float),
                                                       ("amp1_raw", float),
                                                       ("amp2_raw", float),
                                                       ("mean1_deconv", float),
                                                       ("mean2_deconv", float),
                                                       ("event_corr_r", float),
                                                       ("event_corr_p", float),
                                                       ("noise_corr_r", float),
                                                       ("noise_corr_p", float)]))

    # df_pair_events = pd.DataFrame({"exp_id": pd.Series(dtype='string'),
    #                                "pair_id": pd.Series(dtype='string'),
    #                                "roi_index_1": pd.Series(dtype='int'),
    #                                "roi_type_1": pd.Series(dtype='string'),
    #                                "roi_index_2": pd.Series(dtype='int'),
    #                                "roi_type_2": pd.Series(dtype='string'),
    #                                "onset_index": pd.Series(dtype='int'),
    #                                "offset_index": pd.Series(dtype='int'),
    #                                "type": pd.Series(dtype='string'),
    #                                "amp1_norm": pd.Series(dtype='float'),
    #                                "amp2_norm": pd.Series(dtype='float'),
    #                                "amp1_raw": pd.Series(dtype='float'),
    #                                "amp2_raw": pd.Series(dtype='float')})

    # For some reason I made pair_id = exp_id + roi1 + ro2
    #df_pair_events = df_pair_events.set_index(["pair_id", "onset_index"])

    return df_pair_events


def get_roi_ca_data(cfg: config.M2PConfig, roi_type="soma", primary_exp=True, exclude=False,
                    exclude_bad_2p=True):
    df_animals = pd.read_csv(cfg.meta_animals_file)
    df_exps = pd.read_csv(cfg.meta_exps_file)
    df_roi = pd.read_hdf(cfg.db_ca_roi_file)
    df_ca = pd.read_hdf(cfg.db_ca_file)

    # Get the animal id from the experiment id
    df_exps['animal_id'] = df_exps['exp_id'].str.split("_").str[-1].astype(int)

    # Filter experiments we don't want
    df_exps = df_exps.loc[np.logical_and(df_exps["primary_exp"] == primary_exp,
                                         df_exps["exclude"] == exclude)]

    if roi_type:
        df_roi = df_roi.loc[df_roi["roi_type"] == roi_type]

    # Loop through experiments to get experiment specific data, such as bad frames, frame rate etc.
    df_exps["fps_2p"] = [np.NaN] * len(df_exps)
    df_ca["bad_2p"] = [False] * len(df_ca)
    for exp_id in df_exps['exp_id'].unique():
        ca_exp_indexes = df_ca["exp_id"] == exp_id
        df_ca_exp = df_ca.loc[ca_exp_indexes]

        fps_2p = float(get_sci_ini_config(cfg, exp_id)['_']['frames.p.sec'])
        n_frames = df_ca_exp["roi_id"].value_counts().iloc[0]

        bad_s2p_indexes = metautils.get_bad_2p_indexes(cfg, exp_id, n_frames)

        df_exps.loc[ca_exp_indexes, "fps_2p"] = fps_2p

        for roi_id in df_ca_exp['roi_id'].unique():
            roi_indexes = np.logical_and(ca_exp_indexes, df_ca["roi_id"] == roi_id)
            df_ca.loc[roi_indexes, "bad_2p"] = bad_s2p_indexes

    n_frames = df_ca.shape[0]
    if exclude_bad_2p:
        df_ca = df_ca.loc[df_ca["bad_2p"] == False]
        exclude_frames = n_frames - df_ca.shape[0]
        print("Excluded {} bad 2p frames {:.2f}%".format(exclude_frames,
                                                                          100 * exclude_frames / n_frames))

    # Join the animal and experiment frames
    df_exps_animals = df_exps.merge(df_animals, on='animal_id', how='inner')
    df_roi_exps_animals = df_roi.merge(df_exps_animals, on='exp_id', how='inner')
    df_ca_roi_exps_animals = df_ca.merge(df_roi_exps_animals, on=['exp_id', 'roi_id'], how='inner')

    df_roi = df_roi_exps_animals
    df_ca = df_ca_roi_exps_animals

    return df_exps, df_roi, df_ca


def get_ca_behave_data(cfg: config.M2PConfig,
                       roi_type="soma",
                       primary_exp=True,
                       exclude=False,
                       exclude_bad_behave=True,
                       exclude_bad_2p=True,
                       max_speed=None,
                       max_ahv_abs=None,
                       min_ear_dist=None,
                       max_ear_dist=None,
                       calc_extra_behave=False,):

    df_exps, df_roi, df_ca = get_roi_ca_data(cfg=cfg, roi_type=roi_type, primary_exp=primary_exp, exclude=exclude,
                                             exclude_bad_2p=exclude_bad_2p)

    df_behave = pd.read_hdf(cfg.db_behave_frames_file)
    df_behave[bu.BAD_BEHAVE] = [False] * len(df_behave)
    df_behave[bu.IS_ACTIVE] = [True] * len(df_behave)
    df_behave[bu.MAZE_VISIT] = [0] * len(df_behave)
    df_behave[bu.MAZE_CUMDIST] = [0] * len(df_behave)

    df_behave[bu.HEAD_X_FILT_MAZE_INT] = np.ceil(df_behave[bu.HEAD_X_FILT_MAZE]).astype(int)
    df_behave[bu.HEAD_Y_FILT_MAZE_INT] = np.ceil(df_behave[bu.HEAD_Y_FILT_MAZE]).astype(int)

    for exp_id in df_exps['exp_id'].unique():
        fps_2p = float(get_sci_ini_config(cfg, exp_id)['_']['frames.p.sec'])
        n_frames = np.sum(df_behave["exp_id"] == exp_id)

        # Use 2p frame rate because behaviour has been resampled to that.
        bad_behave_indexes = metautils.get_bad_behav_indexes(cfg, exp_id, fps_2p, n_frames)

        behave_exp_indexes = df_behave["exp_id"] == exp_id
        df_behave.loc[behave_exp_indexes, bu.BAD_BEHAVE] = bad_behave_indexes

        # Get inactive bouts
        df_behave_exp = df_behave.loc[behave_exp_indexes]
        move_indexes = bu.get_moving_indexes(df_behave_exp)
        mov_series = pd.Series(move_indexes)
        win = bu.INACTIVE_MOVE_FRAME_THRESH

        # Find all the rows at the centre of sequence of inactivity and set to false
        act_series = mov_series.rolling(window=win, center=False, min_periods=1).sum() != 0
        act_series = act_series.rolling(window=win, center=False, min_periods=1).sum().shift(win-1) != 0
        act_series = act_series.rolling(window=win, center=False, min_periods=1).apply(lambda x: x.mode()[0]) != 0

        df_behave.loc[behave_exp_indexes, bu.IS_ACTIVE] = act_series.values

        if calc_extra_behave:
            maze_vist = [0] * len(df_behave_exp)
            maze_visit_dict = {}
            prev_xy = None
            for i in range(len(df_behave_exp)):
                row = df_behave_exp.iloc[i]
                maze_x = int(round(row[bu.HEAD_X_FILT_MAZE]))
                maze_y = int(round(row[bu.HEAD_Y_FILT_MAZE]))
                if prev_xy is None:
                    maze_visit_dict[(maze_x, maze_y)] = 1
                else:
                    if (maze_x, maze_y) != prev_xy:
                        if (maze_x, maze_y) not in maze_visit_dict:
                            maze_visit_dict[(maze_x, maze_y)] = 1
                        else:
                            maze_visit_dict[(maze_x, maze_y)] += 1

                maze_vist[i] = maze_visit_dict[(maze_x, maze_y)]

                prev_xy = (maze_x, maze_y)


            df_behave.loc[behave_exp_indexes, bu.MAZE_VISIT] = np.array(maze_vist)


    n_frames = df_behave.shape[0]
    if exclude_bad_behave:
        df_behave = df_behave.loc[df_behave[bu.BAD_BEHAVE] == False]
        exclude_frames = n_frames - df_behave.shape[0]
        print("Bad behave - excluded {} bad behave frames {:.2f}%".format(exclude_frames, 100*exclude_frames/n_frames))

    n_frames = df_behave.shape[0]
    if max_speed:
        df_behave = df_behave.loc[df_behave[bu.SPEED_FILT_GRAD] <= max_speed]
        exclude_frames = n_frames - df_behave.shape[0]
        print("Max speed - excluded {} bad behave frames {:.2f}%".format(exclude_frames, 100 * exclude_frames / n_frames))

    n_frames = df_behave.shape[0]
    if max_ahv_abs:
        df_behave = df_behave.loc[np.abs(df_behave[bu.AHV_FILT_GRAD]) <= max_ahv_abs]
        exclude_frames = n_frames - df_behave.shape[0]
        print("Max AHV - excluded {} bad behave frames {:.2f}%".format(exclude_frames, 100 * exclude_frames / n_frames))


    n_frames = df_behave.shape[0]
    if min_ear_dist:
        ear_positions_mm = np.empty((4, len(df_behave)))
        ear_positions_mm[0, :] = df_behave[bu.EAR_LEFT_MM_X]
        ear_positions_mm[1, :] = df_behave[bu.EAR_LEFT_MM_Y]
        ear_positions_mm[2, :] = df_behave[bu.EAR_RIGHT_MM_X]
        ear_positions_mm[3, :] = df_behave[bu.EAR_RIGHT_MM_Y]
        ear_dist = bu.calc_ear_dist(ear_positions_mm)
        df_behave = df_behave.loc[ear_dist >= min_ear_dist]
        exclude_frames = n_frames - df_behave.shape[0]
        print("Min ear dist - excluded {} bad behave frames {:.2f}%".format(exclude_frames, 100 * exclude_frames / n_frames))

    if max_ear_dist:
        ear_positions_mm = np.empty((4, len(df_behave)))
        ear_positions_mm[0, :] = df_behave[bu.EAR_LEFT_MM_X]
        ear_positions_mm[1, :] = df_behave[bu.EAR_LEFT_MM_Y]
        ear_positions_mm[2, :] = df_behave[bu.EAR_RIGHT_MM_X]
        ear_positions_mm[3, :] = df_behave[bu.EAR_RIGHT_MM_Y]
        ear_dist = bu.calc_ear_dist(ear_positions_mm)
        df_behave = df_behave.loc[ear_dist <= max_ear_dist]
        exclude_frames = n_frames - df_behave.shape[0]
        print("Max ear dist - excluded {} bad behave frames {:.2f}%".format(exclude_frames,
                                                                            100 * exclude_frames / n_frames))

    df_exps_animals_behave = df_exps.merge(df_behave, on='exp_id', how='inner')

    df_ca_behave = df_ca.merge(df_behave, on=['exp_id', 'frame_id'], how='inner')

    # todo make sure I didn't lose or data or have nulls etc.

    df_behave = df_exps_animals_behave

    return df_exps, df_roi, df_ca, df_behave, df_ca_behave

def get_ca_behave_pair_data(cfg: config.M2PConfig, roi_type="soma", primary_exp=True, exclude=False,
                            exclude_bad_behave=True, exclude_bad_2p=True):

    df_exps, df_roi, df_ca, df_behave, df_ca_behave = get_ca_behave_data(cfg=cfg, roi_type=roi_type,
                                                                         primary_exp=primary_exp, exclude=exclude,
                                                                         exclude_bad_behave=exclude_bad_behave,
                                                                         exclude_bad_2p=exclude_bad_2p)

    df_pair = pd.read_hdf(cfg.db_somadend_pairs_file)
    df_pair_events = pd.read_hdf(cfg.db_somadend_events_file)
    df_pair_ca = pd.read_hdf(cfg.db_somadend_ca_file)

    # Join to all the roi's we are interested in.
    df_roi_pair = df_pair.merge(df_roi,                      
                                left_on=['exp_id', 'roi_index_1'],
                                right_on=['exp_id', 'roi_id'],
                                how='left')

    #
    # df_roi_pair_events = df_roi.merge(df_pair_events,
    #                                   left_on=['exp_id', 'roi_id'],
    #                                   right_on=['exp_id', 'roi_index_1'],
    #                                   how='left')
    # df_roi_pair_events = df_roi_pair_events.merge(df_pair_events,
    #                                               left_on=['exp_id', 'roi_id'],
    #                                               right_on=['exp_id', 'roi_index_2'],
    #                                               how='left')
    #
    # df_roi_pair_ca = df_roi.merge(df_pair_ca,
    #                               left_on=['exp_id', 'roi_id'],
    #                               right_on=['exp_id', 'roi_index_1'],
    #                               how='left')
    # df_roi_pair_ca = df_roi_pair_ca.merge(df_pair_ca,
    #                                       left_on=['exp_id', 'roi_id'],
    #                                       right_on=['exp_id', 'roi_index_2'],
    #                                       how='left')
    #
    #
    #
    df_pair = df_roi_pair
    # df_pair_events = df_roi_pair_events
    # df_pair_ca = df_roi_pair_ca

    return df_exps, df_roi, df_ca, df_behave, df_ca_behave, df_pair, df_pair_events, df_pair_ca

def get_exp_meta_config(cfg, exp_id):

    m2p_paths = ProcPath(cfg, exp_id)

    config_file = misc.get_filetype(m2p_paths.raw_data_path, "*.meta.txt")

    config_meta = configparser.ConfigParser()
    config_meta.read(config_file)

    return config_meta


def get_sci_ini_config(cfg, exp_id):
    m2p_paths = ProcPath(cfg, exp_id)
    config_meta = get_exp_meta_config(cfg, exp_id)
    inifile = os.path.join(m2p_paths.raw_data_path, misc.path_leaf(config_meta["SciScan"]["inifile"]))

    config_ini = configparser.ConfigParser()
    config_ini.read(inifile)

    return config_ini

def get_exp_animals(cfg: config.M2PConfig, primary=None, exclude=None):


    df_animals = pd.read_csv(cfg.meta_animals_file)
    df_exps = pd.read_csv(cfg.meta_exps_file)

    # Get the animal id from the experiment id
    df_exps['animal_id'] = df_exps['exp_id'].str.split("_").str[-1].astype(int)

    # Filter experiments we don't want
    if primary is not None:
        df_exps = df_exps.loc[df_exps["primary_exp"] == primary]

    if exclude is not None:
        df_exps = df_exps.loc[df_exps["exclude"] == exclude]
    
    df_exps_animals = df_exps.merge(df_animals, on='animal_id', how='inner')

    return df_exps_animals
