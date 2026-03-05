import numpy as np
import pandas as pd
from paths import config

def get_exps(cfg:config.M2PConfig, get_excluded=True, primary_only=False, exp_ids=None, exp_indexes=None):

    dfcsv = pd.read_csv(cfg.meta_exps_file)

    if exp_ids:
        df = dfcsv.loc[dfcsv['exp_id'].isin(exp_ids)]
    elif exp_indexes:
        df = dfcsv.loc[dfcsv['exp_index'].isin(exp_ids)]
    else:
        df = dfcsv

    if not get_excluded:
        df = df.loc[df['exclude'] == 0]

    if primary_only:
        df = df.loc[df['primary_exp'] == 1]

    return df

def get_exp_ids(cfg:config.M2PConfig, get_excluded=True, primary_only=False):

    exps_df = get_exps(cfg, get_excluded=get_excluded, primary_only=primary_only)


    return exps_df['exp_id'].values

def get_animals(cfg:config.M2PConfig):

    dfcsv = pd.read_csv(cfg.meta_animals_file)

    dfcsv['animal_id'] = dfcsv['animal_id'].astype(str)

    return dfcsv

def get_bad_2p(df):

    bad_2p_str = df["bad_2p_frames"]

    if isinstance(bad_2p_str, pd.Series):
        bad_2p_str = bad_2p_str.iloc[0]

    bad_index_list = []
    if not pd.isnull(bad_2p_str):
        for badpart in bad_2p_str.split(";"):
            if badpart == "":
                continue
            start_time, end_time = badpart.split("-")
            start_frame = int(start_time)
            if end_time == "end":
                end_frame = -1
            else:
                end_frame = int(end_time)
            bad_index_list.append((start_frame, end_frame))

    return bad_index_list


def get_bad_2p_indexes(cfg, exp_id, n_frames):

    df_exp = get_exps(cfg, exp_ids=[exp_id])

    badpartlist = get_bad_2p(df_exp)

    # I thought bad indexes was a list of indexes but it seems its a bool array now
    # bad_indexes = np.array([])
    # for badpart in badpartlist:
    #     bad_indexes = np.concatenate((bad_indexes, np.arange(badpart[0]-1, badpart[1]-1)))
    bad_indexes = np.full((n_frames), False)
    for badpart in badpartlist:
        bad_indexes[badpart[0]-1: badpart[1]] = True

    return bad_indexes

def get_bad_behav(df, fps, n_frames=None):
    """

    :param df: dataframe of experiments
    :param fpd: the frame rate so time can be converted to frames
    :param n_frames: the total number of frames in the data, so 'end' is converted to the
    actual last frame rather than -1
    :return: List of tuples of periods of bad behaviour in frames. -1 means last
    """
    bad_str = df["bad_behav_times"]

    if isinstance(bad_str, pd.Series):
        bad_str = bad_str.iloc[0]
        
    bad_frame_list = []
    bad_index_list = []

    # If the bad_str is empty there are no bad frames, if it's a ? then it hasn't been checked.
    if not pd.isnull(bad_str) and bad_str != "?":
        for badpart in bad_str.split(";"):
            if badpart == "":
                continue
            start_time, end_time = badpart.split("-")
            start_frame = int(round(get_secs(start_time) * fps))
            if end_time == "end":
                if n_frames is not None:
                    end_frame = n_frames
                    end_time = get_mmss(end_frame / fps)
                else:
                    end_frame = -1
            else:
                end_frame = int(round(get_secs(end_time) * fps))
            bad_frame_list.append((start_frame, end_frame))
            bad_index_list.append((start_time, end_time))

    return bad_frame_list, bad_index_list


def get_bad_behav_indexes(cfg, exp_id, fps, n_frames):

    df_exp = get_exps(cfg, exp_ids=[exp_id])
    bad_frame_list, bad_index_list = get_bad_behav(df_exp, fps, n_frames)

    bad_indexes = np.full((n_frames), False)
    for badpart in bad_frame_list:
        bad_indexes[badpart[0]-1: badpart[1]] = True

    return bad_indexes

def get_good_bad_behav_time(df, fps, n_frames):
    bad_frame_list, bad_time_list = get_bad_behav(df, fps, n_frames)

    time_bad = 0
    for t in bad_time_list:
        time_bad += get_secs(t[1]) - get_secs(t[0])

    time_total = n_frames / fps

    time_good = time_total - time_bad

    return time_good, time_bad

def get_secs(mmss):
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)

def get_mmss(seconds):
    m, s = divmod(int(round(seconds, 0)), 60)
    return "{:02d}:{:02d}".format(int(m), int(round(s, 0)))

def mmss_lead_zeros(mmss):
    m, s = mmss.split(":")
    return "{:02d}:{:02d}".format(int(m), int(s))


def get_vid_orientation(cfg:config.M2PConfig, exp_id):
    df_exps = get_exps(cfg, exp_ids=[exp_id])

    return float(df_exps['orientation'].values[0])

def get_animal_id_from_exp_id(exp_id):
    return exp_id.split("_")[-1]