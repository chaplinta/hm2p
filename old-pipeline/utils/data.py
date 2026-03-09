import numpy as np
import pandas as pd
from scipy.stats import bootstrap

def df_grp_bin(df, grp_col, data_col, metrics, bins, absolute=False):

    if absolute:
        df_grp = pd.cut(df[grp_col].abs(), bins=bins)
    else:
        df_grp = pd.cut(df[grp_col], bins=bins)
    df_grp_by = df.groupby(df_grp)[data_col].aggregate(metrics)

    return df_grp_by

def df_grp_bin_ci(df, grp_col, data_col, bins, n_resamples, absolute=False):

    if absolute:
        df_grp = pd.cut(df[grp_col].abs(), bins=bins)
    else:
        df_grp = pd.cut(df[grp_col], bins=bins)

    df_grp_data = df.groupby(df_grp)[data_col]
    df_grp_by = df.groupby(df_grp)[data_col].aggregate(["mean"])
    df_grp_by_ci = df_grp_data.apply(lambda x: bootstrap(data=(x.values,), statistic=np.mean, n_resamples=n_resamples))

    # Too stupid to figure out how to access each bootstrap result with out a loop.
    ci_lo = np.zeros(df_grp_by_ci.shape[0])
    ci_hi = np.zeros(df_grp_by_ci.shape[0])
    for i_row in range(df_grp_by_ci.shape[0]):
        ci_lo[i_row] = df_grp_by_ci[i_row].confidence_interval.low
        ci_hi[i_row] = df_grp_by_ci[i_row].confidence_interval.high

    return df_grp_by, ci_lo, ci_hi


def df_grp_bin_roll(df, grp_col, data_col, metrics, bins, n_rolls, min_roll=None, max_roll=None):


    df_copy = df.copy()
    data_orig = df_copy[data_col]
    boot_data = np.zeros((len(metrics), bins.size - 1, n_rolls * 2))
    if min_roll == None:
        min_roll = 1
    if max_roll == None:
        max_roll = df_copy.shape[0]
    rolls_rand_pos = np.random.randint(min_roll, max_roll, n_rolls)
    rolls_rand_neg = np.random.randint(-max_roll, -min_roll, n_rolls)
    rolls_rand = np.hstack((rolls_rand_neg, rolls_rand_pos))
    for i_roll, offset in enumerate(rolls_rand):

        df_copy[data_col] = np.roll(data_orig, offset)
        df_grp_by = df_grp_bin(df_copy, grp_col, data_col, metrics, bins)
        for i_met, met in enumerate(metrics):
            binned_met_vals = df_grp_by[met].values
            boot_data[i_met, :, i_roll] = binned_met_vals

    boot_means = np.mean(boot_data, axis=2)
    boot_ci_lo = np.percentile(boot_data, 2.5, axis=2)
    boot_ci_hi = np.percentile(boot_data, 97.5, axis=2)

    df[data_col] = data_orig

    return (boot_means, boot_ci_lo, boot_ci_hi)