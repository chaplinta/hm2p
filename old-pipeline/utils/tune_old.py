import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import utils.data
from utils import misc
import os
import pycircstat as pcs
from dataclasses import dataclass


@dataclass
class CircTestData:
    cv1: float
    cv2: float

    cv1_sig: bool
    cv2_sig: bool

    cv1_bi: float
    cv2_bi: float

    cv1_bi_sig: bool
    cv2_bi_sig: bool

    is_bi1: bool
    is_bi2: bool


def cirv_var(df1, df2, grp_col, val_col, bins, smooth_sigma, n_boots_1tail):

    df_grp1 = utils.data.df_grp_bin(df1, grp_col, val_col, ["mean"], bins)
    df_grp2 = utils.data.df_grp_bin(df2, grp_col, val_col, ["mean"], bins)

    grp1_mean = df_grp1["mean"].values
    grp2_mean = df_grp2["mean"].values

    if smooth_sigma:
        grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
        grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')

    grp1_mean, grp1_mean_min, grp1_mean_max = norm_circ(grp1_mean)
    grp2_mean, grp2_mean_min, grp2_mean_max = norm_circ(grp2_mean)

    angles = np.deg2rad(bins[1:])
    bin_size = angles[1] - angles[0]
    angles = angles - bin_size/2
    cv1 = pcs.descriptive.var(angles, w=grp1_mean, d=bin_size)
    cv2 = pcs.descriptive.var(angles, w=grp2_mean, d=bin_size)
    p1, z1 = pcs.tests.rayleigh(angles, w=grp1_mean, d=bin_size)
    p2, z2 = pcs.tests.rayleigh(angles, w=grp2_mean, d=bin_size)

    # Bootstrap
    min_roll = 100
    max_roll = df1.shape[0]
    rolls_pos = np.random.randint(min_roll, max_roll, n_boots_1tail)
    rolls_neg = -rolls_pos #np.random.randint(-min_roll, -max_roll, n_rolls)
    both_rolls = np.hstack((rolls_neg, rolls_pos))
    n_boots = n_boots_1tail * 2

    df1_boot = df1.copy()
    df2_boot = df2.copy()

    # Fold angles for bidirectional test
    df1_boot["fold"] = np.copy(df1_boot[grp_col].values)
    df2_boot["fold"] = np.copy(df2_boot[grp_col].values)

    df1_half = df1_boot["fold"] > 180
    df2_half = df2_boot["fold"] > 180

    df1_boot["fold"][df1_half] = 180 - (df1_boot["fold"][df1_half].values - 180)
    df2_boot["fold"][df2_half] = 180 - (df2_boot["fold"][df2_half].values - 180)

    df1_boot["fold"] = df1_boot["fold"] * 2
    df2_boot["fold"] = df2_boot["fold"] * 2

    # Calc bidirectional cv
    df_grp1 = utils.data.df_grp_bin(df1_boot, "fold", val_col, ["mean"], bins)
    df_grp2 = utils.data.df_grp_bin(df2_boot, "fold", val_col, ["mean"], bins)

    grp1_mean = df_grp1["mean"].values
    grp2_mean = df_grp2["mean"].values

    if smooth_sigma:
        grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
        grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')

    grp1_mean, grp1_bimean_min, grp1_bimean_max = norm_circ(grp1_mean)
    grp2_mean, grp2_bimean_min, grp2_bimean_max = norm_circ(grp2_mean)

    cv1_bi = pcs.descriptive.var(angles, w=grp1_mean, d=bin_size)
    cv2_bi = pcs.descriptive.var(angles, w=grp2_mean, d=bin_size)
    p1_bi, z1_bi = pcs.tests.rayleigh(angles, w=grp1_mean, d=bin_size)
    p2_bi, z2_bi = pcs.tests.rayleigh(angles, w=grp2_mean, d=bin_size)

    # Get ready for bootstrap
    data_vals_orig1 = df1_boot[val_col].values
    data_vals_orig2 = df2_boot[val_col].values

    cv1_boots = np.zeros(n_boots)
    cv2_boots = np.zeros(n_boots)
    cv1_bi_boots = np.zeros(n_boots)
    cv2_bi_boots = np.zeros(n_boots)
    pd1_boots = np.zeros(n_boots)
    pd2_boots = np.zeros(n_boots)
    pd1_bi_boots = np.zeros(n_boots)
    pd2_bi_boots = np.zeros(n_boots)
    for i_boot in range(n_boots):

        roll_val = both_rolls[i_boot]
        df1_boot[val_col] = np.roll(data_vals_orig1, roll_val)
        df2_boot[val_col] = np.roll(data_vals_orig2, roll_val)

        df_grp1 = utils.data.df_grp_bin(df1_boot, grp_col, val_col, ["mean"], bins)
        df_grp2 = utils.data.df_grp_bin(df2_boot, grp_col, val_col, ["mean"], bins)
        df_grp1_fold = utils.data.df_grp_bin(df1_boot, "fold", val_col, ["mean"], bins)
        df_grp2_fold = utils.data.df_grp_bin(df2_boot, "fold", val_col, ["mean"], bins)

        grp1_mean = df_grp1["mean"].values
        grp2_mean = df_grp2["mean"].values
        grp1_fold_mean = df_grp1_fold["mean"].values
        grp2_fold_mean = df_grp2_fold["mean"].values

        if smooth_sigma:
            grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
            grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')
            grp1_fold_mean = sp.ndimage.gaussian_filter1d(grp1_fold_mean, sigma=smooth_sigma, mode='wrap')
            grp2_fold_mean = sp.ndimage.gaussian_filter1d(grp2_fold_mean, sigma=smooth_sigma, mode='wrap')

        grp1_mean = norm_circ(grp1_mean, min_val=grp1_mean_min, max_val=grp1_mean_max)[0]
        grp2_mean = norm_circ(grp2_mean, min_val=grp2_mean_min, max_val=grp2_mean_max)[0]
        grp1_fold_mean = norm_circ(grp1_fold_mean, min_val=grp1_bimean_min, max_val=grp1_bimean_max)[0]
        grp2_fold_mean = norm_circ(grp2_fold_mean, min_val=grp2_bimean_min, max_val=grp2_bimean_max)[0]

        cv1_boot = pcs.descriptive.var(angles, w=grp1_mean)
        cv2_boot = pcs.descriptive.var(angles, w=grp2_mean)
        cv1_bi_boot = pcs.descriptive.var(angles, w=grp1_fold_mean)
        cv2_bi_boot = pcs.descriptive.var(angles, w=grp2_fold_mean)

        cv1_boots[i_boot] = cv1_boot
        cv2_boots[i_boot] = cv2_boot
        cv1_bi_boots[i_boot] = cv1_bi_boot
        cv2_bi_boots[i_boot] = cv2_bi_boot

        pd1_boot = pcs.descriptive.mean(angles, w=grp1_mean)
        pd2_boot = pcs.descriptive.mean(angles, w=grp2_mean)
        pd1_bi_boot = pcs.descriptive.mean(angles, w=grp1_fold_mean)
        pd2_bi_boot = pcs.descriptive.mean(angles, w=grp2_fold_mean)

        pd1_boots[i_boot] = pd1_boot
        pd2_boots[i_boot] = pd2_boot
        pd1_bi_boots[i_boot] = pd1_bi_boot
        pd2_bi_boots[i_boot] = pd2_bi_boot

    cv1_lower = np.percentile(cv1_boots, 5)
    cv2_lower = np.percentile(cv2_boots, 5)
    cv1_bi_lower = np.percentile(cv1_bi_boots, 5)
    cv2_bi_lower = np.percentile(cv2_bi_boots, 5)

    cv1_sig = cv1 < cv1_lower
    cv2_sig = cv2 < cv2_lower

    cv1_bi_sig = cv1_bi < cv1_bi_lower
    cv2_bi_sig = cv2_bi < cv2_bi_lower

    # cv1_sig = p1 < 0.05
    # cv2_sig = p2 < 0.05
    #
    # cv1_bi_sig = p1_bi < 0.05
    # cv2_bi_sig = p2_bi < 0.05

    is_bi1 = cv1_bi_sig and (not cv1_sig or cv1_bi < cv1)
    is_bi2 = cv2_bi_sig and (not cv2_sig or cv2_bi < cv2)

    same_tuning = False
    if cv1_sig and cv2_sig:
        same_tuning = True  # If dir same
    if cv1_bi_sig and cv2_bi_sig:
        same_tuning = True # If dir same

    return CircTestData(cv1=cv1,
                        cv2=cv2,
                        cv1_sig=cv1_sig,
                        cv2_sig=cv2_sig,
                        cv1_bi=cv1_bi,
                        cv2_bi=cv2_bi,
                        cv1_bi_sig=cv1_bi_sig,
                        cv2_bi_sig=cv2_bi_sig,
                        is_bi1=is_bi1,
                        is_bi2=is_bi2)

def norm_circ(data, min_val=None, max_val=None):

    data_norm = np.copy(data)
    if min_val is None:
        min_val = np.min(data)

    if min_val < 0:
        data_norm -= min_val
    elif min_val > 0:
        data_norm -= min_val * 0.9

    if max_val is None:
        max_val = np.max(data)
    data_norm = data_norm / max_val

    return data_norm, min_val, max_val

def curve_corr(df, grp_col, val_col, bins, sub_time, fps, smooth_sigma=None):

    # sub_time is the sub section of time to split the data.
    # e.g. sub_time = 30 splits the data into 30 second chunks, calculates 2 tuning curves, and then the correlation
    # between the two.
    # Too stupid to figure out how to do this without a loop
    n_subsamples = int(np.round(sub_time * fps))
    sub_indexes1 = np.full(df.shape[0], False, dtype=bool)
    n_subs = int(np.round(df.shape[0] / n_subsamples))
    for i in range(n_subs - 1):
        if i % 2 == 0:
            start = i * n_subsamples
            sub_indexes1[start:start+n_subsamples] = True
    sub_indexes2 = np.logical_not(sub_indexes1)

    df1 = df.copy().iloc[sub_indexes1]
    df2 = df.copy().iloc[sub_indexes2]

    df_grp1 = utils.data.df_grp_bin(df1, grp_col, val_col, ["mean"], bins)
    df_grp2 = utils.data.df_grp_bin(df2, grp_col, val_col, ["mean"], bins)

    grp1_mean = df_grp1["mean"].values
    grp2_mean = df_grp2["mean"].values

    if smooth_sigma:
        grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
        grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')

    (r, p) = sp.stats.pearsonr(grp1_mean, grp2_mean)

    return (grp1_mean, grp2_mean, r, p)


def curve_corr_boot(df, grp_col, val_col, bins, sub_time, fps,
                    smooth_sigma=None, min_roll=100, max_roll=3000, n_rolls=100):

    # sub_time is the sub section of time to split the data.
    # e.g. sub_time = 30 splits the data into 30 second chunks, calculates 2 tuning curves, and then the correlation
    # between the two.
    # Too stupid to figure out how to do this without a loop
    n_subsamples = int(np.round(sub_time * fps))
    sub_indexes1 = np.full(df.shape[0], False, dtype=bool)
    n_subs = int(np.round(df.shape[0] / n_subsamples))
    for i in range(n_subs - 1):
        if i % 2 == 0:
            start = i * n_subsamples
            sub_indexes1[start:start + n_subsamples] = True
    sub_indexes2 = np.logical_not(sub_indexes1)

    df1 = df.copy().iloc[sub_indexes1]
    df2 = df.copy().iloc[sub_indexes2]

    min_sub_roll = 1
    max_sub_roll = 3000
    sub_rolls_pos = np.random.randint(min_sub_roll, max_sub_roll, n_rolls)
    sub_rolls_neg = np.random.randint(-max_sub_roll, -min_sub_roll,  n_rolls)
    sub_rolls = np.hstack((sub_rolls_neg, sub_rolls_pos))

    # time_rolls_pos = np.random.randint(min_roll, max_roll, n_rolls)
    # time_rolls_neg = np.random.randint(-max_roll, -min_roll, n_rolls)
    # time_rolls = np.hstack((time_rolls_pos, time_rolls_neg))

    n_boots = n_rolls * 2
    r_vals_sub = np.zeros(n_boots)
    r_vals_time = np.zeros(n_boots)
    data_vals_orig1 = df1[val_col].values
    data_vals_orig2 = df2[val_col].values
    for i_boot in range(n_boots):

        roll_sub = sub_rolls[i_boot]
        df1[val_col] = np.roll(data_vals_orig1, roll_sub)
        df2[val_col] = np.roll(data_vals_orig2, roll_sub)

        df_grp1 = utils.data.df_grp_bin(df1, grp_col, val_col, ["mean", "sem", "count"], bins)
        df_grp2 = utils.data.df_grp_bin(df2, grp_col, val_col, ["mean", "sem", "count"], bins)

        grp1_mean = df_grp1["mean"].values
        grp2_mean = df_grp2["mean"].values

        if smooth_sigma:
            grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
            grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')

        (r_sub, p_sub) = sp.stats.pearsonr(grp1_mean, grp2_mean)

        r_vals_sub[i_boot] = r_sub

        # # Shift the time and see what we get.
        # roll_time = time_rolls[i_boot]
        # df_move[val_col] = np.roll(data_vals_orig, roll_time)
        #
        # df_move = df_move.copy().iloc[sub_indexes1_boot]
        # df2 = df_move.copy().iloc[sub_indexes2_boot]
        #
        # df_grp1 = m2putils.df_grp_agg(df_move, grp_col, val_col, ["mean", "sem", "count"], bins)
        # df_grp2 = m2putils.df_grp_agg(df2, grp_col, val_col, ["mean", "sem", "count"], bins)
        #
        # grp1_mean = df_grp1["mean"]
        # grp2_mean = df_grp2["mean"]
        #
        # if smooth_sigma:
        #     grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
        #     grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')
        #
        # (r_time, p_time) = sp.stats.pearsonr(grp1_mean, grp2_mean)
        #
        # r_vals_time[i_boot] = r_time

    r_sub_mean = np.mean(r_vals_sub)
    r_sub_low = np.percentile(r_vals_sub, 2.5)
    r_time_hi = np.percentile(r_vals_sub, 97.5)
    #sig =

    print(r_sub_mean, r_sub_low, r_time_hi)
    plt.figure()
    plt.hist(r_vals_sub)
    plt.show()

    return (r_sub_mean, r_sub_low, r_time_hi)

# def spat_mi_boot(df_move, grp_col, val_col, x_bins, y_bins, sub_time, fps,
#                     smooth_sigma=None, min_roll=100, max_roll=3000, n_rolls=100):
#
#     # Too stupid to figure out how to do this without a loop
#     n_subsamples = int(np.round(sub_time * fps))
#     sub_indexes1 = np.full(df_move.shape[0], False, dtype=bool)
#     n_subs = int(np.round(df_move.shape[0] / n_subsamples))
#     for i in range(n_subs - 1):
#         if i % 2 == 0:
#             start = i * n_subsamples
#             sub_indexes1[start:start+n_subsamples] = True
#     sub_indexes2 = np.logical_not(sub_indexes1)
#
#     min_sub_roll = 0
#     max_sub_roll = df_move.shape[0]
#     sub_rolls_pos = np.random.randint(min_sub_roll, max_sub_roll, n_rolls)
#     sub_rolls_neg = np.random.randint(-max_sub_roll, -min_sub_roll,  n_rolls)
#     sub_rolls = np.hstack((sub_rolls_neg, sub_rolls_pos))
#
#     time_rolls_pos = np.random.randint(min_roll, max_roll, n_rolls)
#     time_rolls_neg = np.random.randint(-max_roll, -min_roll, n_rolls)
#     time_rolls = np.hstack((time_rolls_pos, time_rolls_neg))
#
#     n_boots = n_rolls * 2
#     r_vals_sub = np.zeros(n_boots)
#     r_vals_time = np.zeros(n_boots)
#     data_vals_orig = df_move[val_col].values
#     for i_boot in range(n_boots - 1):
#
#         df_move[val_col] = data_vals_orig
#
#         roll_sub = sub_rolls[i_boot]
#         sub_indexes1_boot = np.roll(sub_indexes1, roll_sub)
#         sub_indexes2_boot = np.logical_not(sub_indexes1_boot)
#         df_move = df_move.copy().iloc[sub_indexes1_boot]
#         df2 = df_move.copy().iloc[sub_indexes2_boot]
#
#
#         df_grp1 = m2putils.df_grp_agg(df_move, grp_col, val_col, ["mean", "sem", "count"], bins)
#         df_grp2 = m2putils.df_grp_agg(df2, grp_col, val_col, ["mean", "sem", "count"], bins)
#
#         grp1_mean = df_grp1["mean"].values
#         grp2_mean = df_grp2["mean"].values
#
#         if smooth_sigma:
#             grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
#             grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')
#
#         (r_sub, p_sub) = sp.stats.pearsonr(grp1_mean, grp2_mean)
#
#         r_vals_sub[i_boot] = r_sub
#
#         # Shift the time and see what we get.
#         roll_time = time_rolls[i_boot]
#         df_move[val_col] = np.roll(data_vals_orig, roll_time)
#
#         df_move = df_move.copy().iloc[sub_indexes1_boot]
#         df2 = df_move.copy().iloc[sub_indexes2_boot]
#
#         df_grp1 = m2putils.df_grp_agg(df_move, grp_col, val_col, ["mean", "sem", "count"], bins)
#         df_grp2 = m2putils.df_grp_agg(df2, grp_col, val_col, ["mean", "sem", "count"], bins)
#
#         grp1_mean = df_grp1["mean"]
#         grp2_mean = df_grp2["mean"]
#
#         if smooth_sigma:
#             grp1_mean = sp.ndimage.gaussian_filter1d(grp1_mean, sigma=smooth_sigma, mode='wrap')
#             grp2_mean = sp.ndimage.gaussian_filter1d(grp2_mean, sigma=smooth_sigma, mode='wrap')
#
#         (r_time, p_time) = sp.stats.pearsonr(grp1_mean, grp2_mean)
#
#         r_vals_time[i_boot] = r_time
#
#     r_sub_mean = np.mean(r_vals_sub)
#     r_sub_low = np.percentile(r_vals_sub, 2.5)
#     r_time_hi = np.percentile(r_vals_time, 97.5)
#     sig = r_sub_low > r_time_hi
#
#     print(r_sub_mean, r_sub_low, r_time_hi, sig)
#     plt.figure()
#     plt.hist(r_vals_sub)
#     plt.hist(r_vals_time)
#     plt.show()
#
#     return (r_sub_mean, r_sub_low, r_time_hi, sig)

def plot_hdheading(df_comb, df_light, df_dark, hd_label, plot_type, roi_label, bins_hd,
                   roi_type, i_roi, ca_data_label, time_offset,
                   plot_combined, plot_tune_dir, plot_lightvsdark, plot_cart, plot_polar,
                   fps,
                   soma_dend_pairs, plot_tune_pairs_dir, plot_tune_hdmod_dir,
                   ylabel_text, linewidth, dpi):

    is_soma = roi_type == "soma"

    df_grp = utils.data.df_grp_bin(df_comb, hd_label, roi_label, ["mean", "sem", "count"],
                                   bins_hd)
    df_grp_light = utils.data.df_grp_bin(df_light, hd_label, roi_label, ["mean", "sem", "count"],
                                         bins_hd)
    df_grp_dark = utils.data.df_grp_bin(df_dark, hd_label, roi_label, ["mean", "sem", "count"],
                                        bins_hd)

    # (boot_mean, boot_ci_lo, boot_ci_hi) = m2putils.df_grp_agg_roll(df_comb,
    #                                                                label_hd,
    #                                                                roi_label,
    #                                                                ["mean"],
    #                                                                bins_hd,
    #                                                                100,
    #                                                                min_roll=1000,
    #                                                                max_roll=10000)
    #
    # hd_boot_sig = boot_mean[0, :]

    # (boot_light_mean, boot_light_ci_lo, boot_light_ci_hi) = m2putils.df_grp_agg_roll(df_light,
    #                                                                                  bu.HD_ABS_FILT,
    #                                                                                  roi_label,
    #                                                                                  ["mean"],
    #                                                                                  bins_hd,
    #                                                                                  200,
    #                                                                                  max_roll=3000)
    #
    # (boot_dark_mean, boot_dark_ci_lo, boot_dark_ci_hi) = m2putils.df_grp_agg_roll(df_dark,
    #                                                                               bu.HD_ABS_FILT,
    #                                                                               roi_label,
    #                                                                               ["mean"],
    #                                                                               bins_hd,
    #                                                                               200,
    #                                                                               max_roll=3000)
    # hd_boot_light_sig = boot_light_ci_hi[0, :]
    # hd_boot_dark_sig = boot_dark_ci_hi[0, :]

    # Smooth it lol
    resp_hd_mean = df_grp["mean"].values
    resp_hd_light_mean = df_grp_light["mean"].values
    resp_hd_dark_mean = df_grp_dark["mean"].values

    hd_sigma_deg = 10
    hd_sigma = hd_sigma_deg * (bins_hd.size - 1) / 360
    resp_hd_mean = sp.ndimage.gaussian_filter1d(resp_hd_mean, sigma=hd_sigma, mode='wrap')
    resp_hd_light_mean = sp.ndimage.gaussian_filter1d(resp_hd_light_mean, sigma=hd_sigma, mode='wrap')
    resp_hd_dark_mean = sp.ndimage.gaussian_filter1d(resp_hd_dark_mean, sigma=hd_sigma, mode='wrap')

    circ_test_data = cirv_var(df_light, df_dark, hd_label, roi_label, bins_hd, hd_sigma, 100)

    (resp_hd1_mean, resp_hd2_mean, hd_r, hd_p) = curve_corr(df=df_comb,
                                                            grp_col=hd_label,
                                                            val_col=roi_label,
                                                            bins=bins_hd,
                                                            sub_time=30,
                                                            fps=fps,
                                                            smooth_sigma=hd_sigma)

    # curve_corr_boot(df_move=df_comb,
    #                 grp_col=bu.HD_ABS_FILT,
    #                 val_col=roi_label,
    #                 bins=bins_hd,
    #                 sub_time=30,
    #                 fps=fps,
    #                 smooth_sigma=hd_sigma)

    (ld_r, ld_p) = sp.stats.pearsonr(resp_hd_light_mean, resp_hd_dark_mean)

    resp_hd_mean_norm = np.copy(resp_hd_mean)
    resp_hd_light_mean_norm = np.copy(resp_hd_light_mean)
    resp_hd_dark_mean_norm = np.copy(resp_hd_dark_mean)

    resp_hd_min = np.min(resp_hd_mean)
    if resp_hd_min < 0:
        resp_hd_mean_norm -= resp_hd_min
    elif resp_hd_min > 0:
        resp_hd_mean_norm -= resp_hd_min * 0.9

    resp_hd_lightdark_min = np.min(np.hstack([resp_hd_light_mean, resp_hd_dark_mean]))
    if resp_hd_lightdark_min < 0:
        resp_hd_light_mean_norm -= resp_hd_lightdark_min
        resp_hd_dark_mean_norm -= resp_hd_lightdark_min
    elif resp_hd_lightdark_min > 0:
        resp_hd_light_mean_norm -= resp_hd_lightdark_min * 0.9
        resp_hd_dark_mean_norm -= resp_hd_lightdark_min * 0.9

    resp_hd_max = np.max(resp_hd_mean_norm)
    resp_hd_lightdark_max = np.max(np.hstack([resp_hd_light_mean_norm, resp_hd_dark_mean_norm]))

    resp_hd_mean_norm = resp_hd_mean_norm / resp_hd_max
    resp_hd_light_mean_norm = resp_hd_light_mean_norm / resp_hd_lightdark_max
    resp_hd_dark_mean_norm = resp_hd_dark_mean_norm / resp_hd_lightdark_max

    angles = bins_hd[:-1]

    # Combined
    if plot_cart:
        if plot_combined:
            fig = plt.figure(tight_layout=False)
            plt.plot(angles, resp_hd_mean, linewidth=linewidth, color='0')
            plt.plot(angles, resp_hd1_mean, linestyle=":", linewidth=linewidth, color='r')
            plt.plot(angles, resp_hd2_mean, linestyle=":", linewidth=linewidth, color='b')
            plt.ylabel(ylabel_text)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().set_xlabel("Direction (°)")
            plt.title('r={:.2f} p={:.3f}'.format(hd_r, hd_p))
            plt.tight_layout()

            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.{ptype}.cart.{ca}.offset_{off:+000}.png"
                                         .format(roi=roi_type,
                                                 i=i_roi,
                                                 ptype=plot_type,
                                                 ca=ca_data_label,
                                                 off=time_offset))
            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)
            misc.copy_rsig(hd_r, hd_p, plot_img_path, plot_tune_hdmod_dir)

        # Light vs dark
        if plot_lightvsdark:
            fig = plt.figure(tight_layout=False)
            plt.plot(angles, resp_hd_light_mean, label='light', linewidth=linewidth, color='0.5')
            plt.plot(angles, resp_hd_dark_mean, label='dark', linewidth=linewidth, color='0')
            plt.ylabel(ylabel_text)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().yaxis.set_ticks_position('left')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().set_xlabel("Direction (°)")
            plt.title('r={:.2f} p={:.3f}'.format(ld_r, ld_p))
            plt.tight_layout()

            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.{ptype}.cart.{ca}.offset_{off:+000}.ld.png"
                                         .format(roi=roi_type,
                                                 i=i_roi,
                                                 ptype=plot_type,
                                                 ca=ca_data_label,
                                                 off=time_offset))
            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)
            misc.copy_rsig(hd_r, hd_p, plot_img_path, plot_tune_hdmod_dir)
            plt.cla()
            plt.clf()
            plt.close('all')

    # Polar plot.
    if plot_polar:
        # Combined
        if plot_combined:
            fig = plt.figure(tight_layout=False)
            ax = plt.subplot(111, projection='polar', aspect='equal')
            polar_theta = np.append(angles, angles[0])
            polar_r = np.append(resp_hd_mean_norm, resp_hd_mean_norm[0])
            plt.plot(np.deg2rad(polar_theta), polar_r, linewidth=linewidth, color='0')
            ax.set_rmax(1)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.gca().yaxis.grid(False)
            plt.gca().xaxis.grid(linewidth=2)
            plt.tight_layout()

            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.{ptype}.polar.{ca}.offset_{off:+000}.png"
                                         .format(roi=roi_type,
                                                 i=i_roi,
                                                 ptype=plot_type,
                                                 ca=ca_data_label,
                                                 off=time_offset))
            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)
            misc.copy_rsig(hd_r, hd_p, plot_img_path, plot_tune_hdmod_dir)

            plt.cla()
            plt.clf()
            plt.close('all')

        # Light vs dark
        if plot_lightvsdark:
            fig = plt.figure(tight_layout=False)
            ax = plt.subplot(111, projection='polar', aspect='equal')
            polar_theta = np.append(angles, angles[0])

            polar_r_light = np.append(resp_hd_light_mean_norm, resp_hd_light_mean_norm[0])
            plt.plot(np.deg2rad(polar_theta), polar_r_light, linewidth=linewidth, color='0.5')
            polar_r_dark = np.append(resp_hd_dark_mean_norm, resp_hd_dark_mean_norm[0])
            plt.plot(np.deg2rad(polar_theta), polar_r_dark, linewidth=linewidth, color='0')
            title_str = ""
            sig1_text = ""
            sig1_bi_text = ""
            sig2_text = ""
            sig2_bi_text = ""
            if circ_test_data.cv1_sig:
                sig1_text = "*"
            if circ_test_data.cv1_bi_sig:
                sig1_bi_text = "*"
            if circ_test_data.cv2_sig:
                sig2_text = "*"
            if circ_test_data.cv2_bi_sig:
                sig2_bi_text = "*"

            title_str += "Light cv={:0.2f}".format(circ_test_data.cv1) + sig1_text
            title_str += "/{:0.2f}".format(circ_test_data.cv1_bi) + sig1_bi_text
            title_str += " Dark cv={:0.2f}".format(circ_test_data.cv2) + sig2_text
            title_str += "/{:0.2f}".format(circ_test_data.cv2_bi) + sig2_bi_text

            plt.title(title_str)
            ax.set_rmax(1)
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.gca().yaxis.grid(False)
            plt.gca().xaxis.grid(linewidth=2)
            plt.tight_layout()

            plot_img_path = os.path.join(plot_tune_dir, "{roi}.roi-{i}.{ptype}.polar.{ca}.offset_{off:+000}.ld.png"
                                         .format(roi=roi_type,
                                                 i=i_roi,
                                                 ptype=plot_type,
                                                 ca=ca_data_label,
                                                 off=time_offset))
            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
            misc.copy_pair(is_soma, i_roi, soma_dend_pairs, plot_img_path, plot_tune_pairs_dir)
            misc.copy_rsig(hd_r, hd_p, plot_img_path, plot_tune_hdmod_dir)

            plt.cla()
            plt.clf()
            plt.close('all')


