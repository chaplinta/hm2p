import pandas as pd
from utils import data as du
import scipy
import numpy as np
import pycircstat as pcs
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score


def calc_index(x1, x2):
    return (x1 - x2) / (x1 + x2)

def get_dfs(cfg):
    df_roi = pd.read_hdf(cfg.db_ca_roi_file)
    df_ca = pd.read_hdf(cfg.db_ca_file)
    df_behave = pd.read_hdf(cfg.db_behave_frames_file)

    df = pd.merge(df_roi, df_ca, on=["exp_id", "roi_id"], how="inner")
    df = pd.merge(df, df_behave, on=["exp_id", "frame_id"], how="left")

    return df, df_roi, df_ca, df_behave


@dataclass
class HD_stats:
    resp_mean: np.array
    resp_mean_lo: np.array
    resp_mean_hi: np.array
    resp_mean_boot_mean: np.array
    resp_mean_norm: np.array
    resp_mean_norm_lo: np.array
    resp_mean_norm_hi: np.array
    resp_mean_norm_boot_mean: np.array
    prefdir: float
    cv: float
    ray_p: float
    ray_z: float
    stab_r: float
    stab_p: float
    cv_lo: float
    stab_r_hi: float


@dataclass
class AHV_stats:
    resp_mean: np.array = None
    resp_mean_lo: np.array = None
    resp_mean_hi: np.array = None
    resp_mean_boot_mean: np.array = None
    left_r: float = None
    left_p: float = None
    left_r_hi: float = None
    left_m: float = None
    left_c: float = None
    left_m_hi: float = None
    right_r: float = None
    right_p: float = None
    right_r_hi: float = None
    right_m: float = None
    right_c: float = None
    right_m_hi: float = None
    di: float = None
    di_lo: float = None
    di_hi: float = None


def calc_HD(df,
            df_move,
            label_hd,
            label_ahv,
            label_ca,
            bins_hd,
            bins_ahv,
            n_boots=10,
            min_roll=20 * 10):

    hd_resp_mean = calc_HD_curve(df_move, label_hd, label_ca, bins_hd)
    #hd_resp_mean, hd_resp_mean_hi, hd_resp_mean_lo = calc_HD_curve(df_move, label_hd, label_ca, bins_hd, n_resamples=n_boots)

    hd_resp_mean_norm, hd_prefdir, hd_cv, hd_ray_p, hd_ray_z = calc_HD_circstats(hd_resp_mean, bins_hd)

    ahv_resp_mean = calc_AHV_curve(df, label_ahv, label_ca, bins_ahv)
    ahv_main_stats = calc_AHV_stats(ahv_resp_mean, bins_ahv, df[label_ca].values, df[label_ahv].values)

    # Shuffle tests
    df_copy = df_move.copy()
    data_vals_orig = df_copy[label_ca].values
    ca_boot = get_boots(n_boots, min_roll, df_copy[label_ca].values)

    hd_resp_dist = np.zeros((n_boots, hd_resp_mean.shape[0]))
    hd_resp_norm_dist = np.zeros((n_boots, hd_resp_mean.shape[0]))
    hd_cv_dist = np.zeros(n_boots)

    ahv_resp_dist = np.zeros((n_boots, ahv_resp_mean.shape[0]))
    ahv_left_r_dist = np.zeros(n_boots)
    ahv_right_r_dist = np.zeros(n_boots)
    ahv_left_m_dist = np.zeros(n_boots)
    ahv_right_m_dist = np.zeros(n_boots)
    ahv_di_dist = np.zeros(n_boots)
    for i_boot in range(n_boots):
        df_copy[label_ca] = ca_boot[i_boot, :]

        boot_hd_resp_mean = calc_HD_curve(df_copy, label_hd, label_ca, bins_hd)
        boot_hd_resp_mean_norm, boot_prefdir, boot_cv, boot_p, boot_z = calc_HD_circstats(boot_hd_resp_mean, bins_hd)
        hd_resp_dist[i_boot, :] = boot_hd_resp_mean
        hd_resp_norm_dist[i_boot, :] = boot_hd_resp_mean_norm
        hd_cv_dist[i_boot] = boot_cv

        boot_ahv_resp_mean = calc_AHV_curve(df_copy, label_ahv, label_ca, bins_ahv)
        ahv_resp_dist[i_boot, :] = boot_ahv_resp_mean
        boot_ahv_stats = calc_AHV_stats(boot_ahv_resp_mean, bins_ahv, df_copy[label_ca].values, df_copy[label_ahv].values)
        ahv_left_r_dist[i_boot] = boot_ahv_stats.left_r
        ahv_right_r_dist[i_boot] = boot_ahv_stats.right_r
        ahv_left_m_dist[i_boot] = boot_ahv_stats.left_m
        ahv_right_m_dist[i_boot] = boot_ahv_stats.right_m
        ahv_di_dist[i_boot] = boot_ahv_stats.di

    hd_resp_lo = np.percentile(hd_resp_dist, 5, axis=0)
    hd_resp_hi = np.percentile(hd_resp_dist, 95, axis=0)
    hd_resp_boot_mean = np.mean(hd_resp_dist, axis=0)
    hd_resp_norm_lo = np.percentile(hd_resp_norm_dist, 5, axis=0)
    hd_resp_norm_hi = np.percentile(hd_resp_norm_dist, 95, axis=0)
    hd_cv_lo = np.percentile(hd_cv_dist, 5)
    hd_resp_norm_boot_mean = np.mean(hd_resp_norm_dist, axis=0)

    ahv_resp_lo = np.percentile(ahv_resp_dist, 5, axis=0)
    ahv_resp_hi = np.percentile(ahv_resp_dist, 95, axis=0)
    ahv_resp_boot_mean = np.mean(ahv_resp_dist, axis=0)
    ahv_left_r_hi = np.percentile(ahv_left_r_dist, 95)
    ahv_right_r_hi = np.percentile(ahv_right_r_dist, 95)
    ahv_left_m_hi = np.percentile(ahv_left_m_dist, 95)
    ahv_right_m_hi = np.percentile(ahv_right_m_dist, 95)
    ahv_di_lo = np.percentile(ahv_di_dist, 5)
    ahv_di_hi = np.percentile(ahv_di_dist, 95)

    # Stability tests. Correlated first half of session to last, bootstrap.
    n_pts_half = int(round(len(data_vals_orig) / 2))

    df_copy1 = df_move.copy().iloc[0:n_pts_half]
    df_copy2 = df_move.copy().iloc[n_pts_half + 1:-1]

    resp_mean1 = calc_HD_curve(df_copy1, label_hd, label_ca, bins_hd)
    resp_mean2 = calc_HD_curve(df_copy2, label_hd, label_ca, bins_hd)

    ca_boot1 = get_boots(n_boots, min_roll, df_copy1[label_ca].values)
    ca_boot2 = get_boots(n_boots, min_roll, df_copy2[label_ca].values)

    (stable_r, stable_p) = scipy.stats.pearsonr(resp_mean1, resp_mean2)

    r_dist = np.zeros(n_boots)
    for i_boot in range(n_boots):
        df_copy1[label_ca] = ca_boot1[i_boot, :]
        df_copy2[label_ca] = ca_boot2[i_boot, :]

        boot_resp_mean1 = calc_HD_curve(df_copy1, label_hd, label_ca, bins_hd)
        boot_resp_mean2 = calc_HD_curve(df_copy2, label_hd, label_ca, bins_hd)

        (boot_r, boot_p) = scipy.stats.pearsonr(boot_resp_mean1, boot_resp_mean2)

        r_dist[i_boot] = boot_r

    stable_r_hi = np.percentile(r_dist, 95)

    hd_stats = HD_stats(resp_mean=hd_resp_mean,
                        resp_mean_lo=hd_resp_lo,
                        resp_mean_hi=hd_resp_hi,
                        resp_mean_boot_mean=hd_resp_boot_mean,
                        resp_mean_norm=hd_resp_mean_norm,
                        resp_mean_norm_lo=hd_resp_norm_lo,
                        resp_mean_norm_hi=hd_resp_norm_hi,
                        resp_mean_norm_boot_mean=hd_resp_norm_boot_mean,
                        prefdir=hd_prefdir,
                        cv=hd_cv,
                        ray_p=hd_ray_p,
                        ray_z=hd_ray_z,
                        stab_r=stable_r,
                        stab_p=stable_p,
                        cv_lo=hd_cv_lo,
                        stab_r_hi=stable_r_hi)

    ahv_stats = AHV_stats(resp_mean=ahv_resp_mean,
                          resp_mean_lo=ahv_resp_lo,
                          resp_mean_hi=ahv_resp_hi,
                          resp_mean_boot_mean=ahv_resp_boot_mean,
                          left_r=ahv_main_stats.left_r,
                          left_p=ahv_main_stats.left_p,
                          left_r_hi=ahv_left_r_hi,
                          left_m=ahv_main_stats.left_m,
                          left_c=ahv_main_stats.left_c,
                          left_m_hi=ahv_left_m_hi,
                          right_r=ahv_main_stats.right_r,
                          right_p=ahv_main_stats.right_p,
                          right_r_hi=ahv_right_r_hi,
                          right_m=ahv_main_stats.right_m,
                          right_c=ahv_main_stats.right_c,
                          right_m_hi=ahv_right_m_hi,
                          di=ahv_main_stats.di,
                          di_lo=ahv_di_lo,
                          di_hi=ahv_di_hi)

    return hd_stats, ahv_stats


def calc_HD_curve(df, hd_label, ca_label, bins_hd, n_resamples=None, frame_int=0.1):

    # if n_resamples is None:
    #     df_grp = du.df_grp_agg(df, hd_label, ca_label, ["mean"], bins_hd)
    #     resp_mean_low = np.zeros(df_grp["mean"].values.shape)
    #     resp_mean_hi = np.zeros(df_grp["mean"].values.shape)
    # else:
    #     df_grp, resp_mean_low, resp_mean_hi = du.df_grp_agg_ci(df, hd_label, ca_label, bins_hd, n_resamples)

    df_grp = du.df_grp_bin(df, hd_label, ca_label, ["mean"], bins_hd)

    resp_mean = df_grp["mean"].values

    # Divide by frame interval to get it in ca per second.
    resp_mean = resp_mean / frame_int
    # resp_mean_low = resp_mean_low / frame_int
    # resp_mean_hi = resp_mean_hi / frame_int

    hd_sigma_deg = 15
    bin_size = np.abs(np.abs(bins_hd[0]) - np.abs(bins_hd[1]))
    hd_sigma_bin = hd_sigma_deg / bin_size

    resp_mean = scipy.ndimage.gaussian_filter1d(resp_mean, sigma=hd_sigma_bin, mode='wrap')
    # resp_mean_low = scipy.ndimage.gaussian_filter1d(resp_mean_low, sigma=hd_sigma_bin, mode='wrap')
    # resp_mean_hi = scipy.ndimage.gaussian_filter1d(resp_mean_hi, sigma=hd_sigma_bin, mode='wrap')

    #return resp_mean, resp_mean_low, resp_mean_hi
    return resp_mean


def calc_AHV_curve(df, ahv_label, ca_label, bins_AHV, frame_int=0.1):
    df_grp = du.df_grp_bin(df, ahv_label, ca_label, ["mean"], bins_AHV)

    resp_mean = df_grp["mean"].values

    # Divide by frame interval to get it in ca per second.
    resp_mean = resp_mean / frame_int

    ahv_sigma_degs = 10
    bin_size = np.abs(bins_AHV[0]) - np.abs(bins_AHV[1])
    ahv_sigma_bins = ahv_sigma_degs / bin_size
    resp_mean = scipy.ndimage.gaussian_filter1d(resp_mean, sigma=ahv_sigma_bins, mode='reflect')

    return resp_mean


def calc_HD_circstats(resp_mean, bins_hd):
    resp_mean_norm, resp_mean_min, resp_mean_max = norm_circ(resp_mean)

    angles = np.deg2rad(bins_hd[1:])
    bin_size = angles[1] - angles[0]
    angles = angles - bin_size / 2

    prefdir = pcs.descriptive.mean(angles, w=resp_mean_norm, d=bin_size)
    prefdir = np.rad2deg(prefdir)

    cv = pcs.descriptive.var(angles, w=resp_mean_norm, d=bin_size)

    p, z = pcs.tests.rayleigh(angles, w=resp_mean_norm, d=bin_size)

    return resp_mean_norm, prefdir, cv, p, z


def calc_AHV_stats(resp_mean, bins_ahv, responses, ahvs):
    # This is not quite right, I need to figure out bins properly.
    n_bins1 = int(np.round(bins_ahv.shape[0] / 2))
    resp_mean_left = resp_mean[0:n_bins1 - 1]
    resp_mean_right = resp_mean[n_bins1 - 1:]
    speeds_left = bins_ahv[0:n_bins1 - 1]
    speeds_right = bins_ahv[n_bins1 - 1:-1]

    (ahv_left_r, ahv_left_p) = scipy.stats.pearsonr(np.abs(speeds_left), resp_mean_left)
    (ahv_right_r, ahv_right_p) = scipy.stats.pearsonr(speeds_right, resp_mean_right)

    if np.isnan(ahv_left_r):
        ahv_left_r = 0
    if np.isnan(ahv_left_p):
        ahv_left_p = 1
    if np.isnan(ahv_right_r):
        ahv_right_r = 0
    if np.isnan(ahv_right_p):
        ahv_right_p = 1

    left_linreg = scipy.stats.linregress(np.abs(speeds_left), resp_mean_left)
    right_linreg = scipy.stats.linregress(speeds_right, resp_mean_right)

    # Calculating a DI with slopes doesn't work well because of negative slopes.
    # ml = left_linreg.slope
    # mr = right_linreg.slope
    # di = (mr - ml) / (np.abs(ml) + np.abs(mr))
    # Use ROC instead, however I still have not done the binning right and zero is probably included.
    # Hardly ever passes sig.
    # n_bins1nz = n_bins1 - 1
    # roc_classes = np.zeros(n_bins1nz * 2, dtype=bool)
    # roc_classes[n_bins1nz:] = True
    # roc_score = np.concatenate((resp_mean_left, resp_mean_right))
    # di = roc_auc_score(roc_classes, roc_score)
    # di = di * 2 - 1

    # # Just using the tuning curve doesn't work well. Try using the whole trace?
    # # Too many zeros. Try removing them? No still sucks.
    # left_indexes = ahvs < 0
    # right_indexes = ahvs > 0
    # resp_nz = responses > 0
    # roc_classes = np.concatenate((np.full(np.sum(left_indexes & resp_nz), False, dtype=bool),
    #                               np.full(np.sum(right_indexes & resp_nz), True, dtype=bool)))
    # roc_score = np.concatenate((responses[left_indexes & resp_nz], responses[right_indexes & resp_nz]))
    # di = roc_auc_score(roc_classes, roc_score)
    # di = di * 2 - 1

    # # Try a mean based DI? Bit weak.
    # ml = np.mean(resp_mean_left)
    # mr = np.mean(resp_mean_right)
    # di = (mr - ml) / (np.abs(ml) + np.abs(mr))

    # Try a max based DI?
    ml = np.nan_to_num(np.max(resp_mean_left))
    mr = np.nan_to_num(np.max(resp_mean_right))
    denom = (np.abs(ml) + np.abs(mr))
    di = 0
    if denom > 0:
        di = (mr - ml) / denom

    return AHV_stats(left_r=ahv_left_r,
                     left_p=ahv_left_p,
                     right_r=ahv_right_r,
                     right_p=ahv_right_p,
                     left_m=left_linreg.slope,
                     left_c=left_linreg.intercept,
                     right_m=right_linreg.slope,
                     right_c=right_linreg.intercept,
                     di=di)


def get_boots(n_boots, min_roll, data_vals):
    # We roll both directions so halve whatever is asked for, if possible.
    n_boots_1tail = 1
    if n_boots > 1:
        n_boots_1tail = int(round(n_boots / 2))

    n_data_vals = data_vals.shape[0]
    boot_rolls = get_boot_rolls(n_data_vals, n_boots_1tail, min_roll)
    n_boots = len(boot_rolls)

    boot_mat = np.zeros((n_boots, n_data_vals))
    for i_boot in range(n_boots):
        roll_val = boot_rolls[i_boot]
        boot_mat[i_boot, :] = np.roll(data_vals, roll_val)

    return boot_mat


def get_boot_rolls(n_pts, n_boots_1tail, min_roll):

    rolls_pos = np.random.randint(min_roll, n_pts - min_roll, n_boots_1tail)
    rolls_neg = -rolls_pos
    both_rolls = np.hstack((rolls_neg, rolls_pos))

    return both_rolls


"""
Normalises circular data
"""


def norm_circ(data, min_val=None, max_val=None):
    data_norm = np.copy(data)
    if min_val is None:
        min_val = np.nan_to_num(np.min(data))

    data_norm -= min_val

    if max_val is None:
        max_val = np.nan_to_num(np.max(data_norm))

    if max_val != 0:
        data_norm = data_norm / max_val

    return data_norm, min_val, max_val


def filt_img_nan(img, nan_indexes=None, sigma = 2, truncate = 5):
    # Black magic to filter an image with nans
    # https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

    if nan_indexes is not None:
        img[nan_indexes] = np.nan

    img1 = img.copy()
    img1[np.isnan(img)] = 0
    img1_filt = scipy.ndimage.gaussian_filter(img1, sigma=sigma, truncate=truncate)

    img2 = 0 * img.copy() + 1
    img2[np.isnan(img)] = 0
    img2_filt = scipy.ndimage.gaussian_filter(img2, sigma=sigma, truncate=truncate)

    img_filt = img1_filt / img2_filt

    if nan_indexes is not None:
        img_filt[nan_indexes] = np.nan

    return img_filt
