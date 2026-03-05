# Match CA data to behaviour data (resample etc)
from utils.db import get_tune_df, get_roistat_df, add_roi_stat, get_pairstat_df, get_tune_pair_df
from classes import Experiment
import matplotlib.pyplot as plt
from utils import ca as cautils, behave as beutils, tune as tuneutils
import pandas as pd
from paths.config import M2PConfig
from classes.ProcPath import ProcPath
from utils import db
import numpy as np
import enum

def proc_resample_exp(cfg:M2PConfig, exp_id):

    m2p_paths = ProcPath(cfg, exp_id)

    print("Loading experiment")
    exp = Experiment.Experiment(m2p_paths.raw_data_path)
    print("Loading experiment done.")

    print("Checking sciscan and frames")
    exp.check_sci_frames()
    print("Done checking sciscan and frames")

    df = pd.read_hdf(m2p_paths.behave_file)

    df_resampled = beutils.resample_to_frames(exp, df)

    df_resampled.to_hdf(m2p_paths.behave_frames_file, key="df_resampled")

    db.add_exp_data(df_resampled, cfg.db_behave_frames_file, exp_id)


# Using enum class create enumerations
class TuneProcType(enum.Enum):
   roi = 1
   somadend = 2

# def proc_tuning(cfg:M2PConfig, plot, proc_type:TuneProcType=TuneProcType.roi,
#                 proc_exp_ids=None, n_boots=200, hd_dirs=[1]):
#
#     plot_dir = None
#     if plot:
#         if proc_type == TuneProcType.roi:
#             plot_dir = cfg.proc_raw_tune_path
#         else:
#             plot_dir = cfg.proc_raw_tune_pair_path
#
#     # Sepi used 6 degree bins (60!)
#     bins_hd = np.linspace(0, 360, 40 + 1, endpoint=True)
#
#     bins_ahv_pos = np.linspace(0, 60, 10 + 1, endpoint=True)
#     bins_ahv_size = bins_ahv_pos[1] - bins_ahv_pos[0]
#     bins_ahv_pos = bins_ahv_pos[1:]
#     bin_ahv_extra = bins_ahv_pos[-1] + bins_ahv_size
#     # Add a negative range, some values close to zero, then an extra one to capture the last values
#     bins_ahv = np.hstack((np.flip(-bins_ahv_pos), -1, 1, bins_ahv_pos, bin_ahv_extra))
#
#     speed_min = 1
#     speed_max = 20
#     bins_speed = np.append([0], np.linspace(speed_min, speed_max, 7 + 1, endpoint=True))
#
#     if proc_type == TuneProcType.roi:
#         df_roi = pd.read_hdf(cfg.db_ca_roi_file)
#         df_ca = pd.read_hdf(cfg.db_ca_file)
#         df_behave = pd.read_hdf(cfg.db_behave_frames_file)
#
#         df = pd.merge(df_roi, df_ca, on=["exp_id", "roi_id"], how="inner")
#         df = pd.merge(df, df_behave, on=["exp_id", "frame_id"], how="left")
#
#         response_types = [cautils.CA_DECONV_NORM_CLEAN]
#     else:
#
#         df_roi = pd.read_hdf(cfg.db_somadend_pairs_file)
#         df_ca = pd.read_hdf(cfg.db_somadend_ca_file)
#         df_behave = pd.read_hdf(cfg.db_behave_frames_file)
#
#         df = pd.merge(df_roi, df_ca, on=["exp_id", "pair_id"], how="inner")
#         df = pd.merge(df, df_behave, on=["exp_id", "frame_id"], how="left")
#
#         response_types = ["event_joint", "event_somatic", "event_dendritic"]
#
#     exp_ids = df.exp_id.unique()
#
#     for exp_id in exp_ids:
#
#         if proc_exp_ids is not None and exp_id not in proc_exp_ids:
#             continue
#
#         if proc_type == TuneProcType.roi:
#             df_tune = get_tune_df()
#             df_roistat = get_roistat_df()
#             id_col = "roi_id"
#         else:
#             id_col = "pair_id"
#             df_tune = get_tune_pair_df()
#             df_roistat = get_pairstat_df()
#
#         # if np.isnan(exp_id) or pd.isna(exp_id):
#         #     raise Exception("Warning NAN exp id found!")
#
#         print("Calculating tuning curves for experiment {}".format(exp_id))
#
#         df_this_exp = df.loc[df.exp_id == exp_id]
#
#         roi_ids = df_this_exp[id_col].unique()
#
#         for roi_id in roi_ids:
#
#             if proc_type == TuneProcType.roi:
#                 roi_id = int(roi_id)
#             df_this_roi = df_this_exp.loc[df_this_exp[id_col] == roi_id]
#
#             if proc_type == TuneProcType.roi:
#                 this_roi_type = df_this_roi.iloc[0]["roi_type"]
#                 if not this_roi_type == "soma":
#                     continue
#             else:
#                 this_roi_type = "pair"
#
#             if proc_type == TuneProcType.roi:
#                 print("Calculating tuning curves for experiment {} roi {:d}".format(exp_id, roi_id))
#             elif proc_type == TuneProcType.somadend:
#                 print("Calculating tuning curves for experiment {} pair {}".format(exp_id, roi_id))
#
#             light_indexes = df_this_roi[beutils.LIGHT_ON].values == 1
#             dark_indexes = np.logical_not(light_indexes)
#             df_light = df_this_roi.loc[light_indexes]
#             df_dark = df_this_roi.loc[dark_indexes]
#
#             df_moving = beutils.get_active_frames(df_this_roi)
#
#             move_light_indexes = df_moving[beutils.LIGHT_ON].values == 1
#             move_dark_indexes = np.logical_not(move_light_indexes)
#             df_moving_light = df_moving.loc[move_light_indexes]
#             df_moving_dark = df_moving.loc[move_dark_indexes]
#
#             for resp_type in response_types:
#
#                 for heading_col in [beutils.HD_ABS_FILT]:#, beutils.HEADING_EGO_ABS_FILT, beutils.HEADING_ALLO_ABS_FILT]:
#
#                     # Uni, bi, etc HD tuning
#                     for hd_dir in hd_dirs:
#                         df_tune, df_roistat = calc_tuning(exp_id,
#                                                           id_col,
#                                                           roi_id,
#                                                           this_roi_type,
#                                                           df_this_roi,
#                                                           df_moving,
#                                                           df_tune,
#                                                           df_roistat,
#                                                           heading_col,
#                                                           resp_type,
#                                                           "comb",
#                                                           bins_hd,
#                                                           bins_ahv,
#                                                           bins_speed,
#                                                           plot_dir,
#                                                           n_boots,
#                                                           n_dirs=hd_dir)
#
#                         df_tune, df_roistat = calc_tuning(exp_id,
#                                                           id_col,
#                                                           roi_id,
#                                                           this_roi_type,
#                                                           df_light,
#                                                           df_moving_light,
#                                                           df_tune,
#                                                           df_roistat,
#                                                           heading_col,
#                                                           resp_type,
#                                                           "light",
#                                                           bins_hd,
#                                                           bins_ahv,
#                                                           bins_speed,
#                                                           plot_dir,
#                                                           n_boots,
#                                                           n_dirs=hd_dir)
#
#                         df_tune, df_roistat = calc_tuning(exp_id,
#                                                           id_col,
#                                                           roi_id,
#                                                           this_roi_type,
#                                                           df_dark,
#                                                           df_moving_dark,
#                                                           df_tune,
#                                                           df_roistat,
#                                                           heading_col,
#                                                           resp_type,
#                                                           "dark",
#                                                           bins_hd,
#                                                           bins_ahv,
#                                                           bins_speed,
#                                                           plot_dir,
#                                                           n_boots,
#                                                           n_dirs=hd_dir)
#
#         if proc_type == TuneProcType.roi:
#             db.add_exp_data(df_tune, cfg.db_roi_tune_file, exp_id)
#             db.add_exp_data(df_roistat, cfg.db_roi_stat_file, exp_id)
#         elif proc_type == TuneProcType.somadend:
#             db.add_exp_data(df_tune, cfg.db_somadend_tune_file, exp_id)
#             db.add_exp_data(df_roistat, cfg.db_somadend_stat_file, exp_id)

def calc_tuning(exp_id,
                id_col,
                roi_id,
                roi_type,
                df_data,
                df_data_move,
                df_tune,
                df_roistat,
                heading_col,
                resp_type,
                light_cond,
                bins_hd,
                bins_ahv,
                bins_speed,
                plot_dir,
                n_boots,
                n_dirs=1):

    # todo This function is a bit fucked because it takes so many parameters.

    df_copy = df_data.copy(deep=True)
    df_move_copy = df_data_move.copy(deep=True)

    if heading_col == beutils.HD_ABS_FILT:
        head_type_name = "HD"
    elif heading_col == beutils.HEADING_EGO_ABS_FILT:
        head_type_name = "HEgo"
    elif heading_col == beutils.HEADING_ALLO_ABS_FILT:
        head_type_name = "HAllo"
    else:
        raise Exception("Unknown heading type {}".format(heading_col))

    if n_dirs > 1:
        if not n_dirs in [2, 4]:
            raise Exception("This number of HD is not supported because I'm too lazy to figure out how to do it.")

        df_move_copy[heading_col] = (df_move_copy[heading_col] * 2) % 360

        if n_dirs == 4:
            df_move_copy[heading_col] = (df_move_copy[heading_col] * 2) % 360


    # This function has been bastardised to include other types of tuning, such as AHV.
    # Todo fix that.
    hd_tune, ahv_tune = tuneutils.calc_HD(df_copy,
                                          df_move_copy,
                                          heading_col,
                                          beutils.AHV_FILT_GRAD,
                                          resp_type,
                                          bins_hd,
                                          bins_ahv,
                                          n_boots=n_boots,
                                          min_roll=30*10)


    # Add HD tuning
    # Head direction type: 1, 2, 4 direction etc.
    hd_tune_type = "{}{}_{}".format(head_type_name, n_dirs, light_cond)

    hd_n_pts = len(hd_tune.resp_mean)

    # Add raw responses.
    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type+"_lo"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_lo,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type + "_hi"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_hi,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type + "_boot_mean"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_boot_mean,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    # Add norm responses
    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type + "_norm"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_norm,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type + "_norm_lo"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_norm_lo,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type + "_norm_hi"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_norm_hi,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * hd_n_pts,
                 id_col: [roi_id] * hd_n_pts,
                 "tune_type": [hd_tune_type + "_norm_boot_mean"] * hd_n_pts,
                 "tune_param": bins_hd[:-1],
                 "resp_type": [resp_type] * hd_n_pts,
                 "resp_mean": hd_tune.resp_mean_norm_boot_mean,
                 "resp_sem": [np.nan] * hd_n_pts,
                 "resp_n": [np.nan] * hd_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    # Add HD stats
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_prefdir", hd_tune.prefdir)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_cv", hd_tune.cv)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_cv_lo", hd_tune.cv_lo)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_ray_p", hd_tune.ray_p)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_ray_z", hd_tune.ray_z)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_stab_r", hd_tune.stab_r)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, hd_tune_type+"_stab_hi", hd_tune.stab_r_hi)

    # Add AHV tuning
    ahv_n_pts = len(ahv_tune.resp_mean)
    data_dict = {"exp_id": [exp_id] * ahv_n_pts,
                 id_col: [roi_id] * ahv_n_pts,
                 "tune_type": ["AHV"] * ahv_n_pts,
                 "tune_param": bins_ahv[:-1],
                 "resp_type": [resp_type] * ahv_n_pts,
                 "resp_mean": ahv_tune.resp_mean,
                 "resp_sem": [np.nan] * ahv_n_pts,
                 "resp_n": [np.nan] * ahv_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * ahv_n_pts,
                 id_col: [roi_id] * ahv_n_pts,
                 "tune_type": ["AHV_lo"] * ahv_n_pts,
                 "tune_param": bins_ahv[:-1],
                 "resp_type": [resp_type] * ahv_n_pts,
                 "resp_mean": ahv_tune.resp_mean_lo,
                 "resp_sem": [np.nan] * ahv_n_pts,
                 "resp_n": [np.nan] * ahv_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * ahv_n_pts,
                 id_col: [roi_id] * ahv_n_pts,
                 "tune_type": ["AHV_hi"] * ahv_n_pts,
                 "tune_param": bins_ahv[:-1],
                 "resp_type": [resp_type] * ahv_n_pts,
                 "resp_mean": ahv_tune.resp_mean_hi,
                 "resp_sem": [np.nan] * ahv_n_pts,
                 "resp_n": [np.nan] * ahv_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    data_dict = {"exp_id": [exp_id] * ahv_n_pts,
                 id_col: [roi_id] * ahv_n_pts,
                 "tune_type": ["AHV_boot_mean"] * ahv_n_pts,
                 "tune_param": bins_ahv[:-1],
                 "resp_type": [resp_type] * ahv_n_pts,
                 "resp_mean": ahv_tune.resp_mean_boot_mean,
                 "resp_sem": [np.nan] * ahv_n_pts,
                 "resp_n": [np.nan] * ahv_n_pts}

    df_tune = pd.concat([df_tune, pd.DataFrame(data_dict)])

    ahv_tune_type = "AHV_{}".format(light_cond)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_left_r", ahv_tune.left_r)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_left_p", ahv_tune.left_p)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_left_r_hi", ahv_tune.left_r_hi)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_left_m", ahv_tune.left_m)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_left_m_hi", ahv_tune.left_m_hi)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_left_c", ahv_tune.left_c)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_right_r", ahv_tune.right_r)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_right_p", ahv_tune.right_p)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_right_r_hi", ahv_tune.right_r_hi)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_right_m", ahv_tune.right_m)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_right_m_hi", ahv_tune.right_m_hi)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_right_c", ahv_tune.right_c)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_di", ahv_tune.di)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_di_lo", ahv_tune.di_lo)
    df_roistat = add_roi_stat(df_roistat, exp_id, id_col, roi_id, resp_type, ahv_tune_type+"_di_hi", ahv_tune.di_hi)

    # Add speed tuning

    if plot_dir:
        dpi = 75

        # Plot HD
        angles = bins_hd[:-1]

        fig = plt.figure(tight_layout=True)
        ax = plt.subplot(121)
        ax.plot(angles, hd_tune.resp_mean)
        plt.plot(angles, hd_tune.resp_mean_lo, color=(0.5, 0.5, 0.5))
        plt.plot(angles, hd_tune.resp_mean_hi, color=(0.5, 0.5, 0.5))
        plt.plot(angles, hd_tune.resp_mean_boot_mean, color=(0.5, 0.5, 0.5), linestyle=":")

        ax.set_xlabel("Head direction (degrees)")
        ax.set_ylabel(cautils.get_ca_axis_label_occ(resp_type))

        ax.set_xlim(left=0, right=360)
        ax.set_ylim(bottom=0)
        (y_min, y_max) = ax.get_ylim()
        cv_line = y_max
        plt.plot([hd_tune.prefdir, hd_tune.prefdir], [y_min, cv_line], 'k')


        plt.title("CV={:.2f} CVLo={:.2f} Stabr={:.2f} StabrHi={:.2f}".format(
            hd_tune.cv,
            hd_tune.cv_lo,
            hd_tune.stab_r,
            hd_tune.stab_r_hi))

        ax = plt.subplot(122, projection='polar', aspect='equal')
        polar_theta = np.append(angles, angles[0])
        polar_r = np.append(hd_tune.resp_mean_norm, hd_tune.resp_mean_norm[0])
        plt.plot(np.deg2rad(polar_theta), polar_r)
        plt.plot([0, np.deg2rad(hd_tune.prefdir)], [0, 1-hd_tune.cv], 'k')
        polar_r_hi = np.append(hd_tune.resp_mean_norm_hi, hd_tune.resp_mean_norm_hi[0])
        plt.plot(np.deg2rad(polar_theta), polar_r_hi, color=(0.5, 0.5, 0.5))
        polar_r_boot_mean = np.append(hd_tune.resp_mean_norm_boot_mean, hd_tune.resp_mean_norm_boot_mean[0])
        plt.plot(np.deg2rad(polar_theta), polar_r_boot_mean, color=(0.5, 0.5, 0.5), linestyle=":")
        ax.set_rmax(1)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().yaxis.grid(False)
        plt.gca().xaxis.grid(linewidth=2)
        plt.tight_layout()

        plot_img_path = plot_dir / "{}-{}-{}-{}-{}.png".format(exp_id,
                                                                 roi_id,
                                                                 roi_type,
                                                                 resp_type,
                                                                 hd_tune_type)


        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        # Plot AHV
        ahvs = bins_ahv[:-1]

        fig = plt.figure(tight_layout=True)
        ax = plt.gca()

        ax.plot(ahvs, ahv_tune.resp_mean)
        plt.plot(ahvs, ahv_tune.resp_mean_lo, color=(0.5, 0.5, 0.5))
        plt.plot(ahvs, ahv_tune.resp_mean_hi, color=(0.5, 0.5, 0.5))
        plt.plot(ahvs, ahv_tune.resp_mean_boot_mean, color=(0.5, 0.5, 0.5), linestyle=":")
        ax.set_xlabel("Angular head velocity (degrees/s)")
        ax.set_ylabel(cautils.get_ca_axis_label_occ(resp_type))

        #ax.set_xlim(left=0, right=360)
        ax.set_ylim(bottom=0)

        plt.title("Lr={:.2f}({:.2f}) Rr={:.2f}({:.2f}) DI={:.2f}({:.2f},{:.2f}) ".format(
            ahv_tune.left_r,
            ahv_tune.left_r_hi,
            ahv_tune.right_r,
            ahv_tune.right_r_hi,
            ahv_tune.di,
            ahv_tune.di_lo,
            ahv_tune.di_hi))

        plot_img_path = plot_dir / "{}-{}-{}-{}-{}-{}.png".format(exp_id,
                                                                    roi_id,
                                                                    roi_type,
                                                                    resp_type,
                                                                    "AHV",
                                                                    light_cond)

        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    return df_tune, df_roistat





