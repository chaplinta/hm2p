import pandas as pd
from utils import ca as cu, plot as pu, behave as bu, tune as tu
import numpy as np
from paths.config import M2PConfig
import matplotlib.pyplot as plt
import scipy
import pycircstat as pcs
from proc import proc_ca_behave

def sum_tune(cfg: M2PConfig):



    df_roi = pd.read_hdf(cfg.db_ca_roi_file)
    df_roi_stats = pd.read_hdf(cfg.db_roi_stat_file)

    df_roi_stats = pd.merge(df_roi, df_roi_stats, on=["exp_id", "roi_id"], how="inner")

    df_roi_stats = df_roi_stats.loc[df_roi_stats.resp_type == cu.CA_DECONV_NORM_CLEAN]

    df_ct = pd.crosstab(index=[df_roi_stats['exp_id'], df_roi_stats['roi_id'], df_roi_stats['roi_type']],
                        columns=df_roi_stats['stat_name'],
                        values=df_roi_stats['stat'],
                        aggfunc=np.mean)

    n_rois = df_ct.shape[0]
    print(n_rois)

    sum_HD_AHV(cfg, df_ct)

    sum_AHV(cfg, df_ct)

    sum_heading(cfg, df_ct, "HD1")
    # sum_heading(cfg, df_ct, "HD2")
    # sum_heading(cfg, df_ct, "HEgo1")
    # sum_heading(cfg, df_ct, "HEgo2")
    # sum_heading(cfg, df_ct, "HAllo1")
    # sum_heading(cfg, df_ct, "HAllo2")



def sum_HD_AHV(cfg, df_ct):
    dpi = 300

    idx_soma = df_ct.index.get_level_values("roi_type") == "soma"
    df_soma = df_ct[idx_soma]

    ahv_r_best = df_soma[["AHV_comb_left_r", "AHV_comb_right_r"]].max(axis=1)

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_soma["AHV_comb_di"].abs(),
                df_soma["HD1_comb_cv"])


    plt.xlabel("Direction selectivity (DI)")
    plt.ylabel("Circular variance")

    plot_img_path = cfg.sum_tune_path / "hd_ahv_di_cv.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(ahv_r_best,
                df_soma["HD1_comb_cv"])

    plt.xlabel("Speed correlation (r)")
    plt.ylabel("Circular variance")

    plot_img_path = cfg.sum_tune_path / "hd_ahv_r_cv.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

def sum_AHV(cfg, df_ct):

    dpi = 300

    idx_soma = df_ct.index.get_level_values("roi_type") == "soma"
    df_soma = df_ct[idx_soma]

    # DI


    idx_comb_left_r_sig = df_soma["AHV_comb_left_r"] > df_soma["AHV_comb_left_r_hi"]
    idx_comb_right_r_sig = df_soma["AHV_comb_right_r"] > df_soma["AHV_comb_right_r_hi"]
    idx_comb_ahv_sig = idx_comb_left_r_sig | idx_comb_right_r_sig
    ahv_r_best = df_soma[["AHV_comb_left_r", "AHV_comb_right_r"]].max(axis=1)

    idx_comb_sig_di = ((df_soma["AHV_comb_di"] > df_soma["AHV_comb_di_hi"]) |
                       (df_soma["AHV_comb_di"] < df_soma["AHV_comb_di_lo"]))
    idx_comb_sig_mot_only = ~idx_comb_sig_di & idx_comb_ahv_sig
    idx_comb_not_sig_di = ~(idx_comb_sig_di | idx_comb_sig_mot_only)

    print(np.sum(idx_soma))
    print(np.sum(idx_comb_sig_di))
    print(np.sum(idx_comb_ahv_sig))
    print(np.sum(idx_comb_sig_mot_only))

    fig = plt.figure(tight_layout=True)
    plt.hist(df_soma["AHV_comb_di"])
    plt.xlabel("Direction selectivity (DI)")
    plt.ylabel("# soma")
    plt.legend()
    plot_img_path = cfg.sum_tune_path / "ahv_di_hist.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(ahv_r_best)
    plt.xlabel("Speed correlation (r)")
    plt.ylabel("# soma")
    plt.legend()
    plot_img_path = cfg.sum_tune_path / "ahv_rbest_hist.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_soma.loc[idx_comb_not_sig_di]["AHV_comb_di"].abs(),
                ahv_r_best[idx_comb_not_sig_di],
                label="Not sig.", color=(0.5, 0.5, 0.5))
    plt.scatter(df_soma.loc[idx_comb_sig_di]["AHV_comb_di"].abs(),
                ahv_r_best[idx_comb_sig_di],
                label="Sig.", color='k')
    plt.scatter(df_soma.loc[idx_comb_sig_mot_only]["AHV_comb_di"].abs(),
                ahv_r_best[idx_comb_sig_mot_only],
                label="Sig.", color='k', marker='x')



    plt.xlabel("Direction selectivity (DI)")
    plt.ylabel("Speed correlation (r)")
    plt.legend()

    plot_img_path = cfg.sum_tune_path / "ahv_di_r.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    # Correlation
    idx_comb_left_r_sig = df_soma["AHV_comb_left_r"] > df_soma["AHV_comb_left_r_hi"]
    idx_comb_right_r_sig = df_soma["AHV_comb_right_r"] > df_soma["AHV_comb_right_r_hi"]

    idx_comb_r_sig = idx_comb_left_r_sig & idx_comb_right_r_sig
    idx_comb_leftonly_r_sig = idx_comb_left_r_sig & ~idx_comb_right_r_sig
    idx_comb_rightonly_r_sig = ~idx_comb_left_r_sig & idx_comb_right_r_sig
    idx_comb_not_r_sig = ~idx_comb_left_r_sig & ~idx_comb_right_r_sig

    # Slope
    idx_comb_left_m_sig = df_soma["AHV_comb_left_m"] > df_soma["AHV_comb_left_m_hi"]
    idx_comb_right_m_sig = df_soma["AHV_comb_right_m"] > df_soma["AHV_comb_right_m_hi"]

    idx_comb_m_sig = idx_comb_left_m_sig & idx_comb_right_m_sig
    idx_comb_leftonly_m_sig = idx_comb_left_m_sig & ~idx_comb_right_m_sig
    idx_comb_rightonly_m_sig = ~idx_comb_left_m_sig & idx_comb_right_m_sig
    idx_comb_not_m_sig = ~idx_comb_left_m_sig & ~idx_comb_right_m_sig


    print(np.sum(idx_soma))
    print(np.sum(idx_comb_r_sig))
    print(np.sum(idx_comb_leftonly_r_sig))
    print(np.sum(idx_comb_rightonly_r_sig))

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_soma.loc[idx_comb_not_r_sig]["AHV_comb_left_r"],
                df_soma.loc[idx_comb_not_r_sig]["AHV_comb_right_r"],
                label="Not sig.", color=(0.5, 0.5, 0.5))
    plt.scatter(df_soma.loc[idx_comb_r_sig]["AHV_comb_left_r"],
                df_soma.loc[idx_comb_r_sig]["AHV_comb_right_r"],
                label="Sig.", color='k')
    plt.scatter(df_soma.loc[idx_comb_leftonly_r_sig]["AHV_comb_left_r"],
                df_soma.loc[idx_comb_leftonly_r_sig]["AHV_comb_right_r"],
                label="CCW sig.", color='r')
    plt.scatter(df_soma.loc[idx_comb_rightonly_r_sig]["AHV_comb_left_r"],
                df_soma.loc[idx_comb_rightonly_r_sig]["AHV_comb_right_r"],
                label="CW sig.", color='b')
    plt.xlabel("Counter clockwise r")
    plt.ylabel("Clockwise r")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.legend()
    pu.square_plot(diag=True)

    plot_img_path = cfg.sum_tune_path / "ahv_r_lr.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_soma.loc[idx_comb_not_m_sig]["AHV_comb_left_m"],
                df_soma.loc[idx_comb_not_m_sig]["AHV_comb_right_m"],
                label="Not sig.", color=(0.5, 0.5, 0.5))
    plt.scatter(df_soma.loc[idx_comb_m_sig]["AHV_comb_left_m"],
                df_soma.loc[idx_comb_m_sig]["AHV_comb_right_m"],
                label="Sig.", color='k')
    plt.scatter(df_soma.loc[idx_comb_leftonly_m_sig]["AHV_comb_left_m"],
                df_soma.loc[idx_comb_leftonly_m_sig]["AHV_comb_right_m"],
                label="CCW sig.", color='r')
    plt.scatter(df_soma.loc[idx_comb_rightonly_m_sig]["AHV_comb_left_m"],
                df_soma.loc[idx_comb_rightonly_m_sig]["AHV_comb_right_m"],
                label="CW sig.", color='b')
    plt.xlabel("Counter clockwise slope")
    plt.ylabel("Clockwise slope")
    plt.legend()
    pu.square_plot(diag=True)

    plot_img_path = cfg.sum_tune_path / "ahv_m_lr.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')


def sum_heading(cfg, df_ct, ht):

    dpi = 300

    idx_soma = df_ct.index.get_level_values("roi_type") == "soma"
    idx_comb_cv_sig = df_ct["{}_comb_cv".format(ht)] < df_ct["{}_comb_cv_lo".format(ht)]
    idx_comb_stab_sig = df_ct["{}_comb_stab_r".format(ht)] > df_ct["{}_comb_stab_hi".format(ht)]
    idx_comb_sig = idx_comb_cv_sig & idx_comb_stab_sig

    idx_soma_comb_sig = idx_soma & idx_comb_sig
    idx_soma_not_comb_sig = idx_soma & ~idx_comb_sig & ~idx_comb_stab_sig
    idx_soma_stabr_sig_not_comb_sig = idx_soma & ~idx_comb_cv_sig & idx_comb_stab_sig

    print(np.sum(idx_soma))
    print(np.sum(idx_soma_comb_sig))
    print(np.sum(idx_soma_stabr_sig_not_comb_sig))

    idx_light_cv_sig = df_ct["{}_light_cv".format(ht)] < df_ct["{}_light_cv_lo".format(ht)]
    idx_light_stab_sig = df_ct["{}_light_stab_r".format(ht)] > df_ct["{}_light_stab_hi".format(ht)]
    idx_light = idx_light_cv_sig & idx_light_stab_sig

    idx_soma_light_sig = idx_soma & idx_light
    idx_soma_not_light_sig = idx_soma & ~idx_light & ~idx_light_stab_sig
    idx_soma_stabr_sig_not_light_sig = idx_soma & ~idx_light_cv_sig & idx_light_stab_sig

    idx_dark_cv_sig = df_ct["{}_dark_cv".format(ht)] < df_ct["{}_dark_cv_lo".format(ht)]
    idx_dark_stab_sig = df_ct["{}_dark_stab_r".format(ht)] > df_ct["{}_dark_stab_hi".format(ht)]
    idx_dark = idx_dark_cv_sig & idx_dark_stab_sig

    idx_soma_dark_sig = idx_soma & idx_dark
    idx_soma_not_dark_sig = idx_soma & ~idx_dark & ~idx_dark_stab_sig
    idx_soma_stabr_sig_not_dark_sig = idx_soma & ~idx_dark_cv_sig & idx_dark_stab_sig

    idx_lightdark_cv_sig = idx_soma_light_sig & idx_soma_dark_sig
    idx_lightonly_cv_sig = idx_soma_light_sig & ~idx_soma_dark_sig
    idx_darkonly_cv_sig = ~idx_soma_light_sig & idx_soma_dark_sig

    idx_lightdark_stab_sig = idx_soma_stabr_sig_not_light_sig & idx_soma_stabr_sig_not_dark_sig
    idx_lightonly_stab_sig = idx_soma_stabr_sig_not_light_sig & ~idx_soma_stabr_sig_not_dark_sig
    idx_darkonly_stab_sig = ~idx_soma_stabr_sig_not_light_sig & idx_soma_stabr_sig_not_dark_sig

    idx_ldnotsig_notstab_sig = ~idx_comb_sig & \
                          ~idx_soma_light_sig & \
                          ~idx_soma_dark_sig & \
                          ~idx_lightdark_stab_sig & \
                          ~idx_lightonly_stab_sig & \
                          ~idx_darkonly_stab_sig

    idx_ldnotsig_cv_sig = ~idx_comb_sig & \
                          ~idx_soma_light_sig & \
                          ~idx_soma_dark_sig

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_ct.loc[idx_soma_not_comb_sig].HD1_comb_cv,
                df_ct.loc[idx_soma_not_comb_sig].HD1_comb_stab_r,
                label="Not sig.", color=(0.5, 0.5, 0.5))
    plt.scatter(df_ct.loc[idx_soma_comb_sig].HD1_comb_cv,
                df_ct.loc[idx_soma_comb_sig].HD1_comb_stab_r,
                label="Sig.", color='r')
    plt.scatter(df_ct.loc[idx_soma_stabr_sig_not_comb_sig].HD1_comb_cv,
                df_ct.loc[idx_soma_stabr_sig_not_comb_sig].HD1_comb_stab_r,
                label="Stable", color='b', marker="x")
    plt.xlabel("Circular variance")
    plt.ylabel("Stability r")
    plt.xlim(left=0, right=1)
    plt.ylim(top=1)
    plt.legend()

    plot_img_path = cfg.sum_tune_path / "{}-cv-stabr.png".format(ht)
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_ct.loc[idx_ldnotsig_notstab_sig].HD1_light_cv,
                df_ct.loc[idx_ldnotsig_notstab_sig].HD1_dark_cv,
                label="Not sig.", color=(0.5, 0.5, 0.5))

    plt.scatter(df_ct.loc[idx_lightdark_cv_sig].HD1_light_cv,
                df_ct.loc[idx_lightdark_cv_sig].HD1_dark_cv,
                label="Light&dark sig.", color='r')
    plt.scatter(df_ct.loc[idx_lightonly_cv_sig].HD1_light_cv,
                df_ct.loc[idx_lightonly_cv_sig].HD1_dark_cv,
                label="Light sig.", color='b')
    plt.scatter(df_ct.loc[idx_darkonly_cv_sig].HD1_light_cv,
                df_ct.loc[idx_darkonly_cv_sig].HD1_dark_cv,
                label="Dark sig.", color='k')

    plt.scatter(df_ct.loc[idx_lightdark_stab_sig].HD1_light_cv,
                df_ct.loc[idx_lightdark_stab_sig].HD1_dark_cv,
                label="Light&dark stable", color='r', marker='x')
    plt.scatter(df_ct.loc[idx_lightonly_stab_sig].HD1_light_cv,
                df_ct.loc[idx_lightonly_stab_sig].HD1_dark_cv,
                label="Light stable", color='b', marker='x')
    plt.scatter(df_ct.loc[idx_darkonly_stab_sig].HD1_light_cv,
                df_ct.loc[idx_darkonly_stab_sig].HD1_dark_cv,
                label="Dark stable", color='k', marker='x')

    plt.xlabel("Circular variance (light)")
    plt.ylabel("Circular variance (dark)")
    plt.xlim(left=0.2, right=1)
    plt.ylim(bottom=0.2, top=1)
    plt.plot([0, 1], [0, 1], 'k')
    plt.gca().set_aspect('equal')
    plt.legend()

    plot_img_path = cfg.sum_tune_path / "{}-cv-lightdark.png".format(ht)
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    ### no stab

    fig = plt.figure(tight_layout=True)
    plt.scatter(df_ct.loc[idx_ldnotsig_cv_sig].HD1_light_cv,
                df_ct.loc[idx_ldnotsig_cv_sig].HD1_dark_cv,
                label="Not sig.", color=(0.5, 0.5, 0.5))

    plt.scatter(df_ct.loc[idx_lightdark_cv_sig].HD1_light_cv,
                df_ct.loc[idx_lightdark_cv_sig].HD1_dark_cv,
                label="Light&dark sig.", color='r')
    plt.scatter(df_ct.loc[idx_lightonly_cv_sig].HD1_light_cv,
                df_ct.loc[idx_lightonly_cv_sig].HD1_dark_cv,
                label="Light sig.", color='b')
    plt.scatter(df_ct.loc[idx_darkonly_cv_sig].HD1_light_cv,
                df_ct.loc[idx_darkonly_cv_sig].HD1_dark_cv,
                label="Dark sig.", color='k')

    plt.xlabel("Circular variance (light)")
    plt.ylabel("Circular variance (dark)")
    plt.xlim(left=0.2, right=1)
    plt.ylim(bottom=0.2, top=1)
    plt.plot([0, 1], [0, 1], 'k')
    plt.gca().set_aspect('equal')
    #plt.legend()

    plot_img_path = cfg.sum_tune_path / "{}-cv-lightdark.png".format(ht)
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    plt.hist(df_ct.loc[idx_soma].HD1_light_cv - df_ct.loc[idx_soma].HD1_dark_cv)

    plt.xlabel("Light - dark CV")
    plt.ylabel("# soma")


    plot_img_path = cfg.sum_tune_path / "{}-cv-lightdark-hist.png".format(ht)
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

# def plot_pairs(cfg, proc_type:proc_ca_behave.TuneProcType=proc_ca_behave.TuneProcType.roi):
#
#     if proc_type == proc_ca_behave.TuneProcType.roi:
#         plot_dir = cfg.sum_tune_roi_path
#         df = pd.read_hdf(cfg.db_roi_tune_file)
#         df_tune_stats = pd.read_hdf(cfg.db_roi_stat_file)
#         df_roi = pd.read_hdf(cfg.db_somadend_pairs_file)
#         # df = pd.merge(df, df_roi, on=["exp_id", "roi_id"], how="left")
#         id_col = "roi_id"
#         response_types = [cu.CA_DECONV_NORM_CLEAN]
#     else:
#         plot_dir = cfg.sum_tune_pair_path
#         df = pd.read_hdf(cfg.db_somadend_tune_file)
#         df_tune_stats = pd.read_hdf(cfg.db_somadend_stat_file)
#         id_col = "pair_id"
#
#
#     exp_ids = df.exp_id.unique()
#
#     for exp_id in exp_ids:
#
#         df_this_exp = df.loc[df.exp_id == exp_id]
#
#         roi_ids = df_this_exp[id_col].unique()
#
#         for roi_id in roi_ids:
#
#             indexes_roi = df_this_exp[id_col] == roi_id
#             df_this_roi = df_this_exp.loc[indexes_roi]
#
#
#
#             if proc_type == proc_ca_behave.TuneProcType.roi:
#
#                 # this_roi_type = df_this_roi.iloc[0]["roi_type"]
#                 # if not this_roi_type == "soma":
#                 #     continue
#                 this_roi_type = "unk"
#
#                 hd_tune_type_comb = "HD1_comb"
#                 hd_tune_type_light = "HD1_light"
#                 hd_tune_type_dark = "HD1_dark"
#
#                 hd_comb_indexes = (df_this_roi["tune_type"] == hd_tune_type_comb)
#                 hd_light_indexes = (df_this_roi["tune_type"] == hd_tune_type_light)
#                 hd_dark_indexes = (df_this_roi["tune_type"] == hd_tune_type_dark)
#
#                 hd_angles = df_this_roi.loc[hd_comb_indexes]["tune_param"]
#                 hd_resp_comb = df_this_roi.loc[hd_comb_indexes]["resp_mean"].values
#                 hd_resp_light = df_this_roi.loc[hd_light_indexes]["resp_mean"].values
#                 hd_resp_dark = df_this_roi.loc[hd_dark_indexes]["resp_mean"].values
#
#                 hd_resp_max = np.max([np.nan_to_num(np.max(hd_resp_light)),
#                                       np.nan_to_num(np.max(hd_resp_dark))])
#
#                 hd_resp_min = np.max([np.nan_to_num(np.min(hd_resp_light)),
#                                       np.nan_to_num(np.min(hd_resp_dark))])
#
#                 hd_resp_comb_norm, _, _ = tu.norm_circ(hd_resp_comb)
#                 hd_resp_light_norm, _, _ = tu.norm_circ(hd_resp_light, min_val=hd_resp_min, max_val=hd_resp_max)
#                 hd_resp_dark_norm, _, _ = tu.norm_circ(hd_resp_dark, min_val=hd_resp_min, max_val=hd_resp_max)
#
#                 polar_theta = np.append(hd_angles, hd_angles[0])
#                 polar_r_comb = np.append(hd_resp_comb_norm, hd_resp_comb_norm[0])
#                 polar_r_light = np.append(hd_resp_light_norm, hd_resp_light_norm[0])
#                 polar_r_dark = np.append(hd_resp_dark_norm, hd_resp_dark_norm[0])
#
#                 # Combined plot
#                 fig = plt.figure(tight_layout=True)
#                 ax = plt.subplot(111, projection='polar')
#                 plt.plot(np.deg2rad(polar_theta), polar_r_comb, linewidth=2)
#                 ax.set_rmax(1)
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 ax.yaxis.grid(False)
#                 ax.xaxis.grid(linewidth=2)
#                 plt.tight_layout()
#
#                 plot_img_path = plot_dir / "comb-{}-{}-{}.png".format(exp_id,
#                                                                     roi_id,
#                                                                     this_roi_type)
#
#                 fig.savefig(plot_img_path, dpi=300, facecolor='white')
#                 plt.cla()
#                 plt.clf()
#                 plt.close('all')
#
#                 # Light dark plot
#                 fig = plt.figure(tight_layout=True)
#                 ax = plt.subplot(111, projection='polar')
#                 plt.plot(np.deg2rad(polar_theta), polar_r_light, linewidth=2, color=(0.5, 0.5, 0.5))
#                 plt.plot(np.deg2rad(polar_theta), polar_r_dark, linewidth=2,color='k')
#                 ax.set_rmax(1)
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 ax.yaxis.grid(False)
#                 ax.xaxis.grid(linewidth=2)
#                 plt.tight_layout()
#
#                 plot_img_path = plot_dir / "ld-{}-{}-{}.png".format(exp_id,
#                                                                       roi_id,
#                                                                       this_roi_type)
#
#                 fig.savefig(plot_img_path, dpi=300, facecolor='white')
#                 plt.cla()
#                 plt.clf()
#                 plt.close('all')
#
#                 # AHV plot
#                 ax_font_ize = 16
#                 ahv_tune_type_comb = "AHV"
#                 ahv_tune_type_light = "AHV_light"
#                 ahv_tune_type_dark = "AHV_dark"
#
#                 ahv_comb_indexes = (df_this_roi["tune_type"] == ahv_tune_type_comb) & (df_this_roi["resp_type"] == cu.CA_DECONV_NORM_CLEAN)
#                 ahv_light_indexes = (df_this_roi["tune_type"] == ahv_tune_type_light) & (df_this_roi["resp_type"] == cu.CA_DECONV_NORM_CLEAN)
#                 ahv_dark_indexes = (df_this_roi["tune_type"] == ahv_tune_type_dark) & (df_this_roi["resp_type"] == cu.CA_DECONV_NORM_CLEAN)
#
#                 # todo fix hacks these numbers are trippled up for some reason
#                 ahv_speeds = df_this_roi.loc[ahv_comb_indexes]["tune_param"].values[0:21]
#                 ahv_resp_comb = df_this_roi.loc[ahv_comb_indexes]["resp_mean"].values[0:21]
#                 # ahv_resp_light = df_this_roi.loc[ahv_light_indexes]["resp_mean"].values
#                 # ahv_resp_dark = df_this_roi.loc[ahv_dark_indexes]["resp_mean"].values
#
#                 # Combined plot
#                 fig = plt.figure(tight_layout=True)
#                 plt.plot(ahv_speeds, ahv_resp_comb, linewidth=2)
#                 plt.xlabel("Angular head velocity (deg/s)", fontsize=ax_font_ize)
#                 plt.ylabel("Mean response (norm df/F0)/s", fontsize=ax_font_ize)
#                 plt.tight_layout()
#
#                 plot_img_path = plot_dir / "comb-ahv-{}-{}-{}.png".format(exp_id,
#                                                                       roi_id,
#                                                                       this_roi_type)
#
#                 fig.savefig(plot_img_path, dpi=300, facecolor='white')
#                 plt.cla()
#                 plt.clf()
#                 plt.close('all')
#
#                 # # Light dark plot
#                 # fig = plt.figure(tight_layout=True)
#                 # plt.plot(ahv_speeds, ahv_resp_light, linewidth=2, color=(0.5, 0.5, 0.5))
#                 # plt.plot(ahv_speeds, ahv_resp_dark, linewidth=2, color='k')
#                 # plt.xlabel("Angular head velocity (deg/s)")
#                 # plt.ylabel("Mean response (norm df/F)/s")
#                 # plt.tight_layout()
#                 #
#                 # plot_img_path = plot_dir / "ld-ahv-{}-{}-{}.png".format(exp_id,
#                 #                                                     roi_id,
#                 #                                                     this_roi_type)
#                 #
#                 # fig.savefig(plot_img_path, dpi=300, facecolor='white')
#                 # plt.cla()
#                 # plt.clf()
#                 # plt.close('all')
#
#             else:
#                 this_roi_type = "pair"
#
#                 hd_tune_type = "HD1_comb"
#
#                 hd_joint_indexes = (df_this_roi["tune_type"] == hd_tune_type) & (df_this_roi["resp_type"] == "event_joint")
#                 hd_somatic_indexes = (df_this_roi["tune_type"] == hd_tune_type) & (df_this_roi["resp_type"] == "event_somatic")
#                 hd_dendritic_indexes = (df_this_roi["tune_type"] == hd_tune_type) & (df_this_roi["resp_type"] == "event_dendritic")
#
#
#                 hd_angles = df_this_roi.loc[hd_joint_indexes]["tune_param"]
#                 hd_resp_joint = df_this_roi.loc[hd_joint_indexes]["resp_mean"].values
#                 hd_resp_somatic = df_this_roi.loc[hd_somatic_indexes]["resp_mean"].values
#                 hd_resp_dendritic = df_this_roi.loc[hd_dendritic_indexes]["resp_mean"].values
#
#                 hd_resp_max = np.max([np.nan_to_num(np.max(hd_resp_joint)),
#                                       np.nan_to_num(np.max(hd_resp_somatic)),
#                                       np.nan_to_num(np.max(hd_resp_dendritic))])
#
#                 hd_resp_min = np.max([np.nan_to_num(np.min(hd_resp_joint)),
#                                       np.nan_to_num(np.min(hd_resp_somatic)),
#                                       np.nan_to_num(np.min(hd_resp_dendritic))])
#
#                 hd_resp_joint_norm, _, _ = tu.norm_circ(hd_resp_joint, min_val=hd_resp_min, max_val=hd_resp_max)
#                 hd_resp_somatic_norm, _, _ = tu.norm_circ(hd_resp_somatic, min_val=hd_resp_min, max_val=hd_resp_max)
#                 hd_resp_dendritic_norm, _, _ = tu.norm_circ(hd_resp_dendritic, min_val=hd_resp_min, max_val=hd_resp_max)
#
#                 polar_theta = np.append(hd_angles, hd_angles[0])
#                 polar_r_joint = np.append(hd_resp_joint_norm, hd_resp_joint_norm[0])
#                 polar_r_somatic = np.append(hd_resp_somatic_norm, hd_resp_somatic_norm[0])
#                 polar_r_dendritic = np.append(hd_resp_dendritic_norm, hd_resp_dendritic_norm[0])
#
#                 fig = plt.figure(tight_layout=True)
#                 ax = plt.subplot(111, projection='polar')
#
#
#                 plt.plot(np.deg2rad(polar_theta), polar_r_joint)
#                 plt.plot(np.deg2rad(polar_theta), polar_r_somatic)
#                 plt.plot(np.deg2rad(polar_theta), polar_r_dendritic)
#
#                 #plt.plot([0, np.deg2rad(hd_tune.prefdir)], [0, 1 - hd_tune.cv], 'k')
#                 #polar_r_hi = np.append(hd_tune.resp_mean_norm_hi, hd_tune.resp_mean_norm_hi[0])
#                 #plt.plot(np.deg2rad(polar_theta), polar_r_hi, color=(0.5, 0.5, 0.5))
#                 #polar_r_boot_mean = np.append(hd_tune.resp_mean_norm_boot_mean, hd_tune.resp_mean_norm_boot_mean[0])
#                 #plt.plot(np.deg2rad(polar_theta), polar_r_boot_mean, color=(0.5, 0.5, 0.5), linestyle=":")
#
#                 ax.set_rmax(1)
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 ax.yaxis.grid(False)
#                 ax.xaxis.grid(linewidth=2)
#                 plt.tight_layout()
#
#                 plot_img_path = plot_dir / "{}-{}-{}-{}.png".format(exp_id,
#                                                                        roi_id,
#                                                                        this_roi_type,
#                                                                        hd_tune_type)
#
#                 fig.savefig(plot_img_path, dpi=300, facecolor='white')
#                 plt.cla()
#                 plt.clf()
#                 plt.close('all')
#
#
#
#
#
#




