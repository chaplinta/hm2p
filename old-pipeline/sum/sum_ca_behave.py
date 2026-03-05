from utils import behave as bu, tune as tu
import numpy as np
import scipy
from paths.config import M2PConfig
import matplotlib.pyplot as plt

def sum_dist_ca(cfg: M2PConfig):

    dpi = 300

    df, df_roi, df_ca, df_behave = tu.get_dfs(cfg)

    exp_ids = df.exp_id.unique()

    for exp_id in exp_ids:
        df_exp = df[df.exp_id == exp_id]
        df_behave_exp = df_behave[df_behave.exp_id == exp_id]

        dist_inst = df_behave_exp[bu.DIST_INST_CM]
        dist_filt = df_behave_exp[bu.DIST_FILT_CM]

        dist_inst_cum = np.cumsum(dist_inst)
        dist_filt_cum = np.cumsum(dist_filt)

        smooth_sigma = 2 * 10 #2 seconds 10fps
        dist_inst_smooth = scipy.ndimage.gaussian_filter1d(dist_inst, sigma=smooth_sigma, mode='nearest')
        dist_filt_smooth = scipy.ndimage.gaussian_filter1d(dist_filt, sigma=smooth_sigma, mode='nearest')

        dist_inst_smooth_cum = np.cumsum(dist_inst_smooth)
        dist_filt_smooth_cum = np.cumsum(dist_filt_smooth)

        delta_filt_x = df_behave_exp[bu.DELTA_X_FILT]
        delta_filt_y = df_behave_exp[bu.DELTA_Y_FILT]

        smooth_sigma = 2 * 10  # 30 seconds 10fps
        delta_filt_x = scipy.ndimage.gaussian_filter1d(delta_filt_x, sigma=smooth_sigma, mode='nearest')
        delta_filt_y = scipy.ndimage.gaussian_filter1d(delta_filt_y, sigma=smooth_sigma, mode='nearest')

        smooth_dist = np.sqrt(np.square(delta_filt_x) + np.square(delta_filt_y)) / 10

        smooth_dist_cum = np.cumsum(smooth_dist)

        speed = df_behave_exp[bu.SPEED_FILT_GRAD]

        smooth_sigma = 10 * 10  # 10 seconds 10fps
        speed_smooth = scipy.ndimage.gaussian_filter1d(speed, sigma=smooth_sigma, mode='nearest')

        fig = plt.figure(tight_layout=True)
        plt.plot(dist_inst_cum)
        plt.plot(dist_filt_cum)
        plt.xlabel("Time")
        plt.ylabel("Cumulative distance (cm)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        #plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        #plt.plot(dist_inst)
        plt.plot(dist_filt_smooth)
        plt.xlabel("Time")
        plt.ylabel("Distance per frame (cm)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        # plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist-trace.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.plot(dist_inst_smooth_cum)
        plt.plot(dist_filt_smooth_cum)
        plt.xlabel("Time")
        plt.ylabel("Cumulative distance (cm)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        # plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist-smooth.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.plot(smooth_dist_cum)
        plt.xlabel("Time")
        plt.ylabel("Cumulative distance (cm)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        # plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist-presmooth.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.plot(smooth_dist)
        plt.xlabel("Time")
        plt.ylabel("Distance (cm)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        # plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist-presmooth-trace.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.plot(speed)
        plt.xlabel("Time")
        plt.ylabel("Speed (cm/s)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        # plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist-speed.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        fig = plt.figure(tight_layout=True)
        plt.plot(speed_smooth)
        plt.xlabel("Time")
        plt.ylabel("Speed (cm/s)")

        # plt.xlim(left=0, right=max_axis)
        # plt.ylim(bottom=0, top=max_axis)

        # plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = cfg.sum_dist_plots / "{}-dist-speed-smooth.png".format(exp_id)
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')


