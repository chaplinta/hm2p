import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils.img
import imageio
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from enum import Enum
from scipy.ndimage import filters

class CaDataType(Enum):
    DFONF0 = 1
    DECONV = 2
    EVENTS_BIN = 3
    EVENTS_AMP = 4

def getCaDataTypeLabelShort(cadt: CaDataType):
    if cadt == CaDataType.DFONF0:
        label = "dFOnF"
    elif cadt == CaDataType.DECONV:
        label = "deconv"
    elif cadt == CaDataType.EVENTS_BIN:
        label = "events"
    elif cadt == CaDataType.EVENTS_AMP:
        label = "event-dFOnF"
    else:
        raise Exception()

    return label

class S2PData:

    def __init__(self,
                 mode: str = None,
                 ops: dict = None,
                 F: np.array = None,
                 Fneu: np.array = None,
                 spks: np.array = None,
                 stat: np.array = None,
                 iscell: np.array = None,
                 dir: str = None):

        self.mode: str = mode
        self.ops: dict = ops
        self.F: np.array = F
        self.Fneu: np.array = Fneu
        self.deconv: np.array = spks
        self.stat: np.array = stat
        self.iscell: np.array = iscell
        self.dir: str = dir

        seconds_per_frame = 1.0 / self.ops["fs"]
        self.time = np.arange(0, self.ops["nframes"] * seconds_per_frame, seconds_per_frame)

        self.FCorr = None
        self.F0 = None
        self.dFonF0 = None

        if self.F is not None and self.ops["neucoeff"] is not None and self.Fneu is not None:
            self.FCorr = self.F - self.ops["neucoeff"] * self.Fneu
            if self.ops["win_baseline"] is not None and self.ops["sig_baseline"] is not None:
                self.F0 = get_f0(self.FCorr, self.ops["fs"], self.ops["win_baseline"], self.ops["sig_baseline"])
                self.dFonF0 = (self.FCorr - self.F0) / self.F0
                dFonF0_max = np.max(self.dFonF0, 1)
                self.dFonF0_norm = self.dFonF0 / dFonF0_max[:, None]

        self.deconv_norm = None
        if self.deconv is not None:
            deconv_max = np.max(self.deconv, 1)
            self.deconv_norm = self.deconv / deconv_max[:, None]

    def plot_traces(self, img_plot_dir, dataset_name, plot_good=True, plot_bad=False, dpi=300, line_width=1,
                    font_size=4):

        n_cell_cand = self.iscell.shape[0]
        n_cells_good = np.sum(self.iscell[:, 0])

        mean_img_vect = self.ops["meanImg"].flatten()
        mean_img_vect.sort()
        # img_base = np.mean(mean_img_vect[1:100])
        img_base = 2 ** 15 / 2



        for i_cell in range(0, n_cell_cand):

            is_good = self.iscell[i_cell, 0] == 1
            is_bad = not is_good
            if is_good and not plot_good:
                continue
            if is_bad and not plot_bad:
                continue

            n_plot_rows = 5
            n_plot_cols = 2

            plt.rcParams.update({'font.size': font_size})
            fig = plt.figure()

            gs = GridSpec(n_plot_rows, n_plot_cols, figure=fig)
            ax_img = fig.add_subplot(gs[0, 0])
            ax_zoom = fig.add_subplot(gs[0, 1])
            ax_rawF = fig.add_subplot(gs[1, :])
            ax_corrF= fig.add_subplot(gs[2, :])
            ax_dF = fig.add_subplot(gs[3, :])

            ###############
            ### Plot image
            ###############
            ax = ax_img
            ax.axis("off")

            mean_img = self.ops["meanImg"]
            img_min = min(mean_img.flatten())
            img_max = max(mean_img.flatten())
            mean_img = (mean_img - img_min) / (img_max - img_min)
            ax.imshow(mean_img, cmap=plt.cm.gray, interpolation="none")

            cell_img = np.zeros((self.ops['Ly'], self.ops['Lx']))
            cell_img[self.stat[i_cell]["ypix"], self.stat[i_cell]["xpix"]] = 1

            cell_img_mask = np.ma.masked_where(cell_img == 0, cell_img)

            ax.imshow(cell_img_mask, cmap=plt.cm.Reds_r, interpolation="none", alpha=0.5)

            pad_pix = 3
            bbox = getbbox(cell_img)
            box_x = bbox[2] - pad_pix
            box_y = bbox[0] - pad_pix
            box_w = bbox[3] - bbox[2] + pad_pix * 2
            box_h = bbox[1] - bbox[0] + pad_pix * 2

            if box_x < 0: box_x = 0
            if box_y < 0: box_y = 0

            rect = patches.Rectangle((box_x, box_y), box_w, box_h,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            ###############
            ### Plot zoom image
            ###############

            ax = ax_zoom
            ax.axis("off")

            mean_img = self.ops["meanImg"]
            mean_img = mean_img[box_y:box_y + box_h, box_x:box_x + box_w]
            img_min = min(mean_img.flatten())
            img_max = max(mean_img.flatten())
            mean_img = (mean_img - img_min) / (img_max - img_min)
            ax.imshow(mean_img, cmap=plt.cm.gray, interpolation="none")

            cell_img = np.zeros((self.ops['Ly'], self.ops['Lx']))
            cell_img[self.stat[i_cell]["ypix"], self.stat[i_cell]["xpix"]] = 1
            cell_img = cell_img[box_y:box_y + box_h, box_x:box_x + box_w]

            cell_img_mask = np.ma.masked_where(cell_img == 0, cell_img)
            ax.imshow(cell_img_mask, cmap=plt.cm.Reds_r, interpolation="none", alpha=0.3)

            ###############
            ### Plot raw trace
            ###############

            ax = ax_rawF
            ax.xaxis.set_visible(False)

            cell_F = self.F[i_cell]
            cell_F_neu = self.Fneu[i_cell]

            ax.plot(self.time, cell_F, linewidth=line_width, label="F")
            ax.plot(self.time, cell_F_neu, linewidth=line_width, label="neuropil")
            # ax.hlines(cell_F_base, ax.get_xlim()[0], ax.get_xlim()[1], label="cell_base_F", linewidth=line_width)
            ax.set(xlabel='Time (s)', ylabel='F', title='Raw signals')
            ax.grid()
            ax.legend()

            ###############
            ### Plot corrected + F0 trace
            ###############
            ax = ax_corrF
            ax.xaxis.set_visible(False)

            cell_FCorr = self.FCorr[i_cell]
            cell_F0 = self.F0[i_cell]

            ax.plot(self.time, cell_FCorr, linewidth=line_width, label="F corr.")
            ax.plot(self.time, cell_F0, linewidth=line_width, label="F0")
            ax.set(xlabel='Time (s)', ylabel='F', title='Neuropill corrected')
            ax.grid()
            ax.legend()

            ###############
            ### Plot dF trace
            ###############
            ax = ax_dF

            cell_dF = self.dFonF0[i_cell]
            cell_decon = self.deconv[i_cell] / np.max(self.deconv[i_cell])

            ax.plot(self.time, cell_dF, linewidth=line_width)
            ax.set(xlabel='Time (s)', ylabel='dF/F', title='Corrected')
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            ax.tick_params(axis='y', labelcolor=colors[0])

            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Deconvolved', color=colors[1])  # we already handled the x-label with ax1
            ax2.plot(self.time, cell_decon, color=colors[1], linewidth=line_width / 2)
            ax2.tick_params(axis='y', labelcolor=colors[1])

            ax.set_ylim(ymin=-0.1)
            ax2.set_ylim(ymin=-0.1)
            ax.grid()

            ###############
            ### Save plot
            ###############

            plot_file = os.path.join(img_plot_dir, "{}.roi-{}.png".format(dataset_name, i_cell))
            fig.savefig(plot_file, dpi=dpi)
            # fig.show()
            plt.clf()
            plt.close('all')

    def plot_movement_hists(self, regmetrics_dir, pix_size):

        fig, axs = plt.subplots(2, 2, tight_layout=True)

        nbins = np.linspace(-10, 10, 21)
        i_plot = (0, 0)
        axs[i_plot].hist(self.ops['xoff'] * pix_size, bins=nbins, alpha=0.75, density=True)
        axs[i_plot].set_xlabel('Shift (um)')
        axs[i_plot].set_ylabel('%')
        axs[i_plot].set_title("X shift")
        axs[i_plot].grid(True)

        i_plot = (0, 1)
        axs[i_plot].hist(self.ops['yoff'] * pix_size, bins=nbins, alpha=0.75, density=True)
        axs[i_plot].set_xlabel('Shift (um)')
        axs[i_plot].set_ylabel('%')
        axs[i_plot].set_title("Y shift")
        axs[i_plot].grid(True)

        nbins = np.linspace(-30, 30, 21)
        i_plot = (1, 0)
        axs[i_plot].hist(self.ops['xoff'], bins=nbins, alpha=0.75, density=True)
        axs[i_plot].set_xlabel('Shift (pixels)')
        axs[i_plot].set_ylabel('%')
        axs[i_plot].grid(True)

        i_plot = (1, 1)
        axs[i_plot].hist(self.ops['yoff'], bins=nbins, alpha=0.75, density=True)
        axs[i_plot].set_xlabel('Shift (pixels)')
        axs[i_plot].set_ylabel('%')
        axs[i_plot].grid(True)

        plot_img_path = os.path.join(regmetrics_dir, "rigid_shifts.png")
        fig.savefig(plot_img_path, dpi=300, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    def save_ref_img(self, regmetrics_dir):
        ref_img = utils.img.normalize_img_8bit(self.ops['refImg'], 0.1)
        imageio.imwrite(os.path.join(regmetrics_dir, "reg_ref.png"), ref_img)

    def plot_pcas(self, plot_dir, gifs_dir):
        plt.close('all')

        n_pca_comp = self.ops['regDX'].shape[0]

        for i_pc in range(n_pca_comp):
            # print("PCA #{i}".format(i_pc))

            fig = plt.figure(tight_layout=False)
            # fig, axs = plt.subplots(3, 1, tight_layout=True)
            gs = GridSpec(3, 2, figure=fig)

            pc_hi = self.ops['regPC'][0, i_pc, :, :]
            pc_lo = self.ops['regPC'][1, i_pc, :, :]

            pc_diff = pc_lo - pc_hi

            ax = fig.add_subplot(gs[0, :])
            ax.plot(self.ops['tPC'][:, i_pc])
            ax.set_title("PC across times")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Magnitude")

            ax = fig.add_subplot(gs[1, 0])
            ax.imshow(pc_hi, cmap='gray')
            ax.set_title("PC high")
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(gs[1, 1])
            ax.imshow(pc_lo, cmap='gray')
            ax.set_title("PC low")
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(gs[2, 0])
            ax.imshow(pc_diff, cmap='bwr')
            ax.set_title("Diff")
            ax.set_xticks([])
            ax.set_yticks([])

            fig.suptitle('PCA #{p}'.format(p=i_pc))

            plot_img_path = os.path.join(plot_dir, "PC_{i}.png".format(i=str(i_pc).zfill(1)))
            plt.savefig(plot_img_path, dpi=300, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            # Make a gif of the top and bottom PCs
            gif_path = os.path.join(gifs_dir, "PC_{i}.gif".format(i=str(i_pc).zfill(1)))
            with imageio.get_writer(gif_path, mode='I', fps=2) as writer:
                pc_cat = np.dstack((pc_hi, pc_lo))
                pc_cat = np.moveaxis(pc_cat, -1, 0)
                pc_cat = utils.img.normalize_img_8bit(pc_cat, 0.1)

                writer.append_data(pc_cat[0, :, :])
                writer.append_data(pc_cat[1, :, :])

    def plot_pca_shifts(self, data_rigid, data_notreg, plot_img_path, pix_size=None):
        # Plot the shift for each PC high vs low
        i_plot = -1
        n_plots = 1
        if data_rigid:
            n_plots += 1
        if data_notreg:
            n_plots += 1
        fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, tight_layout=True)

        scale = 1
        unit = "pixels"
        if pix_size:
            scale = pix_size
            unit = "um"

        i_plot += 1
        axs[i_plot].plot(self.ops["regDX"][:, 0]*scale, color='r', label='rigid')
        axs[i_plot].plot(self.ops["regDX"][:, 1]*scale, color='g', label='avg')
        axs[i_plot].plot(self.ops["regDX"][:, 2]*scale, color='b', label='max')
        axs[i_plot].set_ylabel('Shift ({0})'.format(unit))
        axs[i_plot].set_title('Registered')
        axs[i_plot].legend()

        if data_rigid:
            i_plot += 1
            axs[i_plot].plot(data_rigid.ops["regDX"][:, 0]*scale, color='r')
            axs[i_plot].plot(data_rigid.ops["regDX"][:, 1]*scale, color='g')
            axs[i_plot].plot(data_rigid.ops["regDX"][:, 2]*scale, color='b')
            axs[i_plot].set_ylabel('Shift ({0})'.format(unit))
            axs[i_plot].set_title('Rigid (broken)')

        if data_notreg:
            i_plot += 1
            axs[i_plot].plot(data_notreg.ops["regDX"][:, 0]*scale, color='r')
            axs[i_plot].plot(data_notreg.ops["regDX"][:, 1]*scale, color='g')
            axs[i_plot].plot(data_notreg.ops["regDX"][:, 2]*scale, color='b')
            axs[i_plot].set_ylabel('Shift ({0})'.format(unit))
            axs[i_plot].set_title('Raw')

        axs[i_plot].set_xlabel('Principal component # (highs vs lows)')

        fig.suptitle('Shifts for each PC')

        fig.savefig(plot_img_path, dpi=300, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    def load_image(self):

        tif_reg_path = os.path.join(self.dir, "reg.tif")
        img_data = utils.img.read_tif_vol(tif_reg_path)

        return img_data

    def load_tif_dir(self):

        tif_dir = os.path.join(self.dir, "reg_tif")
        tif_data = utils.img.load_tif_dir(tif_dir, file_filter="file*_chan0.tif")
        return tif_data

    def save_tif(self, tif_path=None, prc_clip=None):

        tif_data = self.load_tif_dir()
        if tif_path is None:
            tif_path = os.path.join(self.dir, "reg.tif")

        if not prc_clip is None:
            tif_data = utils.img.normalize_img_16bit(tif_data, prc_clip)
        imageio.volwrite(tif_path, tif_data, format="TIFF")

        return tif_path



    def save_zproj(self, images_dir, mode, use_s2p=False):

        tif_reg_path = os.path.join(self.dir, "reg.tif")
        if use_s2p:
            z_proj_mean = self.ops["meanImg"]
            z_proj_max = self.ops["max_proj"] # missing wtf?
        else:
            reg_data = utils.img.read_tif_vol(tif_reg_path)
            z_proj_mean = reg_data.mean(axis=0)
            z_proj_max = reg_data.max(axis=0)

        save_zproj_img(z_proj_mean, z_proj_max, images_dir, mode)




    def create_movie(self, video_file):
        tif_path = os.path.join(self.dir, "reg.tif")
        utils.img.tif_to_movie(tif_path, self.ops["fs"], video_file)

    def create_roi_image(self, plot_img_path, plot_good=True, plot_bad=False, good_vs_bad=False):

        n_roi_cand = self.iscell.shape[0]
        n_roi_good = np.sum(self.iscell[:, 0])
        n_roi_bad = n_roi_cand - np.sum(self.iscell[:, 0])

        n_roi = 0
        if plot_good:
            n_roi = n_roi + n_roi_good
        if plot_bad:
            n_roi = n_roi + n_roi_bad

        if good_vs_bad:
            cell_colours = plt.cm.get_cmap('bwr', 2)
        else:
            cell_colours = plt.cm.get_cmap('hsv', n_roi)

        s2p_img = self.ops["meanImg"] # wtf max_proj missing?
        img_min = min(s2p_img.flatten())
        img_max = max(s2p_img.flatten())
        s2p_img = (s2p_img - img_min) / (img_max - img_min)
        fig = plt.figure(tight_layout=False)
        plt.imshow(s2p_img, cmap=plt.cm.gray, interpolation="none")

        i_roi_good = -1
        for i_roi in range(0, n_roi_cand):

            is_good = self.iscell[i_roi, 0] == 1
            if not plot_good and is_good:
                continue
            if not plot_bad and not is_good:
                continue
            i_roi_good += 1

            cell_img = np.zeros((self.ops['Ly'], self.ops['Lx']))
            cell_img[self.stat[i_roi]["ypix"], self.stat[i_roi]["xpix"]] = 1

            cell_img_mask = np.ma.masked_where(cell_img == 0, cell_img)

            col = cell_colours(i_roi_good)
            if good_vs_bad:
                if is_good:
                    col = cell_colours(1)
                else:
                    col = cell_colours(0)

            plt.imshow(cell_img_mask,
                       cmap=ListedColormap(col),
                       interpolation="none", alpha=0.3)

        plt.tick_params(
            axis='both',  # changes apply to the both axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off

        fig.savefig(plot_img_path, dpi=300, facecolor='white')

        plt.cla()
        plt.clf()
        plt.close('all')

    def plot_evoked(self, plot_dir, ca_data_type: CaDataType,
                    s2p_start_time, trial_times, time_trial, time_pre, time_post, plot_good=True, plot_bad=False,
                    ignore_no_pre_time=False, ignore_no_post_time=False, plot_label="", pre_f0=True, norm_deconv=True,
                    plot_as_trace=False, behave_trace=None, behave_trace_time=None):

        trial_data, dFOnF0, deconv, behave_trials, behave_trial_data = self.get_trial_data(s2p_start_time=s2p_start_time,
                                                                                           trial_times=trial_times,
                                                                                           time_trial=time_trial,
                                                                                           time_pre=time_pre,
                                                                                           time_post=time_post,
                                                                                           ignore_no_pre_time=ignore_no_pre_time,
                                                                                           ignore_no_post_time=ignore_no_post_time,
                                                                                           pre_f0=pre_f0,
                                                                                           norm_deconv=norm_deconv,
                                                                                           behave_trace=behave_trace,
                                                                                           behave_trace_time=behave_trace_time)

        if behave_trace is not None:
            fig = plt.figure(tight_layout=False)
            ax = plt.gca()
            if plot_as_trace:
                ax.plot(behave_trial_data.trial_time, behave_trials, linewidth=1, alpha=0.5, color='grey')
                mean_trace = np.mean(behave_trials, 1)
                ax.plot(behave_trial_data.trial_time, mean_trace, linewidth=2, color='black')

                if time_trial > 0:
                    ymax = np.max(np.max(behave_trials))
                    stim_bar_y = ymax * 1.1
                    ax.plot([0, time_trial], [stim_bar_y, stim_bar_y], linewidth=2, alpha=1, color='blue')

                #ax.set(ylabel=ylabel)
            else:
                im = ax.imshow(behave_trials, aspect='auto')
                ax.set(ylabel='Bout')
                plt.colorbar(im)

            ax.set(xlabel='Time (s)', title='Behaviour')
            ax.grid()

            plot_img_path = os.path.join(plot_dir,
                                         "001.behav.{}.png".format(plot_label))
            fig.savefig(plot_img_path, dpi=300, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

        # raise Exception()

        n_roi_cand = self.iscell.shape[0]
        i_roi_good = -1
        for i_roi in range(0, n_roi_cand):

            if not self.do_roi_plot(i_roi, plot_good=plot_good, plot_bad=plot_bad):
                continue
            i_roi_good += 1

            fig = plt.figure(tight_layout=False)
            ax = plt.gca()

            if ca_data_type == CaDataType.DFONF0:
                plot_data = dFOnF0
                ylabel = 'dF/F'

            elif ca_data_type == CaDataType.DECONV:
                plot_data = deconv
                ylabel = 'Deconvolved'

            ca_label = getCaDataTypeLabelShort(ca_data_type)

            if plot_as_trace:
                ax.plot(trial_data.trial_time, plot_data[i_roi, :, :], linewidth=1, alpha=0.5, color='grey')
                mean_trace = np.mean(plot_data[i_roi, :, :], 1)
                ax.plot(trial_data.trial_time, mean_trace, linewidth=2, color='black')

                if time_trial > 0:
                    ymax = np.max(np.max(plot_data[i_roi, :, :]))
                    stim_bar_y = ymax * 1.1
                    ax.plot([0, time_trial], [stim_bar_y, stim_bar_y], linewidth=2, alpha=1, color='blue')

                ax.set(ylabel=ylabel)
            else:
                # tried to normalize by the peak in each bout but its looks totally wrong, or just shit.
                #trial_peak = np.max(plot_data[i_roi, :, 0:trial_data.n_pre_time_indexes], axis=1)
                #trial_peak = np.max(plot_data[i_roi, :, :], axis=1)
                #print(trial_peak)
                #plot_data_norm# = plot_data[i_roi, :, :] / trial_peak[:, None]
                #print(np.max(plot_data_norm[:, :], axis=1))
                #im = ax.imshow(plot_data_norm, aspect='auto')

                im = ax.imshow(plot_data[i_roi, :, :], aspect='auto')
                if behave_trace is None:
                    ax.set(ylabel='Trial')
                else:
                    ax.set(ylabel='Bout')
                plt.colorbar(im)

            ax.set(xlabel='Time (s)', title='Evoked')

            #ax.set_ylim(ymin=-1)
            ax.grid()

            plot_img_path = os.path.join(plot_dir, "{}.roi-{}.evoked.{}.{}.png".format(self.mode, i_roi, plot_label, ca_label))
            fig.savefig(plot_img_path, dpi=300, facecolor='white')

            plt.cla()
            plt.clf()
            plt.close('all')

            # raise Exception()




    def get_trial_data(self, s2p_start_time, trial_times, time_trial, time_pre, time_post, ignore_no_pre_time,
                       ignore_no_post_time, pre_f0=True, norm_deconv=True, behave_trace=None, behave_trace_time=None):

        s2p_trial_data = get_trial_indexes(self.time + s2p_start_time, trial_times, time_trial, time_pre, time_post,
                                           ignore_no_pre_time, ignore_no_post_time)

        if behave_trace is not None:
            behave_trial_data = get_trial_indexes(behave_trace_time, trial_times, time_trial, time_pre, time_post,
                                                  ignore_no_pre_time, ignore_no_post_time)


        Fcorr = self.F - self.ops["neucoeff"] * self.Fneu
        deconv = self.deconv

        #Fcorr = np.roll(Fcorr, 100)



        if pre_f0:
            #Fcorr_trials = Fcorr[:, s2p_trial_data.trial_indexes]
            Fcorr_trials = get_trace_trials(Fcorr, s2p_trial_data.trial_indexes)
            # Use trial prestim as f0
            f0 = np.mean(Fcorr_trials[:, :, 0:s2p_trial_data.n_pre_time_indexes], axis=2)
            dfOnF_trials = (Fcorr_trials - f0[:, :, None])
            dfOnF_trials = dfOnF_trials / f0[:, :, None]
        else:
            #dfOnF_trials = self.dFonF0[:, s2p_trial_data.trial_indexes]
            dfOnF_trials = get_trace_trials(self.dFonF0, s2p_trial_data.trial_indexes)

        # # Use prestim mean as f0, not useful
        # f0 = np.mean(np.mean(Fcorr_trials[:, :, 0:s2p_trial_data.n_pre_time_indexes], axis=2), axis=1)
        # dfOnF_trials = (Fcorr_trials - f0[:, None, None])
        # dfOnF_trials = dfOnF_trials / f0[:, None, None]

        if norm_deconv:
            max_deconv = np.max(deconv, axis=1)
            deconv = (deconv.T / max_deconv).T

        #deconv_trials = deconv[:, s2p_trial_data.trial_indexes]
        deconv_trials = get_trace_trials(deconv, s2p_trial_data.trial_indexes)

        if behave_trace is not None:
            #behave_trials = behave_trace[s2p_trial_data.trial_indexes]
            behave_trials = get_trace_trials(behave_trace, s2p_trial_data.trial_indexes)

        if behave_trace is not None:
            return s2p_trial_data, dfOnF_trials, deconv_trials, behave_trials, behave_trial_data
        else:
            return s2p_trial_data, dfOnF_trials, deconv_trials



    def get_n_roi_plot(self, plot_good=True, plot_bad=False):
        n_roi_cand = self.iscell.shape[0]
        n_roi_good = np.sum(self.iscell[:, 0])
        n_roi_bad = n_roi_cand - np.sum(self.iscell[:, 0])

        n_roi_plot = 0
        if plot_good:
            n_roi_plot = n_roi_plot + n_roi_good
        if plot_bad:
            n_roi_plot = n_roi_plot + n_roi_bad

        return n_roi_plot, n_roi_cand, n_roi_good, n_roi_bad

    def do_roi_plot(self, i_roi, plot_good=True, plot_bad=False):
        is_good = self.iscell[i_roi, 0] == 1
        is_bad = not is_good
        if not plot_good and is_good:
            return False
        if not plot_bad and is_bad:
            return False
        return True



class TrialData:

    def __init__(self, trial_indexes, trial_time, n_pre_time_indexes, n_duration_indexes, n_post_time_indexes):

        self.trial_indexes: [] = trial_indexes
        self.n_trials: int = len(trial_indexes)
        self.trial_time: np.array = trial_time
        self.n_pre_time_indexes: int = n_pre_time_indexes
        self.n_duration_indexes: int = n_duration_indexes
        self.n_post_time_indexes: int = n_post_time_indexes
        self.n_total_indexes: int = n_pre_time_indexes + n_duration_indexes + n_post_time_indexes



def get_mode_path(path, mode):
    return os.path.join(path, 'suite2p_{mode}'.format(mode=mode))

def load_mode(path, mode) -> S2PData:
    s2p_data = load(get_mode_path(path, mode))
    s2p_data.mode = mode
    return s2p_data


def load(s2p_dir) -> S2PData:
    s2p_data_dir = os.path.join(s2p_dir, "plane0")

    ops = np.load("{}.npy".format(os.path.join(s2p_data_dir, "ops")), allow_pickle=True)
    ops = ops.item()
    if ops["roidetect"]:
        F = np.load(os.path.join(s2p_data_dir, "F.npy"), allow_pickle=True)
        Fneu = np.load(os.path.join(s2p_data_dir, "Fneu.npy"), allow_pickle=True)
        spks = np.load(os.path.join(s2p_data_dir, "spks.npy"), allow_pickle=True)
        stat = np.load(os.path.join(s2p_data_dir, "stat.npy"), allow_pickle=True)
        iscell = np.load(os.path.join(s2p_data_dir, "iscell.npy"), allow_pickle=True)

        n_cell_cand = iscell.shape[0]
        n_cells_good = int(np.sum(iscell[:, 0]))
        print("# roi cand={c}, # roi good={g}".format(c=n_cell_cand, g=n_cells_good))
        print("Suite 2p data Loaded")

        return S2PData(ops=ops,
                       F=F,
                       Fneu=Fneu,
                       spks=spks,
                       stat=stat,
                       iscell=iscell,
                       dir=s2p_data_dir)
    else:
        return S2PData(ops=ops, dir=s2p_data_dir)


# Blindy copied from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def getbbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def get_trial_indexes(time, trial_times, time_trial, time_pre, time_post, ignore_no_pre_time, ignore_no_post_time):

    frame_interval = time[1] - time[0]

    # Construct the time array such that there is a point at exactly zero
    trial_time_pre = np.flip(np.arange(-frame_interval, -time_pre, -frame_interval))
    trial_time_post = np.arange(0, time_trial + time_post, frame_interval)
    trial_time = np.hstack((trial_time_pre, trial_time_post))

    n_pre_time_indexes = trial_time_pre.size
    n_total_indexes = trial_time.size
    if time_trial == 0:
        n_trial_indexes = 0
    else:
        n_trial_indexes = np.where(trial_time_post <= time_trial)[0][-1] + 1
    n_post_stim_indexes = n_total_indexes - n_pre_time_indexes - n_trial_indexes
    # Probably could get rid of this loop
    trial_indexes = []
    for t in trial_times:
        zero_time_index = int(np.argmin(np.abs(time - t)))

        pre_time_index = zero_time_index - n_pre_time_indexes
        trial_end_index = zero_time_index + n_trial_indexes
        post_time_index = trial_end_index + n_post_stim_indexes

        if pre_time_index < 0:
            if ignore_no_pre_time:
                continue
            else:
                raise Exception("Pre time extended before the first sci frame")
        if post_time_index > time.size:
            if ignore_no_post_time:
                continue
            else:
                raise Exception("Post time extended after the last sci frame")

        trial_indexes.append(np.arange(pre_time_index, post_time_index))

    return TrialData(trial_indexes=trial_indexes,
                     trial_time=trial_time,
                     n_pre_time_indexes=n_pre_time_indexes,
                     n_duration_indexes=n_trial_indexes,
                     n_post_time_indexes=n_post_stim_indexes)


def get_f0(F, fs, win_baseline, sig_baseline):
    # Taken from https://suite2p.readthedocs.io/en/latest/_modules/suite2p/extraction/dcnv.html#preprocess
    win_size = int(win_baseline * fs)
    sigma = int(sig_baseline * fs)


    #Flow = filters.gaussian_filter(F, [0., sig_baseline])
    # This is surely a bug? Original code doesn't change time to frames.
    Flow = filters.gaussian_filter(input=F, sigma=[0., sigma], mode='nearest')
    Flow = filters.minimum_filter1d(input=Flow, size=win_size, mode='nearest')
    Flow = filters.maximum_filter1d(input=Flow, size=win_size, mode='nearest')


    return Flow


def get_trace_trials(trace_data, trial_indexes):

    trial_length = len(trial_indexes[0])

    if trace_data.ndim == 2:
        n_rois = trace_data.shape[0]
        trace_trials = np.zeros((n_rois, trial_length, len(trial_indexes)))
        for i, t in enumerate(trial_indexes):
            trace_trials[:, :, i] = trace_data[:, t]
    else:
        trace_trials = np.zeros((trial_length, len(trial_indexes)))
        for i, t in enumerate(trial_indexes):
            trace_trials[:, i] = trace_data[t]

    return trace_trials

def save_zproj_img(z_proj_mean, z_proj_max, images_dir, mode):

    prc_clip = 0.1
    z_proj_mean = utils.img.normalize_img(z_proj_mean, prc_clip)
    z_proj_max = utils.img.normalize_img(z_proj_max, prc_clip)

    z_proj_mean = z_proj_mean * 255
    z_proj_max = z_proj_max * 255

    z_proj_mean = z_proj_mean.astype('uint8')
    z_proj_max = z_proj_max.astype('uint8')

    z_proj_mean_path = os.path.join(images_dir, "z_proj_mean_{m}.png".format(m=mode))
    z_proj_max_path = os.path.join(images_dir, "z_proj_max_{m}.png".format(m=mode))
    imageio.imwrite(z_proj_mean_path, z_proj_mean)
    imageio.imwrite(z_proj_max_path, z_proj_max)









