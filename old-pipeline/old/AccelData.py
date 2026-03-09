# import matplotlib
# matplotlib.use("Agg")
from nptdms import TdmsFile
from scipy import signal
import numpy as np
from utils import misc
from classes import Experiment
import os
from enum import Enum
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def load_acc_data(exp: Experiment,
                  axis_swap=None,
                  axis_sign=[1, 1, 1],
                  sensitivity=1000 / 420,
                  zero_g_bias=1.5,
                  acc_filt_f=550,
                  acc_filt_order=2,
                  grav_filt_f=2,
                  grav_filt_order=2,
                  calc_pitch_roll=True):

    tdms_ai_file = os.path.join(exp.data_raw_path, misc.path_leaf(exp.DAQSettings.file_ai))
    group_name = exp.DAQSettings.groupname
    xchan = exp.DAQSettings.accelxchanname
    ychan = exp.DAQSettings.accelychanname
    zchan = exp.DAQSettings.accelzchanname

    print("AccData: Loading tdms file")
    with TdmsFile.read(tdms_ai_file) as ai_file:
        x = ai_file[group_name][xchan].data
        y = ai_file[group_name][ychan].data
        z = ai_file[group_name][zchan].data

        if axis_swap:
            m = np.vstack((x, y, z))
            m = m[axis_swap, :]
            x = m[0, :]
            y = m[1, :]
            z = m[2, :]

        if axis_sign:
            x = x * axis_sign[0]
            y = y * axis_sign[1]
            z = z * axis_sign[2]

        x = x - zero_g_bias
        y = y - zero_g_bias
        z = z - zero_g_bias

        x = sensitivity * x
        y = sensitivity * y
        z = sensitivity * z

        time = ai_file[group_name][xchan].time_track()

        return AccelData(exp.DAQSettings.sf,
                         time,
                         x,
                         y,
                         z,
                         acc_filt_f,
                         acc_filt_order,
                         grav_filt_f,
                         grav_filt_order,
                         calc_pitch_roll)


class AccDataType(Enum):
    LINEAR_ACC = 1
    LINEAR_ACC_RAW = 2
    PITCH = 3
    ROLL = 4
    PITCH_AND_ROLL = 5
    PITCH_AND_ROLL_VELOCITY = 6

class AccelData:

    def __init__(self,
                 fs,
                 time,
                 x,
                 y,
                 z,
                 acc_filt_f=None,
                 acc_filt_order=None,
                 grav_filt_f=None,
                 grav_filt_order=None,
                 calc_pitch_roll=True):

        self.fs = fs
        self.time = time

        n = x.size

        self.x_raw = x
        self.y_raw = y
        self.z_raw = z
        self.x = x
        self.y = y
        self.z = z

        self.x_nograv = np.zeros((n))
        self.y_nograv = np.zeros((n))
        self.z_nograv = np.zeros((n))
        self.x_grav = np.zeros((n))
        self.y_grav = np.zeros((n))
        self.z_grav = np.zeros((n))
        self.pitch = np.zeros((n))
        self.roll = np.zeros((n))
        self.pitch_vel = np.zeros((n))
        self.roll_vel = np.zeros((n))

        if grav_filt_f:
            self.pitch = np.zeros((n))
            self.roll = np.zeros((n))

            print("AccData: gravity filter")
            grav_filt_w = grav_filt_f / (self.fs / 2)  # Normalize the frequency to nyquist
            grav_filt_low_b, grav_filt_low_a = signal.butter(grav_filt_order, grav_filt_w, 'low', analog=False)
            grav_filt_hi_b, grav_filt_hi_a = signal.butter(grav_filt_order, grav_filt_w, 'high', analog=False)

            grav_x = signal.filtfilt(grav_filt_low_b, grav_filt_low_a, self.x_raw)
            grav_y = signal.filtfilt(grav_filt_low_b, grav_filt_low_a, self.y_raw)
            grav_z = signal.filtfilt(grav_filt_low_b, grav_filt_low_a, self.z_raw)

            self.x_grav = grav_x
            self.y_grav = grav_y
            self.z_grav = grav_z

            # Save a copy of the gravity subtracted values before doing this filtering.
            self.x_nograv = signal.filtfilt(grav_filt_hi_b, grav_filt_hi_a, self.x_raw)
            self.y_nograv = signal.filtfilt(grav_filt_hi_b, grav_filt_hi_a, self.y_raw)
            self.z_nograv = signal.filtfilt(grav_filt_hi_b, grav_filt_hi_a, self.z_raw)

            self.x = self.x_nograv
            self.y = self.y_nograv
            self.z = self.z_nograv

            if calc_pitch_roll:

                print("AccData: calculating pitch and roll")
                grav = np.vstack((self.x_grav, self.y_grav, self.z_grav))

                nv = np.linalg.norm(grav, axis=0)
                grav_uv = np.divide(grav, nv).transpose()

                # These values don't really make sense to me but they work.
                pitch_plane = np.array([0, 1, 0])
                roll_plane = np.array([1, 0, 0])

                pitch_dp = np.dot(grav_uv, pitch_plane)
                roll_dp = np.dot(grav_uv, roll_plane)

                pitch_angle = np.rad2deg(np.arccos(pitch_dp)) - 90
                roll_angle = np.rad2deg(np.arccos(roll_dp)) - 90

                self.pitch = pitch_angle
                self.roll = roll_angle

                self.pitch_vel = np.gradient(self.pitch) * self.fs
                self.roll_vel = np.gradient(self.roll) * self.fs

        if acc_filt_f:
            # First do a low pass filter because DAQ has a higher SF thant the accelerometer, 550 Hz.
            # X&Y can do 1600, not sure why.
            print("AccData: low pass filter")

            acc_filt_w = acc_filt_f / (self.fs / 2)  # Normalize the frequency to nyquist
            acc_filt_b, acc_filt_a = signal.butter(acc_filt_order, acc_filt_w, 'low', analog=False)

            x_filt = signal.filtfilt(acc_filt_b, acc_filt_a, self.x)
            y_filt = signal.filtfilt(acc_filt_b, acc_filt_a, self.y)
            z_filt = signal.filtfilt(acc_filt_b, acc_filt_a, self.z)

            self.x = x_filt
            self.y = y_filt
            self.z = z_filt








        print("AccData: data processing complete.")

    def get_crossings(self, threshold, time_exclude, data_type: AccDataType = AccDataType.LINEAR_ACC):

        if data_type == AccDataType.LINEAR_ACC:
            x_crossings = misc.get_crossings(np.absolute(self.x), threshold)
            y_crossings = misc.get_crossings(np.absolute(self.y), threshold)
            z_crossings = misc.get_crossings(np.absolute(self.z), threshold)
            crossings = np.concatenate((x_crossings, y_crossings, z_crossings))
            crossings.sort()
        elif data_type == AccDataType.PITCH:
            crossings = misc.get_crossings(np.absolute(self.pitch), threshold)
        elif data_type == AccDataType.ROLL:
            crossings = misc.get_crossings(np.absolute(self.roll), threshold)
        elif data_type == AccDataType.PITCH_AND_ROLL:
            p_crossings = misc.get_crossings(np.absolute(self.pitch), threshold)
            r_crossings = misc.get_crossings(np.absolute(self.roll), threshold)
            crossings = np.concatenate((p_crossings, r_crossings))
            crossings.sort()
        elif data_type == AccDataType.PITCH_AND_ROLL_VELOCITY:
            p_crossings = misc.get_crossings(np.absolute(self.pitch_vel), threshold)
            r_crossings = misc.get_crossings(np.absolute(self.roll_vel), threshold)
            crossings = np.concatenate((p_crossings, r_crossings))
            crossings.sort()

        samples_exclude = int(np.round(self.fs * time_exclude))
        close_events = []
        for i_event in range(1, crossings.size):
            if crossings[i_event] - crossings[i_event - 1] < samples_exclude:
                close_events.append(i_event)
        crossings = np.delete(crossings, close_events)

        return crossings

    def get_events(self, threshold, time_exclude, pre_time, post_time, data_type: AccDataType = AccDataType.LINEAR_ACC):

        samples_pre = int(np.round(self.fs * pre_time))
        samples_post = int(np.round(self.fs * post_time))
        samples_total = samples_pre+samples_post

        crossings = self.get_crossings(threshold, time_exclude, data_type)

        event_mat = np.zeros((len(crossings), 6, samples_total))

        for i_crossing, i_sample in enumerate(crossings):
            index_start = i_sample - samples_pre
            index_end = i_sample + samples_post
            if index_end >= self.time.size:
                break
                event_mat = event_mat[:-1, :, :]

            t = self.time[index_start:index_end]
            x_cross = self.x[index_start:index_end]
            y_cross = self.y[index_start:index_end]
            z_cross = self.z[index_start:index_end]
            p_cross = self.pitch[index_start:index_end]
            r_cross = self.roll[index_start:index_end]

            event_mat[i_crossing, 0, :] = t
            event_mat[i_crossing, 1, :] = x_cross
            event_mat[i_crossing, 2, :] = y_cross
            event_mat[i_crossing, 3, :] = z_cross
            event_mat[i_crossing, 4, :] = p_cross
            event_mat[i_crossing, 5, :] = r_cross

        return event_mat

    def make_animated_plot(self, fps, start_time_index, end_time_index, n_plot_frames, plot_window=2,
                           data_type: AccDataType = AccDataType.LINEAR_ACC, yax_max=None):

        t = self.time

        fig = plt.Figure()
        canvas = FigureCanvasAgg(fig)

        #ax = plt.axes()
        ax = fig.add_subplot()
        halfpwin = plot_window / 2


        ntcks = 11
        xlbls = np.round(np.linspace(0, plot_window, ntcks), 1)
        ax.set_xlim(left=0, right=plot_window)
        ax.axes.xaxis.set_ticks(xlbls)
        ax.axes.xaxis.set_ticklabels(xlbls)
        ax.axes.grid()
        ax.set_xlabel("Time (s)")
        if data_type == AccDataType.LINEAR_ACC or data_type == AccDataType.LINEAR_ACC_RAW:
            line_x, = ax.plot([], [], lw=2, label='x')
            line_y, = ax.plot([], [], lw=2, label='y')
            line_z, = ax.plot([], [], lw=2, label='z')
            if not yax_max:
                if data_type == AccDataType.LINEAR_ACC:
                    yax_max = max(np.max(np.abs(self.x)), np.max(np.abs(self.y)), np.max(np.abs(self.z)))
                elif data_type == AccDataType.LINEAR_ACC_RAW:
                    yax_max = max(np.max(np.abs(self.x_raw)), np.max(np.abs(self.y_raw)), np.max(np.abs(self.z_raw)))
            if data_type == AccDataType.LINEAR_ACC:
                ax.set_ylabel("Linear acceleration (g)")
            elif data_type == AccDataType.LINEAR_ACC_RAW:
                ax.set_ylabel("Raw linear acceleration (g)")
            ax.set_ylim(bottom=-yax_max, top=yax_max)
        elif data_type == AccDataType.PITCH_AND_ROLL:
            line_p, = ax.plot([], [], lw=2, label='pitch')
            line_r, = ax.plot([], [], lw=2, label='roll')
            # yax_max = max(np.max(np.abs(self.roll)), np.max(np.abs(self.pitch)))
            # ax.set_ylim(bottom=-yax_max, top=yax_max)
            ax.set_ylim(bottom=-90, top=90)
            ax.set_ylabel("Pitch & roll (degrees)")
        elif data_type == AccDataType.PITCH_AND_ROLL_VELOCITY:
            line_p, = ax.plot([], [], lw=2, label='pitch')
            line_r, = ax.plot([], [], lw=2, label='roll')
            if not yax_max:
                max_r = np.max(np.abs(self.pitch_vel))
                max_p = np.max(np.abs(self.roll_vel))
                yax_max = np.max([max_r, max_p])
            ax.set_ylim(bottom=-yax_max, top=yax_max)
            ax.set_ylabel("Angular velocity (degrees/s)")
        else:
            raise Exception("Data type not supported")

        ax.legend(loc='lower right')


        # Init only required for blitting to give a clean slate.
        def init():

            if data_type == AccDataType.LINEAR_ACC or data_type == AccDataType.LINEAR_ACC_RAW:
                line_x.set_data([], [])
                line_y.set_data([], [])
                line_z.set_data([], [])
                return line_x, line_y, line_z
            elif data_type == AccDataType.PITCH_AND_ROLL:
                line_p.set_data([], [])
                line_r.set_data([], [])
                return line_p, line_r
            elif data_type == AccDataType.PITCH_AND_ROLL_VELOCITY:
                line_p.set_data([], [])
                line_r.set_data([], [])
                return line_p, line_r
            else:
                raise Exception("Data type not supported")


        frame_interval = 1 / fps
        frame_interval_ms = frame_interval * 1000
        frame_interval_indexes = int(round(frame_interval * self.fs))
        plot_window_indexes = int(round(plot_window * self.fs))
        halfpwin_indexes = int(round(plot_window_indexes/2))

        def animate_acc_plot(i):

            # Still confused about how this should work.
            # Currently whatever is happening in the current frame appears on the left side of the plot, and moves to
            # right. This requires that the time values be flipped, see bellow.
            # Not sure if this is a good idea, or is similar to how a ephys trace works.

            i_frame = start_time_index + int(i * frame_interval_indexes)
            i_start = i_frame - plot_window_indexes
            i_end = i_frame

            if i_start < 0:
                i_start = 0
            if i_end > t.size:
                i_end = end_time_index

            t_plot = t[i_start:i_end]
            # Set to start at zero
            t_plot = t_plot - t_plot[0]
            # NOTE the flip here to make it go left to right:
            t_plot = np.flip(t_plot)

            # pwin_end_time = t[i_end]
            # pwin_start_time = pwin_end_time - plot_window
            #
            # ax.set_xlim(left=pwin_start_time, right=pwin_end_time)

            if data_type == AccDataType.LINEAR_ACC:
                line_x.set_data(t_plot, self.x[i_start:i_end])
                line_y.set_data(t_plot, self.y[i_start:i_end])
                line_z.set_data(t_plot, self.z[i_start:i_end])
                return line_x, line_y, line_z
            elif data_type == AccDataType.LINEAR_ACC_RAW:
                line_x.set_data(t_plot, self.x_raw[i_start:i_end])
                line_y.set_data(t_plot, self.y_raw[i_start:i_end])
                line_z.set_data(t_plot, self.z_raw[i_start:i_end])
                return line_x, line_y, line_z
            elif data_type == AccDataType.PITCH_AND_ROLL:
                line_p.set_data(t_plot, self.pitch[i_start:i_end])
                line_r.set_data(t_plot, self.roll[i_start:i_end])
                return line_p, line_r
            elif data_type == AccDataType.PITCH_AND_ROLL_VELOCITY:
                line_p.set_data(t_plot, self.pitch_vel[i_start:i_end])
                line_r.set_data(t_plot, self.roll_vel[i_start:i_end])
                return line_p, line_r
            else:
                raise Exception("Data type not supported")

        animator = ani.FuncAnimation(fig,
                                     animate_acc_plot,
                                     init_func=init,
                                     frames=n_plot_frames,
                                     interval=frame_interval_ms,
                                     blit=True,
                                     repeat=False)

        plt.close()

        return animator




