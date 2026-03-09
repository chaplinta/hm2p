from utils import misc
import numpy as np
import matplotlib.pyplot as plt
from old import AccelData
import os

def analyse(exp, proc_data_path, imaging_data=None, clearout=False, plot=False):



    plot_summary = True
    plot_events = False
    hp_filter = 550

    print("Loading accelerometer data")
    acc_data = AccelData.load_acc_data(exp,
                                       axis_swap=[2, 1, 0],
                                       acc_filt_f=hp_filter)

    if plot_summary:

        print("Plotting acceleration summaries")
        misc.setup_dir(proc_data_path, clearout=clearout)
        fig, ax = plt.subplots()

        # ax.plot(acc_data.time, acc_data.x, label='x', linewidth=1, alpha=0.3)
        # ax.plot(acc_data.time, acc_data.y, label='y', linewidth=1, alpha=0.3)
        # ax.plot(acc_data.time, acc_data.z, label='z', linewidth=1, alpha=0.3)
        # ax.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Acceleration (g)')
        # plt.title('Linear acceleration')
        # plt.savefig(os.path.join(proc_data_path, 'linear-trace.png'), dpi=300, facecolor='white')
        #
        # plt.clf()
        # ax.plot(acc_data.time, acc_data.grav_x, label='x', linewidth=1, alpha=0.3)
        # ax.plot(acc_data.time, acc_data.grav_y, label='y', linewidth=1, alpha=0.3)
        # ax.plot(acc_data.time, acc_data.grav_z, label='z', linewidth=1, alpha=0.3)
        # ax.legend()
        # plt.xlabel('Time (s)')
        # plt.ylabel('Gravity (g)')
        # plt.title('Gravity')
        # plt.savefig(os.path.join(proc_data_path, 'grav-trace.png'), dpi=300, facecolor='white')
        #
        # # Plot the magnitude spectrums
        # plt.clf()
        # plt.magnitude_spectrum(acc_data.x, Fs=acc_data.fs, linewidth=1)
        # plt.xlim(left=0, right=1000)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude')
        # plt.title('Linear X spectrum')
        # plt.savefig(os.path.join(proc_data_path, 'linear-spectrum-x.png'), dpi=300, facecolor='white')
        #
        # plt.clf()
        # plt.magnitude_spectrum(acc_data.x, Fs=acc_data.fs, linewidth=1)
        # plt.xlim(left=0, right=1000)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude')
        # plt.title('Linear Y spectrum')
        # plt.savefig(os.path.join(proc_data_path, 'linear-spectrum-y.png'), dpi=300, facecolor='white')
        #
        # plt.clf()
        # plt.magnitude_spectrum(acc_data.x, Fs=acc_data.fs, linewidth=1)
        # plt.xlim(left=0, right=1000)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude')
        # plt.title('Linear Z spectrum')
        # plt.savefig(os.path.join(proc_data_path, 'linear-spectrum-z.png'), dpi=300, facecolor='white')

        # Plot spectrogram.
        plt.clf()
        spectrum, freqs, t, im = plt.specgram(acc_data.x,
                                              Fs=acc_data.fs,
                                              NFFT=int(acc_data.fs*0.005),
                                              noverlap=int(acc_data.fs*0.0025))
        print(freqs)
        plt.ylim(top=100)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Linear X spectrogram')
        plt.savefig(os.path.join(proc_data_path, 'linear-spectrogram-x.png'), dpi=300, facecolor='white')

        # plt.clf()
        # plt.specgram(acc_data.y, Fs=acc_data.fs)
        # #plt.ylim(top=100)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Frequency (Hz)')
        # plt.title('Linear Y spectrogram')
        # plt.savefig(os.path.join(proc_data_path, 'linear-spectrogram-y.png'), dpi=300, facecolor='white')
        #
        # plt.clf()
        # plt.specgram(acc_data.z, Fs=acc_data.fs)
        # #plt.ylim(top=100)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Frequency (Hz)')
        # plt.title('Linear Z spectrogram')
        # plt.savefig(os.path.join(proc_data_path, 'linear-spectrogram-z.png'), dpi=300, facecolor='white')

        plt.close('all')

    if plot_events:
        print("Plotting acceleration events")
        events_dir = os.path.join(proc_data_path, "events")
        misc.setup_dir(events_dir, clearout=True)
        event_thresh = 0.5

        crossings = acc_data.get_crossings(event_thresh, 0.2)

        time_pre = 0.1
        time_post = 0.5
        samples_pre = int(np.round(exp.DAQSettings.sf * time_pre))
        samples_post = int(np.round(exp.DAQSettings.sf * time_post))

        for i_event, i_sample in enumerate(crossings):
            index_start = i_sample - samples_pre
            index_end = i_sample + samples_post
            t = acc_data.time[index_start:index_end] - acc_data.time[index_start + samples_pre]
            x_cross = acc_data.x[index_start:index_end]
            y_cross = acc_data.y[index_start:index_end]
            z_cross = acc_data.z[index_start:index_end]

            fig, ax = plt.subplots()
            ax.plot(t, x_cross, label='x', linewidth=2, alpha=1)
            ax.plot(t, y_cross, label='y', linewidth=2, alpha=1)
            ax.plot(t, z_cross, label='z', linewidth=2, alpha=1)
            ax.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (g)')
            plt.title('Acceleration event {0}'.format(i_event))
            plt.savefig(os.path.join(events_dir, 'linear-event-{:02d}.png'.format(i_event)), dpi=300, facecolor='white')
            plt.close()

    return acc_data


print("Analysis complete.")
