# import matplotlib; matplotlib.use("TkAgg")
import utils.video
from classes import Experiment
from old import AccelData
import matplotlib.pyplot as plt
import os
from utils import misc
import ffmpeg
import time

axis_swap = None #[2, 1, 0]
axis_sign = None #[1, 1, -1]
acc_ymax = None

raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_06_15/20210615_13_50_47_CAA-1112353"
proc_data_path = "J:/Data/Accel/20210615_13_50_47_CAA-1112353"
axis_swap = [2, 1, 0]
acc_ymax = 0.5

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_07_12/20210712_11_24_22_acc-test"
# proc_data_path = "C:/Data/Accel/2021_07_12/20210712_11_24_22_acc-test"
# axis_swap = [0, 1, 2]

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_07_12/20210712_11_26_00_acc-test"
# proc_data_path = "C:/Data/Accel/2021_07_12/20210712_11_26_00_acc-test"
# axis_swap = [2, 1, 0]

events_dir = os.path.join(proc_data_path, "gifs")
misc.setup_dir(events_dir, clearout=True)


make_downsampled_mov = True

make_acc_plot_mov = True
make_raw_plot_mov = True
make_pr_plot_mov = True
make_prvel_plot_mov = True

stitch_movies = False


cam_name = "overhead"

clearout = False
acc_event_thresh = 0.5
acc_event_time_exlcude = 2
acc_event_time_pre = 3
acc_event_time_post = 3

plot_acc_mov_path = os.path.join(proc_data_path, 'plot-acc.mp4')
plot_raw_mov_path = os.path.join(proc_data_path, 'plot-raw.mp4')
plot_pr_mov_path = os.path.join(proc_data_path, 'plot-pitrol.mp4')
plot_prvel_mov_path = os.path.join(proc_data_path, 'plot-pitrolvel.mp4')
downsampled_mov_file = os.path.join(proc_data_path, cam_name + "-small.mp4")
stitched_acc_mov_file = os.path.join(proc_data_path, cam_name + "-stiched-acc.mp4")
stitched_raw_mov_file = os.path.join(proc_data_path, cam_name + "-stiched-raw.mp4")
stitched_pr_mov_file = os.path.join(proc_data_path, cam_name + "-stiched-pitrol.mp4")
stitched_prvel_mov_file = os.path.join(proc_data_path, cam_name + "-stiched-pitrolvel.mp4")

if not os.path.exists(plot_acc_mov_path):
    make_acc_plot_mov = True
if not os.path.exists(plot_raw_mov_path):
    make_raw_plot_mov = True
if not os.path.exists(plot_pr_mov_path):
    make_pr_plot_mov = True
if not os.path.exists(plot_prvel_mov_path):
    make_prvel_plot_mov = True
if not os.path.exists(downsampled_mov_file):
    make_downsampled_mov = True

exp = Experiment.Experiment(raw_data_path)

print("Checking camera timings and frames")
exp.check_cameras(ignore_cams=["side_left"])


acc_filt_f = None
print("Loading accelerometer data")
acc_data = AccelData.load_acc_data(exp,
                                   axis_swap=axis_swap,
                                   axis_sign=axis_sign,
                                   acc_filt_f=acc_filt_f,
                                   calc_pitch_roll=True)

# plt.psd(acc_data.x_nograv, Fs=acc_data.fs, NFFT=2**12)
zoom_f = 80
plt.magnitude_spectrum(acc_data.x_raw, Fs=acc_data.fs)
plt.xlim(left=0, right=250)
plt.savefig(os.path.join(proc_data_path, 'spec-mag-01-x-raw.png'), dpi=300, facecolor='white')
plt.close()

plt.magnitude_spectrum(acc_data.x_nograv, Fs=acc_data.fs)
plt.xlim(left=0, right=250)
plt.savefig(os.path.join(proc_data_path, 'spec-mag-02-x-nograv.png'), dpi=300, facecolor='white')
plt.close()

plt.magnitude_spectrum(acc_data.x_nograv, Fs=acc_data.fs)
plt.xlim(left=0, right=zoom_f)
plt.savefig(os.path.join(proc_data_path, 'spec-mag-03-x-nograv-zoom.png'), dpi=300, facecolor='white')
plt.close()

plt.magnitude_spectrum(acc_data.x, Fs=acc_data.fs)
plt.xlim(left=0, right=zoom_f)
plt.savefig(os.path.join(proc_data_path, 'spec-mag-03-x.png'), dpi=300, facecolor='white')
plt.close()

#acc_events = acc_data.get_events(acc_event_thresh, acc_event_time_exlcude, acc_event_time_pre, acc_event_time_post)



# https://trac.ffmpeg.org/wiki/Encode/H.264
# Controls compression/speed trade off.
h264_preset = 'ultrafast'
# Tuning for type of movie, animation is should be good for this image?
h264_tune = 'animation'
# H264 quality, apparently 17 is indistinguishable from lossless. 23 is default. +6 is half the bitrate/size.
h264_crf = '29'

fps_reduce = 3
ani_fps = int(round(exp.tracking_video.fps / fps_reduce))
ani_dpi = 100
cam_index_start = exp.cam_trigger_indexes[0]
cam_index_end = exp.cam_trigger_indexes[-1]
n_plot_frames = int(round(exp.tracking_video.cameras[cam_name].out_event_times.size / fps_reduce))
plot_window = 2
if make_acc_plot_mov:
    print("Creating acc animated plot")
    animator = acc_data.make_animated_plot(fps=ani_fps,
                                           start_time_index=cam_index_start,
                                           end_time_index=cam_index_end,
                                           n_plot_frames=n_plot_frames,
                                           plot_window=plot_window,
                                           data_type=AccelData.AccDataType.LINEAR_ACC,
                                           yax_max=acc_ymax)
    print("Saving acc animated plot")
    t = time.time()
    animator.save(plot_acc_mov_path, dpi=ani_dpi, fps=ani_fps)
    print("Time elapsed={0}".format(time.time() - t))
    # writer = FasterFFMpegWriter.FasterFFMpegWriter(fps=ani_fps, codec='h264')
    # #writer = FFMpegFileWriter(fps=ani_fps, codec='h264')
    # print(writer.isAvailable())
    # t = time.time()
    # animator.save(plot_acc_mov_path, writer=writer, dpi=ani_dpi)
    # print("Time elapsed={0}".format(time.time() - t))


    print("Animated acc plot complete.")

if make_raw_plot_mov:
    print("Creating raw animated plot")
    animator = acc_data.make_animated_plot(fps=ani_fps,
                                           start_time_index=cam_index_start,
                                           end_time_index=cam_index_end,
                                           n_plot_frames=n_plot_frames,
                                           plot_window=plot_window,
                                           data_type=AccelData.AccDataType.LINEAR_ACC_RAW)
    print("Saving raw animated plot")
    t = time.time()
    animator.save(plot_raw_mov_path, dpi=ani_dpi, fps=ani_fps)
    print("Time elapsed={0}".format(time.time() - t))
    print("Animated acc plot complete.")

if make_pr_plot_mov:
    print("Creating pitch/roll  animated plot")
    animator = acc_data.make_animated_plot(fps=ani_fps,
                                           start_time_index=cam_index_start,
                                           end_time_index=cam_index_end,
                                           n_plot_frames=n_plot_frames,
                                           plot_window=plot_window,
                                           data_type=AccelData.AccDataType.PITCH_AND_ROLL)
    print("Saving pitch/roll animated plot")
    t = time.time()
    animator.save(plot_pr_mov_path, dpi=ani_dpi, fps=ani_fps)
    print("Time elapsed={0}".format(time.time() - t))
    print("Animated pitch/roll plot complete.")

if make_prvel_plot_mov:
    print("Creating pitch/roll velocity animated plot")
    animator = acc_data.make_animated_plot(fps=ani_fps,
                                           start_time_index=cam_index_start,
                                           end_time_index=cam_index_end,
                                           n_plot_frames=n_plot_frames,
                                           plot_window=plot_window,
                                           data_type=AccelData.AccDataType.PITCH_AND_ROLL_VELOCITY)
    print("Saving pitch/roll velocity animated plot")
    t = time.time()
    animator.save(plot_prvel_mov_path, dpi=ani_dpi, fps=ani_fps)
    print("Time elapsed={0}".format(time.time() - t))
    print("Animated pitch/roll plot complete.")

n_plot_mov_frames = utils.video.count_frames(plot_acc_mov_path)
(plot_pix_x, plot_pix_y) = utils.video.get_mov_res(plot_acc_mov_path)
print(n_plot_mov_frames, plot_pix_x, plot_pix_y)



if make_downsampled_mov:

    print("Resampling camera {c}".format(c=cam_name))

    (
        ffmpeg
            .input(exp.tracking_video.cameras[cam_name].file)
            .filter('fps', fps=ani_fps)
            # .filter('codec', codec="h264")
            .filter('scale', plot_pix_x, plot_pix_y)
            .output(downsampled_mov_file, vcodec='h264', format='mp4', preset=h264_preset, tune=h264_tune,
                    crf=str(h264_crf))
            .overwrite_output()
            .run()
    )


n_mov_frames = utils.video.count_frames(downsampled_mov_file)
(mov_pix_x, mov_pix_y) = utils.video.get_mov_res(downsampled_mov_file)
print(n_mov_frames, mov_pix_x, mov_pix_y)

if stitch_movies or make_downsampled_mov or make_acc_plot_mov:
    print("Stitching overhead and acc plot")
    # look up hstack for ffmpeg
    in0 = ffmpeg.input(plot_acc_mov_path)
    in1 = ffmpeg.input(downsampled_mov_file)
    out = ffmpeg.filter([in0, in1], 'hstack').output(stitched_acc_mov_file).overwrite_output()
    out.run()

if stitch_movies or make_downsampled_mov or make_raw_plot_mov:
    print("Stitching overhead and raw plot")
    in0 = ffmpeg.input(plot_raw_mov_path)
    in1 = ffmpeg.input(downsampled_mov_file)
    out = ffmpeg.filter([in0, in1], 'hstack').output(stitched_raw_mov_file).overwrite_output()
    out.run()

if stitch_movies or make_downsampled_mov or make_pr_plot_mov:
    print("Stitching overhead and pitch/roll plot")
    in0 = ffmpeg.input(plot_pr_mov_path)
    in1 = ffmpeg.input(downsampled_mov_file)
    out = ffmpeg.filter([in0, in1], 'hstack').output(stitched_pr_mov_file).overwrite_output()
    out.run()

if stitch_movies or make_downsampled_mov or make_prvel_plot_mov:
    print("Stitching overhead and pitch/roll velocity plot")
    in0 = ffmpeg.input(plot_prvel_mov_path)
    in1 = ffmpeg.input(downsampled_mov_file)
    out = ffmpeg.filter([in0, in1], 'hstack').output(stitched_prvel_mov_file).overwrite_output()
    out.run()




# # print("Creating animated plot")
# i_event = 10
# acc_evt = acc_events[i_event, :, :]
# t = acc_evt[0, :]
# x = acc_evt[1, :]
# y = acc_evt[2, :]
# z = acc_evt[3, :]

#
# fig, ax = plt.subplots()
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration (g)')
# plt.title('Acceleration event {0}'.format(i_event))
# plt.xlim(left=0, right=t.max())
# plt.ylim(bottom=-2, top=2)
# line, = ax.plot(t, x, color='blue')
#
# def animate_acc_plot(i, t, x, y, z, line):
#     line.set_data(t[0:i], x[0:i])
#     return line,
#     #plt.plot(t[0:i], x[0:i], label='x', linewidth=2, alpha=1, color="blue")
#     #plt.plot(t[0:i], y[0:i], label='y', linewidth=2, alpha=1)
#     #plt.plot(t[0:i], z[0:i], label='z', linewidth=2, alpha=1)
#     #ax.legend()
#
# animator = ani.FuncAnimation(fig, animate_acc_plot, frames=t.size, fargs=[t, x, y, z, line], interval=100)
# #plt.savefig(os.path.join(events_dir, 'linear-event-{:02d}.png'.format(i_event)), dpi=300, facecolor='white')
# animator.save(os.path.join(events_dir, 'linear-event-{:02d}.gif'.format(i_event)), dpi=300)
# plt.show()
# plt.close()



# print("Creating event movies")
# event_start_time = t[0]
# event_end_time = t[-1]
#
# cam_start_frame = (np.abs(exp.cam_trigger_times - event_start_time)).argmin()
# cam_end_frame = (np.abs(exp.cam_trigger_times - event_end_time)).argmin()
#
# with imageio.get_reader(exp.tracking_video.cameras["side_left"].file,
#                         pixelformat='gray') as mov_reader:
#     cam_res = mov_reader.get_meta_data()["size"]
#     with imageio.get_writer(os.path.join(events_dir, 'linear-event-{:02d}.mp4'.format(i_event)),
#                             mode="I",
#                             fps=round(exp.tracking_video.fps),
#                             pixelformat='gray',
#                             codec='h264',
#                             format='mp4',
#                             output_params=['-s', '{0}x{1}'.format(cam_res[0], # height and width need to be flipped
#                                                                   cam_res[1]),
#                                            '-preset', exp.tracking_video.h264preset,
#                                            '-tune', exp.tracking_video.h264tune,
#                                            '-crf', str(exp.tracking_video.h264crf)]) as mov_writer:
#
#         for i_frame in range(cam_start_frame, cam_end_frame):
#
#             mov_writer.append_data(mov_reader.get_data(i_frame)[:, :, 0])

# c = acc_data.get_crossings(0.5, 1)
#
# x1 = acc_data.grav_x[i]
# y1 = acc_data.grav_y[i]
# z1 = acc_data.grav_z[i]
#
# grav = np.array([x1, y1, z1])
# grav_unit_vector = grav / np.linalg.norm(grav)
#
# pitch_plane = np.array([1, 0, 0])
# roll_plane = np.array([0, 1, 0])
#
# pitch_dp = np.dot(grav_unit_vector, pitch_plane)
# roll_dp = np.dot(grav_unit_vector, roll_plane)
#
# pitch_angle = np.rad2deg(np.arccos(pitch_dp))
# roll_angle = np.rad2deg(np.arccos(roll_dp))
#
# print(pitch_angle, roll_angle)

# print(acc_data.x.size)
# plt.plot(acc_data.time, acc_data.pitch)
# plt.plot(acc_data.time, acc_data.roll)

# plt.plot(acc_data.time, acc_data.x)
# plt.plot(acc_data.time, acc_data.y)
# plt.plot(acc_data.time, acc_data.z)

# plt.plot(acc_data.time[50000:100000], acc_data.x[50000:100000])
# plt.plot(acc_data.time[50000:100000], acc_data.y[50000:100000])
# plt.plot(acc_data.time[50000:100000], acc_data.z[50000:100000])
# plt.show()

# print("Creating event movies")
# pitch_event_thresh = 88
# pitch_event_time_exlcude = 2
# pitch_event_time_pre = 3
# pitch_event_time_post = 3
# i_event = 0
# pitch_events = acc_data.get_events(pitch_event_thresh,
#                                    pitch_event_time_exlcude,
#                                    pitch_event_time_pre,
#                                    pitch_event_time_post,
#                                    event_type=AccelData.EventType.PITCH)
# pitch_events = pitch_events[i_event, :, :]
# t = pitch_events[0, :]
# x = pitch_events[1, :]
# y = pitch_events[2, :]
# z = pitch_events[3, :]
#
# event_start_time = t[0]
# event_end_time = t[-1]
#
# cam_start_frame = (np.abs(exp.cam_trigger_times - event_start_time)).argmin()
# cam_end_frame = (np.abs(exp.cam_trigger_times - event_end_time)).argmin()
#
# with imageio.get_reader(exp.tracking_video.cameras["side_left"].file,
#                         pixelformat='gray') as mov_reader:
#     cam_res = mov_reader.get_meta_data()["size"]
#     with imageio.get_writer(os.path.join(events_dir, 'pitch-event-{:02d}.mp4'.format(i_event)),
#                             mode="I",
#                             fps=round(exp.tracking_video.fps),
#                             pixelformat='gray',
#                             codec='h264',
#                             format='mp4',
#                             output_params=['-s', '{0}x{1}'.format(cam_res[0], # height and width need to be flipped
#                                                                   cam_res[1]),
#                                            '-preset', exp.tracking_video.h264preset,
#                                            '-tune', exp.tracking_video.h264tune,
#                                            '-crf', str(exp.tracking_video.h264crf)]) as mov_writer:
#
#         for i_frame in range(cam_start_frame, cam_end_frame):
#
#             mov_writer.append_data(mov_reader.get_data(i_frame)[:, :, 0])

print("Done")



