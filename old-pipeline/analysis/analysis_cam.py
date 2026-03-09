import utils.video
from utils import misc
from classes import Experiment

def analyse(exp: Experiment, proc_data_path, clearout=False, plot=False):

    for cam_name in exp.tracking_video.cameras:

        cam = exp.tracking_video.cameras[cam_name]

        # Check the number of trigger pulses matches the number of output pulses.
        n_mov_frames = utils.video.count_frames(cam.file)
        n_cam_pulses = len(cam.out_event_times)
        n_trg_pulses = len(exp.cam_trigger_times)
        if n_cam_pulses < n_trg_pulses:
            raise Exception("Camera {0} has too many output pulses ({1})".format(cam_name, n_trg_pulses - n_cam_pulses))
        elif n_cam_pulses > n_trg_pulses:
            raise Exception("Camera {0} has too few output pulses ({1})".format(cam_name, n_cam_pulses - n_trg_pulses))

        # Check the number of movie file frames is correct.
        if n_mov_frames > n_trg_pulses:
            raise Exception("Camera movie file {0} has too many frames ({1})".format(cam_name, n_mov_frames - n_trg_pulses))
        elif n_mov_frames < n_trg_pulses:
            # This happens when a camera stops early, might be salveagable.
            print("Camera movie file {0} has too few frames ({1})".format(cam_name, n_trg_pulses - n_mov_frames))

        # Check that each output pulse is close enough to the trigger pulse.
        for i_pulse in range(n_cam_pulses):

            master_time = exp.cam_trigger_times[i_pulse]
            camera_time = cam.out_event_times[i_pulse]

            # Sanity check that no camera output pulse comes before the trigger.
            timing_err = 0.00001
            if master_time - camera_time > timing_err:
                raise Exception("Camera {0} produced an output pulse before the trigger somehow?".format(cam_name))

            # Make sure the frame takes place before the next trigger(frame)
            if camera_time - master_time > (1 / exp.tracking_video.fps):
                raise Exception("Camera {0} frame event was too slow.".format(cam_name))

