import os
import configparser

import utils.img
import utils.video
from utils import misc, img as imutils
from classes import TrackingCamera
from dataclasses import dataclass
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
import imageio
import datetime
import scipy



@dataclass
class DAQSettings:
    devicename: str
    file_di: str
    file_ai: str
    groupname: str
    sf: str

    sciscanchanname: str
    sciscanlinename: str
    sciscantriglinename: str

    lightschanname: str

    cameratriggercounter: str
    cameratriggerline: str
    cameratriggerchanname: str


@dataclass
class TrackingVideo:
    fps: float
    duration: float
    n_cameras: int
    threaded_writer: bool
    daq_camera_trigger: bool
    h264preset: str
    h264tune: str
    h264crf: int
    exposure_percent: float
    exposure_time: float

    cam_start_time: float
    cam_end_time: float
    cam_duration: float

    cameras: dict = None

@dataclass
class SciscanSettings:
    image_file: str = None
    image_n_frames: int = None
    ini_file: str = None
    sci_line_indexes: np.array = None
    sci_line_times: np.array = None
    sci_frame_indexes: np.array = None
    sci_frame_times: np.array = None
    ms_p_line: float = None
    frames_p_sec: float = None
    x_pixels: int = None
    y_pixels: int = None
    x_pixel_sz: float = None
    bad_frames: np.array = None
    bad_frames_indexes: np.array = None

@dataclass
class Lighting:
    time_on: float = None
    time_off: float = None
    line_clock_pulses_on: int = None
    line_clock_pulses_off: int = None

    on_pulse_indexes: np.array = None
    off_pulse_indexes: np.array = None
    on_pulse_times: np.array = None
    off_pulse_times: np.array = None

    on_indexes: np.array = None

class Experiment:

    def __init__(self, exp_path, remake_tif=False):
        config_file = misc.get_filetype(exp_path, "*.meta.txt")

        config_meta = configparser.ConfigParser()
        config_meta.read(config_file)

        self.data_raw_path = exp_path
        self.config_meta = config_meta

        self.date = config_meta["Experiment"]["date"]
        self.name = config_meta["Experiment"]["name"]

        self.type = config_meta["Experiment"]["type"]
        self.complete = "True" == config_meta["Experiment"]["complete"]
        self.file_index = config_meta["Experiment"]["fileindex"]
        self.directory = config_meta["Experiment"]["directory"]
        self.script_version = config_meta["Experiment"]["scriptversion"]
        self.start_time = config_meta["Experiment"]["starttime"]
        self.end_time = config_meta["Experiment"]["endtime"]
        self.duration = config_meta["Experiment"]["duration"]
        self.quit = "True" == config_meta["Experiment"]["quit"]
        self.sciscan_stopped = "True" == config_meta["Experiment"]["sciscanstopped"]

        self.useaccelerometer = "True" == config_meta["DAQ"]["useaccelerometer"]
        self.usenosepoke = "True" == config_meta["DAQ"]["usenosepoke"]
        self.uselights = "True" == config_meta["DAQ"]["uselights"]

        lightschanname = None
        if self.uselights:
            lightschanname = config_meta["DAQ"]["lightschanname"]



        self.DAQSettings = DAQSettings(devicename=config_meta["DAQ"]["devicename"],
                                       file_di=os.path.join(exp_path, misc.path_leaf(config_meta["DAQ"]["file-di"])),
                                       file_ai=os.path.join(exp_path, misc.path_leaf(config_meta["DAQ"]["file-ai"])),
                                       groupname=config_meta["DAQ"]["groupname"],
                                       sf=int(config_meta["DAQ"]["sf"]),
                                       sciscanchanname=config_meta["DAQ"]["sciscanchanname"],
                                       sciscanlinename=config_meta["DAQ"]["sciscanlinename"],
                                       sciscantriglinename=config_meta["DAQ"]["sciscantriglinename"],
                                       lightschanname=lightschanname,
                                       cameratriggercounter=config_meta["DAQ"]["cameratriggercounter"],
                                       cameratriggerline=config_meta["DAQ"]["cameratriggerline"],
                                       cameratriggerchanname=config_meta["DAQ"]["cameratriggerchanname"])

        self.check_sci_running = "True" == config_meta["SciScan"]["checkrunning"]
        self.usesciscandirforalldata = "True" == config_meta["SciScan"]["usesciscandirforalldata"]



        # Camera stuff
        self.cam_start_time = config_meta["Experiment"]["camstarttime"]
        self.cam_end_time = config_meta["Experiment"]["camendtime"]
        self.cam_duration = config_meta["Experiment"]["camduration"]

        video_duration = config_meta["Video"]["duration"]
        if video_duration == "None":
            video_duration = None
        else:
            video_duration = float(video_duration)

        with TdmsFile.read(self.DAQSettings.file_di) as di_file:
            cam_trg_chan = misc.get_tdms_di_channel(di_file,
                                                    self.DAQSettings.groupname,
                                                    self.DAQSettings.cameratriggerchanname)
            cam_trg_data = cam_trg_chan.data
            self.daq_time = cam_trg_chan.time_track()
            self.cam_trigger_indexes = misc.get_crossings(np.absolute(cam_trg_data), 0.9)
            self.cam_trigger_times = self.daq_time[self.cam_trigger_indexes]

            exposure_percent = None
            exposure_time = None
            if "exposurepercent" in config_meta["Video"]:
                exposure_percent = float(config_meta["Video"]["exposurepercent"])
            else:
                exposure_time = int(config_meta["Video"]["exposuretime"])

            self.tracking_video = TrackingVideo(fps=float(config_meta["Video"]["fps"]),
                                                duration=video_duration,
                                                n_cameras=int(config_meta["Video"]["numberofcameras"]),
                                                threaded_writer="True" == config_meta["Video"]["threadedwriter"],
                                                daq_camera_trigger="True" == config_meta["Video"]["daqcameratrigger"],
                                                h264preset=config_meta["Video"]["h264preset"],
                                                h264tune=config_meta["Video"]["h264tune"],
                                                h264crf=int(config_meta["Video"]["h264crf"]),
                                                exposure_percent=exposure_percent,
                                                exposure_time=exposure_time,
                                                cam_start_time=config_meta["Experiment"]["camstarttime"],
                                                cam_end_time=config_meta["Experiment"]["camendtime"],
                                                cam_duration=config_meta["Experiment"]["camduration"])


            imgfile = None
            inifile = None
            sci_indexes = None
            sci_times = None
            ms_p_line = None
            frames_p_sec = None
            x_pixels = None
            y_pixels = None
            x_pixel_sz = None
            if self.check_sci_running:

                inifile = os.path.join(exp_path, misc.path_leaf(config_meta["SciScan"]["inifile"]))
                config_ini = configparser.ConfigParser()
                config_ini.read(inifile)

                imgfile = os.path.join(exp_path, misc.path_leaf(config_meta["SciScan"]["imgfile"]))
                base, ext = os.path.splitext(imgfile)
                if ext == ".raw":

                    tif_path = base + ".tif"
                    if remake_tif or not os.path.exists(tif_path):
                        print("Converting raw to tif ...")
                        tif_paths = misc.sci_raw_2_tif(imgfile, config_ini)
                        tif_path = tif_paths[0]
                        print("Converting raw to tif done.")
                    imgfile = tif_path

                with TdmsFile.read(self.DAQSettings.file_di) as di_file:
                    sci_chan = misc.get_tdms_di_channel(di_file,
                                                        self.DAQSettings.groupname,
                                                        self.DAQSettings.sciscanchanname)

                sci_chan_data = sci_chan.data
                self.daq_time = sci_chan.time_track()



                # Assume this is the line clock, and convert to frame clock.
                y_pix = int(float(config_ini['_']['y.pixels']))
                sci_line_indexes = misc.get_crossings(np.absolute(sci_chan_data), 0.5)
                sci_line_times = self.daq_time[sci_line_indexes]
                print("Line pulses={0} ypix={1} frames={2} remainder pulses={3}".format(sci_line_indexes.size,
                                                                                 y_pix,
                                                                                 int(np.floor(sci_line_indexes.size / y_pix)),
                                                                                 sci_line_indexes.size % y_pix))

                #sci_indexes = np.hstack((sci_indexes[0], sci_indexes[y_pix-1::y_pix]))
                # Take the first frame time as when it finishes the last line, minus one because the array is zero
                # indexed. Then get every y_pix value.
                # Therefore, frame times are when the frame finishes, not starts.
                start_index = y_pix - 1
                sci_frame_indexes = sci_line_indexes[start_index::y_pix]
                sci_frame_times = self.daq_time[sci_frame_indexes]


                ms_p_line = float(config_ini['_']['ms.p.line'])
                frames_p_sec = float(config_ini['_']['frames.p.sec'])
                x_pixels = int(float(config_ini['_']['x.pixels']))
                y_pixels = y_pix
                x_pixel_sz = float(config_ini['_']['x.pixel.sz'])

            else:
                raise Exception("Sciscan was not set to run?!?")

            self.SciscanSettings = SciscanSettings(image_file=imgfile,
                                                   image_n_frames=imutils.count_tif_frames(imgfile),
                                                   ini_file=inifile,
                                                   sci_line_indexes=sci_line_indexes,
                                                   sci_line_times=sci_line_times,
                                                   sci_frame_indexes=sci_frame_indexes,
                                                   sci_frame_times=sci_frame_times,
                                                   ms_p_line=ms_p_line,
                                                   frames_p_sec=frames_p_sec,
                                                   x_pixels=x_pixels,
                                                   y_pixels=y_pixels,
                                                   x_pixel_sz=x_pixel_sz)


            self.tracking_video.cameras = {}
            for sec in config_meta:
                if sec.startswith("Camera"):
                    tc = TrackingCamera.TrackingCamera(name=config_meta[sec]["name"],
                                                       device=config_meta[sec]["id"],
                                                       daqchanname=config_meta[sec]["daqchanname"],
                                                       daqphyschan=config_meta[sec]["daqphyschan"],
                                                       is_master=config_meta[sec]["ismaster"] == "True",
                                                       resolution=config_meta[sec]["resolution"],
                                                       file=os.path.join(exp_path,
                                                                         misc.path_leaf(config_meta[sec]["file"])))

                    cam_out_chan = misc.get_tdms_di_channel(di_file,
                                                            self.DAQSettings.groupname,
                                                            tc.daqchanname)
                    cam_out_data = cam_out_chan.data
                    out_indexes = misc.get_crossings(np.absolute(cam_out_data), 0.9)
                    out_times = self.daq_time[out_indexes]
                    tc.add_daq_data(self.cam_trigger_indexes, self.cam_trigger_times, out_indexes, out_times)


                    self.tracking_video.cameras[tc.name] = tc


        # Lights stuff
        light_time_on = None
        light_time_off = None
        light_on_indexes = None
        light_on_times = None
        lineclockpulseson = None
        lineclockpulsesoff = None
        if self.uselights:
            light_time_on = float(config_meta["Lights"]["timeon"])
            light_time_off = float(config_meta["Lights"]["timeoff"])

            lineclockpulseson = int(float(config_meta["Lights"]["lineclockpulseson"]))
            lineclockpulsesoff = int(float(config_meta["Lights"]["lineclockpulsesoff"]))

            with TdmsFile.read(self.DAQSettings.file_di) as di_file:
                light_chan = misc.get_tdms_di_channel(di_file,
                                                      self.DAQSettings.groupname,
                                                      self.DAQSettings.lightschanname)
            light_chan_data = light_chan.data
            self.daq_time = light_chan.time_track()
            light_on_pulse_indexes = misc.get_crossings(np.absolute(light_chan_data), 0.9)
            light_off_pulse_indexes = misc.get_crossings(1 - np.absolute(light_chan_data), 0.9)
            light_on_pulse_times = self.daq_time[light_on_pulse_indexes]
            light_off_pulse_times = self.daq_time[light_off_pulse_indexes]
            light_on_indexes = np.zeros(light_chan_data.shape)
            for i, on_index in enumerate(light_on_pulse_indexes):
                if i >= light_off_pulse_indexes.size:
                    off_index = light_on_indexes.size - 1
                else:
                    off_index = light_off_pulse_indexes[i] - 1
                if off_index < on_index:
                    raise Exception("wtf")
                light_on_indexes[on_index:off_index] = 1


        self.Lighting = Lighting(time_on=light_time_on,
                                 time_off=light_time_off,
                                 line_clock_pulses_on=lineclockpulseson,
                                 line_clock_pulses_off=lineclockpulsesoff,
                                 on_pulse_indexes=light_on_pulse_indexes,
                                 off_pulse_indexes=light_off_pulse_indexes,
                                 on_pulse_times=light_on_pulse_times,
                                 off_pulse_times=light_off_pulse_times,
                                 on_indexes=light_on_indexes)



    def check_cameras(self, ignore_cams = []):

        print("WARNING CHECK CAMERAS DISABLED DUE TO MISSING FILES")
        return

        for cam_name in self.tracking_video.cameras:

            if cam_name in ignore_cams:
                continue
            cam = self.tracking_video.cameras[cam_name]

            # Check the number of trigger pulses matches the number of output pulses.
            print("Checking camera file {0}".format(cam.file))
            n_frm_frames = utils.video.count_frames(cam.file)
            n_out_pulses = len(cam.out_event_times)
            n_in_pulses = len(cam.frm_event_times)

            out_in_diff = n_out_pulses - n_in_pulses
            frm_in_diff = n_frm_frames - n_in_pulses
            frm_out_diff = n_frm_frames - n_out_pulses

            init_checks_good = out_in_diff == 0 and frm_in_diff == 0 and frm_out_diff == 0

            if out_in_diff == 1 and frm_in_diff == 1 and frm_out_diff == 0:
                # There appears to be a missing input pulse but not a missing file frame or output pulse.
                # Find the pulse that is missing and add it.
                missing_frame = -1
                for i_frame in range(cam.frm_event_times.size - 1):
                    input_t = cam.frm_event_times[i_frame]
                    output_t = cam.out_event_times[i_frame]
                    inoutdiff = input_t - output_t
                    if inoutdiff > 0.001:
                        t = cam.out_event_times[i_frame]
                        i = cam.out_event_indexes[i_frame]
                        cam.frm_event_times = np.insert(cam.frm_event_times, i_frame, t)
                        cam.frm_event_indexes = np.insert(cam.frm_event_indexes, i_frame, i)

                        missing_frame = i_frame
                        break

                if missing_frame == -1:
                    # Must be the last one
                    cam.frm_event_times = np.append(cam.frm_event_times, cam.out_event_times[-1])
                    cam.frm_event_indexes = np.append(cam.frm_event_indexes, cam.out_event_indexes[-1])
                    missing_frame = cam.out_event_times.size - 1

                print(
                    "Camera {0} has 1 missing intput pulse vs input pulses and file frames, so inserting the missing pulse {1}".format(
                        cam_name, missing_frame))

                self.cam_trigger_times = cam.frm_event_times
                self.cam_trigger_indexes = cam.frm_event_indexes

                n_out_pulses = len(cam.out_event_times)
                n_in_pulses = len(cam.frm_event_times)

                out_in_diff = n_out_pulses - n_in_pulses
                frm_in_diff = n_frm_frames - n_in_pulses
                frm_out_diff = n_frm_frames - n_out_pulses

            if out_in_diff == -1 and frm_in_diff == -1 and frm_out_diff == 0:
                # There appears to be a missing output pulse and frame.
                # If it's the first or last frame that's ok, the input pulse can be ignored.
                first_input_t = cam.frm_event_times[0]
                first_output_t = cam.out_event_times[0]
                first_diff = first_input_t - first_output_t

                # See if the first difference in pulse times is too long.
                # If it is, it could be that the first input pulse was ignored.
                # But this is only true if the next input pulse comes before the first output pulse.
                if first_diff < -0.001 and cam.frm_event_times[1] < cam.out_event_times[0]:
                    # The first one
                    print(
                        "Camera {0} has 1 missing out pulse & frame vs input pulses, it missed the first input so delete it".format(
                            cam_name))

                    cam.frm_event_times = np.delete(cam.frm_event_times, 0)
                    cam.frm_event_indexes = np.delete(cam.frm_event_indexes, 0)
                else:
                    input_t = cam.frm_event_times[-1]
                    output_t = cam.out_event_times[-1]
                    inoutdiff = input_t - output_t
                    if inoutdiff > 0.001:
                        # It was the last.
                        print(
                            "Camera {0} has 1 missing out pulse & frame vs input pulses, it missed the last input so delete it".format(
                                cam_name))
                        cam.frm_event_times = np.delete(cam.frm_event_times, -1)
                        cam.frm_event_indexes = np.delete(cam.frm_event_indexes, -1)

                    else:
                        msg = "Camera {0} has 1 missing out pulse & frame vs input pulses, it wasn't the first or last".format(
                                cam_name)
                        raise Exception(msg)


                self.cam_trigger_times = cam.frm_event_times
                self.cam_trigger_indexes = cam.frm_event_indexes

                n_out_pulses = len(cam.out_event_times)
                n_in_pulses = len(cam.frm_event_times)

                out_in_diff = n_out_pulses - n_in_pulses
                frm_in_diff = n_frm_frames - n_in_pulses
                frm_out_diff = n_frm_frames - n_out_pulses


            if out_in_diff < 0:
                raise Exception(
                    "Camera {0} has too many output pulses ({1}) vs input pulses".format(cam_name, out_in_diff))

            if out_in_diff > 0:
                raise Exception(
                    "Camera {0} has too few output pulses ({1}) vs input pulses".format(cam_name, out_in_diff))

            # Check that each output pulse is close enough to the trigger pulse.
            for i_pulse in range(n_out_pulses):

                master_time = self.cam_trigger_times[i_pulse]
                camera_time = cam.out_event_times[i_pulse]

                # Sanity check that no camera output pulse comes before the trigger.
                timing_err = 0.00001
                if master_time - camera_time > timing_err:
                    next_master_time = self.cam_trigger_times[i_pulse + 1]
                    next_camera_time = cam.out_event_times[i_pulse + 1]
                    next_pulse_ok = next_master_time - next_camera_time > timing_err
                    if init_checks_good and i_pulse == 0 and next_pulse_ok:
                        msg = "Camera {0} produced an output pulse before the first pulse, even " \
                              "though the number of pulses was correct to begin with and the next pulse is ok." \
                              "".format(cam_name)
                        print(msg)
                        cam.out_event_times[i_pulse] = self.cam_trigger_times[i_pulse]
                        master_time = self.cam_trigger_times[i_pulse]
                        camera_time = cam.out_event_times[i_pulse]
                    else:
                        raise Exception(
                            "Camera {0} produced an output pulse before the trigger somehow?".format(cam_name))

                # Make sure the frame takes place before the next trigger(frame)
                if camera_time - master_time > (1 / self.tracking_video.fps):
                    raise Exception("Camera {0} frame event was too slow.".format(cam_name))




            # Check the number of movie file frames is correct compared to output pulses.
            if frm_in_diff < 0:
                raise Exception(
                   "Camera movie file {0} has too few frames ({1}) vs trigger pulses".format(cam_name, frm_in_diff))
            elif frm_in_diff < 0:
                raise Exception(
                    "Camera movie file {0} has too many frames ({1}) vs trigger pulses".format(cam_name, frm_in_diff))

            # Check the number of movie file frames is correct compared to output pulses.
            if frm_out_diff < 0:
                raise Exception(
                    "Camera movie file {0} has too few frames ({1}) vs output pulses".format(cam_name, frm_out_diff))
            elif frm_out_diff > 0:
                raise Exception(
                    "Camera movie file {0} has too many frames ({1}) vs output pulses".format(cam_name, frm_out_diff))

        # Check if cameras started around the same time as sci scan.
        sci_cam_diff = self.cam_trigger_times[0] - self.SciscanSettings.sci_line_times[0]
        if sci_cam_diff > 0.001:
            msg = "Cameras were too slow to start after sciscan {0}ms".format((sci_cam_diff))
            raise Exception(msg)
        # elif sci_cam_diff < 0.0:
        #     msg = "Cameras were started before sciscan somehow {0}ms".format((sci_cam_diff))
        #     raise Exception(msg)

    def check_sci_frames(self, max_frame_pulse_diff=1):
        n_tif_frames = utils.img.count_tif_frames(self.SciscanSettings.image_file)
        n_sci_pulses = self.SciscanSettings.sci_frame_indexes.size
        print("Sci tif frames={0} pulses={1}".format(n_tif_frames, n_sci_pulses))

        if n_tif_frames < n_sci_pulses:
            diff = n_sci_pulses - n_tif_frames

            msg = "There were an extra {0} sci frame pulses vs frames".format(diff)
            self.SciscanSettings.sci_frame_indexes = self.SciscanSettings.sci_frame_indexes[0:n_tif_frames]
            self.SciscanSettings.sci_frame_times = self.SciscanSettings.sci_frame_times[0:n_tif_frames]
            n_sci_pulses = self.SciscanSettings.sci_frame_indexes.size
            if diff > max_frame_pulse_diff:
                raise Exception(msg)
            else:
                print(msg + ", just drop the last ones")

        elif n_tif_frames > n_sci_pulses:
            msg = "There were too few sci pulses ({0}), frames={1} pulses={2}".format(n_tif_frames-n_sci_pulses,
                                                                                      n_tif_frames,
                                                                                      n_sci_pulses)
            raise Exception(msg)


        sci_intervals = self.SciscanSettings.sci_frame_times[1:-1] - self.SciscanSettings.sci_frame_times[0:-2]
        max_pulse_delay = 0.001
        frame_interval = (1.0 / self.SciscanSettings.frames_p_sec)
        bad_pulse_indexes = np.abs(sci_intervals - frame_interval) > max_pulse_delay
        bad_intervals = sci_intervals[bad_pulse_indexes]
        n_bad_pulses = np.sum(bad_pulse_indexes)
        if n_bad_pulses > 0:
            raise Exception(
                "There were {0} sci pulses more than {1}ms out of time.".format(n_tif_frames,max_pulse_delay*1000))



    def check_light_times(self, sync_with_sciscan=True):
        if sync_with_sciscan:
            n_light_pulses = int(self.Lighting.line_clock_pulses_on + self.Lighting.line_clock_pulses_off)
            st = self.SciscanSettings.sci_line_times
            start_index = int(self.Lighting.line_clock_pulses_off) - 1
            sci_light_times = self.SciscanSettings.sci_line_times[start_index::n_light_pulses]

            diffs = self.Lighting.on_pulse_times - sci_light_times
            max_diff = 0.001
            n_out_of_sync = np.sum(np.abs(diffs) > max_diff)
            if n_out_of_sync > 0:
                raise Exception(
                    "There were {0} light pulses intervals more than {1}ms out of time with sciscan.".format(n_out_of_sync, max_diff*1000))


            light_intervals = self.Lighting.on_pulse_times[1:-1] - self.Lighting.on_pulse_times[0:-2]
            sci_light_intervals = sci_light_times[1:-1] - sci_light_times[0:-2]

            max_interval_lines = 4
            max_interval = (max_interval_lines * self.SciscanSettings.ms_p_line) / 1000
            light_interval_time = (self.Lighting.time_on + self.Lighting.time_off) / 1000
            n_out_of_sync_light = np.sum(np.abs(light_intervals - light_interval_time) > max_interval)
            n_out_of_sync_light_sci = np.sum(np.abs(sci_light_intervals - light_interval_time) > max_interval)

            if n_out_of_sync_light > 0:
                print(light_intervals)
                print(light_interval_time)
                raise Exception(
                    "There were {0} light pulses more than {1}ms out of time ({2} lines)".format(n_out_of_sync_light,
                                                                                               max_interval * 1000,
                                                                                               max_interval_lines))

            if n_out_of_sync_light_sci > 0:
                raise Exception(
                    "There were {0} light pulse *INTERVALS* more than {1}ms out of time with sciscan ({2} lines)".format(n_out_of_sync_light_sci,
                                                                                                             max_interval * 1000,
                                                                                                             max_interval_lines))

        else:
            raise Exception("todo checks when there is no sciscan.")


    def write_s2p_sum_file(self, sum_text_file):
        # Write some important values out for quick reference

        config_ini = configparser.ConfigParser()
        config_ini.read(self.SciscanSettings.ini_file)

        sci_fps = float(config_ini['_']['frames.p.sec'])

        sci_npix_x = float(config_ini['_']['nomin.x.pixels'])
        sci_npix_y = float(config_ini['_']['y.pixels'])

        # X is the true size, y is set differently to adjust aspect ratio
        sci_pix_size_x = float(config_ini['_']['x.pixel.sz'])
        sci_pix_size_y = float(config_ini['_']['y.pixel.sz'])

        # FOV probably not quite right, should be estimated from voltage and emperical measurements but ok for now
        sci_fov_x = float(config_ini['_']['x.fov'])
        sci_fov_y = float(config_ini['_']['y.fov'])
        sci_volt_x = float(config_ini['_']['x.voltage'])
        sci_volt_y = float(config_ini['_']['y.voltage'])

        sci_ms_per_line = float(config_ini['_']['ms.p.line'])
        sci_line_freq = ((1 / sci_ms_per_line) / 2) * 1000

        sci_zoom = float(config_ini['_']['ZOOM'])
        sci_dwell_time = float(config_ini['_']['pixel.dwell.time.in.sec'])

        max_voltage = 10
        max_degrees = 20
        scan_angle = (sci_volt_x / max_voltage) * max_degrees
        sci_line_speed = scan_angle / sci_ms_per_line

        sci_fill_frac = float(config_ini['_']['fill.fraction'])

        sci_duration = utils.img.count_tif_frames(self.SciscanSettings.image_file) / sci_fps

        with open(sum_text_file, "w") as f:
            f.writelines("Sciscan\n")

            f.writelines("FPS        = {:.1f}\n".format(round(sci_fps, 1)))

            f.writelines("Pix size   = {:.1f} um\n".format(round(sci_pix_size_x * 1000000, 1)))

            f.writelines("FOV        = {:.0f}x{:.0f} um\n".format(round(sci_fov_x * 1000000, 0), round(sci_fov_y * 1000000, 0)))

            f.writelines("Pix        = {:.0f}x{:.0f}\n".format(round(sci_npix_x, 0), round(sci_npix_y, 0)))

            f.writelines("Line freq  = {:.0f} Hz\n".format(round(sci_line_freq, 1)))

            f.writelines("Line speed = {:.1f} deg/ms\n".format(round(sci_line_speed, 1)))

            f.writelines("Fill frac  = {:.0f} %\n".format(round(sci_fill_frac, 0)))

            f.writelines("Dwell      = {:.1f} us\n".format(round(sci_dwell_time * 1000000, 0)))

            f.writelines("Duration   = {d}\n".format(d=str(datetime.timedelta(seconds=round(sci_duration)))))

    def get_bad_frames(self, plot_dir, reg_tif=None, corr_cutoff=0.1):

        # Check for bad frames
        if not reg_tif:
            reg_tif = self.SciscanSettings.image_file
        tif_data = utils.img.read_tif_vol(reg_tif)

        # Why do I use the max projection and not the mean?
        z_proj = tif_data.max(axis=0)
        z_proj_flat = z_proj.flatten()

        img_path = os.path.join(plot_dir, "01-zproj-max.png")
        imageio.imwrite(img_path, utils.img.normalize_img_8bit(z_proj, prc_clip=0))

        n_frames = tif_data.shape[0]
        corrs = np.zeros(n_frames)
        pvals = np.zeros(n_frames)
        for i_frame in range(n_frames):
            # cov_mat = np.corrcoef(z_proj_flat, tif_data[i_frame, :, :].flatten())
            # corrs[i_frame] = cov_mat[0, 1]

            (r, p) = scipy.stats.spearmanr(z_proj_flat, tif_data[i_frame, :, :].flatten())
            corrs[i_frame] = r
            pvals[i_frame] = p

        win = 5
        corrs = scipy.signal.medfilt(corrs, kernel_size=win)

        bad_frames = corrs < corr_cutoff
        bad_frames_indexes = np.where(bad_frames)[0]

        if plot_dir:
            f = plt.figure()
            frame_indexes = np.arange(n_frames)
            plt.plot(frame_indexes, corrs, label='r')
            plt.plot(frame_indexes, pvals, label='p')
            plt.plot(frame_indexes[bad_frames], corrs[bad_frames], 'r', label='bad')
            plt.plot(frame_indexes, np.ones(n_frames) * corr_cutoff, 'k', label='thresh')

            plt.ylim([0, 1])
            plt.xlabel('Frame #')
            plt.ylabel('Correlation with max z-proj image')
            plt.legend()
            plot_img_path = os.path.join(plot_dir, "02-corr.png")
            f.savefig(plot_img_path, dpi=300, facecolor='white')
            plt.cla()
            plt.clf()
            plt.close('all')

            f = plt.figure()
            plt.hist(corrs, bins=50)
            plt.xlabel('Correlation')
            plt.ylabel('Number of frames')
            plot_img_path = os.path.join(plot_dir, "03-corr-hist.png")
            f.savefig(plot_img_path, dpi=300, facecolor='white')
            plt.cla()
            plt.clf()
            plt.close('all')

            # for i_frame in bad_frames_indexes:
            #     img_path = os.path.join(plot_dir, "04-badframe-{}.png".format(str(i_frame).zfill(6)))
            #     imageio.imwrite(img_path, utils.img.normalize_img_8bit(np.squeeze(tif_data[i_frame, :, :]), prc_clip=0.1))




        # # Look at means and stds. Seems ok for PMT dropout but not mems failure.
        # # This seems fast enough
        # with imageio.get_reader(self.SciscanSettings.image_file) as tif_reader:
        #     means = np.zeros((tif_reader.get_length()))
        #     stds = np.zeros((tif_reader.get_length()))
        #     for i_frame, frame in enumerate(tif_reader):
        #         f_mean = np.mean(frame)
        #         f_std = np.std(frame)
        #         means[i_frame] = f_mean
        #         stds[i_frame] = f_std
        #
        #
        # print(
        #     "Frame mean mean={:0.2f}, std mean={:0.2f}, mean std={:0.2f}, std std={:0.2f} ".format(np.mean(means),
        #                                                                                            np.std(means),
        #                                                                                            np.mean(stds),
        #                                                                                            np.std(stds)))
        #
        # bad_frames = np.logical_and(means < (np.mean(means) - nstds_mean * np.std(means)),
        #                             stds < (np.mean(stds) - nstds_stdd * np.std(stds)))
        # bad_frames_indexes = np.where(bad_frames)[0]
        #
        #
        # f = plt.figure()
        # plt.hist(means, bins=50)
        # plt.title('Frame means')
        # f.show()
        #
        # f = plt.figure()
        # plt.hist(stds, bins=50)
        # plt.title('Frame stds')
        # f.show()
        #
        # f = plt.figure()
        # plt.scatter(means, stds)
        # plt.xlabel('Frame means')
        # plt.ylabel('Frame stds')
        # f.show()

        # # Mess around trying to get PCA to find outliers, doesn't work easily.
        # print("Reading whole tif array")
        # tif_array =  m2putils.read_tif_array(self.SciscanSettings.image_file)
        # print(tif_array.shape)
        #
        # print("Calculating PCA")
        # pca = PCA(n_components=1)
        # projected = pca.fit_transform(tif_array)
        # print(pca.explained_variance_)
        # print(projected.shape)
        #
        # # f = plt.figure()
        # # plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5)
        # # plt.xlabel('PCA 1')
        # # plt.ylabel('PCA 2')
        # # f.show()
        #
        # f = plt.figure()
        # plt.plot(projected[:, 0])
        # plt.ylabel('PCA 1')
        # f.show()
        #
        # plt.cla()
        # plt.clf()
        # plt.close()
        #
        # raise Exception()

        print(
            "Found potentially {0} bad frames where the signal was lost (PMT trip or MEMS failure)".format(bad_frames_indexes.size))


        self.SciscanSettings.bad_frames = bad_frames
        self.SciscanSettings.bad_frames_indexes = bad_frames_indexes

    def save_bad_frames(self, overwrite=False, bad_frames_file=None, plot_dir=None, reg_tif=None, corr_cutoff=None):

        if bad_frames_file is None:
            bad_frames_file = os.path.join(os.path.split(self.SciscanSettings.image_file)[0], "bad_frames.npy")

        if overwrite or not os.path.exists(bad_frames_file):
            self.get_bad_frames(plot_dir=plot_dir, reg_tif=reg_tif, corr_cutoff=corr_cutoff)
            np.save(bad_frames_file, self.SciscanSettings.bad_frames_indexes)

        bf = np.load(bad_frames_file)
        print("There are {} bad frames (PMT trip or MEMS failure)".format(bf.size))

    def load_bad_frames(self, bad_frames_file=None):
        if bad_frames_file is None:
            bad_frames_file = os.path.join(os.path.split(self.SciscanSettings.image_file)[0], "bad_frames.npy")
        bf = np.load(bad_frames_file)
        return bf
