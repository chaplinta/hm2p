from classes import Experiment, S2PData
import os
from utils import misc

proc_base_path = "J:/Data/light/"
s2p_base_path = "J:/Data/s2p/"
base_raw_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze"

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/maze-testing/2021_06_16/20210616_11_55_34_CAA-1113250"
# suite2p_path = "C:/Data/s2p/20210616_11_55_34_CAA-1113250/"

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_07_12/20210712_11_24_22_acc-test"
# # #suite2p_path = "C:/Data/s2p/20210616_11_55_34_CAA-1113250/"

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_07_12/20210712_11_26_00_acc-test"
# #suite2p_path = "C:/Data/s2p/20210616_11_55_34_CAA-1113250/"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_07_15/20210715_17_45_58_1113252"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_07_15/20210715_17_37_51_1113252"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/02 anesth-lights/2021_08_02/20210802_15_10_58_1113251"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/02 anesth-lights/2021_08_02/20210802_16_13_05_1114353"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_08_11/20210811_17_12_25_1113251"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_08_12/20210812_13_45_35_1113251"

#raw_data_path = os.path.join(base_raw_path, "2021_08_13", "20210813_15_55_56_1113251")

raw_data_path = misc.get_exp_path(base_raw_path, "20210920_11_41_16_1114356")


clearout = True
plot_bad = False


print("Loading experiment data")
exp = Experiment.Experiment(raw_data_path)
proc_data_path = os.path.join(proc_base_path, misc.path_leaf(exp.directory))
s2p_path = os.path.join(s2p_base_path, misc.path_leaf(exp.directory))
misc.setup_dir(proc_data_path, clearout=clearout)


evoke_soma_dFonF_dir = os.path.join(proc_data_path, "evoked-soma-dFonF")
evoke_soma_deconv_dir = os.path.join(proc_data_path, "evoked-soma-deconv")
evoke_soma_dFonF_bad_dir = os.path.join(evoke_soma_dFonF_dir, "bad")
evoke_soma_deconv_bad_dir = os.path.join(evoke_soma_deconv_dir, "bad")

misc.setup_dir(evoke_soma_dFonF_dir, clearout=clearout)
misc.setup_dir(evoke_soma_deconv_dir, clearout=clearout)
misc.setup_dir(evoke_soma_dFonF_bad_dir, clearout=clearout)
misc.setup_dir(evoke_soma_deconv_bad_dir, clearout=clearout)

evoke_dend_dFonF_dir = os.path.join(proc_data_path, "evoked-dend-dFonF")
evoke_dend_deconv_dir = os.path.join(proc_data_path, "evoked-dend-deconv")
evoke_dend_dFonF_bad_dir = os.path.join(evoke_dend_dFonF_dir, "bad")
evoke_dend_deconv_bad_dir = os.path.join(evoke_dend_deconv_dir, "bad")

misc.setup_dir(evoke_dend_dFonF_dir, clearout=clearout)
misc.setup_dir(evoke_dend_deconv_dir, clearout=clearout)
misc.setup_dir(evoke_dend_dFonF_bad_dir, clearout=clearout)
misc.setup_dir(evoke_dend_deconv_bad_dir, clearout=clearout)

print("Checking camera timings and frames")
exp.check_cameras(ignore_cams="side_left")

print("Checking sciscan and frames")
exp.check_sci_frames(max_frame_pulse_diff=50)
# events_dir = os.path.join(proc_data_path, "gifs")
# m2putils.setup_dir(events_dir, clearout=True)

exp.check_light_times(sync_with_sciscan=True)

print("Loading suite2p data")
soma_data = S2PData.load_mode(s2p_path, "soma")
dend_data = S2PData.load_mode(s2p_path, "dend")

print("Plotting light evoked activity")
time_pre = 1.0
time_post = 3.0
pre_f0 = True
plot_label = "flash"
plot_as_trace = True

print("Plotting light evoked dFOnF for good soma")
soma_data.plot_evoked(plot_dir=evoke_soma_dFonF_dir,
                      ca_data_type=S2PData.CaDataType.DFONF0,
                      s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                      trial_times=exp.Lighting.on_pulse_times[1:-1],
                      time_pre=time_pre,
                      time_trial=exp.Lighting.time_on / 1000,
                      time_post=time_post,
                      plot_good=True,
                      plot_bad=False,
                      pre_f0=pre_f0,
                      plot_label=plot_label,
                      plot_as_trace=plot_as_trace)

print("Plotting light evoked deconv for good soma")
soma_data.plot_evoked(plot_dir=evoke_soma_deconv_dir,
                      ca_data_type=S2PData.CaDataType.DECONV,
                      s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                      trial_times=exp.Lighting.on_pulse_times[1:-1],
                      time_pre=time_pre,
                      time_trial=exp.Lighting.time_on / 1000,
                      time_post=time_post,
                      plot_good=True,
                      plot_bad=False,
                      pre_f0=pre_f0,
                      plot_label=plot_label,
                      plot_as_trace=plot_as_trace)

print("Plotting light evoked dFOnF for good dendrites")
dend_data.plot_evoked(plot_dir=evoke_dend_dFonF_dir,
                      ca_data_type=S2PData.CaDataType.DFONF0,
                      s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                      trial_times=exp.Lighting.on_pulse_times[1:-1],
                      time_pre=time_pre,
                      time_trial=exp.Lighting.time_on / 1000,
                      time_post=time_post,
                      plot_good=True,
                      plot_bad=False,
                      pre_f0=pre_f0,
                      plot_label=plot_label,
                      plot_as_trace=plot_as_trace)

print("Plotting light evoked deconv for good dendrites")
dend_data.plot_evoked(plot_dir=evoke_dend_deconv_dir,
                      ca_data_type=S2PData.CaDataType.DECONV,
                      s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                      trial_times=exp.Lighting.on_pulse_times[1:-1],
                      time_pre=time_pre,
                      time_trial=exp.Lighting.time_on / 1000,
                      time_post=time_post,
                      plot_good=True,
                      plot_bad=False,
                      pre_f0=pre_f0,
                      plot_label=plot_label,
                      plot_as_trace=plot_as_trace)

if plot_bad:

    print("Plotting light evoked dFOnF for bad soma")
    soma_data.plot_evoked(plot_dir=evoke_soma_dFonF_bad_dir,
                          ca_data_type=S2PData.CaDataType.DFONF0,
                          s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                          trial_times=exp.Lighting.on_pulse_times[1:-1],
                          time_pre=time_pre,
                          time_trial=exp.Lighting.time_on / 1000,
                          time_post=time_post,
                          plot_good=False,
                          plot_bad=True,
                          pre_stim_f0=pre_f0,
                          plot_label=plot_label,
                          plot_as_trace=plot_as_trace)

    print("Plotting light evoked deconv for bad soma")
    soma_data.plot_evoked(plot_dir=evoke_soma_deconv_bad_dir,
                          ca_data_type=S2PData.CaDataType.DECONV,
                          s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                          trial_times=exp.Lighting.on_pulse_times[1:-1],
                          time_pre=time_pre,
                          time_trial=exp.Lighting.time_on / 1000,
                          time_post=time_post,
                          plot_good=False,
                          plot_bad=True,
                          pre_stim_f0=pre_f0,
                          plot_label=plot_label,
                          plot_as_trace=plot_as_trace)

    print("Plotting light evoked dFOnF for bad dendrites")
    dend_data.plot_evoked(plot_dir=evoke_dend_dFonF_bad_dir,
                          ca_data_type=S2PData.CaDataType.DFONF0,
                          s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                          trial_times=exp.Lighting.on_pulse_times[1:-1],
                          time_pre=time_pre,
                          time_trial=exp.Lighting.time_on / 1000,
                          time_post=time_post,
                          plot_good=False,
                          plot_bad=True,
                          pre_stim_f0=pre_f0,
                          plot_label=plot_label,
                          plot_as_trace=plot_as_trace)

    print("Plotting light evoked deconv for bad dendrites")
    dend_data.plot_evoked(plot_dir=evoke_dend_deconv_bad_dir,
                          ca_data_type=S2PData.CaDataType.DECONV,
                          s2p_start_time=exp.SciscanSettings.sci_frame_times[0],
                          trial_times=exp.Lighting.on_pulse_times[1:-1],
                          time_pre=time_pre,
                          time_trial=exp.Lighting.time_on / 1000,
                          time_post=time_post,
                          plot_good=False,
                          plot_bad=True,
                          pre_stim_f0=pre_f0,
                          plot_label=plot_label,
                          plot_as_trace=plot_as_trace)


print("Done")