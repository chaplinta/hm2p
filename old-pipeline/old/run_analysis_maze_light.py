from classes import Experiment
import os
from utils import misc

proc_base_path = "J:/Data/s2p/"

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/maze-testing/2021_06_16/20210616_11_55_34_CAA-1113250"
# suite2p_path = "C:/Data/s2p/20210616_11_55_34_CAA-1113250/"

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_07_12/20210712_11_24_22_acc-test"
# # #suite2p_path = "C:/Data/s2p/20210616_11_55_34_CAA-1113250/"

# raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/accel/testing/2021_07_12/20210712_11_26_00_acc-test"
# #suite2p_path = "C:/Data/s2p/20210616_11_55_34_CAA-1113250/"

#raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_07_15/20210715_17_45_58_1113252"

raw_data_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_07_15/20210715_17_37_51_1113252"



# TODO make stiched movies

clearout = False

print("Loading experiment data")
exp = Experiment.Experiment(raw_data_path)
proc_data_path = os.path.join(proc_base_path, misc.path_leaf(exp.directory))

misc.setup_dir(proc_data_path, clearout=False)

print("Checking camera timings and frames")
exp.check_cameras(ignore_cams="side_left")

print("Checking sciscan and frames")
exp.check_sci_frames()
# events_dir = os.path.join(proc_data_path, "gifs")
# m2putils.setup_dir(events_dir, clearout=True)



print("Loading suite2p data")
#s2p_data_soma = s2putils.load_mode(suite2p_path, "soma")
#s2p_data_dend = s2putils.load_mode(suite2p_path, "dend")

print("Done")