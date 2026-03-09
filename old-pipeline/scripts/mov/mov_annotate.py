import os
import imageio
from classes import Experiment
from utils import misc
import napari
from paths.config import M2PConfig
from classes.ProcPath import ProcPath

# Crop: Draw a rectangular crop region.
# Scale: Draw a line between two holes on the air table.
# ROI: Draw a rectangle on the floor of the maze. Rotate a bit if needed, although rotation is not used yet.
# 'Save all layers' as 'meta' in video folder, as napari builtin Save to FOlder (*.*), NOT SVG!
# This saves as csv in a folder called meta.

cfg = M2PConfig()

exp_id = "20221003_14_36_54_1118020"

m2p_paths = ProcPath(cfg=cfg, exp_id=exp_id)

raw_data_path = misc.get_exp_path(cfg.raw_path, exp_id)

exp = Experiment.Experiment(raw_data_path)

proc_data_path = os.path.join(cfg.video_path, misc.path_leaf(exp.directory))
misc.setup_dir(proc_data_path, clearout=False)

cam_name = "overhead"
undistort_file_name = os.path.splitext(os.path.split(exp.tracking_video.cameras[cam_name].file)[1])[0] + "-undistort.mp4"
mov_undistort_file = os.path.join(proc_data_path, undistort_file_name)

with imageio.get_reader(mov_undistort_file) as mov_reader:
    frame_image = mov_reader.get_next_data()

# Create the viewer
viewer = napari.Viewer()

# Add image layer
layer_image = viewer.add_image(frame_image, name='movie-frame')

layer_names = ["roi", "scale", "crop"]
layers = []
for ln in layer_names:
    layers.append(viewer.add_shapes(name=ln))

napari.run()
#napari.save_layers(proc_data_path, layers)
print("Done annotating")

