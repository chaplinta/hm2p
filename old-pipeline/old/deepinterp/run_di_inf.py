import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib
import h5py
import numpy as np
import imageio

model_path = "J:/Data/deepinterp/examples/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"
input_path = "J:/Data/s2p/20210823_16_59_50_1114353/suite2p_soma/plane0/reg.pad.tif"
output_path = "J:/Data/s2p/20210823_16_59_50_1114353/suite2p_soma/plane0/reg.denoise.h5"

generator_param = {}
inferrence_param = {}

# # We are reusing the data generator for training here.
# generator_param["type"] = "generator"
# generator_param["name"] = "SingleTifGenerator"
# generator_param["pre_post_frame"] = 30
# generator_param["pre_post_omission"] = 0
# generator_param[
#     "steps_per_epoch"
# ] = -1  # No steps necessary for inference as epochs are not relevant. -1 deactivate it.
#
# # generator_param["train_path"] = os.path.join(
# #     pathlib.Path(__file__).parent.absolute(),
# #     "..",
# #     "sample_data",
# #     "ophys_tiny_761605196.tif",
# # )
#
# #generator_param["train_path"] = "J:/Data/deepinterp/examples/ophys_tiny_761605196.tif"
# generator_param["train_path"] = input_path
#
#
# generator_param["batch_size"] = 5
# generator_param["start_frame"] = 0
# generator_param["end_frame"] = -1 #99  # -1 to go until the end.
# generator_param[
#     "randomize"
# ] = 0  # This is important to keep the order and avoid the randomization used during training
#
#
# inferrence_param["type"] = "inferrence"
# inferrence_param["name"] = "core_inferrence"
#
# # Replace this path to where you stored your model
# # inferrence_param[
# #     "model_path"
# # ] = "/Users/jeromel/Documents/Work documents/Allen Institute/Projects/Deep2P/repos/public/deepinterpolation/examples/unet_single_1024_mean_absolute_error_2020_11_12_21_33_2020_11_12_21_33/2020_11_12_21_33_unet_single_1024_mean_absolute_error_2020_11_12_21_33_model.h5"
# inferrence_param[
#     "model_path"
# ] = model_path
#
#
#
#
# # # Replace this path to where you want to store your output file
# # inferrence_param[
# #     "output_file"
# # ] = "/Users/jeromel/test/ophys_tiny_continuous_deep_interpolation.h5"
# # inferrence_param[
# #     "output_file"
# # ] = "J:/Data/deepinterp/examples/ophys_tiny_761605196_denoised.h5"
# inferrence_param[
#     "output_file"
# ] = output_path
#
# jobdir = "J:/Data/test/"
#
# try:
#     os.mkdir(jobdir)
# except:
#     print("folder already exists")
#
# path_generator = os.path.join(jobdir, "generator.json")
# json_obj = JsonSaver(generator_param)
# json_obj.save_json(path_generator)
#
# path_infer = os.path.join(jobdir, "inferrence.json")
# json_obj = JsonSaver(inferrence_param)
# json_obj.save_json(path_infer)
#
# generator_obj = ClassLoader(path_generator)
# data_generator = generator_obj.find_and_build()(path_generator)
#
# inferrence_obj = ClassLoader(path_infer)
# inferrence_class = inferrence_obj.find_and_build()(path_infer, data_generator)
#
# print("Running inference")
#
# # Except this to be slow on a laptop without GPU. Inference needs parallelization to be effective.
# inferrence_class.run()
#
# print("Done")

print("Converting to tif")
tif_out_file = os.path.splitext(output_path)[0] + ".tif"
f = h5py.File(output_path, 'r')
img_data = f["data"]
n_frames = img_data.shape[0]

#bytes_to_read = pix_x * pix_y * raw_dtype.itemsize
with imageio.get_writer(tif_out_file, format='TIFF', mode='v', bigtiff=True) as tif_writer:

    for i_frame in range(n_frames):
        #print(i_frame)
        #data = f.read(bytes_to_read)
        #pix_array = np.frombuffer(data, dtype=raw_dtype)
        #img = np.reshape(pix_array, (pix_y, pix_x)).astype(raw_dtype).byteswap()
        tif_writer.append_data(np.squeeze(img_data[i_frame, :, :, :]))

print("Done")

