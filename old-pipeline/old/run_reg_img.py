from suite2p.registration import rigid
import imageio
import os
import numpy as np
from utils import s2p

base_path = "C:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/relocation-test-11132456"
img_path = os.path.join(base_path, "crop-reloc-05-remount1-20210916_17_39_15_1114356_XYT.tif")
ref_img_path = os.path.join(base_path, "reloc-06-remount2-20210916_17_40_58_1114356_XYT.tif")

maxregshift = 0.4

img_data = imageio.imread(img_path)
ref_img_data = imageio.imread(ref_img_path)

print("Registering")

# Standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the
# frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV).
smooth_sigma = 1.15

ymax, xmax, cmax = s2p.img_reg(img_data, ref_img_data, smooth_sigma=smooth_sigma, maxregshift=maxregshift)

ymax = int(np.round(ymax))
xmax = int(np.round(xmax))
img_data_reg = rigid.shift_frame(frame=img_data, dy=ymax, dx=xmax)

imageio.imwrite(os.path.join(base_path, "test.tif"), img_data_reg)

print("Done")

print("Process complete")