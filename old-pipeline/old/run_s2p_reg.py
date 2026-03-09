from classes import S2PData
import configparser
import os
from utils import misc, s2p

raw_data_path = "J:/Users/Tristan/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze/2021_08_11/20210811_16_46_31_1113251"

inifile = misc.get_filetype(raw_data_path, "*.ini")
sci_ini = configparser.ConfigParser()
sci_ini.read(inifile)

imgfile = misc.get_filetype(raw_data_path, "*.raw")
if not imgfile:
    imgfile = misc.get_filetype(raw_data_path, "*.raw")
else:
    imgfile = misc.sci_raw_2_tif(imgfile, config_ini_reader=sci_ini)
if not imgfile:
    raise Exception("Couldn't find any image files")

is_zstack = "XYTZ" in imgfile
print(imgfile)
print(is_zstack)
if is_zstack:
    proc_base_path = "C:/Data/z-stacks/"
else:
    proc_base_path = "C:/Data/reg-imgs/"

proc_data_path = os.path.join(proc_base_path, misc.path_leaf(raw_data_path))

misc.setup_dir(proc_data_path)



basename = os.path.basename(imgfile)
basename, ext = os.path.splitext(basename)

ops, db = s2p.create_ops_reg("reg",
                             fps=float(sci_ini['_']['frames.hd_p.sec']),
                             image_file=imgfile,
                             proc_data_path=proc_data_path)

s2p.process(ops, db)

s2p_reg = S2PData.load_mode(proc_data_path, "reg")

s2p_reg.save_tif(tif_path=os.path.join(proc_data_path, "reg.tif"), prc_clip=0)



