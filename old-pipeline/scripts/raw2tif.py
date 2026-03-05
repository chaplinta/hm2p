from paths import config
from utils import misc as m2putils
import configparser

# Script to convert sciscan raw files to tifs. Useful for converting z-stack datasets to tif.

exp_id = "20220406_18_11_37_1116663"
cfg = config.M2PConfig()


exp_path = m2putils.get_exp_path(cfg.raw_path, exp_id)

#exp_path = "/Users/tristan/Neuro/hm2p/z-stacks/20220406_18_11_37_1116663"


inifile = m2putils.get_filetype(exp_path, "*.ini")
config_ini = configparser.ConfigParser()
config_ini.read(inifile)

imgfile = m2putils.get_filetype(exp_path, "*.raw")

m2putils.sci_raw_2_tif(imgfile, config_ini)

print("Done")

