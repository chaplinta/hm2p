import os
import shutil
from paths import config
from utils import misc as m2putils, metadata as mdutils
import configparser
from pathlib import Path

# Script to take a list of experiment ids and create tifs in a separate directory for inspection.

out_path = Path("/Users/tristan/Neuro/hm2p/inspect-2p")

exp_ids = None  # Set to none to get the all from meta data

# These should be good
# exp_ids = ["20210823_16_59_50_1114353",
#            "20210920_11_09_37_1114356",
#            "20211203_15_10_27_1115464",
#            "20211216_14_36_39_1115816",
#            "20220408_15_01_57_1116663",
#            "20220601_13_53_18_1117217",
#            "20221003_14_36_54_1118020",
#            "20221004_10_42_58_1118023",
#            "20221010_11_43_35_1118018",
#            "20221115_13_27_42_1118213",
#            "20221116_14_31_12_1118320",
#            "20221117_13_20_31_1118317"]
#
# out_path = Path("/Users/tristan/Neuro/hm2p/inspect-2p")

# # some maybe crappy ones to check
# exp_ids = ["20210923_15_05_14_1114356",  # Could be useful if 20/09 exp has issues
#            "20210924_16_09_21_1114356",  # Could be useful if 20/09 exp has issues
#            "20211028_10_22_38_1114356",  # Could be useful if 20/09 exp has issues
#            "20211028_11_25_50_1115465",  #usable?
#            "20211029_13_50_08_1115465",  #usable?
#            "20211102_15_11_34_1115465",  #usable?
#            "20211124_10_31_16_1115465",  #not good
#            "20211212_13_59_33_1115464",  #usable?
#            "20220411_16_45_08_1116663",  #usable if other day bad
#            "20220531_11_06_13_1117217",  #usable if other day bad
#            "20220608_15_27_32_1117217",  #usable if other day bad
#            "20220608_16_22_06_1116994",
#            "20220802_15_06_53_1117646",
#            "20221018_10_56_17_1117788"]

# # not usable - box or short etc
# exp_ids = ["20210715_17_45_58_1113252",
#            "20210802_15_10_58_1113251",
#            "20210802_16_13_05_1114353",
#            "20210811_17_12_25_1113251",
#            "20210812_13_45_35_1113251",
#            "20210812_13_51_22_1113251",
#            "20210813_16_34_58_1113251",
#            "20210920_11_41_16_1114356"]

# exp_ids = ["20220804_11_21_59_1117646",
#            "20220804_13_52_02_1117646"]

# # Another to check
# exp_ids = ["20220815_15_01_41_1117646", "20221003_14_36_54_1118020"]

# more to check: 20220815_15_01_41_1117646,


cfg = config.M2PConfig()

if exp_ids is None:
    exp_ids = mdutils.get_exp_ids(cfg)
    print(exp_ids)


out_path.mkdir(exist_ok=True)

for exp_id in exp_ids:

    print(exp_id)

    exp_path = m2putils.get_exp_path(cfg.raw_path, exp_id)

    inifile = m2putils.get_filetype(exp_path, "*.ini")
    config_ini = configparser.ConfigParser()
    config_ini.read(inifile)

    imgfile = m2putils.get_filetype(exp_path, "*.raw", allow_missing=True)
    if imgfile:
        m2putils.sci_raw_2_tif(imgfile, config_ini, out_path=out_path, save_red=False, overwrite=False)
    else:
        imgfile = m2putils.get_filetype(exp_path, "*.tif", allow_missing=False)
        shutil.copyfile(imgfile, os.path.join(out_path, os.path.split(imgfile)[1]))



print("Done")

