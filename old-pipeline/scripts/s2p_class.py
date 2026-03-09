import os
from paths.config import M2PConfig
from utils import metadata as mdutils

# Builds traning files for suite 2p roi classifier.
# Still have to build classifier in the gui because I can't see how to call it from here.

cfg = M2PConfig()


data_dirs = []

exps_df = mdutils.get_exps(cfg)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]
    data_dirs.append(os.path.join(cfg.s2p_path, exp_id))



dend_iscells = []
soma_iscells = []

list_file_soma = os.path.join(cfg.s2p_class_path, "train_soma.txt")
list_file_dend = os.path.join(cfg.s2p_class_path, "train_dend.txt")

with open(list_file_soma, "w") as f_soma, open(list_file_dend, "w") as f_dend:
    for dd in data_dirs:
        soma_file = os.path.join(dd, "suite2p_soma/plane0/iscell.npy")
        dend_file = os.path.join(dd, "suite2p_dend/plane0/iscell.npy")

        print(soma_file)

        f_soma.write("{f}\n".format(f=soma_file))
        f_dend.write("{f}\n".format(f=dend_file))
        soma_iscells.append(soma_file)
        dend_iscells.append(dend_file)


# Can't see how to run the classifer from here, just use the txt file above and the gui.
# class_soma_file = os.path.join(video_path, "classifier_soma.npy")
# class_dend_file = os.path.join(video_path, "classifier_dend.npy")

print("Done")