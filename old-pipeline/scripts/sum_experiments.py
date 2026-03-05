import os
from classes import Experiment, S2PData
from utils import misc, behave as bu, metadata as mdutils, video as vidutils
from paths.config import M2PConfig
import numpy as np
import pandas as pd
from classes.ProcPath import ProcPath

# Script to summarise experimental data, number of animals, cells, behaviour time etc.

cfg = M2PConfig()
exps_df = mdutils.get_exps(cfg, get_excluded=False)
animals_df = mdutils.get_animals(cfg)
# Create an animal id in the experiment frame by extracting it from the exp id.
exps_df['animal_id'] = exps_df['exp_id'].str.split("_").str[-1]

if not exps_df['animal_id'].isin(animals_df['animal_id'].astype(str)).all():
    raise Exception("There are experiments with no corresponding animal id")

exp_ani_df = pd.merge(exps_df, animals_df, left_on='animal_id', right_on='animal_id', how='inner')

# Use only primary experiments
exps_ani_pri_df = exp_ani_df.loc[exp_ani_df["primary_exp"] == 1]
animal_pri_counts = exps_ani_pri_df['animal_id'].value_counts()
if np.max(animal_pri_counts) > 1:
    raise Exception("At least one animal has more than one primary experiment \n{}".format(animal_pri_counts))

# Data frame of animals that were used in the included experiments
animals_pri_df = animals_df.loc[animals_df['animal_id'].astype(str).isin(exps_ani_pri_df['animal_id'])]

n_experiments = len(exps_ani_pri_df)
n_animals = exps_ani_pri_df['animal_id'].nunique()
n_sfb = exps_ani_pri_df['fibre'].value_counts()['SFB']
n_tfb = exps_ani_pri_df['fibre'].value_counts()['TFB']
n_has_zstack = exps_ani_pri_df['zstack_id'].count()
n_has_bad_2p = exps_ani_pri_df['bad_2p_frames'].count()
n_has_nobad_behav = exps_ani_pri_df['bad_behav_times'].isna().sum()
n_has_bad_behav = n_experiments - n_has_nobad_behav

# Some useful indexes
idx_penk = exps_ani_pri_df["celltype"] == "penk"
idx_nonpenk = exps_ani_pri_df["celltype"] == "nonpenk"
idx_sfb = exps_ani_pri_df["fibre"] == "SFB"
idx_tfb = exps_ani_pri_df["fibre"] == "TFB"

idx_penk_animals_used = animals_pri_df["celltype"] == "penk"
idx_nonpenk_animals_used = animals_pri_df["celltype"] == "nonpenk"

with open(cfg.sum_primary_file, "w") as f:
    f.writelines("Experiments\n")

    f.writelines("n                 = {:.0f}\n".format(round(n_experiments, 0)))
    f.writelines("n penk            = {:.0f}\n".format(round(exps_ani_pri_df.loc[idx_penk]['exp_id'].nunique(), 0)))
    f.writelines("n nonpenk         = {:.0f}\n".format(round(exps_ani_pri_df.loc[idx_nonpenk]['exp_id'].nunique(), 0)))
    f.writelines("n sfb             = {:.0f}\n".format(round(n_sfb, 0)))
    f.writelines("n tfb             = {:.0f}\n".format(round(n_tfb, 0)))
    f.writelines("n with z-stacks   = {:.0f}\n".format(round(n_has_zstack, 0)))
    f.writelines("n some bad 2p     = {:.0f}\n".format(round(n_has_bad_2p, 0)))
    f.writelines("n bad behav       = {:.0f}\n".format(round(n_has_bad_behav, 0)))

    f.writelines("\n")

    f.writelines("Animals\n")

    f.writelines("n                 = {:.0f}\n".format(round(n_animals, 0)))
    f.writelines("n penk            = {:.0f}\n".format(round(len(animals_pri_df.loc[idx_penk_animals_used]['animal_id']), 0)))
    f.writelines("n nonpenk         = {:.0f}\n".format(round(len(animals_pri_df.loc[idx_nonpenk_animals_used]['animal_id']), 0)))



# Add new columns for analysis
exp_ani_df["n_soma"] = np.nan
exp_ani_df["n_dend"] = np.nan
exp_ani_df["behave_time_good"] = np.nan
exp_ani_df["behave_time_bad"] = np.nan
for index, exp_row in exp_ani_df.iterrows():

    exp_id = exp_row["exp_id"]

    print("Loading s2p for {}".format(exp_id))

    m2p_paths = ProcPath(cfg, exp_id)

    soma_data = S2PData.load_mode(m2p_paths.proc_s2p_path, "soma")
    dend_data = S2PData.load_mode(m2p_paths.proc_s2p_path, "dend")

    exp_ani_df.loc[index, "n_soma"] = np.sum(soma_data.iscell[:, 0] == 1)
    exp_ani_df.loc[index, "n_dend"] = np.sum(dend_data.iscell[:, 0] == 1)

    # Loading an experiment is slow so don't do it if needed.
    # todo a better way of doing this.
    if exp_row["bad_behav_times"] != "?":

        exp = Experiment.Experiment(m2p_paths.raw_data_path)

        n_frames = vidutils.count_frames(exp.tracking_video.cameras["overhead"].file)
        good_time, bad_time = mdutils.get_good_bad_behav_time(exp_row, exp.tracking_video.fps, n_frames)

        exp_ani_df.loc[index, "behave_time_good"] = good_time / 60
        exp_ani_df.loc[index, "behave_time_bad"] = bad_time / 60

# Get primary experiments again since there are new stats
exps_ani_pri_df = exp_ani_df.loc[exp_ani_df["primary_exp"] == 1]

with open(cfg.sum_primary_file, "a") as f:
    f.writelines("\n")
    f.writelines("Cells\n")
    f.writelines("n soma            = {:.0f}\n".format(round(np.sum(exps_ani_pri_df["n_soma"]), 0)))
    f.writelines("n soma penk       = {:.0f}\n".format(round(np.sum(exps_ani_pri_df.loc[idx_penk]["n_soma"]), 0)))
    f.writelines("n soma nonpenk    = {:.0f}\n".format(round(np.sum(exps_ani_pri_df.loc[idx_nonpenk]["n_soma"]), 0)))

    f.writelines("n dend            = {:.0f}\n".format(round(np.sum(exps_ani_pri_df["n_dend"]), 0)))
    f.writelines("n dend penk       = {:.0f}\n".format(round(np.sum(exps_ani_pri_df.loc[idx_penk]["n_dend"]), 0)))
    f.writelines("n dend nonpenk    = {:.0f}\n".format(round(np.sum(exps_ani_pri_df.loc[idx_nonpenk]["n_dend"]), 0)))

    f.writelines("\n")
    f.writelines("Behaviour\n")
    f.writelines("min good behav (mins)     = {:.1f}\n".format(round(np.nanmin(exps_ani_pri_df["behave_time_good"]), 0)))
    f.writelines("max good behavior (mins)  = {:.1f}\n".format(round(np.nanmax(exps_ani_pri_df["behave_time_good"]), 0)))
    f.writelines("mean good behavior (mins) = {:.1f}\n".format(round(np.nanmean(exps_ani_pri_df["behave_time_good"]), 0)))

    f.writelines("\n")
    f.writelines("Behaviour penk vs nonpenk\n")
    f.writelines("min good behav (mins) {:.1f} vs {:.1f}\n".format(
        round(np.nanmin(exps_ani_pri_df.loc[idx_penk]["behave_time_good"]), 0),
        round(np.nanmin(exps_ani_pri_df.loc[idx_nonpenk]["behave_time_good"]), 0)))
    f.writelines("max good behav (mins) {:.1f} vs {:.1f}\n".format(
        round(np.nanmax(exps_ani_pri_df.loc[idx_penk]["behave_time_good"]), 0),
        round(np.nanmax(exps_ani_pri_df.loc[idx_nonpenk]["behave_time_good"]), 0)))
    f.writelines("mean good behav (mins) {:.1f} vs {:.1f}\n".format(
        round(np.nanmean(exps_ani_pri_df.loc[idx_penk]["behave_time_good"]), 0),
        round(np.nanmean(exps_ani_pri_df.loc[idx_nonpenk]["behave_time_good"]), 0)))

    f.writelines("\n")
    f.writelines("Behaviour SFB vs TFB\n")
    f.writelines("min good behav (mins) {:.1f} vs {:.1f}\n".format(
        round(np.nanmin(exps_ani_pri_df.loc[idx_sfb]["behave_time_good"]), 0),
        round(np.nanmin(exps_ani_pri_df.loc[idx_tfb]["behave_time_good"]), 0)))
    f.writelines("max good behav (mins) {:.1f} vs {:.1f}\n".format(
        round(np.nanmax(exps_ani_pri_df.loc[idx_sfb]["behave_time_good"]), 0),
        round(np.nanmax(exps_ani_pri_df.loc[idx_tfb]["behave_time_good"]), 0)))
    f.writelines("mean good behav (mins) {:.1f} vs {:.1f}\n".format(
        round(np.nanmean(exps_ani_pri_df.loc[idx_sfb]["behave_time_good"]), 0),
        round(np.nanmean(exps_ani_pri_df.loc[idx_tfb]["behave_time_good"]), 0)))

# Save all experiments, whether they are primary or not
exp_ani_df.to_csv(cfg.sum_exp_csv_file)

# Save only primary experiments
exps_ani_pri_df.to_csv(cfg.sum_exp_pri_csv_file)

# Save only animals used in primary experiments
animals_pri_df.to_csv(cfg.sum_animals_pri_csv_file)


