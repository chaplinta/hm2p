from proc import proc_s2p
from paths import config
from utils import metadata as mdutils

# Doesn't work? Does all experiments! todo
exp_ids = []  # run on all experiments
# exp_ids = ["20220804_11_21_59_1117646",
#            "20220804_13_52_02_1117646"]

run_suite2p = False
run_all_exps = False
use_z_stack = False
overwrite = True
make_images = False
make_zproj = False
make_movies = True  # Not really needed if speed is an issue


cfg = config.M2PConfig()

# do_rigid_and_nonreg must best set to false because doing rigid or nonreg seems to be broken in suite2p
# see suite2p/registration/register.py", line 615, in registration_wrapper
#     yoff1, xoff1, corrXY1 = nonrigid_offsets

exps_df = mdutils.get_exps(cfg, exp_ids)

for index, exp_row in exps_df.iterrows():

    exp_id = exp_row["exp_id"]
    exp_index = exp_row["exp_index"]
    zstack_id = exp_row["zstack_id"]

    is_primary = exp_row["primary_exp"]
    is_good_exp = not exp_row["exclude"]


    if run_all_exps or (is_primary and is_good_exp):
    #if not (is_primary and is_good_exp):
        print(exp_id)
        proc_s2p.proc_single(cfg,
                             exp_id,
                             zstack_id=zstack_id,
                             run_suite2p=run_suite2p,
                             do_rigid_and_nonreg=False,
                             plot_traces=False,
                             plot_traces_bad=False,
                             make_images=make_images,
                             make_zproj=make_zproj,
                             plot_reg_metrics=False,
                             plot_pcas=False,
                             make_movies=make_movies,
                             overwrite_movies=False,
                             do_zstack=use_z_stack,
                             overwrite=overwrite)

