from utils import misc as m2putils, metadata as mdutils
from proc import proc_behave, proc_somadend, proc_ca_behave, proc_ca
from sum import sum_behave, sum_ca_behave, sum_tune, sum_events, sum_somadend
from paths import config



expids = None

# Or manually set some of the ids to run
#expids = ["20220802_15_06_53_1117646"]

cfg = config.M2PConfig()

if expids is None:
    expids = mdutils.get_exp_ids(cfg, get_excluded=False, primary_only=True)
    print(expids)

for exp_id in expids:

    print(exp_id)

    proc_ca.proc_ca(cfg, exp_id, plot_events=True)

    proc_behave.proc_single(cfg, exp_id, rebuild_behav_data=True)
    
    sum_behave.sum_exp(cfg, exp_id, create_bigspeed_videos=False)
    
    proc_ca_behave.proc_resample_exp(cfg, exp_id)

    #proc_ca_behave.proc_tuning(cfg, proc_exp_ids=[exp_id], plot=True, n_boots=200, hd_dirs=[1])

    # todo try tightening up parameters?
    # event_onset_p = 0.2,
    # event_offset_p = 0.5,
    # smooth_sigma = 3,
    # noise_alpha = 0.5
    #proc_somadend.proc_single(cfg, exp_id, plot_events=True, noise_alpha=0.05)




    # Don't think I need these now
    #proc_somadend.proc_resample_single(cfg, exp_id)
    # proc_ca_behave.proc_tuning(cfg, proc_type=proc_ca_behave.TuneProcType.somadend,
    #                              proc_exp_ids=[exp_id], plot=True, n_boots=200, hd_dirs=[1])



    pass





#sum_events.sum_events(cfg)
#sum_tune.sum_tune(cfg)
#sum_ca_behave.sum_dist_ca(cfg)

#sum_tune.plot_pairs(cfg, proc_ca_behave.TuneProcType.roi)
#
#sum_somadend.sum_all(cfg)
#sum_somadend.sum_all_behave(cfg)
#sum_somadend.sum_type(cfg)
#
# sum_tune.plot_pairs(cfg, proc_ca_behave.TuneProcType.somadend)

print("Run done")