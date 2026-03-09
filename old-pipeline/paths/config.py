from pathlib import Path
from dataclasses import dataclass

@dataclass
class M2PConfig:

    # Input dirs, todo get from config.ini
    raw_path: Path = Path("/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/shared/lab-108/experiments/01 lights-maze")
    #raw_path: Path = Path("/Users/tristan/Neuro/hm2p/raw")
    #meta_path: Path = Path("/Users/tristan/Neuro/hm2p-analysis/metadata")
    meta_path: Path = Path("/Users/tristan/Neuro/hm2p-analysis/metadata")
    s2p_params_path: Path = Path("/Users/tristan/Neuro/hm2p-analysis/s2p")
    # base_path: Path = Path("/Users/tristan/Neuro/hm2p")
    base_path: Path = Path("/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p")
    #dlc_config_path = "L:/Data/dlc/overhead-m2p-Tristan-2021-08-22/config.yaml"
    #dlc_base_path = Path("/Users/tristan/Desktop/hm2p-maze-tristan-2023-02-17")
    dlc_base_path = Path("/Users/tristan/Library/CloudStorage/Dropbox/Neuro/Margrie/hm2p/dlc")
    dlc_path = dlc_base_path / "hm2p-maze-tristan-2023-02-17"
    dlc_config_path = dlc_base_path / "config.yaml"
    dlc_iter_name = "DLC_resnet50_hm2p-mazeFeb17shuffle1_950000"
    cam_calibration_files = {'f4mm': '/Users/tristan/Neuro/hm2p-analysis/cam-calibrations/acA1300-200um_C125-0418-5M.npz',
                             'f6mm': '/Users/tristan/Neuro/hm2p-analysis/cam-calibrations/acA1300-200um_C125-0618-5M.npz'}


    # Meta data files.
    meta_animals_file = meta_path / "animals.csv"
    meta_exps_file = meta_path / "experiments.csv"
    meta_somadend_file = meta_path / "somadend.csv"

    # Video directories
    video_path: Path = base_path / "video"
    video_meta_bak_path: Path = base_path / "video-meta-backup"

    # Video directories
    brains_path: Path = base_path / "brains"
    brains_sorted_path: Path = base_path / "brains-sorted"
    brains_reg_path: Path = base_path / "brains-reg"


    # Suite2p directories
    s2p_path: Path = base_path / "s2p"
    s2p_class_path: Path = s2p_params_path

    # Z-stacks
    zstack_path = base_path / "z-stacks"

    # DLC
    dlc_path = dlc_base_path
    dlc_input_video_path = dlc_path / "videos"
    dlc_tracked_path = dlc_input_video_path

    # Data checking directories
    bad_2p_plots = base_path / "bad-2p-plots"
    exp_check_path = base_path / "exp-check"

    exp_check_camsyncs_file = exp_check_path / "bad_camsyncs.txt"
    exp_check_lightsyncs_file = exp_check_path / "bad_lightsyncs.txt"
    exp_check_2psyncs_file = exp_check_path / "bad_2pyncs.txt"
    exp_check_2pimg_file = exp_check_path / "bad_2ping.txt"

    # Processed directories
    proc_path = base_path / "proc"
    proc_db_path = base_path / "db"
    proc_raw_event_plots_path = proc_db_path / "event-plots"
    proc_raw_tune_path = proc_db_path / "tune-plots"
    proc_raw_tune_pair_path = proc_db_path / "tune-pair-plots"
    proc_raw_dendplots_path = proc_db_path / "dend-event-plots"

    # Summary  directories
    sum_path = base_path / "sum"
    sum_exp_path = sum_path / "experiments"
    sum_primary_file = sum_exp_path / "summary_primary.txt"
    sum_exp_csv_file = sum_exp_path / "experiments.csv"
    sum_exp_pri_csv_file = sum_exp_path / "experiments_primary.csv"
    sum_animals_pri_csv_file = sum_exp_path / "animals_primary.csv"
    sum_dist_plots = sum_path / "dist-plots"
    sum_behave_path = sum_path / "behave"
    sum_tune_path = sum_path / "tune"
    sum_tune_roi_path = sum_tune_path / "roi"
    sum_tune_roi_sig_path = sum_tune_roi_path / "sig"
    sum_tune_roi_nonsig_path = sum_tune_roi_path / "nonsig"
    sum_tune_pair_path = sum_tune_path / "pair"
    sum_events_path = sum_path / "events"
    sum_agg_somadend_path = sum_path / "somadend"
    sum_agg_somadend_pairs_path = sum_agg_somadend_path / "pairs"
    sum_plot_path = sum_path / "sum-plots"
    sum_trace_path = sum_path / "traces"
    sum_dec_path = sum_path / "dec"
    sum_dec_roi_path = sum_dec_path / "roi"
    sum_dec_roi_sig_path = sum_dec_path / "sig"
    sum_dec_roi_nonsig_path = sum_dec_path / "nonsig"

    # Create directories
    raw_path.mkdir(exist_ok=True)
    proc_path.mkdir(exist_ok=True)
    proc_db_path.mkdir(exist_ok=True)
    proc_raw_event_plots_path.mkdir(exist_ok=True)
    proc_raw_tune_path.mkdir(exist_ok=True)
    proc_raw_tune_pair_path.mkdir(exist_ok=True)
    proc_raw_dendplots_path.mkdir(exist_ok=True)

    sum_path.mkdir(exist_ok=True)
    sum_exp_path.mkdir(exist_ok=True)
    sum_dist_plots.mkdir(exist_ok=True)
    sum_tune_path.mkdir(exist_ok=True)
    sum_tune_roi_path.mkdir(exist_ok=True)
    sum_tune_roi_sig_path.mkdir(exist_ok=True)
    sum_tune_roi_nonsig_path.mkdir(exist_ok=True)
    sum_tune_pair_path.mkdir(exist_ok=True)
    sum_events_path.mkdir(exist_ok=True)
    sum_agg_somadend_path.mkdir(exist_ok=True)
    sum_agg_somadend_pairs_path.mkdir(exist_ok=True)
    sum_plot_path.mkdir(exist_ok=True)
    sum_trace_path.mkdir(exist_ok=True)

    video_path.mkdir(exist_ok=True)
    video_meta_bak_path.mkdir(exist_ok=True)

    dlc_path.mkdir(exist_ok=True)
    dlc_input_video_path.mkdir(exist_ok=True)

    bad_2p_plots.mkdir(exist_ok=True)
    exp_check_path.mkdir(exist_ok=True)

    s2p_path.mkdir(exist_ok=True)

    # Database data files
    db_behave_file = proc_db_path / "behave.h5"
    db_behave_frames_file = proc_db_path / "behave_frames.h5"
    db_behave_events_file = proc_db_path / "behave_events.h5"

    db_ca_roi_file = proc_db_path / "ca_rois.h5"
    db_ca_file = proc_db_path / "ca.h5"

    db_roi_tune_file = proc_db_path / "roi_tune.h5"
    db_roi_stat_file = proc_db_path / "roi_stats.h5"
    db_roi_stat_dend_file = proc_db_path / "roi_stats_withdend.h5"

    db_somadend_pairs_file = proc_db_path / "somadend_pairs.h5"
    db_somadend_events_file = proc_db_path / "somadend_events.h5"
    db_somadend_ca_file = proc_db_path / "somadend_ca.h5"

    db_somadend_tune_file = proc_db_path / "somadend_tune.h5"
    db_somadend_stat_file = proc_db_path / "somadend_stats.h5"

    db_conn_pair_file = proc_db_path / "conn_pairs.h5"






