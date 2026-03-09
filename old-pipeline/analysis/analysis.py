import analysis_acc

def analyse_vis_evoke(raw_data_path, proc_data_path)
    exp_data = load_exp_data(raw_data_path)

    imaging_data = load_imaging(exp_data)

    timings = load_timings(exp_data)

    # Plot visually evoked responses
    analyse_vis_evoke.sum(exp_data, timings, imaging_data)

def analyse_acc(raw_data_path, proc_data_path):

    exp_data = load_exp_data(raw_data_path)

    imaging_data = load_imaging(exp_data)

    acc_data = load_acc(exp_data)

    analysis_acc.sum(exp_data, acc_data, imaging_data=imaging_data)
