import pandas as pd
import os
import numpy as np
from paths.config import M2PConfig
import matplotlib.pyplot as plt
import scipy
from utils import behave as beutils, data as dutils, plot as putils


def sum_type(cfg: M2PConfig, dpi=300):

    df_events = pd.read_hdf(cfg.db_somadend_events_file)
    df_behave = pd.read_hdf(cfg.db_behave_events_file)

    df = df_events.merge(df_behave, on=["exp_id", "pair_id", "onset_index", "offset_index"], how='left')

    df_grp = df.groupby(['exp_id', 'pair_id', 'type', beutils.LIGHT_ON])['onset_index'].aggregate(["count"])

    df_pt = pd.pivot_table(df,
                           values=['onset_index'],
                           index=['exp_id', 'pair_id', 'type'],
                           columns=[beutils.LIGHT_ON],
                           aggfunc=np.size)

    indexes_joint = df_pt.index.get_level_values('type').values == 0
    indexes_somatic = df_pt.index.get_level_values('type').values == 1
    indexes_dendritic = df_pt.index.get_level_values('type').values == 2

    n_joint_light = df_pt.loc[indexes_joint]["onset_index"][0].values
    n_joint_dark = df_pt.loc[indexes_joint]["onset_index"][1].values
    n_somatic_light = df_pt.loc[indexes_somatic]["onset_index"][0].values
    n_somatic_dark = df_pt.loc[indexes_somatic]["onset_index"][1].values
    n_dendritic_light = df_pt.loc[indexes_dendritic]["onset_index"][0].values
    n_dendritic_dark = df_pt.loc[indexes_dendritic]["onset_index"][1].values

    _, p_joint = scipy.stats.wilcoxon(n_joint_light, n_joint_dark)
    _, p_somatic = scipy.stats.wilcoxon(n_somatic_light, n_somatic_dark)
    _, p_dendritc = scipy.stats.wilcoxon(n_dendritic_light, n_dendritic_dark)

    plot_index = 1000
    fig = plt.figure(tight_layout=True)
    plt.scatter(n_joint_light,
                n_joint_dark,
                label='joint')
    plt.scatter(n_somatic_light,
                n_somatic_dark,
                label='somatic')
    plt.scatter(n_dendritic_light,
                n_dendritic_dark,
                label='dendritic')
    plt.xlabel("# events light")
    plt.ylabel("# events dark")
    plt.legend()

    putils.square_plot()

    plt.title("jp={:.3f} sp={:.3f} dp={:.3f}".format(p_joint, p_somatic, p_dendritc))

    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-ld-event_types.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    # Percents

    df_pt_2 = pd.pivot_table(df,
                             values=['onset_index'],
                             index=['exp_id', 'pair_id'],
                             columns=[beutils.LIGHT_ON, 'type'],
                             aggfunc=np.size)

    n_joint_light = np.nan_to_num(df_pt_2["onset_index"][1][0].values)
    n_joint_dark = np.nan_to_num(df_pt_2["onset_index"][0][0].values)
    n_somatic_light = np.nan_to_num(df_pt_2["onset_index"][1][1].values)
    n_somatic_dark = np.nan_to_num(df_pt_2["onset_index"][0][1].values)
    n_dendritic_light = np.nan_to_num(df_pt_2["onset_index"][1][2].values)
    n_dendritic_dark = np.nan_to_num(df_pt_2["onset_index"][0][2].values)

    n_light = (n_joint_light + n_somatic_light + n_dendritic_light)
    n_dark = (n_joint_light + n_somatic_light + n_dendritic_light)
    n_total = n_light + n_dark

    per_joint_light = 100 * (n_joint_light / n_light)
    per_joint_dark = 100 * (n_joint_dark / n_dark)
    per_somatic_light = 100 * (n_somatic_light / n_light)
    per_somatic_dark = 100 * (n_somatic_dark / n_dark)
    per_dendritic_light = 100 * (n_dendritic_light / n_light)
    per_dendritic_dark = 100 * (n_dendritic_dark / n_dark)

    _, p_joint_per = scipy.stats.wilcoxon(per_joint_light, per_joint_dark)
    _, p_somatic_per = scipy.stats.wilcoxon(per_somatic_light, per_somatic_dark)
    _, p_dendritc_per = scipy.stats.wilcoxon(per_dendritic_light, per_dendritic_dark)

    print(p_joint_per, p_somatic_per, p_dendritc_per)

    plot_index += 1
    fig = plt.figure(tight_layout=True)
    plt.scatter(per_joint_light,
                per_joint_dark,
                label='joint')
    plt.scatter(per_somatic_light,
                per_somatic_dark,
                label='somatic')
    plt.scatter(per_dendritic_light,
                per_dendritic_dark,
                label='dendritic')
    plt.xlabel("% events light")
    plt.ylabel("% events dark")
    plt.legend()
    putils.square_plot(axmin=0)

    plt.title("jp={:.3f} sp={:.3f} dp={:.3f}".format(p_joint_per, p_somatic_per, p_dendritc_per))

    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-ld-event_types_per.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    # Bar charts comb

    per_joint = 100 * (n_joint_light + n_joint_dark) / n_total
    per_somatic = 100 * (n_somatic_light + n_somatic_dark) / n_total
    per_dendritic = 100 * (n_dendritic_light + n_dendritic_dark) / n_total

    per_joint_mean = np.mean(per_joint)
    per_somatic_mean = np.mean(per_somatic)
    per_dendritic_mean = np.mean(per_dendritic)

    per_joint_sem = scipy.stats.sem(per_joint)
    per_somatic_sem = scipy.stats.sem(per_somatic)
    per_dendritic_sem = scipy.stats.sem(per_dendritic)

    per_mean = [per_joint_mean, per_somatic_mean, per_dendritic_mean]
    per_sem = [per_joint_sem, per_somatic_sem, per_dendritic_sem]


    plot_index += 1
    fig = plt.figure(tight_layout=True)

    ind = np.arange(len(per_mean))  # the x locations for the groups
    width = 0.75  # the width of the bars

    plt.bar(ind, per_mean, width, yerr=per_sem)
    plt.legend()
    plt.gca().set_xticks(ind)
    plt.gca().set_xticklabels(('Joint', 'Somatic', 'Dendritic'))
    plt.ylabel("% events")
    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-comb-event_types_per_bar.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    # Bar charts by light dark
    per_joint_light_mean = np.mean(per_joint_light)
    per_joint_dark_mean = np.mean(per_joint_dark)
    per_somatic_light_mean = np.mean(per_somatic_light)
    per_somatic_dark_mean = np.mean(per_somatic_dark)
    per_dendritic_light_mean = np.mean(per_dendritic_light)
    per_dendritic_dark_mean = np.mean(per_dendritic_dark)

    per_joint_light_sem = scipy.stats.sem(per_joint_light)
    per_joint_dark_sem = scipy.stats.sem(per_joint_dark)
    per_somatic_light_sem = scipy.stats.sem(per_somatic_light)
    per_somatic_dark_sem = scipy.stats.sem(per_somatic_dark)
    per_dendritic_light_sem = scipy.stats.sem(per_dendritic_light)
    per_dendritic_dark_sem = scipy.stats.sem(per_dendritic_dark)

    # per_join_mean = [per_joint_light_mean, per_joint_dark_mean]
    # per_join_sem = [per_joint_light_sem, per_joint_dark_sem]
    # per_somatic_mean = [per_somatic_light_mean, per_somatic_dark_mean]
    # per_somatic_sem = [per_somatic_light_sem, per_somatic_dark_sem]
    # per_dendritc_mean = [per_dendritic_light_mean, per_dendritic_light_sem]
    # per_dendritc_sem = [per_dendritic_dark_mean, per_dendritic_dark_sem]

    per_light_mean = [per_joint_light_mean, per_somatic_light_mean, per_dendritic_light_mean]
    per_light_sem = [per_joint_light_sem, per_somatic_light_sem, per_dendritic_light_sem]
    per_dark_mean = [per_joint_dark_mean, per_somatic_dark_mean, per_dendritic_dark_mean]
    per_dark_sem = [per_joint_dark_sem, per_somatic_dark_sem, per_dendritic_dark_sem]

    plot_index += 1
    fig = plt.figure(tight_layout=True)

    ind = np.arange(len(per_light_mean))  # the x locations for the groups
    width = 0.35  # the width of the bars

    plt.bar(ind - width / 2, per_light_mean, width, yerr=per_light_sem, label='light', color=(0.5, 0.5, 0.5))
    plt.bar(ind + width / 2, per_dark_mean, width, yerr=per_dark_sem, label='dark', color='k')
    plt.legend()
    plt.gca().set_xticks(ind)
    plt.gca().set_xticklabels(('Joint', 'Somatic', 'Dendritic'))
    plt.ylabel("% events")
    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-ld-event_types_per_bar.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    plot_index += 1
    fig = plt.figure(tight_layout=True)
    plt.scatter(per_joint_light,
                per_joint_dark)
    plt.xlabel("% events light")
    plt.ylabel("% events dark")
    plt.legend()
    putils.square_plot(axmin=0)
    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-ld-event_types_per_joint.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    plot_index += 1
    fig = plt.figure(tight_layout=True)
    plt.scatter(per_somatic_light,
                per_somatic_dark)
    plt.xlabel("% events light")
    plt.ylabel("% events dark")
    plt.legend()
    putils.square_plot(axmin=0)
    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-ld-event_types_per_somatic.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    plot_index += 1
    fig = plt.figure(tight_layout=True)
    plt.scatter(per_dendritic_light,
                per_dendritic_dark)
    plt.xlabel("% events light")
    plt.ylabel("% events dark")
    plt.legend()
    putils.square_plot(axmin=0)
    plot_img_path = os.path.join(cfg.sum_agg_somadend_path, "{:04d}-ld-event_types_per_dendritic.png".format(plot_index))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

def sum_all(cfg: M2PConfig, dpi=300, nhistbins=30):
    # Load
    # df_pairs = pd.read_hdf(cfg.db_somadend_pairs_file)
    df_events = pd.read_hdf(cfg.db_somadend_events_file)
    df_events_pair = df_events.groupby(['exp_id', 'pair_id'])['amp1_norm', 'amp2_norm', 'amp1_raw', 'amp2_raw'].mean()

    # Quick check
    n_events = df_events.shape[0]
    n_animals = df_events['exp_id'].nunique()
    n_pairs = df_events['pair_id'].nunique()
    n_events_joint = np.sum(df_events['type'] == 0)
    n_events_soma = np.sum(df_events['type'] == 1)
    n_events_dend = np.sum(df_events['type'] == 2)

    print(n_events, n_animals, n_pairs, n_events_joint, n_events_soma, n_events_dend)

    df_grp_by = df_events.groupby(['exp_id', 'pair_id', 'type'])['onset_index'].aggregate("count")

    df_ct = pd.crosstab(index=[df_events['exp_id'], df_events['pair_id']],
                        columns=df_events['type'],
                        values=df_events['onset_index'],
                        aggfunc=np.size)



    # Plot each pair indvidualually.
    plot_index = 0
    plot_events_cell(df_events,
                     cfg.sum_agg_somadend_pairs_path,
                     plot_index=plot_index,
                     dpi=dpi,
                     nhistbins=nhistbins)

    # Plot each pair together.
    plot_index += 100
    plot_events(df_events,
                cfg.sum_agg_somadend_path,
                plot_index=plot_index,
                plot_label="all",
                dpi=dpi,
                nhistbins=nhistbins)

    # Plot the mean each pair.
    plot_index += 100
    plot_events(df_events_pair,
                cfg.sum_agg_somadend_path,
                plot_index=plot_index,
                plot_label="mean",
                dpi=dpi,
                nhistbins=10)

def plot_events_cell(df, plot_path, plot_index, dpi=300, nhistbins=30):


    pair_ids = df['pair_id'].unique()

    for pair_id in pair_ids:

        df_pair = df[df.pair_id == pair_id]

        amp_soma_raw = df_pair.amp1_raw
        amp_dend_raw = df_pair.amp2_raw

        plot_index = plot_event_somadend(plot_index,
                                         pair_id,
                                         "raw",
                                         amp_soma_raw,
                                         amp_dend_raw,
                                         plot_path,
                                         dpi,
                                         nhistbins,
                                         plot_somadend=True,
                                         plot_somadend_diff=False,
                                         plot_soma_diff=False,
                                         plot_soma_ratio=False)


def plot_events(df,
                plot_path,
                plot_index,
                plot_label,
                dpi=300,
                nhistbins=30,
                plot_somadend=True,
                plot_somadend_diff=True,
                plot_soma_diff=True,
                plot_soma_ratio=True):

    amp_soma_raw = df.amp1_raw
    amp_dend_raw = df.amp2_raw
    amp_soma_norm = df.amp1_norm
    amp_dend_norm = df.amp2_norm

    plot_index = plot_event_somadend(plot_index,
                                     plot_label,
                                     "raw",
                                     amp_soma_raw,
                                     amp_dend_raw,
                                     plot_path,
                                     dpi,
                                     nhistbins,
                                     plot_somadend,
                                     plot_somadend_diff,
                                     plot_soma_diff,
                                     plot_soma_ratio)

    plot_index = plot_event_somadend(plot_index,
                                     plot_label,
                                     "norm",
                                     amp_soma_norm,
                                     amp_dend_norm,
                                     plot_path,
                                     dpi,
                                     nhistbins,
                                     plot_somadend,
                                     plot_somadend_diff,
                                     plot_soma_diff,
                                     plot_soma_ratio)




def plot_event_somadend(plot_index,
                        plot_label,
                        ca_type,
                        amps_soma,
                        amps_dend,
                        plot_path,
                        dpi,
                        nhistbins,
                        plot_somadend=True,
                        plot_somadend_diff=True,
                        plot_soma_diff=True,
                        plot_soma_ratio=True):

    # Plot somatic vs dendritic amplitudes
    if ca_type == "raw":
        unit_str = "dF/F0"
        max_axis = np.max([np.nan_to_num(np.max(amps_soma)),
                           np.nan_to_num(np.max(amps_dend))]) * 1.1

    elif ca_type == "norm":
        unit_str = "norm"
        max_axis = 1

    amp_diff = amps_dend - amps_soma
    # Plus one so dF/F0 is 1 at baseline so we can do ratios.
    amp_ratio = np.log((amps_dend + 1) / (amps_soma + 1))

    diff_median = np.median(amp_diff)
    _, diff_p = scipy.stats.wilcoxon(amps_soma, amps_dend)
    corr_r, corr_p = scipy.stats.pearsonr(amps_soma, amps_dend)

    if plot_somadend:
        plot_index += 1
        fig = plt.figure(tight_layout=True)
        plt.scatter(amps_soma, amps_dend, color='blue')
        plt.xlabel("Somatic event amplitude ({})".format(unit_str))
        plt.ylabel("Dendritic event amplitude ({})".format(unit_str))

        plt.xlim(left=0, right=max_axis)
        plt.ylim(bottom=0, top=max_axis)
        plt.plot([0, max_axis], [0, max_axis], 'k')
        plt.gca().set_aspect('equal')

        plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r, corr_p, diff_median, diff_p))

        plot_img_path = os.path.join(plot_path, "{:04d}-{}-{}-soma-dend-amp.png".format(plot_index, plot_label, ca_type))
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    if plot_somadend_diff:
        plot_index += 1
        fig = plt.figure(tight_layout=True)
        plt.hist(amp_diff, bins=nhistbins)
        plt.xlabel("Dend-soma event amp diff ({})".format(unit_str))
        plt.ylabel("# pairs")
        x_max = np.max(np.abs(plt.gca().get_xlim()))
        plt.xlim(left=-x_max, right=x_max)
        plt.title("median={:0.2f} p={:.3f}".format(diff_median, diff_p))
        plot_img_path = os.path.join(plot_path, "{:04d}-{}-{}-soma-dend-amp-hist.png".format(plot_index, plot_label, ca_type))
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    if plot_soma_diff:
        plot_index = plot_event_somadiff(plot_index, plot_label, ca_type, "diff", amps_soma, amp_diff,
                                         plot_path, dpi)

    if plot_soma_ratio:
        plot_index = plot_event_somadiff(plot_index, plot_label, ca_type, "ratio", amps_soma, amp_ratio,
                                         plot_path, dpi)

    return plot_index

def plot_event_somadiff(plot_index, plot_label, ca_type, diff_type, amps_soma, dend_diff, plot_path, dpi):

    if ca_type == "raw":
        unit_str = "df/F0"
        max_axis = np.nan_to_num(np.max(amps_soma)) * 1.1
    elif ca_type == "norm":
        unit_str = "norm"
        max_axis = 1

    if diff_type == "diff":
        unit_diff_str = "(" + unit_str + ")"
    elif diff_type == "ratio":
        unit_diff_str = ""

    # Plot somatic vs dendritic amplitudes (normalised)
    corr_r, corr_p = scipy.stats.pearsonr(amps_soma, dend_diff)

    plot_index += 1
    fig = plt.figure(tight_layout=True)
    plt.scatter(amps_soma, dend_diff, color='blue')
    plt.xlabel("Somatic event amplitude ({})".format(unit_str))
    plt.ylabel("Dend/soma event amp {} {}".format(diff_type, unit_diff_str))

    plt.xlim(left=0, right=max_axis)

    plt.title("r={:.2f} p={:.3f}".format(corr_r, corr_p))

    plot_img_path = os.path.join(plot_path, "{:04d}-{}-{}-soma-amp-{}.png".format(plot_index, plot_label, ca_type, diff_type))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    return plot_index

def plot_event_somadiff2(plot_index, plot_label, ca_type, diff_type,
                         amps_soma1, dend_diff1, amps_soma2, dend_diff2,
                         color1, color2, plot_path, dpi):

    if ca_type == "raw":
        unit_str = "df/F0"
        max_axis = np.max([np.max(amps_soma1), np.max(amps_soma2)])  * 1.1
    elif ca_type == "norm":
        unit_str = "norm"
        max_axis = 1

    if diff_type == "diff":
        unit_diff_str = "(" + unit_str + ")"
    elif diff_type == "ratio":
        unit_diff_str = ""

    # Plot somatic vs dendritic amplitudes (normalised)
    corr_r1, corr_p1 = scipy.stats.pearsonr(amps_soma1, dend_diff1)

    plot_index += 1
    fig = plt.figure(tight_layout=True)
    plt.scatter(amps_soma1, dend_diff1, color=color1)
    plt.scatter(amps_soma2, dend_diff2, color=color2)
    plt.xlabel("Somatic event amplitude ({})".format(unit_str))
    plt.ylabel("Dend/soma event amp {} {}".format(diff_type, unit_diff_str))

    plt.xlim(left=0, right=max_axis)

    plt.title("r={:.2f} p={:.3f}".format(corr_r1, corr_p1))

    plot_img_path = os.path.join(plot_path, "{:04d}-{}-{}-soma-amp-{}.png".format(plot_index, plot_label, ca_type, diff_type))
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    return plot_index

def sum_all_behave(cfg: M2PConfig, dpi=300, nhistbins=30):
    # Load

    df_events = pd.read_hdf(cfg.db_somadend_events_file)
    df_behave = pd.read_hdf(cfg.db_behave_events_file)

    df = df_events.merge(df_behave, on=["exp_id", "pair_id", "onset_index", "offset_index"])

    df_events_pair = df.groupby(['exp_id', 'pair_id', beutils.LIGHT_ON])['amp1_norm', 'amp2_norm', 'amp1_raw', 'amp2_raw'].mean()

    # Seems that I need to use pivot table to crosstab multiple columns?
    df_events_ct = pd.pivot_table(df,
                                  values=['amp1_norm', 'amp2_norm', 'amp1_raw', 'amp2_raw'],
                                  index=['exp_id', 'pair_id'],
                                  columns=df[beutils.LIGHT_ON],
                                  aggfunc=np.mean)

    # Quick check
    n_events = df.shape[0]
    n_animals = df['exp_id'].nunique()
    n_pairs = df['pair_id'].nunique()

    print(n_events, n_animals, n_pairs)

    light_indexes = df[beutils.LIGHT_ON].values == 1
    dark_indexes = np.logical_not(light_indexes)

    move_indexes = df[beutils.SPEED_FILT_GRAD].values >= 1
    stat_indexes = np.logical_not(move_indexes)

    light_pair_indexes = df_events_pair.index.get_level_values(beutils.LIGHT_ON).values == 1
    dark_pair_indexes = np.logical_not(light_pair_indexes)

    # move_pair_indexes = df_events_pair.index.get_level_values(beutils.SPEED_FILT_GRAD).values >= 1
    # stat_pair_indexes = np.logical_not(move_pair_indexes)

    plot_index = 100
    plot_events2(df,
                 cfg.sum_agg_somadend_path,
                 plot_index=plot_index,
                 plot_label="all",
                 plot_type="ld",
                 indexes1=light_indexes,
                 indexes2=dark_indexes,
                 color1='gray',
                 color2='black',
                 dpi=dpi,
                 nhistbins=nhistbins)


    plot_index += 100
    plot_events2(df,
                 cfg.sum_agg_somadend_path,
                 plot_index=plot_index,
                 plot_label="all",
                 plot_type="move",
                 indexes1=move_indexes,
                 indexes2=stat_indexes,
                 color1='green',
                 color2='red',
                 dpi=dpi,
                 nhistbins=nhistbins)

    plot_index += 100
    plot_type = "ld"
    plot_events_cell2(df,
                   cfg.sum_agg_somadend_pairs_path,
                   plot_index,
                   plot_type,
                   light_indexes,
                   dark_indexes,
                   color1='gray',
                   color2='black',
                   dpi=300,
                   nhistbins=30)

    plot_index += 100
    plot_type = "move"
    plot_events_cell2(df,
                       cfg.sum_agg_somadend_pairs_path,
                       plot_index,
                       plot_type,
                       move_indexes,
                       stat_indexes,
                       color1='green',
                       color2='red',
                       dpi=300,
                       nhistbins=30)

    plot_index = 100
    plot_events2(df_events_pair,
                 cfg.sum_agg_somadend_path,
                 plot_index=plot_index,
                 plot_label="mean",
                 plot_type="ld",
                 indexes1=light_pair_indexes,
                 indexes2=dark_pair_indexes,
                 color1='gray',
                 color2='black',
                 dpi=dpi,
                 nhistbins=nhistbins,
                 plot_somadend_diff=True)

    # plot_index += 100
    # plot_events2(df_events_pair,
    #              cfg.sum_agg_somadend_path,
    #              plot_index=plot_index,
    #              plot_label="mean",
    #              plot_type="move",
    #              indexes1=move_pair_indexes,
    #              indexes2=stat_pair_indexes,
    #              color1='green',
    #              color2='red',
    #              dpi=dpi,
    #              nhistbins=nhistbins)



def plot_events_cell2(df, plot_path, plot_index, plot_type, indexes1, indexes2, color1, color2, dpi=300, nhistbins=30):


    pair_ids = df['pair_id'].unique()

    df1 = df.iloc[indexes1]
    df2 = df.iloc[indexes2]

    for pair_id in pair_ids:

        df_pair1 = df1[df1.pair_id == pair_id]
        df_pair2 = df2[df2.pair_id == pair_id]

        amp_soma_raw1 = df_pair1.amp1_raw
        amp_dend_raw1 = df_pair1.amp2_raw
        amp_soma_raw2 = df_pair2.amp1_raw
        amp_dend_raw2 = df_pair2.amp2_raw

        plot_index = plot_event_somadend2(plot_index,
                                          pair_id,
                                          plot_type,
                                          "raw",
                                          amp_soma_raw1,
                                          amp_dend_raw1,
                                          amp_soma_raw2,
                                          amp_dend_raw2,
                                          color1=color1,
                                          color2=color2,
                                          plot_path=plot_path,
                                          dpi=dpi,
                                          nhistbins=nhistbins,
                                          plot_somadend=True,
                                          plot_somadend_diff=False,
                                          plot_soma_diff=False,
                                          plot_soma_ratio=False)



    return plot_index

def plot_events2(df,
                 plot_path,
                 plot_index,
                 plot_label,
                 plot_type,
                 indexes1,
                 indexes2,
                 color1,
                 color2,
                 dpi=300,
                 nhistbins=30,
                 plot_somadend=True,
                 plot_somadend_diff=True,
                 plot_soma_diff=True,
                 plot_soma_ratio=True):

    df1 = df.iloc[indexes1]
    df2 = df.iloc[indexes2]

    amp_soma_raw1 = df1.amp1_raw
    amp_dend_raw1 = df1.amp2_raw
    amp_soma_norm1 = df1.amp1_norm
    amp_dend_norm1 = df1.amp2_norm

    amp_soma_raw2 = df2.amp1_raw
    amp_dend_raw2 = df2.amp2_raw
    amp_soma_norm2 = df2.amp1_norm
    amp_dend_norm2 = df2.amp2_norm

    plot_index = plot_event_somadend2(plot_index,
                                      plot_label,
                                      plot_type,
                                      "raw",
                                      amp_soma_raw1,
                                      amp_dend_raw1,
                                      amp_soma_raw2,
                                      amp_dend_raw2,
                                      color1,
                                      color2,
                                      plot_path,
                                      dpi,
                                      nhistbins,
                                      plot_somadend,
                                      plot_somadend_diff,
                                      plot_soma_diff,
                                      plot_soma_ratio)

    plot_index = plot_event_somadend2(plot_index,
                                      plot_label,
                                      plot_type,
                                      "norm",
                                      amp_soma_norm1,
                                      amp_dend_norm1,
                                      amp_soma_norm2,
                                      amp_dend_norm2,
                                      color1,
                                      color2,
                                      plot_path,
                                      dpi,
                                      nhistbins,
                                      plot_somadend,
                                      plot_somadend_diff,
                                      plot_soma_diff,
                                      plot_soma_ratio)

def plot_event_somadend2(plot_index,
                         plot_label,
                         plot_type,
                         ca_type,
                         amps_soma1,
                         amps_dend1,
                         amps_soma2,
                         amps_dend2,
                         color1,
                         color2,
                         plot_path,
                         dpi,
                         nhistbins,
                         plot_somadend=True,
                         plot_somadend_diff=True,
                         plot_soma_diff=True,
                         plot_soma_ratio=True):

    # Plot somatic vs dendritic amplitudes
    if ca_type == "raw":
        unit_str = "dF/F0"
        max_axis = np.max([np.nan_to_num(np.max(amps_soma1)),
                           np.nan_to_num(np.max(amps_dend1)),
                           np.nan_to_num(np.max(amps_dend2)),
                           np.nan_to_num(np.max(amps_dend2))]) * 1.1

    elif ca_type == "norm":
        unit_str = "norm"
        max_axis = 1

    amp_diff1 = amps_dend1 - amps_soma1
    amp_diff2 = amps_dend2 - amps_soma2
    # Plus one so dF/F0 is 1 at baseline so we can do ratios.
    amp_ratio1 = np.log((amps_dend1 + 1) / (amps_soma1 + 1))
    amp_ratio2 = np.log((amps_dend2 + 1) / (amps_soma2 + 1))

    diff1_median = np.median(amp_diff1)
    diff2_median = np.median(amp_diff2)

    if amps_soma1.empty or amps_dend1.empty:
        diff1_p = 1
        corr_r1 = 0
        corr_p1 = 1
    else:
        _, diff1_p = scipy.stats.wilcoxon(amps_soma1, amps_dend1)
        corr_r1, corr_p1 = scipy.stats.pearsonr(amps_soma1, amps_dend1)

    if amps_soma2.empty or amps_dend2.empty:
        diff2_p = 1
        corr_r2 = 0
        corr_p2 = 1
    else:
        _, diff2_p = scipy.stats.wilcoxon(amps_soma2, amps_dend2)
        corr_r2, corr_p2 = scipy.stats.pearsonr(amps_soma2, amps_dend2)



    if plot_somadend:
        plot_index += 1
        fig = plt.figure(tight_layout=True)
        plt.scatter(amps_soma1, amps_dend1, color=color1)
        plt.scatter(amps_soma2, amps_dend2, color=color2)
        plt.xlabel("Somatic event amplitude ({})".format(unit_str))
        plt.ylabel("Dendritic event amplitude ({})".format(unit_str))

        plt.xlim(left=0, right=max_axis)
        plt.ylim(bottom=0, top=max_axis)
        plt.plot([0, max_axis], [0, max_axis], 'k')
        plt.gca().set_aspect('equal')

        plt.title("r={:.2f} p={:.3f}; diffmed={:.2f} p={:.3f}".format(corr_r1, corr_p1, diff1_median, diff1_p))

        plot_img_path = os.path.join(plot_path, "{:04d}-{}-{}-{}-soma-dend-amp.png".format(plot_index, plot_label, plot_type, ca_type))
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

    if plot_somadend_diff:
        plot_index += 1
        fig = plt.figure(tight_layout=True)
        plt.hist(amp_diff1, bins=nhistbins)
        plt.hist(amp_diff2, bins=nhistbins)
        plt.xlabel("Dend-soma event amp diff ({})".format(unit_str))
        plt.ylabel("# pairs")
        x_max = np.max(np.abs(plt.gca().get_xlim()))
        plt.xlim(left=-x_max, right=x_max)
        plt.title("median={:0.2f} p={:.3f}".format(diff1_median, diff1_p))
        plot_img_path = os.path.join(plot_path, "{:04d}-{}-{}- {}-soma-dend-amp-hist.png".format(plot_index, plot_label, plot_type, ca_type))
        fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
        plt.cla()
        plt.clf()
        plt.close('all')

        if amp_diff1.size == amp_diff2.size:
            # Can only do scatter then there the same number of points, ie cell means.

            _, diff12_p = scipy.stats.wilcoxon(amp_diff1.values, amp_diff2.values)
            diff12_median = np.median(amp_diff1.values - amp_diff2.values)

            plot_index += 1
            fig = plt.figure(tight_layout=True)

            plt.scatter(amp_diff1, amp_diff2)
            plt.xlabel("Dend-soma event amp diff ({})".format(unit_str))
            plt.ylabel("Dend-soma event amp diff ({})".format(unit_str))
            putils.square_plot(diag=True)
            plt.title("median={:0.2f} p={:.3f}".format(diff12_median, diff12_p))
            plot_img_path = os.path.join(plot_path,
                                         "{:04d}-{}-{}- {}-soma-dend-amp.png".format(plot_index,
                                                                                     plot_label,
                                                                                     plot_type,
                                                                                     ca_type))
            fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
            plt.cla()
            plt.clf()
            plt.close('all')

    if plot_soma_diff:
        plot_index = plot_event_somadiff2(plot_index, plot_label, ca_type, "diff",
                                          amps_soma1, amp_diff1, amps_soma2, amp_diff2,
                                          color1='gray', color2='black',
                                          plot_path=plot_path, dpi=dpi)

    if plot_soma_ratio:
        plot_index = plot_event_somadiff2(plot_index, plot_label, ca_type, "ratio",
                                          amps_soma1, amp_ratio1, amps_soma2, amp_ratio2,
                                          color1='green', color2='red',
                                          plot_path=plot_path, dpi=dpi)

    return plot_index













# f = plt.figure(tight_layout=False)
# plt.plot([0, n_roi-0.5], [n_soma-0.5, n_soma-0.5], 'k', linewidth=2)
# plt.plot([n_soma-0.5, n_soma-0.5], [0, n_roi-0.5], 'k', linewidth=2)
# plt.imshow(mat_joint_events_per, vmin=0, vmax=1, cmap='Reds', aspect=1)
# plt.colorbar()
# plot_img_path = os.path.join(proc_data_path, "ca.joint.percent.mat.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# ass = np.triu(mat_joint_events_per).flatten()
# plt.hist(ass, bins=50)
# plot_img_path = os.path.join(proc_data_path, "ca.joint.percent.hist.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
#
# f = plt.figure(tight_layout=False)
# plt.plot([0, n_roi-0.5], [n_soma-1, n_soma-0.5], 'k', linewidth=2)
# plt.plot([n_soma-0.5, n_soma-0.5], [0, n_roi-0.5], 'k', linewidth=2)
# plt.imshow(mat_ca_pear_r, vmin=-1, vmax=1, cmap='seismic', aspect=1)
# plt.colorbar()
# plot_img_path = os.path.join(proc_data_path, "ca.corr.r.mat.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# plt.plot([0, n_roi], [n_soma-1, n_soma-0.5], 'k', linewidth=2)
# plt.plot([n_soma-0.5, n_soma-0.5], [0, n_roi-0.5], 'k', linewidth=2)
# plt.imshow(mat_ca_pear_p, vmin=0, vmax=1, aspect=1)
# plt.colorbar()
# plot_img_path = os.path.join(proc_data_path, "ca.corr.hd_p.mat.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# plt.plot([0, n_roi-0.5], [n_soma-0.5, n_soma-0.5], 'k', linewidth=2)
# plt.plot([n_soma-0.5, n_soma-0.5], [0, n_roi-0.5], 'k', linewidth=2)
# plt.imshow(mat_ca_act_pear_r, vmin=-1, vmax=1, cmap='seismic', aspect=1)
# plt.colorbar()
# plot_img_path = os.path.join(proc_data_path, "ca.corr.r.mat.act.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# plt.plot([0, n_roi], [n_soma-0.5, n_soma-0.5], 'k', linewidth=2)
# plt.plot([n_soma-0.5, n_soma-0.5], [0, n_roi-0.5], 'k', linewidth=2)
# plt.imshow(mat_ca_act_pear_p, vmin=0, vmax=1, aspect=1)
# plt.colorbar()
# plot_img_path = os.path.join(proc_data_path, "ca.corr.hd_p.mat.act.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# ass = np.triu(mat_ca_pear_r).flatten()
# plt.hist(ass, bins=20)
# plot_img_path = os.path.join(proc_data_path, "ca.corr.f.hist.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# ass = mat_ca.flatten()
# plt.hist(ass, bins=100)
# plot_img_path = os.path.join(proc_data_path, "ca.hist.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
#
# f = plt.figure(tight_layout=False)
# ass = np.std(mat_ca, axis=0)
# plt.hist(ass, bins=100)
# plot_img_path = os.path.join(proc_data_path, "ca.std.hist.act.png")
# f.savefig(plot_img_path, dpi=dpi, facecolor='white')
# plt.cla()
# plt.clf()
# plt.close('all')
