import pandas as pd
from utils import ca as cu, behave as bu, plot as pu
import numpy as np
from paths.config import M2PConfig
import matplotlib.pyplot as plt
import scipy



def sum_events(cfg: M2PConfig):

    dpi = 300

    df_ca = pd.read_hdf(cfg.db_ca_file)
    df_roi = pd.read_hdf(cfg.db_ca_roi_file)
    df_behave = pd.read_hdf(cfg.db_behave_frames_file)

    # Only use soma
    df_roi = df_roi.loc[df_roi["roi_type"] == "soma"]

    df_roi_ca = pd.merge(df_roi, df_ca, on=["exp_id", "roi_id"], how="inner")
    df_event = pd.merge(df_roi_ca, df_behave, on=["exp_id", "frame_id"], how="left")

    # For amplitude measurements only use rows where there is an amplitude.
    # This way you can easily calculate mean amplitudes.
    # For events, you need to leave all the non event rows in there so you can get events per minute.
    df_amp = df_event.loc[df_event["event_amp"] > 0]

    # Get the active periods for the event table
    act_event_indexes = bu.get_active_indexes(df_event)
    inact_event_indexes = np.logical_not(act_event_indexes)
    act_amp_indexes = bu.get_active_indexes(df_amp)
    inact_amp_indexes = np.logical_not(act_amp_indexes)

    # Get the light periods for the amp table
    light_event_indexes = df_event[bu.LIGHT_ON].values == 1
    dark_event_indexes = np.logical_not(light_event_indexes)
    light_amp_indexes = df_amp[bu.LIGHT_ON].values == 1
    dark_amp_indexes = np.logical_not(light_amp_indexes)

    # Todo hardcoded frame rate here
    scale_events_per_min = (1/0.1) * 60

    all_events = df_event.groupby(["exp_id", "roi_id"])["event_onset"].aggregate(["mean"])
    all_amp = df_amp.groupby(["exp_id", "roi_id"])["event_amp"].aggregate(["mean"])

    # Active vs inactive
    act_event = df_event.loc[act_event_indexes].groupby(["exp_id", "roi_id"])["event_onset"].aggregate(
        ["mean"])
    inact_event = df_event.loc[inact_event_indexes].groupby(["exp_id", "roi_id"])["event_onset"].aggregate(
        ["mean"])
    act_amp = df_amp.loc[act_amp_indexes].groupby(["exp_id", "roi_id"])["event_amp"].aggregate(
        ["mean"])
    inact_amp = df_amp.loc[inact_amp_indexes].groupby(["exp_id", "roi_id"])["event_amp"].aggregate(
        ["mean"])

    # Light vs dark
    light_event = df_event.loc[light_event_indexes].groupby(["exp_id", "roi_id"])["event_onset"].aggregate(
        ["mean"])
    dark_event = df_event.loc[dark_event_indexes].groupby(["exp_id", "roi_id"])["event_onset"].aggregate(
        ["mean"])
    light_amp = df_amp.loc[light_amp_indexes].groupby(["exp_id", "roi_id"])["event_amp"].aggregate(
        ["mean"])
    dark_amp = df_amp.loc[dark_amp_indexes].groupby(["exp_id", "roi_id"])["event_amp"].aggregate(
        ["mean"])

    # Outer join to all rois because sometimes an ROI doesn't have events in light/dark etc.
    act_event = pd.merge(df_roi, act_event, on=["exp_id", "roi_id"], how="left").fillna(0)
    inact_event = pd.merge(df_roi, inact_event, on=["exp_id", "roi_id"], how="left").fillna(0)
    light_event = pd.merge(df_roi, light_event, on=["exp_id", "roi_id"], how="left").fillna(0)
    dark_event = pd.merge(df_roi, dark_event, on=["exp_id", "roi_id"], how="left").fillna(0)

    act_amp = pd.merge(df_roi, act_amp, on=["exp_id", "roi_id"], how="left").fillna(0)
    inact_amp = pd.merge(df_roi, inact_amp, on=["exp_id", "roi_id"], how="left").fillna(0)
    light_amp = pd.merge(df_roi, light_amp, on=["exp_id", "roi_id"], how="left").fillna(0)
    dark_amp = pd.merge(df_roi, dark_amp, on=["exp_id", "roi_id"], how="left").fillna(0)

    fig = plt.figure(tight_layout=True)

    plt.scatter(all_events["mean"] * scale_events_per_min,
                all_amp["mean"])
    plt.xlabel("Events per minute")
    plt.ylabel("Mean event amplitude (dF/F0)")

    plot_img_path = cfg.sum_events_path / "event_amps.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    _, p_ld_event = scipy.stats.wilcoxon(light_event["mean"],
                                         dark_event["mean"])
    plt.scatter(light_event["mean"] * scale_events_per_min,
                dark_event["mean"] * scale_events_per_min)
    plt.xlabel("Light events/min")
    plt.ylabel("Dark events/min")
    pu.square_plot()

    plot_img_path = cfg.sum_events_path / "ld_events.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    _, p_ld_amp = scipy.stats.wilcoxon(light_amp["mean"],
                                       dark_amp["mean"])
    plt.scatter(light_amp["mean"],
                dark_amp["mean"])
    plt.xlabel("Light mean event amplitude (dF/F0)")
    plt.ylabel("Dark mean event amplitude (dF/F0)")
    pu.square_plot()

    plot_img_path = cfg.sum_events_path / "ld_amp.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)

    _, p_act_event = scipy.stats.wilcoxon(act_event["mean"],
                                      inact_event["mean"])
    plt.scatter(act_event["mean"] * scale_events_per_min,
                inact_event["mean"] * scale_events_per_min)
    plt.xlabel("Active events/min")
    plt.ylabel("Inactive events/min")
    pu.square_plot()

    plot_img_path = cfg.sum_events_path / "act_events.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')

    fig = plt.figure(tight_layout=True)
    _, p_act_amp = scipy.stats.wilcoxon(act_amp["mean"],
                                        inact_amp["mean"])
    plt.scatter(act_amp["mean"],
                inact_amp["mean"])
    plt.xlabel("Active mean event amplitude (dF/F0)")
    plt.ylabel("Inactive mean event amplitude (dF/F0)")
    pu.square_plot()

    print(p_act_event, p_act_amp, p_ld_event, p_ld_amp)

    plot_img_path = cfg.sum_events_path / "act_amp.png"
    fig.savefig(plot_img_path, dpi=dpi, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')
