import pandas as pd
import os
import numpy as np
from paths.config import M2PConfig
import matplotlib
import matplotlib.pyplot as plt
import scipy
from utils import img as imutils, plot as pltutils, behave as beutils
from paths import config
from classes.ProcPath import ProcPath
from classes import Experiment, S2PData
from matplotlib.colors import ListedColormap

cfg = config.M2PConfig()

df_ca = pd.read_hdf(cfg.db_ca_file)
df_roi = pd.read_hdf(cfg.db_ca_roi_file)
df_events = pd.read_hdf(cfg.db_somadend_events_file)
df_behave_event_resample = pd.read_hdf(cfg.db_behave_events_file)
df_behave = pd.read_hdf(cfg.db_behave_file)


# Filter the data down to the soma and dendrites of interest
exp_id = "20211216_14_36_39_1115816"


roi_list = [(1, (122, 122, 122, 255)),
            (17, (255, 219, 83, 255)),
            (39, (210, 73, 73, 255)),
            (133, (29, 199, 29, 255)),
            (20, (248, 133, 54, 255)),
            (35, (168, 67, 0, 255)),
            (30, (116, 29, 116, 255))]



n_roi = len(roi_list)

roi_id_list = [i[0] for i in roi_list]

soma_id = roi_list[0][0]

m2p_paths = ProcPath(cfg, exp_id)
soma_data = S2PData.load_mode(m2p_paths.proc_s2p_path, "soma")

# ca_indexes = df_ca[(df_ca["exp_id"] == exp_id) &
#                    (df_ca["roi_id"].isin(roi_list))]
# event_indexes = df_events[(df_events["exp_id"] == exp_id) & (df_events["roi_index_1"] == soma_id)]

# df_ca = df_ca[df_ca.index.isin(ca_indexes)]
# df_events = df_events[df_events.index.isin(event_indexes)]

ca_indexes = (df_ca["exp_id"] == exp_id) & (df_ca["roi_id"].isin(roi_id_list))
event_indexes = (df_events["exp_id"] == exp_id) & (df_events["roi_index_1"] == soma_id)
behave_indexes = (df_behave["exp_id"] == exp_id)
df_ca = df_ca.loc[ca_indexes]
df_events = df_events.loc[event_indexes]
df_behave = df_behave.loc[behave_indexes]

df_event_behave = df_events.merge(df_behave_event_resample, on=["exp_id", "pair_id", "onset_index"], how='left')

# Setup directories
plot_dir = cfg.sum_path / "benoit"
rois_dir = plot_dir / "rois"
trace_long_dir = plot_dir / "trace-long"
trace_short_dir = plot_dir / "trace-short"
analysis_dir = plot_dir / "analysis"

plot_dir.mkdir(exist_ok=True)
rois_dir.mkdir(exist_ok=True)
trace_long_dir.mkdir(exist_ok=True)
trace_short_dir.mkdir(exist_ok=True)
analysis_dir.mkdir(exist_ok=True)

# Scatter plots
# Soma vs dend
fig = plt.figure(tight_layout=True)
for i_trace in range(len(roi_list)):
    if i_trace == 0:
        continue

    i_roi = roi_list[i_trace][0]
    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])
    roi_indexes = (df_events["roi_index_1"] == soma_id) & (df_events["roi_index_2"] == i_roi)
    plt.scatter(df_events.loc[roi_indexes]["amp1_norm"],
                df_events.loc[roi_indexes]["amp2_norm"],
                s=4,
                color=plot_colour)

pltutils.square_plot(axmin=0, axmax=1)
plt.xlabel("Somatic response (Norm. dF/F0)")
plt.ylabel("Dendritic response (Norm. dF/F0)")
plot_img_path = analysis_dir / "event_amps.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Local events percentage
fig = plt.figure(tight_layout=True)
roi_per_local = []
bar_colors = []
for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    if i_trace == 0:
        event_type = 1
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id)
    else:
        event_type = 2
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id) & (df_event_behave["roi_index_2"] == i_roi)

    local_indexes = roi_indexes & (df_event_behave["type"] == event_type)

    n_total = df_event_behave.loc[roi_indexes]["onset_index"].size
    n_local = np.sum(local_indexes)

    if i_trace == 0:
        # If it's the soma get the average number of events across dendrites.
        # Otherwise you have * n_dend too many
        n_total = n_total / (n_roi - 1)

    per_local = 0
    if n_total > 0:
        per_local = 100 * (n_local / n_total)
    print(n_total)

    roi_per_local.append(per_local)

    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])
    bar_colors.append(plot_colour)

plt.bar(np.arange(n_roi), roi_per_local, color=bar_colors)
plt.xticks([])

plt.ylabel("Local events (%)")
plot_img_path = analysis_dir / "event_local.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Inactive amplitudes

act_time_indexes = beutils.get_active_indexes(df_behave)
inact_time_indexes = np.logical_not(act_time_indexes)
act_mins = (np.sum(act_time_indexes) * 0.01) / 60
inact_mins = (np.sum(inact_time_indexes) * 0.01) / 60

light_time_indexes = (df_behave[beutils.LIGHT_ON] == 1) & act_time_indexes
dark_time_indexes = (df_behave[beutils.LIGHT_ON] == 0) & act_time_indexes
light_mins = (np.sum(light_time_indexes) * 0.01) / 60
dark_mins = (np.sum(dark_time_indexes) * 0.01) / 60

act_indexes = beutils.get_active_indexes(df_event_behave)
inact_indexes = np.logical_not(act_indexes)

light_indexes = (df_event_behave[beutils.LIGHT_ON] == 1) & act_indexes
dark_indexes = (df_event_behave[beutils.LIGHT_ON] == 0) & act_indexes
fig = plt.figure(tight_layout=True)
for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    if i_trace == 0:
        ampcol = "amp1_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id)
    else:
        ampcol = "amp2_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id) & (df_event_behave["roi_index_2"] == i_roi)


    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])


    act_mean = np.mean(df_event_behave.loc[roi_indexes & act_indexes][ampcol])
    inact_mean = np.mean(df_event_behave.loc[roi_indexes & inact_indexes][ampcol])
    plt.scatter(act_mean,
                inact_mean,
                color=plot_colour)

pltutils.square_plot()
plt.xlabel("Mean active response (dF/F0)")
plt.ylabel("Mean inactive response (dF/F0)")
plot_img_path = analysis_dir / "active_amps.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Inactive events
act_events_permin = []
inact_events_permin = []
for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    if i_trace == 0:
        ampcol = "amp1_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id)
    else:
        ampcol = "amp2_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id) & (df_event_behave["roi_index_2"] == i_roi)


    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])


    act_n = df_event_behave.loc[roi_indexes & act_indexes].shape[0]
    inact_n = df_event_behave.loc[roi_indexes & inact_indexes].shape[0]

    if i_trace == 0:
        act_n /= (n_roi - 1)
        inact_n /= (inact_n - 1)

    act_events_permin.append(act_n / act_mins)
    inact_events_permin.append(inact_n / inact_mins)


x = np.arange(len(act_events_permin))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, act_events_permin, width,  color=bar_colors)
rects2 = ax.bar(x + width/2, inact_events_permin, width, color=bar_colors, edgecolor='k', hatch="//")

# plt.bar(np.arange(n_roi), act_events_permin, color=bar_colors)
# plt.bar(np.arange(n_roi), inact_events_permin, color=bar_colors)
plt.xticks([])

plt.ylabel("Event rate (events/min)")
plot_img_path = analysis_dir / "active_events.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Inactive local events
act_events_permin = []
inact_events_permin = []
for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    if i_trace == 0:
        ampcol = "amp1_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id)& \
                      (df_event_behave["type"] == 1)
    else:
        ampcol = "amp2_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id) & \
                      (df_event_behave["roi_index_2"] == i_roi) & \
                      (df_event_behave["type"] == 2)


    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])


    act_n = df_event_behave.loc[roi_indexes & act_indexes].shape[0]
    inact_n = df_event_behave.loc[roi_indexes & inact_indexes].shape[0]

    if i_trace == 0:
        act_n /= (n_roi - 1)
        inact_n /= (inact_n - 1)

    act_events_permin.append(act_n / act_mins)
    inact_events_permin.append(inact_n / inact_mins)


x = np.arange(len(act_events_permin))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, act_events_permin, width,  color=bar_colors)
rects2 = ax.bar(x + width/2, inact_events_permin, width, color=bar_colors, edgecolor='k', hatch="//")

# plt.bar(np.arange(n_roi), act_events_permin, color=bar_colors)
# plt.bar(np.arange(n_roi), inact_events_permin, color=bar_colors)
plt.xticks([])

plt.ylabel("Local event rate (events/min)")
plot_img_path = analysis_dir / "active_events_local.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Light events
light_events_permin = []
dark_events_permin = []
for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    if i_trace == 0:
        ampcol = "amp1_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id) & \
                      (df_event_behave["type"] == 1)
    else:
        ampcol = "amp2_raw"
        roi_indexes = (df_event_behave["roi_index_1"] == soma_id) & \
                      (df_event_behave["roi_index_2"] == i_roi) & \
                      (df_event_behave["type"] == 2)


    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])


    light_n = df_event_behave.loc[roi_indexes & light_indexes].shape[0]
    dark_n = df_event_behave.loc[roi_indexes & dark_indexes].shape[0]

    if i_trace == 0:
        light_n /= (n_roi - 1)
        dark_n /= (inact_n - 1)

    light_events_permin.append(light_n / light_mins)
    dark_events_permin.append(dark_n / dark_mins)


x = np.arange(len(light_events_permin))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, light_events_permin, width,  color=bar_colors)
rects2 = ax.bar(x + width/2, dark_events_permin, width, color=bar_colors, edgecolor='black', hatch="*")

plt.xticks([])

plt.ylabel("Local event rate (events/min)")
plot_img_path = analysis_dir / "light_events.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Make image
s2p_img = soma_data.ops["meanImg"]

s2p_img = imutils.normalize_img(s2p_img, 0.1)

fig = plt.figure(tight_layout=False)
plt.imshow(s2p_img, cmap=plt.cm.gray, interpolation="none")
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#cell_colours = plt.cm.get_cmap('hsv', n_roi)


for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    cell_img = np.zeros((soma_data.ops['Ly'], soma_data.ops['Lx']))
    cell_img[soma_data.stat[i_roi]["ypix"], soma_data.stat[i_roi]["xpix"]] = 1

    cell_img_mask = np.ma.masked_where(cell_img == 0, cell_img)

    # neuy, neux = np.unravel_index(soma_data.stat[0]['neuropil_mask'], (soma_data.ops['Ly'], soma_data.ops['Lx']))
    # neumask = np.zeros((soma_data.ops['Ly'], soma_data.ops['Lx']))
    # neumask[neuy, neux] = 1
    #
    # neumask = np.ma.masked_where(neumask == 0, neumask)

    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])
    plt.imshow(cell_img_mask,
               interpolation="none", alpha=1,
               cmap=ListedColormap(plot_colour))

plt.gca().axis('off')

plot_img_path = rois_dir / "rois.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Plot long traces
print("Plotting long traces")
fig = plt.figure(tight_layout=True)
plt.gca().axis('off')
trace_offset_y = 1.1
for i_trace in range(len(roi_list)):

    i_roi = roi_list[i_trace][0]

    roi_ca_indexes = df_ca["roi_id"] == i_roi
    dfOnF = df_ca.loc[roi_ca_indexes]["dFonF0"]

    # dfOnF = soma_data.dFonF0[i_roi, :]

    dfOnF = dfOnF[:18000]
    dfOnF_norm = dfOnF / np.max(dfOnF)

    plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])
    plt.plot(dfOnF_norm.values + (n_roi - i_trace) * trace_offset_y, color=plot_colour)

scale_top = n_roi * trace_offset_y + 0.5
plt.plot([scale_top] * 10 * 60, 'k')
plt.plot([0, 0], [scale_top, scale_top + 1], 'k')

plot_img_path = trace_long_dir / "long.png"
fig.savefig(plot_img_path, dpi=300, facecolor='white')
plt.cla()
plt.clf()
plt.close('all')

# Plot short traces
print("Plotting short traces")
time_before = 5
time_after = 8
n_samps_before = round(time_before / 0.1)
n_samps_after = round(time_after / 0.1)
trace_offset_y = 0.5
max_short = 400

onsets = df_events["onset_index"].unique()
for i_event, event_onset in enumerate(onsets):

    if i_event >= max_short:
        break

    event_onset = int(event_onset)
    fig = plt.figure(tight_layout=True)
    plt.gca().axis('off')

    i_start = event_onset - n_samps_before
    i_end = event_onset + n_samps_after
    if i_start < 0:
        continue

    for i_trace in range(len(roi_list)):

        i_roi = roi_list[i_trace][0]

        roi_ca_indexes = df_ca["roi_id"] == i_roi
        dfOnF = df_ca.loc[roi_ca_indexes]["dFonF0"]
        dfOnF_norm = dfOnF / np.max(dfOnF)

        dfOnF_norm = dfOnF_norm[i_start:i_end]

        plot_colour = tuple((ti / 255) for ti in roi_list[i_trace][1])
        plt.plot(dfOnF_norm.values + (n_roi - i_trace) * trace_offset_y, color=plot_colour)

    scale_top = n_roi * trace_offset_y + 0.2
    plt.plot([scale_top] * 10, 'k')
    plt.plot([0, 0], [scale_top, scale_top + 0.5], 'k')

    plot_img_path = trace_short_dir / "{:05d}-event{:05d}.png".format(i_event, event_onset)
    fig.savefig(plot_img_path, dpi=300, facecolor='white')
    plt.cla()
    plt.clf()
    plt.close('all')





print("Invivo Done")