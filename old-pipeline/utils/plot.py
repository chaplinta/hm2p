import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

COLOR_PENK = (0, 0, 1, 1)
COLOR_NONPENK = (1, 0, 0, 1)
LINEWIDTH_BOX_PLOT = 1.5

COLOR_SOMA = 'turquoise'
COLOR_DEND = 'darkorchid'

COLOUR_TUNE_CURVE = 'deepskyblue'

COLOUR_LIGHT = 'orange'
COLOUR_DARK = 'black'

COLOUR_ACTIVE = 'limegreen'
COLOUR_INACTIVE = 'saddlebrown'

COLOUR_EVENTS = 'midnightblue'

COLOR_DEC1 = (0.2, 0.6, 1.0, 1)
COLOR_DEC2 = (1.0, 0.4, 0.4, 1)



def square_plot(diag=True, axmin=None, axmax=None, diag_linewidth=1):
    xlims = plt.xlim()
    ylims = plt.ylim()
    if axmin is None:
        axmin = np.min([xlims[0], ylims[0]])
    if axmax is None:
        axmax = np.max([xlims[1], ylims[1]])
    plt.xlim(left=axmin, right=axmax)
    plt.ylim(bottom=axmin, top=axmax)
    if diag:
        plt.plot([axmin, axmax], [axmin, axmax], 'k', linewidth=diag_linewidth)
    plt.gca().set_aspect('equal')


def jitter_points(x, data, jitter_width):
    jitter = np.random.uniform(-jitter_width / 2, jitter_width / 2, len(data))
    return x + jitter


def plot_box_celltype(df_bp=None,
                      penk_indexes=None,
                      nonpenk_indexes=None,
                      col_name=None,
                      plot_label=None,
                      labels=["Penk", "Non-Penk"],
                      do_ranksums=False,
                      penk_data=None,
                      nonpenk_data=None,
                      create_new_figure=True,
                      show_plot=True):
    if create_new_figure:
        plt.figure()

    # Prepare data for the box plot
    if penk_data is None:
        penk_data = df_bp.loc[penk_indexes][col_name]
    if nonpenk_data is None:
        nonpenk_data = df_bp.loc[nonpenk_indexes][col_name]

    # Box plot
    bp = plt.boxplot([penk_data, nonpenk_data], positions=[1, 2], widths=0.6, patch_artist=True,
                     capprops=dict(linewidth=LINEWIDTH_BOX_PLOT),
                     whiskerprops=dict(linewidth=LINEWIDTH_BOX_PLOT),
                     showfliers=False,
                     medianprops=dict(linewidth=LINEWIDTH_BOX_PLOT))

    # Set box colors and make them unfilled
    for patch, color in zip(bp['boxes'], [COLOR_PENK, COLOR_NONPENK]):
        patch.set_facecolor('none')
        patch.set_edgecolor(color)
        patch.set_linewidth(LINEWIDTH_BOX_PLOT)

    # Set cap, whisker, and median colors
    for cap, whisker, color in zip(bp['caps'][::2], bp['whiskers'][::2], [COLOR_PENK, COLOR_NONPENK]):
        cap.set_color(color)
        whisker.set_color(color)

    for cap, whisker, color in zip(bp['caps'][1::2], bp['whiskers'][1::2], [COLOR_PENK, COLOR_NONPENK]):
        cap.set_color(color)
        whisker.set_color(color)

    for median, color in zip(bp['medians'], [COLOR_PENK, COLOR_NONPENK]):
        median.set_color(color)

    # Data points with jitter
    jitter_width = 0.3
    plt.scatter(jitter_points(1, penk_data, jitter_width), penk_data, color=COLOR_PENK, alpha=0.5)
    plt.scatter(jitter_points(2, nonpenk_data, jitter_width), nonpenk_data, color=COLOR_NONPENK, alpha=0.5)

    # Customize labels and axes
    plt.xticks([1, 2], labels)
    plt.ylabel(plot_label)

    if do_ranksums:
        stat, p_value = ranksums(penk_data, nonpenk_data)
        plt.title(f'Ranksum test: W={stat:.2f}, p={p_value:.3f}')

    # Show the plot
    if show_plot:
        plt.show()

    if do_ranksums:
        return p_value

def plot_scatter_celltype(df,
                            penk_indexes,
                            nonpenk_indexes,
                            col1,
                            col2,
                            plot_label="",
                            labels=["Penk", "Non-Penk"],
                            xlabel="", 
                            ylabel="",
                            do_ranksums=False):
    
    plt.figure()
    plt.scatter(df[penk_indexes][col1],
                df[penk_indexes][col2],
                color=COLOR_PENK,
                label=labels[0],)
    plt.scatter(df[nonpenk_indexes][col1],
                df[nonpenk_indexes][col2],
                color=COLOR_NONPENK,
                label=labels[1],)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_label)
    square_plot()       
