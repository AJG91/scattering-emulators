import numpy as np

import matplotlib as mpl
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


BASIS_KWARGS = dict(
    # c="lightgrey",  # I think light grey == '0.8'
    c="0.785",
    # lw=0.8,
    lw=0.84,
    zorder=-0.1,
)
PRED_KWARGS = dict(
    dash_capstyle="round",
    linestyle=(0, (0.1, 2.7)),
    lw=1.9,
    zorder=1.5,
    # path_effects=[mpe.Stroke(linewidth=2.6, foreground="k"), mpe.Normal()],
    path_effects=[mpe.withStroke(linewidth=2.6, foreground="k")],
)
FULL_KWARGS = dict(lw=0.5, c="k", zorder=1)
ZERO_KWARGS = dict(c="k", lw=0.8, zorder=0)


def setup_rc_params(presentation=False, constrained_layout=True, usetex=True, dpi=400):
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = "k"

    mpl.rcdefaults()  # Set to defaults

    # mpl.rc("text", usetex=True)
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["text.usetex"] = usetex
    # mpl.rcParams["text.latex.preview"] = True
    mpl.rcParams["font.family"] = "serif"

    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["axes.edgecolor"] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams["axes.labelcolor"] = black
    mpl.rcParams["axes.titlesize"] = fontsize

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["xtick.color"] = black
    mpl.rcParams["ytick.color"] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams["xtick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams["xtick.minor.size"] = 2.4  # Default 2.0
    mpl.rcParams["ytick.minor.size"] = 2.4
    mpl.rcParams["xtick.major.size"] = 3.9  # Default 3.5
    mpl.rcParams["ytick.major.size"] = 3.9

    ppi = 72  # points per inch
    # dpi = 150
    mpl.rcParams["figure.titlesize"] = fontsize
    mpl.rcParams["figure.dpi"] = 150  # To show up reasonably in notebooks
    mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams["figure.constrained_layout.wspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.hspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.h_pad"] = 3.0 / ppi  # 3 points
    mpl.rcParams["figure.constrained_layout.w_pad"] = 3.0 / ppi

    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"

    mpl.rcParams["legend.title_fontsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams[
        "legend.edgecolor"
    ] = "inherit"  # inherits from axes.edgecolor, to match
    mpl.rcParams["legend.facecolor"] = (
        1,
        1,
        1,
        0.6,
    )  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.borderaxespad"] = 0.8
    mpl.rcParams[
        "legend.framealpha"
    ] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams[
        "patch.linewidth"
    ] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams["hatch.linewidth"] = 0.5

    # bbox = 'tight' can distort the figure size when saved (that's its purpose).
    # mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.04, dpi=350, format='png')
    mpl.rc("savefig", transparent=False, bbox=None, dpi=dpi, format="png")


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def darken_color(color, amount=0.5):
    """
    Darken the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    return lighten_color(color, 1.0 / amount)


def plot_emulator_results(emulator, lecs, t_lab, nn_online=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
    phase_train = emulator.phase_train

    phase_pred = emulator.predict(lecs, return_phase=True, full_space=False)
    phase_full = emulator.predict(lecs, return_phase=True, full_space=True)

    ax.plot(t_lab, phase_train[:, 0], label="Basis", **BASIS_KWARGS)
    ax.plot(t_lab, phase_train[:, 1:], **BASIS_KWARGS)
    ax.plot(t_lab, phase_pred, label="EC", **PRED_KWARGS)
    ax.plot(t_lab, phase_full, label="Exact", **FULL_KWARGS)
    if nn_online is not None:
        ax.plot(t_lab, nn_online, label="PWA-93", zorder=0.1)
    ax.axhline(0, 0, 1, **ZERO_KWARGS)
    ax.legend()
    ax.set_ylabel(r"$\delta$ [deg]")
    ax.set_xlabel(r"$E_{\mathrm{lab}}$ [MeV]")
    return ax


def plot_coupled_emulator_results(
    emulator, lecs, t_lab, axes=None, clip_upper=None, clip_lower=None
):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    else:
        fig = None

    K_pred = emulator.predict(lecs, full_space=False, return_phase=False)
    K_full = emulator.predict(lecs, full_space=True, return_phase=False)
    K_train = emulator.K_on_shell_train.copy()

    if clip_upper is not None:
        K_full[K_full >= clip_upper] = np.inf
        K_pred[K_pred >= clip_upper] = np.inf
        K_train[K_train >= clip_upper] = np.inf

    if clip_lower is not None:
        K_full[K_full <= clip_lower] = -np.inf
        K_pred[K_pred <= clip_lower] = -np.inf
        K_train[K_train <= clip_lower] = -np.inf

    axes[0].plot(t_lab, K_full[:, 0, 0], **FULL_KWARGS)
    axes[0].plot(t_lab, K_pred[:, 0, 0], **PRED_KWARGS)
    axes[0].plot(t_lab, K_train[:, 0, 0], **BASIS_KWARGS)

    axes[1].plot(t_lab, K_full[:, 1, 1], **FULL_KWARGS)
    axes[1].plot(t_lab, K_pred[:, 1, 1], **PRED_KWARGS)
    axes[1].plot(t_lab, K_train[:, 1, 1], **BASIS_KWARGS)

    axes[2].plot(t_lab, K_train[:, 1, 0, 0], label="Train", **BASIS_KWARGS)
    axes[2].plot(t_lab, K_full[:, 1, 0], label="Exact", **FULL_KWARGS)
    axes[2].plot(t_lab, K_pred[:, 1, 0], label="Emulator", **PRED_KWARGS)

    axes[2].plot(t_lab, K_train[:, 1, 0], **BASIS_KWARGS)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.03, 1), borderaxespad=0)

    labels = [r"$K_{++}$", r"$K_{\vphantom{+} -- \phantom{+}}$", r"$K_{+-}$"]

    for i, ax in enumerate(axes.ravel()):
        ax.set_xlabel(r"$E_{\mathrm{lab}}$ (MeV)")
        ax.axhline(0, 0, 1, **ZERO_KWARGS)
        ax.text(
            0.93,
            # 0.09,
            0.905,
            s=labels[i],
            ha="right",
            va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="w"),
        )
    return fig, axes


def plot_coupled_emulator_residuals(emulator, lecs, t_lab, axes):
    K_pred = emulator.predict(lecs, full_space=False, return_phase=False)
    K_full = emulator.predict(lecs, full_space=True, return_phase=False)
    resid = np.abs(K_full - K_pred)
    # resid = np.abs(K_full - K_pred) / np.where(np.abs(K_full) < 1e-13, 1e-13, np.abs(K_full))

    axes[0].semilogy(t_lab, resid[:, 0, 0], zorder=1)
    axes[1].semilogy(t_lab, resid[:, 1, 1], zorder=1)
    (line,) = axes[2].semilogy(t_lab, resid[:, 1, 0], zorder=1, label=r"Abs.\\Residual")
    axes[2].legend(
        [line],
        ["Abs.\nResidual"],
        loc="upper left",
        bbox_to_anchor=(1.03, 1),
        borderaxespad=0,
        # ma='left',
        # handlelength=1.4, handletextpad=0.5,
    )

    for i, ax in enumerate(axes.ravel()):
        ax.set_xlabel(r"$E_{\mathrm{lab}}$ [MeV]")
        ax.tick_params(which="both", axis="both", top=True, right=True)
        if i > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
    return axes


def plot_coupled_emulator_results_with_residuals(
    emulator,
    lecs,
    t_lab,
    clip_upper=None,
    clip_lower=None,
    emulator_resid=None,
    t_lab_resid=None,
):

    #     fig, axes = plt.subplots(2, 3, figsize=(7, 2.5))
    fig = plt.figure(figsize=(7, 2.3), constrained_layout=True)
    widths = [1, 1, 1]
    heights = [2, 1]
    spec = fig.add_gridspec(
        ncols=3, nrows=2, width_ratios=widths, height_ratios=heights,
    )
    ax0_pred = fig.add_subplot(spec[0, 0])
    ax1_pred = fig.add_subplot(spec[0, 1], sharey=ax0_pred)
    ax2_pred = fig.add_subplot(spec[0, 2], sharey=ax0_pred)
    axes_pred = np.array([ax0_pred, ax1_pred, ax2_pred])

    ax0_resid = fig.add_subplot(spec[1, 0], sharex=ax0_pred)
    ax1_resid = fig.add_subplot(spec[1, 1], sharey=ax0_resid, sharex=ax1_pred)
    ax2_resid = fig.add_subplot(spec[1, 2], sharey=ax0_resid, sharex=ax2_pred)
    axes_resid = np.array([ax0_resid, ax1_resid, ax2_resid])
    axes = np.stack([axes_pred, axes_resid])

    plot_coupled_emulator_results(
        emulator,
        lecs,
        t_lab,
        axes=axes_pred,
        clip_upper=clip_upper,
        clip_lower=clip_lower,
    )

    for i, ax in enumerate(axes_pred.ravel()):
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both", axis="both", top=True, right=True)
        if i > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

            ax.set_xlabel("")

    if emulator_resid is None:
        emulator_resid = emulator
    if t_lab_resid is None:
        t_lab_resid = t_lab
    plot_coupled_emulator_residuals(
        emulator_resid, lecs, t_lab=t_lab_resid, axes=axes_resid
    )
    # axes_resid[-1].set_ylabel(r"Abs.\ Residual", rotation="horizontal", ha="left")
    # axes_resid[-1].legend(
    #     loc="upper left", bbox_to_anchor=(1.03, 1), borderaxespad=0,
    #     # ma='left',
    #     # handlelength=1.4, handletextpad=0.5,
    # )
    axes_resid[-1].yaxis.set_label_position("right")
    return fig, axes


def plot_cross_section_with_residual(
    t_lab,
    y_full,
    y_pred,
    y_full_samples,
    y_pred_samples,
    ax=None,
    ncol=1,
    loc=None,
    borderaxespad=None,
    bbox_to_anchor=None,
):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 3))

    # ax.semilogy(t_lab, sgt_nn_online, c='k', label='PWA-93')
    ax.semilogy(t_lab, y_pred, label="Emulator", **PRED_KWARGS)
    ax.plot(t_lab, y_full, label="Exact", **FULL_KWARGS)
    ax.axhline(0, 0, 1, **ZERO_KWARGS)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 1),
        ncol=3,
        #     columnspacing=1, handlelength=1.5, handletextpad=0.5,
        borderaxespad=0.0,
        mode="expand",
    )
    # ax.set_title("Cross Section")
    ax.set_ylabel(r"$\sigma_{\mathrm{tot}}$ [mb]")
    ax.set_xlabel(r"$E_{\mathrm{lab}}$ [MeV]")

    # Create inset of width 30% and height 40% of the parent axes' bounding box
    # at the lower left corner (loc=3)
    ax2 = inset_axes(ax, width="60%", height="65%", loc="upper right")
    ax2.semilogy(
        t_lab,
        np.mean(np.abs(y_pred_samples - y_full_samples), axis=0),
        c="C3",
        ls="-",
        label="Sampled",
    )
    ax2.semilogy(
        t_lab, np.abs(y_pred - y_full), c="C0", ls="-", label="Best",
    )
    ax2.legend(
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1,
        borderaxespad=borderaxespad,
        ncol=ncol,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
    )
    ax2.tick_params(right=True)

    ax2.axhline(0, 0, 1, c="k", lw=0.8, zorder=0)
    ax2.set_ylabel(r"Abs.\ Residual [mb]")
    ax2.set_xlabel(r"$E_{\mathrm{lab}}$ [MeV]")
    # ax.set_title("Cross Section Simulator Absolute Residual")
    return ax, ax2
