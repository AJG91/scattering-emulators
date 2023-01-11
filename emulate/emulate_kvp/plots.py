"""
File used for plotting.
"""
import matplotlib.pyplot as plt
from numpy import (
    append, unique, arctan, pi, stack,
    degrees, array, concatenate, median,
    delete, zeros, asarray, swapaxes,
    minimum, concatenate, inf
)
from numpy.typing import ArrayLike
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as mpe
from .utils import fix_phases_continuity
from typing import Optional

zero_kwargs = dict(c='black', lw=0.8, zorder=0)
basis_kwargs = dict(c='grey', lw=0.5, zorder=-0.5, alpha=0.3)
sim_kwargs = dict(ls='solid', lw=1, zorder=1)
emu_kwargs = dict(dash_capstyle="round", ls=(0, (0.1, 2)), zorder=2, lw=3, 
                  path_effects=[mpe.withStroke(linewidth=4, foreground="k")])

emu_color = 'lightsalmon'
sim_color = 'black'

emu_label = 'Emulator'
sim_label = 'Simulator'
basis_label = 'Basis'

def setup_rc_params(presentation=False, constrained_layout=True, usetex=True, dpi=400):
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = "k"

    mpl.rcdefaults()  # Set to defaults
    x_minor_tick_size = y_minor_tick_size = 2.4
    x_major_tick_size = y_major_tick_size = 3.9

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
    mpl.rcParams["xtick.minor.size"] = x_minor_tick_size  # Default 2.0
    mpl.rcParams["ytick.minor.size"] = y_minor_tick_size
    mpl.rcParams["xtick.major.size"] = x_major_tick_size  # Default 3.5
    mpl.rcParams["ytick.major.size"] = y_major_tick_size
    plt.rcParams["xtick.minor.visible"] =  True
    plt.rcParams["ytick.minor.visible"] =  True

    ppi = 72  # points per inch
    mpl.rcParams["figure.titlesize"] = fontsize
    mpl.rcParams["figure.dpi"] = 150  # To show up reasonably in notebooks
    mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams["figure.constrained_layout.wspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.hspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.h_pad"] = 0#3.0 / ppi
    mpl.rcParams["figure.constrained_layout.w_pad"] = 0#3.0 / ppi

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
    
    return None

def plot_results(
    E: ArrayLike,
    emulator, 
    sim_results: list, 
    emu_results: list, 
    cutoff: int,
    error: str = 'Rel.',
    dpi: int = 800
) -> None:
    """
    Wrapper for coupled or uncoupled channel functions plotting functions.

    Parameters
    ----------
    E : array
        Energy grid.
    emulator : instance
        Specific emulator instance.
    sim_results : ArrayLike
        Simulator output.
    emu_results : ArrayLike
        Emulator predictions.
    cutoff : int    
        EFT cutoff Lambda.
    error : str
        Specifies relative or absolute error.
        Default: 'Rel.'
    dpi : int
        Dots per inch.
        Used for file name output.
        Default: 800

    Returns
    -------
    None
    """
    wave = emulator.wave
    K0_b = swapaxes(emulator.K0_b, 0, 1)
        
    wave = wave.replace("/", "-")
    path = './plots/' + str(cutoff) + 'MeV/' + 'SMS_Chiral_' + str(cutoff) + 'MeV_' + wave
    
    if emulator.is_coupled:
        plot_coupled_errors(E, K0_b, sim_results, emu_results, 
                            path, error, dpi)
            
    else:
        plot_uncoupled_errors(E, K0_b, sim_results, emu_results, 
                              path, error, dpi)
    
    return None

def compute_phase_shifts(
    K0: ArrayLike
) -> ArrayLike:
    """  
    Calculates the phase shifts for the uncoupled channel.
    
    Parameters
    ----------
    K0 : array
        On-shell K matrix.

    Returns
    -------
    ps : array
        Phase shifts
    """
    phase = arctan(K0) * 180 / pi
    return fix_phases_continuity(phase, is_radians=False)

def compute_error(
    A: ArrayLike, 
    B: ArrayLike, 
    error: str
) -> ArrayLike:
    """
    Used to calculate relative or absolute error.

    Parameters
    ----------
    A : array
        Expected values.
    B : array
        Actual values observed.
    error : str
        Used to choose between relative or absolute errors.
        'Rel.' == relative errors
        'Abs.' == absolute errors

    Returns
    -------
    err : array
        Errors
    """
    if error == 'Rel.':
        return 2 * abs(A - B) / (abs(A) + abs(B))
    elif error == 'Abs.':
        return abs(A - B)
    else:
        raise ValueError('Check error input!')

def plot_uncoupled_errors(
    E: ArrayLike,
    K0_b: ArrayLike,
    K0_sim: ArrayLike,
    K0_emu: ArrayLike,
    path: str,
    error: str,
    dpi: int
) -> None:
    """
    Plots the phase shifts of the training points, testing points, and prediction
    for the non-coupled channel.

    Parameters
    ----------
    E : array
        Energy grid.
    K0_b : array
        Training points on-shell K matrices.
    K0_sim : array
        Testing points on-shell K matrices.
    K0_emu : array
        Predictions of on-shell K matrices.
    path : str
        Path for file output.
    error : str
        Specifies relative or absolute error.
    dpi : int
        Dots per inch.
        Used for file name output.
        Default: 800
    
    Returns
    -------
    None
    """
    shape_test = array(K0_sim).shape[0]
    
    if shape_test == 3:
        K0_emu_nvp = array(K0_emu[2])
        K0_sim_nvp = array(K0_sim[2])
        
        K0_emu = array([K0_emu[0], K0_emu[1]])
        K0_sim = array([K0_sim[0], K0_sim[1]])
        
        phase_shifts_emu_nvp = compute_phase_shifts(K0_emu_nvp)
        phase_shifts_sim_nvp = compute_phase_shifts(K0_sim_nvp)

        min_err_nvp = compute_error(phase_shifts_sim_nvp, 
                                    phase_shifts_emu_nvp, 
                                    error=error)
        
    else:
        K0_emu = array(K0_emu)
        K0_sim = array(K0_sim)
    
    phase_shifts_emu = compute_phase_shifts(K0_emu)
    phase_shifts_sim = compute_phase_shifts(K0_sim)
    
    min_err = compute_error(phase_shifts_sim, phase_shifts_emu, error=error)
    
    if shape_test == 3:
        min_err = concatenate((min_err, min_err_nvp[None, :]), axis=0)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(3.4, 3.2), nrows=2, sharex=True)
    minorLocator = MultipleLocator(20)
    ax1.axhline(0, 0, 1, **zero_kwargs)
            
    for K0_i in K0_b:
        ax1.plot(E, compute_phase_shifts(K0_i), **basis_kwargs)

    ax1.plot([], [], label=basis_label, **basis_kwargs)
    ax1.plot(E, phase_shifts_sim[1], c=sim_color, label=sim_label, **sim_kwargs)
    ax1.plot(E, phase_shifts_emu[1], c=emu_color, label=emu_label, **emu_kwargs)
    ax1.tick_params(axis='y')
    ax1.set_ylabel(r'$\delta$ [deg]')
    ax1.set_xlim(E[0] - 7, E[-1] + 7)
    ax1.set_ylim(-100, 250)
    ax1.legend(bbox_to_anchor=(0.0, 0.05, 1.02, 1), handlelength=1.5, 
               handletextpad=0.5, columnspacing=1, ncol=3, framealpha=0.6)
    
    if shape_test == 2:
        plt_labels = [r'Gl$\mathrm{\ddot{o}}$ckle', f'Standard']
        plt_colors = ['red', 'blue']
        plt_ls = ['dashed', 'solid']
        
    elif shape_test == 3:
        plt_labels = [r'Gl$\mathrm{\ddot{o}}$ckle', f'Standard', f'NVP']
        plt_colors = ['red', 'blue', 'green']
        plt_ls = ['dashed', 'solid', 'dotted']
        
    else:
        raise ValueError('Check input!')
        
    for i, (err_i, lb_i, c_i, ls_i) in enumerate(zip(min_err, plt_labels, plt_colors, plt_ls)):
        ax2.semilogy(E, err_i, color=c_i, ls=ls_i, label=lb_i)
        
    ax2.set_xlabel(r'$E_{\mathrm{lab}}$ [MeV]')
    ax2.set_ylabel(error + ' ' + 'Error')
    ax2.set_xlim(E[0] - 7, E[-1] + 7)
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.xaxis.set_minor_locator(minorLocator)
    ax2.set_ylim(1e-16, 1e1)
    ax2.legend(bbox_to_anchor=(0.0, 0.05, 1.02, 1), handlelength=1.5, 
               handletextpad=0.5, columnspacing=1, ncol=shape_test, framealpha=0.6)
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + '.png', dpi=dpi)
    
    return None

def plot_coupled_errors(
    E: ArrayLike, 
    K0_b: ArrayLike, 
    K0_sim: ArrayLike, 
    K0_emu: ArrayLike, 
    path: str,
    error: str,
    dpi: int
) -> None:
    """
    Plots the phase shifts of the training points, testing points, and prediction
    for the coupled channel.

    Parameters
    ----------
    E : array
        Energy grid.
    K0_b : array
        Training points on-shell K matrices.
    K0_sim : array
        Testing points on-shell K matrices.
    K0_emu : array
        Predictions of on-shell K matrices.
    path : str
        Path for file output.
    error : str
        Specifies if we are using relative or absolute error.
        Default: 'Rel.'
    dpi : int
        Dots per inch.
        Used for file name output.
        Default: 800
    
    Returns
    -------
    None
    """
    clip_upper, clip_lower = 1.8, -1.8
    shape_test, len_k = array(K0_sim).shape[0:2]
    
    if shape_test == 3:
        K0_emu_nvp = array(K0_emu[2])
        K0_sim_nvp = array(K0_sim[2])
        
        K0_sim_nvp = swapaxes(delete(K0_sim_nvp, 2, 1), 1, -1)
        K0_sim_nvp[:, [1, 2]] = K0_sim_nvp[:, [2, 1]]
        K0_emu_nvp = swapaxes(delete(K0_emu_nvp, 2, 1), 1, -1)
        K0_emu_nvp[:, [1, 2]] = K0_emu_nvp[:, [2, 1]]
        
        K0_emu_arr = array([K0_emu[0], K0_emu[1]])
        K0_sim_arr = array([K0_sim[0], K0_sim[1]])
        
        min_err_nvp = swapaxes(compute_error(K0_sim_nvp, K0_emu_nvp, error=error), 0, 1)
        
    else:
        K0_emu_arr = array(K0_emu)
        K0_sim_arr = array(K0_sim)
    
    K0_emu = K0_emu_arr
    K0_sim = swapaxes(delete(K0_sim_arr, 2, 2), 1, -1)
    K0_sim[:, [1, 2]] = K0_sim[:, [2, 1]]
    K0_emu[:, [1, 2]] = K0_emu[:, [2, 1]]
    min_err = compute_error(K0_sim, K0_emu, error=error)
                
    if shape_test == 3:
        min_err = concatenate((min_err, min_err_nvp[None, :]), axis=0)
    
    fig, ax = plt.subplots(2, 3, figsize=(7, 2.3), sharex='col', sharey='row')
    minorLocator = MultipleLocator(20)

    if clip_upper is not None:
        K0_sim[K0_sim >= clip_upper] = inf
        K0_emu[K0_emu >= clip_upper] = inf
        K0_b[K0_b >= clip_upper] = inf

    if clip_lower is not None:
        K0_sim[K0_sim <= clip_lower] = -inf
        K0_emu[K0_emu <= clip_lower] = -inf
        K0_b[K0_b <= clip_lower] = -inf
    
    for K0_i in K0_b:
        K0_i = delete(swapaxes(K0_i.reshape(4, len_k, order='F'), 0, 1).reshape(4, len_k), 1, 0)
        K0_i[[1, 2], :] = K0_i[[2, 1], :]
          
        for j in range(len(K0_i)):
            ax[0][j].plot(E, K0_i[j], **basis_kwargs)

    for i in range(len(K0_i)):
        ax[0][i].axhline(0, 0, 1, **zero_kwargs)
        ax[0][i].plot(E, K0_sim[1][i], c=sim_color, **sim_kwargs)
        ax[0][i].plot(E, K0_emu[1][i], c=emu_color, **emu_kwargs)
        ax[0][i].set_xlim(E[0] - 7, E[-1] + 7)
        ax[0][i].tick_params(axis='y')
        
    if '3S1' in path:
        ax[0][0].set_ylim(-1.2, 1.2)
        ax[0][1].set_ylim(-1.2, 1.2)
        ax[0][2].set_ylim(-1.2, 1.2)
    
    ax[0][0].set_ylabel(r'$K(k_0)$')
    labels = [r"$K_{++}$", r"$K_{--}$", r"$K_{+-}$"]  
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    for i, ax0 in enumerate(ax[0].ravel()):
        ax0.text(0.8, 0.8, labels[i], transform=ax0.transAxes, 
                 verticalalignment='bottom', bbox=props)
        
    ax[0][2].plot([], [], label=basis_label, **basis_kwargs)
    ax[0][2].plot([], [], c=sim_color, label=sim_label, **sim_kwargs)
    ax[0][2].plot([], [], c=emu_color, label=emu_label, **emu_kwargs)
    ax[0][2].legend(bbox_to_anchor=(0.815, 0.705, 0.8, 0.4), #handletextpad=0.3, #numpoints=10, 
                    handlelength=1.5, handleheight=0.2, framealpha=0.6)
    
    if shape_test == 2:
        plt_labels = [r'Gl$\mathrm{\ddot{o}}$ckle', f'Standard']
        plt_colors = ['red', 'blue']
        plt_ls = ['dashed', 'solid']
        
    elif shape_test == 3:
        plt_labels = [r'Gl$\mathrm{\ddot{o}}$ckle', f'Standard', f'NVP']
        plt_colors = ['red', 'blue', 'green']
        plt_ls = ['dashed', 'solid', 'dotted']
        
    else:
        raise ValueError('Check input!')
    
    for i in range(len(K0_i)):
        for j, (err_j, c_j, ls_j) in enumerate(zip(min_err, plt_colors, plt_ls)):
            ax[1][i].semilogy(E, err_j[i], color=c_j, ls=ls_j)

            ax[1][i].set_xlim(E[0] - 7, E[-1] + 7)
            ax[1][i].tick_params(axis='x')
            ax[1][i].set_ylim(1e-16, 1e1)
            ax[1][i].xaxis.set_minor_locator(minorLocator)
            ax[1][i].tick_params(axis='y')
            ax[1][i].set_xlabel(r'$E_{\mathrm{lab}}$ [MeV]')
    
    for lb_i, c_i, ls_i in zip(plt_labels, plt_colors, plt_ls):
        ax[1][2].plot([], [], color=c_i, ls=ls_i, label=lb_i)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    ax[1][0].set_ylabel(error + ' ' + 'Error')
    ax[1][2].legend(bbox_to_anchor=(0.797, 0.705, 0.8, 0.4),
                    handlelength=1.5, handleheight=0.2, framealpha=0.6)
    
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.004)
    fig.savefig(path + '.png', dpi=dpi)
        
    return None

def plot_cross_section(
    E: ArrayLike, 
    cutoff: int,
    jmax: int, 
    sim_results: ArrayLike, 
    emu_results: ArrayLike, 
    err_results: ArrayLike,
    error: str = 'Rel.',
    dpi: int = 800,
    spikes: Optional[bool] = None
) -> None:
    """
    Plots the simulated and emulated cross section.
    Plots the sampled errors for the Glockle, Standard, and NVP emulator.

    Parameters
    ----------
    E : array
        Energy grid
    cutoff : int
        EFT cutoff Lambda.
    jmax : int
        Angular momentum.
    sim_results : ArrayLike
        Simulator output.
    emu_results : ArrayLike
        Emulator predictions.
    err_results : ArrayLike
        Errors between simulator and emulator.
    error : str
        Specifies relative or absolute error.
        Default: 'Rel.'
    dpi : int
        Dots per inch.
        Used for file name output.
        Default: 800
    spikes: boolean
    
    Returns
    -------
    None
    """        
    fig = plt.figure(figsize=(3.4, 3.2))
    ax = fig.add_subplot()

    ax.semilogy(E, sim_results, c=sim_color, label=sim_label, **sim_kwargs)
    ax.semilogy(E, emu_results, c=emu_color, label=emu_label, **emu_kwargs)
    
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.set_ylim(1e1, 4e4)
    ax.set_ylabel(r'$\sigma_{\mathrm{tot}}$ [mb]')
    ax.set_xlabel(r'$E_{\mathrm{lab}}$ [MeV]')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.01, 1, 1), ncol=2)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax_ins = inset_axes(ax, width="130%", height="130%",
                        bbox_to_anchor=(0.505, 0.54, 0.5, 0.38),
                        bbox_transform=ax.transAxes)
    # ax_ins.text(0.49, 0.82, r'$\Lambda =$ ' + str(cutoff) + ' MeV', 
    #             transform=ax_ins.transAxes, verticalalignment='bottom', bbox=props)

    ax_ins.semilogy(E, err_results[1], color='b', label=f'Standard')
    ax_ins.semilogy(E, err_results[0], color='r', 
                    ls='dashed', label=r'Gl$\mathrm{\ddot{o}}$ckle')
    ax_ins.semilogy(E, err_results[2], color='g', 
                    ls='dotted', label=r'NVP')

    ax_ins.set_ylabel('Mean ' + error + ' Error')
    ax_ins.set_xlabel(r'$E_{\mathrm{lab}}$ [MeV]')
    ax_ins.set_xlim(E[0] - 7, E[-1] + 7)
    ax_ins.set_ylim(1e-15, 1e0)
    ax_ins.tick_params(axis='x')
    ax_ins.tick_params(axis='y')
    ax_ins.legend(loc='center', handlelength=2.1,
                  bbox_to_anchor=(-0.241, 0.6, 1, 1), ncol=3)
    
    if spikes:
        string = '_spikes'
    else:
        string = ''
        
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig('./plots/' + str(cutoff) + 'MeV/SMS_Chiral_' + 'jmax_' + str(jmax) 
                + '_sampled_cross_section' + string + '.png', dpi=dpi)
    
    return None

def plot_spin_obs(
    degrees: ArrayLike, 
    E: list,
    cutoff: int,
    jmax: int, 
    ylim: list,
    yarrows: list,
    sim_results: ArrayLike, 
    emu_results: ArrayLike,
    err_results: ArrayLike,
    spin_obs: str,
    error: str = 'Rel.',
    dpi: int = 800,
    nn_obs_data: Optional[ArrayLike] = None
) -> None:
    """
    Plots the simulated and emulated spin observables.
    Plots the sampled errors for the Glockle, Standard, and NVP emulator.

    Parameters
    ----------
    E : array
        Energy grid
    cutoff : int
        EFT cutoff Lambda.
    jmax : int
        Angular momentum.
    ylim : list
        Specifies y axis range for spin observables.
    yarrows : list
        Specifies arrow length for spin observables error.
    sim_results : ArrayLike
        Simulator output.
    emu_results : ArrayLike
        Emulator predictions.
    err_results : ArrayLike
        Errors between simulator and emulator.
    spin_obs : str
        Specifies spin observable.
    error : str
        Specifies relative or absolute error.
        Default: 'Rel.'
    dpi : int
        Dots per inch.
        Used for file name output.
        Default: 800
    
    Returns
    -------
    None
    """
    E1, E2, E3 = E[0], E[1], E[2]
    
    if spin_obs == 'dsg':
        label = r'$d \sigma / d \Omega$ [mb/sr]'
    elif spin_obs == 'Ay':
        label = r'$A_{y}$'
    elif spin_obs == 'D':
        label = r'$D$'
    elif spin_obs == 'Axx':
        label = r'$A_{xx}$'
    elif spin_obs == 'Ayy':
        label = r'$A_{yy}$'
    elif spin_obs == 'A':
        label = r'$A$'
    else:
        raise ValueError('Check observable input!')        

    E1_color = 'deepskyblue'
    E2_color = 'red'
    E3_color = 'yellowgreen'
    
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex='col', sharey='row')

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    text_inset = [[0.95, 0.8], [0.95, 0.1], [0.08, 0.95]]

    ax1.axhline(0, 0, 1, **zero_kwargs)
    ax1.plot(degrees, sim_results[E1 - 1], c=E1_color, label=str(E1) + ' MeV', **sim_kwargs)
    ax1.plot(degrees, emu_results[E1 - 1], markevery=10, c=E1_color, **emu_kwargs)

    ax1.plot(degrees, sim_results[E2 - 1], c=E2_color, label=str(E2) + ' MeV', **sim_kwargs)
    ax1.plot(degrees, emu_results[E2 - 1], markevery=10, c=E2_color, **emu_kwargs)

    ax1.plot(degrees, sim_results[E3 - 1], c=E3_color, label=str(E3) + ' MeV', **sim_kwargs)
    ax1.plot(degrees, emu_results[E3 - 1], markevery=10, c=E3_color, **emu_kwargs)
    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    
    if nn_obs_data is not None:
        ax1.plot(degrees, nn_obs_data[E1 - 1], c=E1_color, ls='dashdot')
        ax1.plot(degrees, nn_obs_data[E2 - 1], c=E2_color, ls='dashdot')
        ax1.plot(degrees, nn_obs_data[E3 - 1], c=E3_color, ls='dashdot')
        

    # ax1.text(0.6, 0.93, r'$\Lambda =$' + str(value) + ' MeV', 
    #          transform=ax1.transAxes, ha='left', va='top', bbox=props)

    ax1.set_xlim(degrees[0] - 3, degrees[-1] + 3)
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_ylabel(label)
    ax1.legend(loc='center', handlelength=1.28, ncol=3, framealpha=1,
               bbox_to_anchor=(0, 0.62, 1, 1))

    ax2.semilogy(degrees, err_results[0][E1 - 1], c=E1_color)
    ax2.semilogy(degrees, err_results[1][E1 - 1], c=E1_color)
    ax2.semilogy(degrees, err_results[2][E1 - 1], c=E1_color)

    ax2.semilogy(degrees, err_results[0][E2 - 1], c=E2_color)
    ax2.semilogy(degrees, err_results[1][E2 - 1], c=E2_color)
    ax2.semilogy(degrees, err_results[2][E2 - 1], c=E2_color)

    ax2.semilogy(degrees, err_results[0][E3 - 1], c=E3_color)
    ax2.semilogy(degrees, err_results[1][E3 - 1], c=E3_color)
    ax2.semilogy(degrees, err_results[2][E3 - 1], c=E3_color)

    opt = dict(color='k', alpha=0.2, 
               arrowstyle = 'simple, head_width=0.5, head_length=0.5',
               connectionstyle = 'arc3, rad=0')
    ax2.annotate('', xy=(16, yarrows[0][0]), xycoords='data', 
                 xytext =(16, yarrows[0][1]), textcoords = 'data', arrowprops=opt)
    ax2.annotate('', xy=(43, yarrows[1][0]), xycoords='data', 
                 xytext =(43, yarrows[1][1]), textcoords = 'data', arrowprops=opt)
    ax2.annotate('', xy=(67, yarrows[2][0]), xycoords='data', 
                 xytext =(67, yarrows[2][1]), textcoords = 'data', arrowprops=opt)
    ax2.text(0.5, 0.85, r'Gl$\mathrm{\ddot{o}}$ckle/NVP/Standard',  
             transform=ax2.transAxes, ha='right', bbox=props)

    ax2.set_xlim(degrees[0] - 5, degrees[-1] + 5)
    ax2.set_ylim(1e-15, 8e0)
    ax2.set_xlabel(r'$\theta_{\mathrm{cm}}$ [deg]')
    ax2.set_ylabel('Mean ' + error + ' Error')
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02)
    fig.savefig('./plots/' + str(cutoff) + 'MeV/SMS_Chiral_' + 'jmax_' + str(jmax) 
                + '_sampled_' + str(spin_obs) + '.png', dpi=dpi)
    
    return None

def plot_errors_spikes(
    E: list,
    cutoff: int,
    wave: str, 
    err_results: ArrayLike,
    err_results_fixed: ArrayLike,
    error: str = 'Rel.',
    dpi: int = 800
) -> None:
    """
    Plots the relative errors showcasing the Kohn anomalies and mesh-induced spikes removal.

    Parameters
    ----------
    E : array
        Energy grid.
    cutoff : int
        EFT cutoff Lambda.
    wave : int
        Partial wave.
    err_results : ArrayLike
        Error results for one boundary condition.
    err_results_fixed : ArrayLike
        Error results using the weighted (mixed) S matrix.
    error : str
        Specifies relative or absolute error.
        Default: 'Rel.'
    dpi : int
        Dots per inch.
        Used for file name output.
        Default: 800
    
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.3), sharey=True)
    minorLocator = MultipleLocator(20)

    ax[0].semilogy(E, err_results[1], color='blue', label=r'Std.', lw=2)
    ax[0].semilogy(E, err_results[0], color='red', ls='--', 
                label=r'Gl$\mathrm{\ddot{o}}$ckle', lw=2)
    ax[0].set_xlabel(r'$E_{\mathrm{lab}}$ [MeV]')
    ax[0].set_ylabel(error + ' ' + 'Error')
    ax[0].set_xlim(0, 210)
    ax[0].tick_params(axis='x')
    ax[0].tick_params(axis='y')
    ax[0].set_ylim(1e-15, 9e-1)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax[0].text(0.06, 0.9, r'$K^{-1}$ emu.', ha='left', va='top', 
               transform=ax[0].transAxes, bbox=props)

    ax[1].semilogy(E, err_results_fixed[0], color='red', ls='--', 
                label=r'Gl$\mathrm{\ddot{o}}$ckle', lw=2)
    ax[1].semilogy(E, err_results_fixed[1], color='blue', label=r'Standard', lw=2)
    ax[1].set_xlabel(r'$E_{\mathrm{lab}}$ [MeV]')
    ax[1].set_xlim(0, 210)
    ax[1].tick_params(axis='x')
    ax[1].tick_params(axis='y')
    ax[1].set_ylim(1e-15, 9e-1)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax[1].text(0.06, 0.9, r'Mixed $S$', ha='left', va='top', 
               transform=ax[1].transAxes, bbox=props)
    ax[1].legend(loc='upper right', ncol=2)

    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    plot_path = 'plots/' + str(cutoff) + 'MeV/' + 'SMS_Chiral_' + str(cutoff) + 'MeV_' + wave
    fig.savefig(plot_path + '_spikes.png', dpi=dpi)

    return None





