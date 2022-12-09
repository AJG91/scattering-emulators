"""
File with functions used in emulator calculation.
Includes weighted (mixed) S functions.
"""
import numpy as np
from numpy import sum as np_sum
from numpy import zeros, zeros_like, pi, append, maximum
from numpy.typing import ArrayLike
from .kinematics import K_to_S, S_to_K

def compute_delta(
    L1: ArrayLike, 
    L2: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Calculates the relative residuals between KVPs.
    
    Parameters
    ----------
    L1 : array
        First KVP prediction (in K-matrix form).
    L2 : array
        Second KVP prediction (in K-matrix form).

    Returns
    -------
    S_L1 : array
        S-matrix form of L1. 
    S_L2 : array
        S-matrix form of L2.
    delta : array
        Maximum relative residual.
    """
    S_L1 = K_to_S(L1)
    S_L2 = K_to_S(L2)

    delta_L1_L2 = abs(S_L1 / S_L2 - 1)
    delta_L2_L1 = abs(S_L2 / S_L1 - 1)
    delta = maximum(delta_L1_L2, delta_L2_L1)
    
    delta[abs(delta) < 1e-15] = 1e-15
    delta[abs(delta) > 1e-1] = 1e15
    
    return S_L1, S_L2, delta

def compute_w(
    delta_L1_L2: ArrayLike, 
    delta_L1_L3: ArrayLike, 
    delta_L2_L3: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Calculates the weights needed for the mixed S-matrix.
    
    Parameters
    ----------
    delta_L1_L2 : array
        Maximum relative residual between L1 and L2.
    delta_L1_L3 : array
        Maximum relative residual between L1 and L3.
    delta_L2_L3 : array
        Maximum relative residual between L2 and L3.

    Returns
    -------
    w_L1_L2 : array
        Weights for L1 and L2 pairs. 
    w_L1_L3 : array
        Weights for L1 and L3 pairs.
    w_L2_L3 : array
        Weights for L2 and L3 pairs.
    """
    denom = 1 / delta_L1_L2 + 1 / delta_L1_L3 + 1 / delta_L2_L3
    w_L1_L2 = 1 / (delta_L1_L2 * denom)
    w_L1_L3 = 1 / (delta_L1_L3 * denom)
    w_L2_L3 = 1 / (delta_L2_L3 * denom)
    
    return w_L1_L2, w_L1_L3, w_L2_L3

def compute_mixed_S(
    L: ArrayLike
) -> ArrayLike:
    """
    Computes the mixed S-matrix used to mitigate the Kohn anomalies.
    Here, we only consider three different boundary conditions.
    
    Parameters
    ----------
    L : array
        Array of multiple on-shell predictions for different boundary conditions.

    Returns
    -------
    S_to_K(S_mix) : array
        Outputs the mixed S-matrix in its K-matrix equivalent.
    """
    L1, L2, L3 = L[0], L[1], L[2]
    S_L1, S_L2, delta_L1_L2 = compute_delta(L1, L2)
    _, S_L3, delta_L1_L3 = compute_delta(L1, L3)
    _, _, delta_L2_L3 = compute_delta(L2, L3)
    
    w_L1_L2, w_L1_L3, w_L2_L3 = compute_w(delta_L1_L2, delta_L1_L3, delta_L2_L3)
    S_mix = 0.5 * (w_L1_L2 * (S_L1 + S_L2) + w_L2_L3 * (S_L2 + S_L3) + w_L1_L3 * (S_L1 + S_L3))
    
    return S_to_K(S_mix)

def compute_G0(
    k: ArrayLike,
    ps: ArrayLike, 
    ws: ArrayLike        
) -> ArrayLike:
    """
    Computes the partial-wave Green's function for free-space scattering.

    Parameters
    ----------
    k : array
        The on-shell center-of-mass momenta.
    ps : array
        The momentum grid in inverse fermi
    ws : array
        The corresponding weights of the mesh points.

    Returns
    -------
    G0 : array
        The free-space Green's function.
    """
    len_k = len(k)
    len_ps = len(ps)
    G0 = zeros((len_k, len_ps + 1), float)  

    for i, k0 in enumerate(k):
        D = ps ** 2 * ws / ( ps ** 2 - k0 ** 2 )
        D_k0_sum = k0 ** 2 * np_sum(ws / (ps ** 2 - k0 ** 2))
        G0[i] = (2 / pi) * append(D, -D_k0_sum)

    return G0

def compute_errors(
    exact: ArrayLike, 
    predict: ArrayLike, 
    obs: str = None
) -> ArrayLike:
    """
    Computes the errors of the observables.

    Parameters
    ----------
    exact : array
        Exact answer.
    predict : array
        Predicted answer.
    obs : str (default=None)
        Specifies which spin observable to compute errors for.
        Only used with the spin observables!

    Returns
    -------
    abs_error : array
        Absolute error.
    rel_error : array
        Relative error.
    """
    if obs != None:
        predict = predict[obs]
        exact = exact[obs]
    
    abs_error = abs(predict - exact)
    rel_error = 2 * abs(predict - exact) / (abs(predict) + abs(exact))
    
    return abs_error, rel_error

def spin_obs_errors(
    E: ArrayLike, 
    deg: ArrayLike, 
    obs_sim: dict, 
    obs_emu: dict
) -> list:
    """
    Computes the spin observables errors.

    Parameters
    ----------
    E : array
        Energy grid.
    deg : array
        Center-of-mass angle grid.
    obs_sim : dict
        Simulator solutions of the spin observables.
    obs_emu : dict
        Emulator solutions of the spin observables.

    Returns
    -------
    spin_observables : list
        List of spin observables relative errors.
    """
    len_E = len(E)
    len_deg = len(deg)
    N = obs_sim.shape[0]
    
    dsg_sim = zeros((N, len_E, len_deg))
    D_sim = zeros_like(dsg_sim)
    A_sim = zeros_like(dsg_sim)
    Ay_sim = zeros_like(dsg_sim)
    Axx_sim = zeros_like(dsg_sim)
    Ayy_sim = zeros_like(dsg_sim)
    
    dsg_emu = zeros((N, len_E, len_deg))
    D_emu = zeros_like(dsg_emu)
    A_emu = zeros_like(dsg_emu)
    Ay_emu = zeros_like(dsg_emu)
    Axx_emu = zeros_like(dsg_emu)
    Ayy_emu = zeros_like(dsg_emu)
    
    for i in range(N):
        dsg_sim[i] = obs_sim[i]['DSG']
        D_sim[i] = obs_sim[i]['D']
        A_sim[i] = obs_sim[i]['A']
        Ay_sim[i] = obs_sim[i]['PB']
        Axx_sim[i] = obs_sim[i]['AXX']
        Ayy_sim[i] = obs_sim[i]['AYY']
        
        dsg_emu[i] = obs_emu[i]['DSG']
        D_emu[i] = obs_emu[i]['D']
        A_emu[i] = obs_emu[i]['A']
        Ay_emu[i] = obs_emu[i]['PB']
        Axx_emu[i] = obs_emu[i]['AXX']
        Ayy_emu[i] = obs_emu[i]['AYY']
    
    dsg_abs_err, dsg_rel_err = compute_errors(dsg_sim, dsg_emu)
    D_abs_err, D_rel_err = compute_errors(D_sim, D_emu)
    A_abs_err, A_rel_err = compute_errors(A_sim, A_emu)
    Ay_abs_err, Ay_rel_err = compute_errors(Ay_sim, Ay_emu)
    Axx_abs_err, Axx_rel_err = compute_errors(Axx_sim, Axx_emu)
    Ayy_abs_err, Ayy_rel_err = compute_errors(Ayy_sim, Ayy_emu)
    
    return [dsg_rel_err, D_rel_err, Ay_rel_err, Axx_rel_err, Ayy_rel_err, A_rel_err]

def glockle_cubic_spline(
    old_mesh: ArrayLike, 
    new_mesh: ArrayLike
) -> ArrayLike:
    """
    Creates a cubic spline for interpolation based on the Glockle paper
    "Numerical Treatment of Few Body Equations in Momentum Space by the 
     Spline Method", Z. Phys. A - Atoms and Nuclei 305, 217-221 (1982).
    
    Parameters
    ----------
    old_mesh : ndarray
        Mesh you want to transform.
        Usually mesh taken from Gauss-Legendre quadrature.
    new_mesh : ndarray
        Mesh you used for transformation.
        Usually mesh of energies in fm.
    
    Returns
    -------
    S : ndarray, shape = (new_mesh, old_mesh)
        Spline polynomials given in paper by Glockle et al.
        Link: https://link.springer.com/article/10.1007/BF01417437
    """
    from numpy import zeros
    
    n = len(old_mesh)
    S = zeros((len(new_mesh), len(old_mesh)), float)

    B = zeros((n, n), float)
    A = zeros((n, n), float)
    C = zeros((n, n), float)
    h = zeros(n + 1, float)
    p = zeros(n, float)
    q = zeros(n, float)
    lam = zeros(n, float)
    mu = zeros(n, float)

    for i in range(1, n):
        h[i] = old_mesh[i] - old_mesh[i - 1]

    for i in range(1, n - 1):
        B[i, i] = -6.0 / (h[i] * h[i + 1])
    for i in range(1, n):
        B[i - 1, i] = 6.0 / ((h[i - 1] + h[i]) * h[i])
        B[i, i - 1] = 6.0 / ((h[i + 1] + h[i]) * h[i])

    for j in range(1, n):
        lam[j] = h[j + 1] / (h[j] + h[j + 1])
        mu[j] = 1.0 - lam[j]
        p[j] = mu[j] * q[j - 1] + 2.0
        q[j] = -lam[j] / p[j]
        A[j, :] = (B[j, :] - mu[j] * A[j - 1, :]) / p[j]

    for i in range(n - 2, -1, -1):
        C[i, :] = q[i] * C[i + 1, :] + A[i, :]

    imin, imax = old_mesh.argmin(), old_mesh.argmax()
    xmin, xmax = old_mesh[imin], old_mesh[imax]
    for yi, y in enumerate(new_mesh):
        if y <= xmin:
            S[yi, :] = 0
            S[yi, imin] = 1.0
        elif y >= xmax:
            S[yi, :] = 0
            S[yi, imax] = 1.0
        else:
            j = 0
            while old_mesh[j + 1] < y:
                j += 1
            dx = y - old_mesh[j]
            S[yi, :] += dx * (
                -(h[j + 1] / 6.0) * (2.0 * C[j, :] + C[j + 1, :])
                + dx
                * (
                    0.5 * C[j, :]
                    + dx * (1.0 / (6.0 * h[j + 1])) * (C[j + 1, :] - C[j, :])
                )
            )
            S[yi, j] += 1.0 - dx / h[j + 1]
            S[yi, j + 1] += dx / h[j + 1]
    return S

def fix_phases_continuity(
    phases: ArrayLike, 
    n0: int = None, 
    is_radians: bool = True
) -> ArrayLike:
    """
    Smoothes out the phase shifts by removing jumps by multiples of pi.

    Parameters
    ----------
    phases : array, shape = (..., N)
        Phase shifts that vary as a function in their right-most length-N axis. arctan2 may
        have caused jumps by multiples of pi in this axis.
    n0 : int, optional
        If given, shifts the initial value of the smooth phases (phases[..., 0]) to be in
        the range (n0-1/2, n0+1/2) * pi. Else, the smooth phase is defined
        to leave phases[..., -1] fixed.
    is_radians : bool
        Expects phases to be in radians if True, otherwise degrees.

    Returns
    -------
    smooth_phases : array, shape = (..., N)
        Phase shifts with jumps of pi smoothed in the right-most axis.
    """
    from numpy import pi, round, zeros_like

    if is_radians:
        factor = pi
    else:
        factor = 180.0
    n = zeros_like(phases)
    # Find all jumps by multiples of pi.
    # Store cumulative number of jumps from beginning to end of phase array
    n[..., 1:] = (round((phases[..., 1:] - phases[..., :-1]) / factor).cumsum(-1) * factor)
    # Make the jumps be relative to the final value of the phase shift
    # i.e., don't adjust phases[..., -1]
    n -= n[..., [-1]]
    # Subtract away the jumps
    smooth_phases = phases.copy()
    smooth_phases[...] -= n
    if n0 is not None:  
        # If the initial (rather than final) value of phases is constrained
        # Now move the entire phase shift at once so it starts in the range (n0-1/2, n0+1/2) * pi.
        smooth_phases[...] -= (round(smooth_phases[..., 0] / factor) - n0) * factor
    return smooth_phases


