"""
File containing different kinematics functions.
"""
from numpy import (
    asarray, sqrt, atleast_2d, isclose, 
    append, arccos, real, log, pi, unique
)
from numpy.typing import ArrayLike
from .constants import hbar_c

def K_to_S(K):
    """
    Transforms the K matrix to the S matrix.
    
    Parameters
    ----------
    K : array
        K matrix.

    Returns
    -------
    (1 + 1j * K) / (1 - 1j * K) : array
        S matrix.
    """
    return (1 + 1j * K) / (1 - 1j * K)

def K_to_T(K):
    """
    Transforms the K matrix to the T matrix.
    
    Parameters
    ----------
    K : array
        K matrix.

    Returns
    -------
    K / (1 - 1j * K) : array
        T matrix.
    """
    return K / (1 - 1j * K)

def T_to_K(T):
    """
    Transforms the T matrix to the K matrix.
    
    Parameters
    ----------
    T : array
        T matrix.

    Returns
    -------
    real(T / (1 + 1j * T)) : array
        K matrix.
    """
    return real(T / (1 + 1j * T))

def T_to_S(T):
    """
    Transforms the T matrix to the S matrix.
    
    Parameters
    ----------
    T : array
        T matrix.

    Returns
    -------
    1 + 2 * 1j * T : array
        S matrix.
    """
    return 1 + 2 * 1j * T

def S_to_K(S):
    """
    Transforms the S matrix to the T matrix.
    
    Parameters
    ----------
    S : array
        S matrix.

    Returns
    -------
    real(1j * (1 - S) / (1 + S)) : array
        K matrix.
    """
    return real(1j * (1 - S) / (1 + S))

def S_to_T(S):
    """
    Transforms the S matrix to the T matrix.
    
    Parameters
    ----------
    S : array
        S matrix.

    Returns
    -------
    1j / 2 * (1 - S) : array
        T matrix.
    """
    return 1j / 2 * (1 - S)

def K_to_phase(K):
    """
    Calculates the phase shifts from the on-shell K matrix
    by transforming to the S matrix.
    
    Parameters
    ----------
    K: array
        On-shell K matrix.

    Returns
    -------
    S_to_phase(K_to_S(K)): array
        Phase shifts.
    """
    return S_to_phase(K_to_S(K))

def T_to_phase(T):
    """
    Calculates the phase shifts from the on-shell T matrix
    by transforming to the S matrix.
    
    Parameters
    ----------
    T : array
        On-shell T matrix.

    Returns
    -------
    S_to_phase(T_to_S(K)) : array
        Phase shifts.
    """
    return S_to_phase(T_to_S(T))

def S_to_phase(S):
    """
    Calculates the phase shifts from the S matrix.
    
    Parameters
    ----------
    S : array
        On-shell S matrix.

    Returns
    -------
    real(log(S) / (2 * 1j)) * 180 / pi : array
        Phase shifts.
    """
    return real(log(S) / (2 * 1j)) * 180 / pi

def avoid_pv_sing(
    E, 
    ps, 
    m1, 
    m2, 
    relativistic, 
    rtol=1e-2, 
    remove=False
) -> tuple[ArrayLike, list]:
    """
    Fixes the mesh to prevent mesh-induced singularities resulting
    from the a mesh point being too close to a k0 point.
    
    Parameters
    ----------
    E : array
        Energy grid.
    ps : array
        Quadrature mesh.
    m1 : float
        Mass of beam.
    m2 : float
        Mass of target
    relativistic : bool
        If True, calculates center-of-mass k using relativistic formula.
        If False, calculates center-of-mass k using non-relativistic formula.
    rtol : float (default=1e-2)
        Relative tolerance between k0 and mesh point.
    remove : bool (default=False)
        If True, removes the energy value.
        If False, pushes the energy value.

    Returns
    -------
    unique(E) : array
        New energy grid.
    idx : list
        List of indices where mesh point was close to k0 point.
    """
    k = e_lab_to_k_cm(E, m1, m2, relativistic)[1]
    E = asarray(E)
    idx = []
    
    for i, k_i in enumerate(k):
        if isclose(k_i, ps, rtol=rtol).any():
            idx = append(idx, i)
            
            if not remove:
                while isclose(k_i, ps, rtol=rtol).any():
                    E[i] = E[i] + 10 / E[i]
                    k_i = e_lab_to_k_cm(E[i], m1, m2, relativistic)[1]
            else:
                E[i] = E[i-1]
            
    idx.tolist()
    idx = [int(k) for k in idx]
    return unique(E), idx

def e_lab_to_k_cm(
    E_lab: ArrayLike,
    mass_beam: float,
    mass_target: float,
    relativistic: bool
) -> tuple[ArrayLike, ArrayLike]:
    """
    Calculates center-of-mass k from laboratory energy.
    
    Parameters
    ----------
    E_lab : array
        Laboratory energy grid.
    mass_beam : float
        Mass of beam.
    mass_target : float
        Mass of target
    relativistic : bool
        If True, calculates center-of-mass k using relativistic formula.
        If False, calculates center-of-mass k using non-relativistic formula.

    Returns
    -------
    E_tot : array
        Energy grid.
    asarray(sqrt(n / d) / hbar_c) : array
        Center-of-mass k.
    """
    E_tot = e_lab_to_e_tot(E_lab, mass_beam, mass_target)
    
    if relativistic:
        n = mass_target**2 * E_lab * (E_lab + 2.0 * mass_beam)
        d = (mass_target + mass_beam)**2 + 2.0 * E_lab * mass_target
    else:
        n = mass_beam * E_tot
        d = 1
    return E_tot, asarray(sqrt(n / d) / hbar_c)

def e_lab_to_e_tot(
    E_lab: ArrayLike,
    mass_beam: float,
    mass_target: float
) -> ArrayLike:
    """
    Calculates the total energy mesh from the laboratory energy.
    
    Parameters
    ----------
    E_lab : array
        Laboratory energy grid.
    mass_beam : float
        Mass of beam.
    mass_target : float
        Mass of target

    Returns
    -------
    E_tot : array
        Energy grid.
    """
    return E_lab / ((mass_beam + mass_target) / mass_target)


def mandelstam(q, x, m1, m2):
    """
    Outputs the mandelstam variables.
    """
    q = atleast_2d(q).T * hbar_c

    E1 = sqrt(q ** 2 + m1 ** 2)
    E2 = sqrt(q ** 2 + m2 ** 2)
    s = (E1 + E2)**2
    t = - 2 * q ** 2 * (1 - x)
    u = (E1 - E2)**2 - 2 * q ** 2 * (1 + x)
    return s, t, u


def wigner_rotations(q, x, m1, m2):
    """
    Performs Wigner rotations.
    """
    s, t, _ = mandelstam(q, x, m1, m2)
    q = atleast_2d(q).T * hbar_c
    rt_s = sqrt(s)
    
    cos_ap = (
            8 * q**4 * s +
            2 * q**2 * t * (3 * s - 4 * m2 * rt_s + m2**2 - m1**2) +
            ((rt_s - m2)**2 - m1**2) * t ** 2
        ) / (4 * q**3 * rt_s * sqrt((s + t - (m2 + m1)**2) * (s + t - (m1 - m2)**2)))
    
    cos_bp = sqrt(-t / (4 * m2**2 - t)) / (4 * q**3 * rt_s) * \
             (2 * q**2 * (s - m1**2 + m2**2 - 4 * m2 * rt_s) + t * (s - m1**2 + m2**2 - 2 * m2 * rt_s))
    
    cos_bm = (s - m1**2 + m2**2) / (2 * q * rt_s) * sqrt(-t / (4 * m2**2 - t))
    
    theta = arccos(x)
    alpha = arccos(cos_ap) - theta / 2
    beta = arccos(cos_bp) - theta / 2
    return alpha, beta

