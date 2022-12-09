import numpy as np
from numpy import asarray, sqrt, atleast_2d, arccos
from .constants import hbar_c


def compute_reduced_mass(m1, m2):
    return (m1 * m2) / (m1 + m2)


def t_lab_to_t_cm(t_lab, mass_beam, mass_target):
    return t_lab / ((mass_beam + mass_target) / mass_target)


def t_cm_to_t_lab(t_cm, mass_beam, mass_target):
    return t_cm * ((mass_beam + mass_target) / mass_target)


def t_lab_to_q_cm_beam_and_target(t_lab, mass_beam, mass_target):
    n = mass_target ** 2 * t_lab * (t_lab + 2 * mass_beam)
    d = (mass_target + mass_beam) ** 2 + 2 * t_lab * mass_target
    return np.sqrt(n / d) / hbar_c


def mandelstam(q, x, m1, m2):
    q = atleast_2d(q).T * hbar_c

    E1 = sqrt(q ** 2 + m1 ** 2)
    E2 = sqrt(q ** 2 + m2 ** 2)
    s = (E1 + E2)**2
    t = - 2 * q ** 2 * (1 - x)
    u = (E1 - E2)**2 - 2 * q ** 2 * (1 + x)
    return s, t, u


def wigner_rotations(q, x, m1, m2):
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