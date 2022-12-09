
import numpy as np
from numpy.typing import ArrayLike
import scipy.special as ss 
from emulate.emulate_kvp.constants import Mn_MeV as M_N, hbar_c


class SeparablePotential():
    
    def __init__(
        self, 
        lam_str: float,
        LAM2: float = 200,
        BE: float = 2 # [MeV]
    ):
        self.lam2_unscaled = lam_str
        self.LAM2 = LAM2
        
    def analytical_S_mat(
        self,
        E: ArrayLike, 
    ):
        """
        This directly computes S matrix for the energy mesh E_lab_MeV 
        Based on the analytical formula. 
        """
        LAM2 = self.LAM2
        lam2_unscaled = self.lam2_unscaled
        
        factor = 8j * np.pi**2 * M_N / (2 * hbar_c**2)

        S_mat = np.vectorize(lambda E: 1- factor * np.sqrt(M_N * E) 
                             * (self.tau_unscaled(lam2_unscaled=lam2_unscaled, 
                                                  LAM2=LAM2, ene_2= E) 
                                * self.formfactor_unscaled(q=np.sqrt(M_N * E))**2))(E)
        return S_mat

    def get_sep_pot(
        self,
        ps
    ):
        """
        """
        LAM2 = self.LAM2
        lam2_unscaled = self.lam2_unscaled
        ps_hbarc = ps * hbar_c
        
        ff_ipext_in_fm = self.formfactor_unscaled(q=ps_hbarc)
        outer_prod = np.outer(ff_ipext_in_fm, ff_ipext_in_fm)

        pot_mat_in_fm = 4 * np.pi**2 * lam2_unscaled * outer_prod * M_N / (2 * hbar_c)

        return pot_mat_in_fm

    def tau_unscaled(
        self,
        lam2_unscaled,
        LAM2,
        ene_2,
    ):
        """
        Calculates the T-matrix (Eq. 17 - 18)
        """        
        if np.iscomplex(ene_2) :  
            sqrtvalue= np.sqrt(-M_N * ene_2)
        elif ene_2.real < 0 : 
            sqrtvalue = np.sqrt(-M_N * ene_2)
        else:
            sqrtvalue = -1j * np.sqrt(M_N * ene_2)

        Csq = LAM2 / (2 * np.pi**1.5 * M_N)
        integral = np.sqrt(np.pi)- sqrtvalue / LAM2 * np.pi * ss.erfcx(sqrtvalue / LAM2)
        return lam2_unscaled / (1. + lam2_unscaled * M_N * 2 * np.pi * Csq / LAM2 * integral) ## dimensionless

    def formfactor_unscaled(
        self,
        q
    ):
        """
        Calculates the Gaussian form factor (Eq. 13)
        """
        LAM2 = self.LAM2

        Csq = LAM2 / (2 * np.pi**1.5 * M_N)
        return np.sqrt(Csq) * hbar_c / LAM2 * np.exp(-0.5 * (q / LAM2)**2) ## fm



