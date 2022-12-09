"""
File used for calculating half-on-shell and on-shell pieces of the potential
using either interpolation or exact calculation.
"""
from numpy import (
    block, rollaxis, array, zeros, 
    swapaxes, reshape, append, meshgrid
)
from numpy.typing import ArrayLike
from typing import Optional
from .constants import V_factor_RME
from scipy.interpolate import RectBivariateSpline

def uncoupled_potential(
    jmom: int, 
    lecs: ArrayLike, 
    spectral: ArrayLike, 
    contacts: ArrayLike, 
    wave: Optional[str] = None
) -> ArrayLike:
    """
    Combines the spectral and contact terms with the corresponding coupling
    contants to obtain the SMS chiral potential for a specified uncoupled channel.

    Parameters
    ----------
    jmom : int
        Angular momentum used to select correct potential.
    lecs : array
        Array of coupling constants.
    spectral: ndarray
        The spectral part of the potential.
    contacts: ndarray
        The contacts part of the potential. These are scaled by the LECs.
    wave: str or None (default=None)
        Specifies which partial wave is being considered. 
        None used when partial wave does not have a coupling constant.
        Ex: '1S0', '3P1', etc.
    
    Returns
    -------
    pot : ndarray
        An array of the potential at the specified partial wave.
    """
    cont = V_factor_RME * contacts[jmom]
    spec = V_factor_RME * spectral[jmom]
    
    if (jmom == 0):
#         if (wave == '1S0'):
        if '1S0' in wave:
            pot_cont, pot_spec = cont[0:3], spec[0]
#         elif (wave == '3P0'):
        elif '3P0' in wave:
            pot_cont, pot_spec = cont[3:5], spec[5]
        else:
            raise Exception("Wrong partial wave!")
        
    elif (jmom == 1):
#         if (wave == '1P1'):
        if '1P1' in wave:
            pot_cont, pot_spec = cont[0:2], spec[0]
#         elif (wave == '3P1'):
        elif '3P1' in wave:
            pot_cont, pot_spec = cont[2:4], spec[1]
        else:
            raise Exception("Wrong partial wave!")
        
    elif (jmom == 2):
#         if (wave == '1D2'):
        if '1D2' in wave:
            pot_cont, pot_spec = cont[0:1], spec[0]
#         elif (wave == '3D2'):
        elif '3D2' in wave:
            pot_cont, pot_spec = cont[1:2], spec[1]
        else:
            raise Exception("Wrong partial wave!")
        
    elif (jmom == 3):
#         if (wave == '1F3'):
        if '1F3' in wave:
            pot_cont, pot_spec = cont[0:1], spec[0]
#         elif (wave == '3F3'):
        elif '3F3' in wave:
            pot_cont, pot_spec = cont[1:2], spec[1]
        else:
            raise Exception("Wrong partial wave!")
    else:
        raise Exception("Wrong value for momentum!")

    pot_cont = rollaxis(pot_cont, 0, 3)
    pot = pot_spec + pot_cont @ array(lecs)
    return pot, pot_spec, pot_cont

def coupled_potential(
    jmom: int, 
    lecs: ArrayLike, 
    spectral: ArrayLike, 
    contacts: ArrayLike, 
    wave: Optional[str] = None
) -> ArrayLike:
    """
    Combine the spectral and contact terms with the corresponding coupling 
    contants to obtain the SMS chiral potential for specified coupled channels.

    Parameters
    ----------
    jmom : int
        Angular momentum used to select correct potential.
    lecs : array
        Array of coupling constants.
    spectral : ndarray
        The spectral part of the potential.
    contacts : ndarray
        The contacts part of the potential. These are scaled by the LECs.
    wave : str or None (default=None)
        Specifies which partial wave is being considered. 
        None used when partial wave does not have a coupling constant.
        Ex: '1S0', '3P1', etc.
    
    Returns
    -------
    pot : ndarray
        An array of the potential at the specified partial wave.
    """
    size = len(spectral[0, 0, 0, :])
    cont = V_factor_RME * contacts[jmom]
    spec = V_factor_RME * spectral[jmom]
    
    pot_spec =  block([[spec[2], -spec[3]], [-spec[4], spec[5]]])
    
    if (jmom == 1):
        if '3S1' in wave:
            lecs_array = array([lecs[0], lecs[1], lecs[3], lecs[2], 
                                lecs[5], lecs[2], lecs[5], lecs[4]])
            pot_cont = zeros((len(lecs_array), 2 * size, 2 * size))
    
            pot_cont[0][:size, :size] = cont[4]
            pot_cont[1][:size, :size] = cont[5]
            pot_cont[2][:size, :size] = cont[6]
            pot_cont[3][:size, size:] = -cont[7]
            pot_cont[4][:size, size:] = -cont[8]
            pot_cont[5][size:, :size] = -cont[9]
            pot_cont[6][size:, :size] = -cont[10]
            pot_cont[7][size:, size:] = cont[11]
        else:
            raise Exception("Wrong partial wave!")
            
    elif (jmom == 2):
        if '3P2' in wave:
            lecs_array = array([lecs[0], lecs[1], lecs[2], lecs[2], lecs[3]])
            pot_cont = zeros((len(lecs_array), 2 * size, 2 * size))
    
            pot_cont[0][:size, :size] = cont[2]
            pot_cont[1][:size, :size] = cont[3]
            pot_cont[2][:size, size:] = -cont[4]
            pot_cont[3][size:, :size] = -cont[5]
            pot_cont[4][size:, size:] = cont[6]
        else:
            raise Exception("Wrong partial wave!")
        
    elif (jmom == 3):
        if '3D3' in wave:
            lecs_array = array([lecs[0]])
            pot_cont = zeros((len(lecs_array), 2 * size, 2 * size))
    
            pot_cont[0][:size, :size] = cont[2]
        else:
            raise Exception("Wrong partial wave!")
        
    elif (jmom == 4):
        if '3F4' in wave:
            lecs_array = array([lecs[0]])
            pot_cont = zeros((len(lecs_array), 2 * size, 2 * size))
    
            pot_cont[0][:size, :size] = cont[0]
        else:
            raise Exception("Wrong partial wave!")
            
    elif (jmom > 4):
        lecs_array = array([0.0])
        pot_cont = zeros((len(lecs_array), 2 * size, 2 * size))

        pot_cont[0][:size, :size] = cont[2]

    else:
        raise Exception("Wrong value for momentum!")
        
    pot_cont = rollaxis(pot_cont, 0, 3)
    pot = pot_spec + pot_cont @ lecs_array
    
    return pot, pot_spec, pot_cont


class GetFullPotential:
    """
    A class used to compute the full potential (including half-on-shell and on-shell piece).
    
    Parameters
    ----------
    k : array
        The k grid.
    ps : array
        The momentum grid in inverse fermi.
    jmax : int
        Maximum angular momentum used for potential calculation.
    potential : instance
        Specific potential instance.
    """  
    def __init__(
        self, 
        k: ArrayLike,
        ps: ArrayLike,
        jmax: int, 
        potential
    ):    
        self.N = len(ps)
        self.len_k = len(k)
        self.k, self.ps = k, ps
        
        self.spec, self.cont = self._compute_spectral_contacts(k, ps, jmax, potential)

    def _compute_spectral_contacts(
        self, 
        k: ArrayLike,
        ps: ArrayLike,
        jmax: int, 
        potential
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the half-on-shell and on-shell pieces of the LEC-dependent
        and LEC-independent potential terms.

        Parameters
        ----------
        k : array
            The k grid.
        ps : array
            The momentum grid in inverse fermi
        jmax : int
            Maximum angular momentum used for potential calculation.
        potential : instance
            Specific potential instance.
            
        Returns
        -------
        spec : array
            LEC-independent part of potential.
        cont : array
            LEC-dependent part of potential.
        """
        len_k, N = self.len_k, self.N
        spec = zeros((len_k, jmax + 1, 6, N + 1, N + 1))
        cont = zeros((len_k, jmax + 1, 12, N + 1, N + 1))
        
        for i, k0 in enumerate(k):
            if ((i + 1) % 25) == 0:
                print('E:', i + 1)
                
            spec[i], cont[i] = potential.get_half_on_shell(append(ps, k0), jmax)
        return spec, cont
        
    def no_interp_pot(
        self, 
        jmom: int, 
        wave: str, 
        lecs: ArrayLike, 
        V: ArrayLike, 
        V0: ArrayLike, 
        V1: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Combines half-on-shell and on-shell pieces of the LEC-dependent
        and LEC-independent potential terms to the pre-calculated off-shell part.
        Checks whether potential is coupled or uncoupled.

        Parameters
        ----------
        jmom : int
            Specific angular momentum being considered.
        wave : str
            Specific partial-wave being considered.
        lecs : array
            Low-energy couplings associated with the partial-wave.
        V : array
            Full potential without half-on-shell and on-shell part.
        V0 : array
            LEC-independent part of potential without half-on-shell and on-shell part.
        V1 : array
            LEC-dependent part of potential without half-on-shell and on-shell part.
            
        Returns
        -------
        V_w_k0 : array
            Full potential.
        V0_w_k0 : array
            Full LEC-independent part of potential.
        V1_w_k0 : array
            Full LEC-dependent part of potential.
        """
        if V.shape[1] == self.N:
            V_w_k0, V0_w_k0, V1_w_k0 = self._get_uncoupled_potential(jmom, wave, 
                                                                     lecs, V, V0, V1)
        else:
            V_w_k0, V0_w_k0, V1_w_k0 = self._get_coupled_potential(jmom, wave, 
                                                                   lecs, V, V0, V1)
        return V_w_k0, V0_w_k0, V1_w_k0

    def _get_uncoupled_potential(
        self, 
        jmom: int, 
        wave: str, 
        lecs: ArrayLike, 
        V: ArrayLike, 
        V0: ArrayLike, 
        V1: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Combines half-on-shell and on-shell pieces of the LEC-dependent
        and LEC-independent potential terms to the pre-calculated off-shell part
        for a non-coupled channel potential.

        Parameters
        ----------
        jmom : int
            Specific angular momentum being considered.
        wave : str
            Specific partial-wave being considered.
        lecs : array
            Low-energy couplings associated with the partial-wave.
        V : array
            Full potential without half-on-shell and on-shell part.
        V0 : array
            LEC-independent part of potential without half-on-shell and on-shell part.
        V1 : array
            LEC-dependent part of potential without half-on-shell and on-shell part.
            
        Returns
        -------
        V_w_k0 : array
            Full potential.
        V0_w_k0 : array
            Full LEC-independent part of potential.
        V1_w_k0 : array
            Full LEC-dependent part of potential.
        """
        spec, cont = self.spec, self.cont
        N, len_k, k = self.N, self.len_k, self.k
        len_lecs = len(lecs)
        
        V_w_k0 = zeros((len_k, N + 1, N + 1))
        V0_w_k0 = zeros((len_k, N + 1, N + 1))
        V1_w_k0 = zeros((len_k, N + 1, N + 1, len_lecs))

        for i, k0 in enumerate(k):
            V_w_k0[i], V0_w_k0[i], V1_w_k0[i] = uncoupled_potential(jmom, lecs, 
                                                                    spec[i], cont[i], wave)
            
            V_w_k0[i][-1, :N + 1] = V_w_k0[i][:N + 1, -1]
            V0_w_k0[i][-1, :N + 1] = V0_w_k0[i][:N + 1, -1]
            V1_w_k0[i][-1, :N + 1] = V1_w_k0[i][:N + 1, -1]
            V_w_k0[i][:N, :N] = V
            V0_w_k0[i][:N, :N] = V0
            V1_w_k0[i][:N, :N] = V1
        
        return V_w_k0, V0_w_k0, V1_w_k0
    
    def uncoupled_potential_no_contacts(
        self, 
        jmom: int, 
        idx: int, 
        V: ArrayLike
    ) -> ArrayLike:
        """
        Combines half-on-shell and on-shell pieces of the LEC-independent potential 
        terms to the pre-calculated off-shell part for a non-coupled channel potential.

        Parameters
        ----------
        jmom : int
            Specific angular momentum being considered.
        idx : int
            Specific index for indexing potential array.
        V : array
            LEC-independent potential without half-on-shell and on-shell part.
            
        Returns
        -------
        V_w_k0 : array
            Full potential.
        """
        spec = self.spec
        N, len_k, k = self.N, self.len_k, self.k
        
        V_w_k0 = zeros((len_k, N + 1, N + 1))

        for i, k0 in enumerate(k):
            V_w_k0[i] = spec[i][jmom][idx]
            
            V_w_k0[i][-1, :N + 1] = V_w_k0[i][:N + 1, -1]
            V_w_k0[i][:N, :N] = V
        
        return V_factor_RME * V_w_k0
    
    def uncoupled_potential_no_contacts_interp(
        self, 
        V: ArrayLike
    ) -> ArrayLike:
        """
        Combines half-on-shell and on-shell pieces of the LEC-independent potential 
        terms to the pre-calculated off-shell part for a non-coupled channel potential
        using interpolation method from simulator.

        Parameters
        ----------
        V : array
            LEC-independent potential without half-on-shell and on-shell part.
            
        Returns
        -------
        V_w_k0 : array
            Full potential.
        """
        return V_factor_RME * self._interpolate_uncoupled_potential(V)
    
    def _interpolate_uncoupled_potential(
        self,
        potential: ArrayLike,
        spl_order: int = 3
    ) -> ArrayLike:
        """
        Interpolates half-on-shell and on-shell part of a non-coupled potential.

        Parameters
        ----------
        potential : array
            Potential without half-on-shell and on-shell pieces.
            
        Returns
        -------
        V : array
            Full potential.
        """
        k, ps = self.k, self.ps
        len_k, N = self.len_k, self.N
        
        V = zeros((len_k, N + 1, N + 1), float)
        V_spline = RectBivariateSpline(ps, ps, potential, kx=spl_order, ky=spl_order)
        
        for i, k0 in enumerate(k):
            k_vals = append(ps, k0)
            grid_kx, grid_ky = meshgrid(k_vals, k_vals)
            V[i] = V_spline.ev(grid_kx, grid_ky)
        
        return V
    
    def _get_coupled_potential(
        self, 
        jmom: int, 
        wave: str, 
        lecs: ArrayLike, 
        V: ArrayLike, 
        V0: ArrayLike, 
        V1: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Combines half-on-shell and on-shell pieces of the LEC-dependent
        and LEC-independent potential terms to the pre-calculated off-shell part
        for a coupled channel potential.

        Parameters
        ----------
        jmom : int
            Specific angular momentum being considered.
        wave : str
            Specific partial-wave being considered.
        lecs : array
            Low-energy couplings associated with the partial-wave.
        V : array
            Full potential without half-on-shell and on-shell part.
        V0 : array
            LEC-independent part of potential without half-on-shell and on-shell part.
        V1 : array
            LEC-dependent part of potential without half-on-shell and on-shell part.
            
        Returns
        -------
        V_w_k0 : array
            Full potential.
        V0_w_k0 : array
            Full LEC-independent part of potential.
        V1_w_k0 : array
            Full LEC-dependent part of potential.
        """
        spec, cont = self.spec, self.cont
        N, len_k = self.N, self.len_k
        k, ps = self.k, self.ps
        len_lecs = len(lecs)
        
        if '3S1' in wave:
            len_lecs += 2
        elif '3P2' in wave:
            len_lecs += 1
        
        V_w_k0 = zeros((len_k, 2 * (N + 1), 2 * (N + 1)))
        V0_w_k0 = zeros((len_k, 2 * (N + 1), 2 * (N + 1)))
        V1_w_k0 = zeros((len_k, 2 * (N + 1), 2 * (N + 1), len_lecs))
        
        for i, k0 in enumerate(k):
            V_w_k0[i], V0_w_k0[i], V1_w_k0[i] = coupled_potential(jmom, lecs, 
                                                                  spec[i], cont[i], wave)
            
            V_w_k0[i][:N, :N] = V[:N, :N] ## 3S1
            V_w_k0[i][:N, N + 1:-1] = V[:N, N:] ## 3S1/3D1
            V_w_k0[i][N + 1:-1, :N] = V[N:, :N] ## 3D1/3S1
            V_w_k0[i][N + 1:-1, N + 1:-1] = V[N:, N:] ## 3D1
                        
            V_w_k0[i][N:N + 1, :N + 1] = reshape(V_w_k0[i][:N + 1, N:N + 1], (N + 1)) ## 3S1
            V_w_k0[i][N:N + 1, N + 1:] = reshape(V_w_k0[i][N + 1:, N:N + 1], (N + 1)) ## 3S1/3D1
            V_w_k0[i][2 * N + 1:2 * N + 2, :N + 1] = V_w_k0[i][:N + 1, -1] ## 3D1/3S1
            V_w_k0[i][2 * N + 1:2 * N + 2, N + 1:] = V_w_k0[i][N + 1:, -1] ## 3D1
            
            V0_w_k0[i][:N, :N] = V0[:N, :N]
            V0_w_k0[i][:N, N + 1:-1] = V0[:N, N:]
            V0_w_k0[i][N + 1:-1, :N] = V0[N:, :N]
            V0_w_k0[i][N + 1:-1, N + 1:-1] = V0[N:, N:]
            
            V0_w_k0[i][N:N + 1, :N + 1] = reshape(V0_w_k0[i][:N + 1, N:N + 1], (N + 1))
            V0_w_k0[i][N:N + 1, N + 1:] = reshape(V0_w_k0[i][N + 1:, N:N + 1], (N + 1))
            V0_w_k0[i][2 * N + 1:2 * N + 2, :N + 1] = V0_w_k0[i][:N + 1, -1]
            V0_w_k0[i][2 * N + 1:2 * N + 2, N + 1:] = V0_w_k0[i][N + 1:, -1]
            
            V1_w_k0[i][:N, :N] = V1[:N, :N]
            V1_w_k0[i][:N, N + 1:-1] = V1[:N, N:]
            V1_w_k0[i][N + 1:-1, :N] = V1[N:, :N]
            V1_w_k0[i][N + 1:-1, N + 1:-1] = V1[N:, N:]
            
            V1_w_k0[i][N:N + 1, :N + 1] = reshape(V1_w_k0[i][:N + 1, N:N + 1], (N + 1, len_lecs))
            V1_w_k0[i][N:N + 1, N + 1:] = reshape(V1_w_k0[i][N + 1:, N:N + 1], (N + 1, len_lecs))
            V1_w_k0[i][2 * N + 1:2 * N + 2, :N + 1] = V1_w_k0[i][:N + 1, -1]
            V1_w_k0[i][2 * N + 1:2 * N + 2, N + 1:] = V1_w_k0[i][N + 1:, -1]
            
            V_w_k0[i] = (V_w_k0[i] + V_w_k0[i].T) / 2
            V0_w_k0[i] = (V0_w_k0[i] + V0_w_k0[i].T) / 2
            V1_w_k0[i] = (V1_w_k0[i] + swapaxes(V1_w_k0[i], 0, 1)) / 2
        
        return V_w_k0, V0_w_k0, V1_w_k0
    
    
    
    
    
    
    
    