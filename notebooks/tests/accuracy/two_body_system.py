"""
File used to two-body scattering calculations.
"""
import numpy as np
from numpy import sum as np_sum
from numpy import(
    pi, zeros, identity, arctan, log,
    stack, ndarray, array, append, squeeze, 
    cos, sin, arcsin, tan, degrees, meshgrid,
    reshape, vstack, hstack, sqrt, asarray,
    tile, concatenate, swapaxes, zeros_like
)
from math import ceil
from scipy.linalg import solve
from numpy.typing import ArrayLike
from typing import Optional
from scipy.interpolate import RectBivariateSpline

from emulate.emulate_kvp.constants import (
    mass_proton, mass_neutron, fm_to_sqrt_mb
)
from emulate.emulate_kvp.kinematics import e_lab_to_k_cm
from emulate.emulate_kvp.utils import glockle_cubic_spline, fix_phases_continuity


class TwoBodyScattering:
    """
    Creates a two-body scattering environment used to calculate phase shifts
    and the total cross section.

    Parameters
    ----------
    E : array
        The energy mesh. If a single value is passed, an array is created for it.
    ps : array
        The momentum grid in inverse fermi
    ws : array
        The corresponding weights of the mesh points.
    is_coupled : boolean, (default=False)
        Used to determine whether the calculations should be done 
        for a single or coupled channel.
    inf_map : boolean, Default=True
        Denotes whether the mesh has an infinite mapping. 
        If true, no contribution from the cutoff is necessary.
    """
    def __init__(
        self, 
        E: ArrayLike, 
        ps: ArrayLike, 
        ws: ArrayLike,
        is_coupled: bool = False,
        inf_map: bool = True,
        relativistic: bool = True
    ):
        if (isinstance(E, ndarray) == False):
            E = array([E])
            
        E_tot, k = e_lab_to_k_cm(E, mass_neutron, mass_proton, relativistic)
        Sp = glockle_cubic_spline(ps, k)
        
        self.E, self.E_tot = E, E_tot
        self.k, self.ps, self.ws = k, ps, ws
        self.inf_map = inf_map
        
        self.N = len(self.ps)
        self.len_k = len(self.k)
        self.k_max = ceil(max(self.ps))
        G0, G0_sp = self.calc_G0(Sp)
        
        if is_coupled:
            self.tau = zeros((self.len_k, 2, 2), float)
            I = identity(2 * self.N, float)
            I_std = identity(2 * (self.N + 1), float)
            K = zeros((self.len_k, 2 * (self.N + 1), 2 * (self.N + 1)), float)
            G0_coup = zeros((self.len_k, 2 * self.N), float)
            G0_coup[:, :self.N] = G0_coup[:, self.N:] = G0_sp
            Sp_coup = zeros((self.len_k, 2, 2 * self.N))
            Sp_coup[:, 0, :self.N] = Sp_coup[:, 1, self.N:] = Sp
            G0_coup_std = zeros_like(K, float)
            
            for i in range(G0.shape[0]):
                G0_coup_std[i] = np.tile(G0[i], (2 * (self.N + 1), 2))
            
            self.I_std = I_std
            self.Sp = Sp_coup
            self.G0_sp = G0_coup
            self.I, self.K = I, K
            self.G0_coup_std = G0_coup_std
            
        else:
            self.tau = zeros(self.len_k)
            self.I = identity(self.N)
            self.I_std = identity(self.N + 1)
            self.K = zeros((self.len_k, self.N + 1), float)
            self.G0_sp = G0_sp
            self.Sp = Sp
            
        self.G0 = G0
        self.is_coupled = is_coupled
        
    def calc_G0(
        self,
        spline: ArrayLike 
    ) -> ArrayLike:
        """
        Computes the partial-wave Green's function for free-space scattering.

        Parameters
        ----------
        spline : array
            The interpolation vector from the Glockle spline.

        Returns
        -------
        G0 : array
            The free-space Green's function.
        """
        inf_map = self.inf_map
        k, ps, ws = self.k, self.ps, self.ws
        N, len_k = self.N, self.len_k
        
        G0 = zeros((len_k, N + 1), float)
        G0_sp = zeros((len_k, N), float)      
        
        for i, k0 in enumerate(k):
            D = ps ** 2 * ws / ( ps ** 2 - k0 ** 2 )
            D_k0_sum = k0 ** 2 * np_sum(ws / (ps ** 2 - k0 ** 2))
            D_k0_sum_sp = D_k0_sum * spline[i]

            if inf_map:
                D_k0 = -D_k0_sum
                D_k0_sp = -D_k0_sum_sp
            else:
                D_k0_ln = 0.5 * k0 * log( ( k_max + k0 ) / ( k_max - k0 ) )
                D_k0 = -( D_k0_sum + D_k0_ln )
                D_k0_sp = -( D_k0_sum_sp + D_k0_ln * spline[i] )

            G0[i] = (2 / pi) * append(D, D_k0)
            G0_sp[i] = (2 / pi) * (D + D_k0_sp)
            
        return G0, G0_sp
    
    def ls_eq_no_interpolate(self, V):
        """
        Wrapper used for the simulator solution.
        Checks if considering coupled or non-coupled channels.

        Parameters
        ----------
        V : array
            Potential.

        Returns
        -------
        K0 : array
            On-shell K.
        K : array
            Full K-matrix.
        """
        if self.is_coupled:
            N = self.N
            K = self.ls_eq_no_interpolate_coupled(V, self.G0_coup_std, self.K)
            tau = array([[K[:, N, N], K[:, N, 2 * N + 1]], 
                         [K[:, 2 * N + 1, N], K[:, 2 * N + 1, 2 * N + 1]]]).T
            
        else:
            K = self.ls_eq_no_interpolate_uncoupled(V, self.G0, self.K)
            tau = K[:, -1]
        
        return tau, K
    
    def ls_eq_no_interpolate_uncoupled(self, V, G0, K):
        """
        Solves the Lippmann-Schwinger equation for the non-coupled channels.

        Parameters
        ----------
        V : array
            Potential.
        G0 : array
            Partial-wave Green's function for free-space scattering.
        K : array
            Pre-defined array for K-matrix.

        Returns
        -------
        K : array
            Full K-matrix.
        """
        I = self.I_std
        
        for i in range(K.shape[0]):
            K[i] = solve(I + G0[i] * V[i], V[i][:, -1])
        return -self.k[:, None] * K
    
    def ls_eq_no_interpolate_coupled(self, V, G0, K):
        """
        Solves the Lippmann-Schwinger equation for the coupled channels.

        Parameters
        ----------
        V : array
            Potential.
        G0 : array
            Partial-wave Green's function for free-space scattering.
        K : array
            Pre-defined array for K-matrix.

        Returns
        -------
        K : array
            Full K-matrix.
        """
        I = self.I_std
        
        for i in range(K.shape[0]):
            K[i] = solve(I + G0[i] * V[i], V[i])
        return -0.5 * self.k[:, None, None] * (K + swapaxes(K, 1, 2))

    def ls_eq_Sp(
        self,
        V: ArrayLike,
        I: ArrayLike, 
        Sp: ArrayLike, 
        G0: ArrayLike, 
        K: ArrayLike
    ) -> ArrayLike:
        """
        Solves the Lippmann-Schwinger equation using Glockle spline method.
        
        Parameters
        ----------
        V : array, shape = (ps, ps)
            Potential matrix.
        I : array
            Identity matrix.
        Sp : array
            The interpolation vector from the Glockle spline.
        G0 : array
            Partial-wave Green's function.
        K : array
            Unfilled K-matrix.

        Returns
        -------
        K : array
            Filled K-matrix.
        """
        k = self.k
        for i, k0 in enumerate(k):
            ket = k0 * solve(I + V * G0[i], V @ Sp[i].T)
            K[i] = Sp[i] @ ket
        return -K
    
    def ls_eq_EC_basis(
        self,
        V: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Solves the Lippmann-Schwinger equation using Glockle spline method.
        
        Parameters
        ----------
        V : array, shape = (ps, ps)
            Potential matrix.

        Returns
        -------
        -K : array
        K_half_on_shell : array
        """
        k, len_k, N = self.k, self.len_k, self.N
        I, Sp, G0_sp = self.I, self.Sp, self.G0_sp
        
        if self.is_coupled:
            K = zeros((len_k, 2, 2), float)
            K_half_on_shell = zeros((len_k, 2 * N, 2), float)
            
        else:
            K = zeros(len_k, float)
            K_half_on_shell = zeros((len_k, N), float)
        
        for i, k0 in enumerate(k):
            ket = -k0 * solve(I + V * G0_sp[i], V @ Sp[i].T)
            K[i] = Sp[i] @ ket # Picks out K[-1][-1]
            K_half_on_shell[i] = ket
    
        return K, K_half_on_shell
    
    def _get_K_matrix(
        self, 
        V: ArrayLike
    ) -> ArrayLike:
        """
        Calculates K-matrix for single and coupled channels.
        
        Parameters
        ----------
        V : array, shape = (ps, ps)
            Potential matrix.
        
        Returns
        -------
        K : array
            Filled K-matrix.
        """
        I, Sp, G0_sp = self.I, self.Sp, self.G0_sp
        
        if self.is_coupled:
            K = zeros((self.len_k, 2, 2), float)
        else:
            K = zeros(self.len_k, float)
            
        K = self.ls_eq_Sp(V, I, Sp, G0_sp, K)
        return K
    
    def interp_uncoupled_pot_std(
        self,
        pot: ArrayLike,
        spl_order: int = 3
    ) -> ArrayLike:
        """
        Interpolates the potential to the k values using a 
        bivariate spline approximation over a rectangular mesh.
        
        Parameters
        ----------
        V : array, shape = (ps, ps)
            Potential matrix.
        spl_order : int
            Order of spline.            

        Returns
        -------
        V : array, shape = (ps + 1, ps + 1)
            Full potential.
        """
        k, ps = self.k, self.ps
        
        V = zeros((self.len_k, self.N + 1, self.N + 1), float)
        V_spline = RectBivariateSpline(ps, ps, pot, kx=spl_order, ky=spl_order)
        
        for i, k0 in enumerate(k):
            k_vals = append(ps, k0)
            grid_kx, grid_ky = meshgrid(k_vals, k_vals)
            V[i] = V_spline.ev(grid_kx, grid_ky)
        
        return V
        
    def interp_uncoupled_pot_glockle(
        self,
        pot: ArrayLike
    ) -> ArrayLike:
        """
        Interpolates the potential to the k values using the 
        Glockle spline approximation method.
        
        Parameters
        ----------
        V : array, shape = (ps, ps)
            Potential matrix.          

        Returns
        -------
        V : array, shape = (ps + 1, ps + 1)
            Full potential.
        """
        k = self.k
        Sp = self.Sp
        N = self.N
        
        V = zeros((self.len_k, N + 1, N + 1), float)
        
        for i, k0 in enumerate(k):
            V_half_on_shell = pot @ Sp[i]
            V_on_shell = Sp[i] @ V_half_on_shell
            
            V[i][-1, :N] = V[i][:N, -1] = V_half_on_shell
            V[i][-1, -1] = V_on_shell
            V[i][:N, :N] = pot
        
        return V
    
    def ls_eq_NC( 
        self,
        pot: ArrayLike,
        spl_order: int = 3
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Solves the Lippmann-Schwinger equation for uncoupled channels
        using build-in Python spline and interpolator.

        Parameters
        ----------
        pot : array, shape = (ps, ps)
            The potential we are using to calculate the phase shifts.
        spl_order : int
            Order of spline.

        Returns
        -------
        phase : array, shape = (1, len_k)
        """
        tau = self.tau
        I, G0 = self.I_std, self.G0
        len_k, N = self.len_k, self.N
        k, ps = self.k, self.ps
        
        K = zeros((len_k, N + 1, N + 1), float)
        V = zeros((len_k, N + 1, N + 1), float)
        V_spline = RectBivariateSpline(ps, ps, pot, kx=spl_order, ky=spl_order)
        
        for i, (k0, G0_i) in enumerate(zip(k, G0)):
            k_vals = append(ps, k0)
            grid_kx, grid_ky = meshgrid(k_vals, k_vals)
            V[i] = V_spline.ev(grid_kx, grid_ky)
            
            K[i] = -k0 * solve(I + G0_i * V[i], V[i])
            K[i] = (K[i] + K[i].T) / 2
            tau[i] = K[i][-1][-1]
            
        return tau, K, V
        
    def ls_eq_C( 
        self,
        pot: ArrayLike,
        spl_order: int = 3
    ) -> ArrayLike:
        """
        Solves the Lippmann-Schwinger equation for coupled channels
        using build-in Python spline and interpolator.

        Parameters
        ----------
        pot : array, shape = (ps, ps)
            The potential we are using to calculate the phase shifts.
        spl_order : int
            Order of spline.

        Returns
        -------
        phase : array, shape = (3, len_k)
        """
        tau = self.tau        
        I, G0 = self.I_std, self.G0
        len_k, N = self.len_k, self.N
        k, ps, ws, k_max = self.k, self.ps, self.ws, self.k_max
        
        K = zeros((len_k, 2 * (N + 1), 2 * (N + 1)), float)
        V = zeros((len_k, 2 * (N + 1), 2 * (N + 1)), float)
        
        V11_spline = RectBivariateSpline(ps, ps, pot[:N, :N], kx=spl_order, ky=spl_order)
        V12_spline = RectBivariateSpline(ps, ps, pot[:N, N:2*N], kx=spl_order, ky=spl_order)
        V21_spline = RectBivariateSpline(ps, ps, pot[N:2*N, :N], kx=spl_order, ky=spl_order)
        V22_spline = RectBivariateSpline(ps, ps, pot[N:2*N, N:2*N], kx=spl_order, ky=spl_order)        
        
        for i, (k0, G0_i) in enumerate(zip(k, G0)):
            k_vals = append(ps, k0)
            grid_kx, grid_ky = meshgrid(k_vals, k_vals)
 
            V11 = V11_spline.ev(grid_kx, grid_ky)
            V12 = V12_spline.ev(grid_kx, grid_ky)
            V21 = V21_spline.ev(grid_kx, grid_ky)
            V22 = V22_spline.ev(grid_kx, grid_ky)
            V[i] = concatenate((concatenate((V11, V12)), 
                                concatenate((V21, V22))), axis=1)
            
            K[i] = -k0 * solve(I + tile(G0_i, (2 * (N + 1), 2)) * V[i], V[i])
            K[i] = (K[i] + K[i].T) / 2

            tau = array([[K[i][N, N], K[i][N, 2 * N + 1]], 
                         [K[i][2 * N + 1, N], K[i][2 * N + 1, 2 * N + 1]]])
            
        return tau, K, V
    
    def phase_shifts(
        self, 
        V: ArrayLike, 
        use_glockle: bool = True,
        fix: bool = True
    ) -> ArrayLike:
        """
        Computes phase shifts in degrees given the solution to the LS equation.
        Can be used with both coupled and uncoupled channels.

        Parameters
        ----------
        V : array
            The on-shell solution to the LS equation.
        use_glockle : boolean (default = True)
            If True, uses the Glocke interpolation. If False, uses the built in Python function.
        fix : bool (default = True)
            Whether to try to make the phase shifts continuous, as opposed to jump by 180 degrees.

        Returns
        -------
        phase_shifts : if_coupled = False, shape = (1, len_k)
                       if_coupled = True, shape = (3, len_k)
        """
        tau = self.get_tau(V, use_glockle)
        
        if self.is_coupled:
            ps = self.phase_coupled_channels(-tau)
        else:
            ps = arctan(tau) * 180 / pi
                
        if fix:
            ps = fix_phases_continuity(ps, is_radians=False)
        return ps
    
    def phase_coupled_channels(
        self, 
        K: ArrayLike
    ) -> ArrayLike:
        """
        Computes phase shifts in degrees given the solution to the LS equation.
        Can be used with both coupled and uncoupled channels.

        Parameters
        ----------
        k0 : array
            The on-shell center-of-mass momenta.
        K : array
            The on-shell solution to the LS equation.
        fix_K : boolean (default=False)
            Used to correctly get half-on-shell piece of K depending on
            the method used for the calculation.

        Returns
        -------
        phase_shifts : shape = (3, len_k)
        """
        from numpy import atleast_1d

        def _fix_phases(delta_minus, delta_plus, eps, bar=True):
            delta_minus, delta_plus, eps = atleast_1d(delta_minus, delta_plus, eps)

            d = delta_minus - delta_plus
            offset = (d + pi / 2) // pi
            dm = delta_minus - offset * pi

            if bar:
                e_offset = (2 * eps + pi / 2) // pi
                e = eps - e_offset * pi / 2
            else:
                e_offset = (epsilon + pi / 2) // pi
                e = eps - e_offset * pi

            return dm, delta_plus, e

        def transform_phases(delta_minus, delta_plus, eps, to_bar=True):
            delta_minus, delta_plus, eps = atleast_1d(delta_minus, delta_plus, eps)

            d = delta_minus - delta_plus
            s = delta_minus + delta_plus

            if to_bar:
                e = 0.5 * arcsin(sin(2 * eps) * sin(d))
                diff = arcsin(tan(2 * e) / tan(2 * eps))
            else:
                e = 0.5 * arctan(tan(2 * eps) / sin(d))
                diff = arcsin(sin(2 * eps) / sin(2 * e))

            dm = 0.5 * (s + diff)
            dp = 0.5 * (s - diff)

            return dm, dp, e
            
        K11, K12, K22 = K[:, 0], K[:, 1], K[:, 3]
            
        e = 0.5 * arctan(2 * K12 / (K11 - K22))
        K_e = (K11 - K22) / cos(2 * e)
        delta_a = -arctan(0.5 * (K11 + K22 + K_e))
        delta_b = -arctan(0.5 * (K11 + K22 - K_e))
        
        delta_a, delta_b, e = transform_phases(delta_a, delta_b, e, to_bar=True)
        delta_a, delta_b, e = _fix_phases(delta_a, delta_b, e, bar=True)

        return degrees(stack([delta_a, delta_b, e], axis=0))

    def get_phase(
        self, 
        tau: ArrayLike
    ) -> ArrayLike:
        r"""
        Function used to calculation tau from the K matrix.
        Tau is defined as -K(k0, k0).
        
        Parameters
        ----------
        k0 : array
            The on-shell center-of-mass momenta.
        tau : array
            Defined as tau = -K(k0, k0).
            
        Returns
        -------
        ps : array
            Phase shifts
        """
        if self.is_coupled:
            ps = self.phase_coupled_channels(-tau)
        elif not self.is_coupled:
            ps = arctan(tau) * 180 / pi
            
        return fix_phases_continuity(ps, is_radians=False)

    def get_tau(
        self, 
        V: ArrayLike, 
        use_glockle: bool = True
    ) -> ArrayLike:
        r"""
        Function used to calculation tau from the K matrix.
        Tau is defined as K(k0, k0).
        
        Parameters
        ----------
        V : array, shape = (ps, ps)
            Potential matrix.
        use_glockle : boolean (default = True)
            If True, uses the Glocke interpolation. If False, uses the built in Python function.
            
        Returns
        -------
        tau : array
        """
        is_coupled = self.is_coupled
        
        if use_glockle is True:
            K = self._get_K_matrix(V)
            
            if is_coupled:
                K00, K02, K20, K22 = K[:, 0, 0], K[:, 0, 1], K[:, 1, 0], K[:, 1, 1]
                tau = vstack((hstack((K00, K02)), hstack((K20, K22))))
            else:
                tau = K
                
        elif use_glockle is False:
            if is_coupled:
                tau = self.ls_eq_C(V)[0]
            else:
                tau = self.ls_eq_NC(V)[0]
                
        else:
            tau = self.ls_eq_no_interpolate(V)[0]
                
        return tau
    
    
