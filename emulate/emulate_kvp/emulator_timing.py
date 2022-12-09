
"""
Used to apply eigenvector continuation (EC) to a potential for
scattering applications.
"""
import numpy as np
from numba import jit, prange
from numpy import sum as np_sum
from numpy import (
    reshape, append, pad, zeros, ndarray, 
    log, pi, arctan, mean, log, conj, real, 
    identity, degrees, meshgrid, column_stack,
    full, block, array, stack, squeeze, ones, 
    swapaxes, moveaxis, einsum, resize, mean,
    tile
)
from numpy.linalg import solve
from scipy.linalg import lstsq, det
from numpy.typing import ArrayLike
from typing import Union, Optional
from math import ceil
from scipy.interpolate import RectBivariateSpline
from .kinematics import K_to_T, K_to_S, T_to_K, S_to_K
from .kinematics import e_lab_to_k_cm
from .utils import glockle_cubic_spline

from .constants import (
    mass_proton, mass_neutron, fm_to_sqrt_mb
)

@jit(nopython=True, parallel=True, fastmath=True)
def lstsq_parallel(mat, a, b, len_b, h):
    for i in prange(a.shape[0]):
        mat[i] = np.linalg.lstsq(a[i], b[i].T, rcond=h)[0][:len_b]

    return mat

def fix_coupled_lecs(
    lecs: ArrayLike,
    wave: Optional[str] = None
) -> ArrayLike:
    """
    """
    if '3S1' in wave:
        lecs = array([lecs[0], lecs[1], lecs[3], lecs[2], 
                      lecs[5], lecs[2], lecs[5], lecs[4]])
        
    elif '3P2' in wave:
        lecs = array([lecs[0], lecs[1], lecs[2], lecs[2], lecs[3]])
        
    return lecs

def calc_G0(
    k: ArrayLike,
    ps: ArrayLike,
    ws: ArrayLike
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
    G0 = zeros((len(k), len(ps) + 1), float)  

    for i, k0 in enumerate(k):
        D = ps ** 2 * ws / ( ps ** 2 - k0 ** 2 )
        D_k0 = -k0 ** 2 * np_sum(ws / (ps ** 2 - k0 ** 2))
        G0[i] = (2 / pi) * append(D, D_k0)

    return G0
    

class KVP_emulator:
    """
    A class that can either simulate or emulate two-body scattering observables 
    in momentum space via Kohn variational principle.
    
    Parameters
    ----------
    wave : str
        Denotes the partial wave.
    V0 : array
        The piece of the potential that does not depend of parameters, in units of energy.
    V1 : array
        The piece of the potential that is linear in the parameters.
    fix_seed : int (default=None)
        Used to seed the random nugget generator for the nugget.
    """
    def __init__(
        self, 
        E: ArrayLike, 
        ps: ArrayLike, 
        ws: ArrayLike, 
        V0: ArrayLike, 
        V1: ArrayLike,
        wave: str,
        is_coupled: bool = False,
    ):
        if (isinstance(E, ndarray) == False):
            E = array([E])
            
        E_tot, k = e_lab_to_k_cm(E, mass_neutron, mass_proton, True)
        self.Sp = glockle_cubic_spline(ps, k)
        G0 = calc_G0(k, ps, ws)
        
        self.len_k = len(k)
        self.len_ps = len(ps)
        
        if is_coupled:
            self.tau = zeros((self.len_k, 2, 2), float)
            I_ls = identity(2 * (self.len_ps + 1), float)
            K = zeros((self.len_k, 2 * (self.len_ps + 1), 2 * (self.len_ps + 1)), float)
            G0_coup_std = zeros((self.len_k, 2 * (self.len_ps + 1), 2 * (self.len_ps + 1)), float)
            
            for i in range(G0.shape[0]):
                G0_coup_std[i] = np.tile(G0[i], (2 * (self.len_ps + 1), 2))
            
            self.I_ls, self.K = I_ls, K
            self.G0_coup_std = G0_coup_std
            
        else:
            self.G0 = G0
            self.tau = zeros(self.len_k)
            self.I_ls = identity(self.len_ps + 1)
            self.K = zeros((self.len_k, self.len_ps + 1), float)
            
        self.k = k
        self.ps = ps
        self.ws = ws
        self.V0 = V0
        self.V1 = V1
        self.wave = wave
        self.is_coupled = is_coupled
        
        if self.is_coupled:
            self.tau_var = zeros((3, self.len_k), dtype=complex)
        
        self.mat_trans_K = array([[1, 0], [0, 1]])
        self.mat_trans_S = array([[-1j, 1], [-1j, -1]])
        self.mat_trans_T = array([[1, 0], [1j, 1]])

    
    def high_fidelity(
        self,
        lecs: ArrayLike
    ) -> ArrayLike:
        """
        """
        if '3S1' in self.wave or '3P2' in self.wave:
            lecs = fix_coupled_lecs(lecs, self.wave)
        
        tau, _ = self.ls_eq_no_interpolate(self.V0 + self.V1 @ lecs)
            
        if self.is_coupled:
            tau = tau.swapaxes(0, 1).reshape(2, 2 * self.len_k, order='F')
            tau = reshape(tau, (4, self.len_k)).T
            
        return tau
    
    def ls_eq_no_interpolate(self, V):
        """
        """
        if self.is_coupled:
            N = self.len_ps
            K = self.ls_eq_no_interpolate_coupled(V, self.G0_coup_std, self.K)
            tau = array([[K[:, N, N], K[:, N, 2 * N + 1]], 
                         [K[:, 2 * N + 1, N], K[:, 2 * N + 1, 2 * N + 1]]]).T
            
        else:
            K = self.ls_eq_no_interpolate_uncoupled(V, self.G0, self.K)
            tau = K[:, -1]
        
        return tau, K
    
    def ls_eq_no_interpolate_uncoupled(self, V, G0, K):
        I = self.I_ls
        
        for i in range(K.shape[0]):
            K[i] = solve(I + G0[i] * V[i], V[i][:, -1])
        return -self.k[:, None] * K
    
    def ls_eq_no_interpolate_coupled(self, V, G0, K):
        I = self.I_ls
        
        for i in range(K.shape[0]):
            K[i] = solve(I + G0[i] * V[i], V[i])
        
        return -0.5 * self.k[:, None, None] * (K + swapaxes(K, 1, 2))
    
    def create_basis(
        self,
        wave: str, 
        lecs: ArrayLike
    ) -> None:  
        """
        Calculates the basis needed to train the emulator.

        Parameters
        ----------
        wave : str
            Specifies which partial wave is being considered.
            Example: '1S0', '3P1', etc.
        lecs : array
            An array containing the LECs for the training points.

        Returns
        -------
        None
        """
        k = self.k
        ps = self.ps
        V0 = self.V0
        V1 = self.V1
        len_k = self.len_k
        len_ps = self.len_ps
        is_coupled = self.is_coupled
        
        len_basis = len(lecs)
        
        if is_coupled:
            tau_b = zeros((len_basis, len_k, 2, 2))
            K_b = zeros((len_basis, len_k, 2 * (len_ps + 1), 2 * (len_ps + 1)))
            V_b = zeros((len_basis, len_k, 2 * (len_ps + 1), 2 * (len_ps + 1)))

        else:
            tau_b = zeros((len_basis, len_k))
            K_b = zeros((len_basis, len_k, len_ps + 1))
            V_b = zeros((len_basis, len_k, len_ps + 1, len_ps + 1))
        
        for i, lec_i in enumerate(lecs):
            if '3S1' in wave or '3P2' in wave:
                lec_i = fix_coupled_lecs(lec_i, wave)

            V_b[i] = V0 + V1 @ lec_i
            tau_b[i], K_b[i] = self.ls_eq_no_interpolate(V_b[i])
            
        self.lecs_test = lecs
        self.len_basis = len_basis
        self.I = resize(identity(len_basis), 
                        (self.len_k, len_basis, len_basis))
        return V_b, swapaxes(tau_b, 0, 1), moveaxis(K_b, 0, 1)
    
    def _compute_U_from_U0_U1(
        self, 
        lecs: ArrayLike, 
        U0: ArrayLike, 
        U1: ArrayLike
    ) -> ArrayLike:
        """
        """
        if '3S1' in self.wave or '3P2' in self.wave:
            lecs = fix_coupled_lecs(lecs, self.wave)
        
        return U0 + U1 @ lecs
    
    def _get_emulation_method(
        self,
        tau_b: ArrayLike, 
        emu_method: str
    ) -> ArrayLike:
        """
        """
        if emu_method == 'K':
            matrix_det = det(self.mat_trans_K)
            factor = 1
            sign = 1
        elif emu_method == '1/K':
            trans_matrix = self.mat_trans_K
            trans_matrix[[0, 1]] = trans_matrix[[1, 0]]
            matrix_det = det(trans_matrix)
            tau_b = 1 / tau_b
            factor = -tau_b
            sign = -1
        elif emu_method == 'T':
            matrix_det = det(self.mat_trans_T).real
            factor = 1 / (1 - 1j * tau_b)
            tau_b = K_to_T(tau_b)
            sign = 1
        elif emu_method == 'S':
            matrix_det = det(self.mat_trans_S)
            factor = 2 * 1j / (-1 + 1j * tau_b)
            tau_b = K_to_S(tau_b)
            sign = 1 / (2 * 1j)
        else:
            raise ValueError('Specify boundary condition!')

        return tau_b, matrix_det, factor, sign
    
    def _fix_tau(
        self,
        tau_i: ArrayLike,
        emu_method: str
    ) -> ArrayLike:
        """
        """
        if emu_method == 'K':
            tau_fix = tau_i
        elif emu_method == '1/K':
            tau_fix = 1 / tau_i
        elif emu_method == 'T':
            tau_fix = T_to_K(tau_i)
        elif emu_method == 'S':
            tau_fix = S_to_K(tau_i)
        else:
            raise ValueError('Specify boundary condition!')
    
        return tau_fix.real
    
    def train(
        self,
        train_lecs: ArrayLike,
        glockle: bool = False,
        method: str = 'all'
    ) -> ArrayLike:
        """
        """
        len_k = self.len_k
        len_basis = len(train_lecs)
        
        V_b, tau_b, K_b = self.create_basis(self.wave, train_lecs)
        U0, U1 = self.train_emulator(V_b, tau_b, K_b, glockle)
        self.tau_b = tau_b
        
        if self.is_coupled:
            U0 = U0 + U0.swapaxes(3, 4)
            U1 = U1 + U1.swapaxes(3, 4)
        else:
            U0 = U0 + U0.swapaxes(1, 2)
            U1 = U1 + U1.swapaxes(1, 2)
        
        if glockle:
            self.U0_glockle, self.U1_glockle = U0, U1
        else:
            self.U0_std, self.U1_std = U0, U1
        
        tau_b_all = self._alt_boundary_conditions(glockle, method, U1.shape[-1])
        
        if self.is_coupled:
            tau = zeros((self.num_emulations, 3, len_k))
        else:
            tau = zeros((self.num_emulations, len_k))
            
        I = resize(identity(U0.shape[-1]), 
                   (U0.shape[0], U0.shape[1], U0.shape[2]))
        npad = ((0, 0), (0, 0), (0, 1), (0, 1))
        tau_expand = pad(tau_b_all / self.k[None, :, None], pad_width=npad[0:3], 
                         mode='constant', constant_values=1)
        dU_expand = pad(zeros((self.num_emulations, len_k, 
                               len_basis, len_basis), dtype=complex), 
                        pad_width=npad, mode='constant', constant_values=1)
        dU_expand[:, :, -1, -1] = 0
        
        self.dU_expand = dU_expand
        self.tau_expand = tau_expand
        self.I = I
            
        self.tau = tau
        self.c_j = zeros((self.len_k, self.len_basis), dtype=complex)
        return None
    
    def _emulation_method_list(
        self,
        method: str
    ) -> list:
        """
        """
        if method == 'all':
            emu_method_list = ['K', '1/K', 'T']
        elif method == 'K':
            emu_method_list = ['K']
        elif method == '1/K':
            emu_method_list = ['1/K']
        elif method == 'T':
            emu_method_list = ['T']
        elif method == 'S':
            emu_method_list = ['S']
        else:
            raise ValueError('Input emulation method!')

        self.num_emulations = len(emu_method_list)
        self.emu_method_list = emu_method_list
        
        return None
    
    def prediction(
        self,
        lecs: ArrayLike,
        glockle: bool, 
        sol: str,
        h: Union[float, ArrayLike] = None, 
    ) -> ArrayLike:
        """
        """
        tau = self.tau
        dU = self.dU_expand
        tau_expand = self.tau_expand
        
        if glockle:
            U0 = self.U0_arb_bc_glockle
            U1 = self.U1_arb_bc_glockle
            tau_b = self.tau_b_arb_bc_glockle
            sign = self.sign_arb_bc_glockle
            matrix_det = self.matrix_det_arb_bc_glockle
        else:
            U0 = self.U0_arb_bc_std
            U1 = self.U1_arb_bc_std
            tau_b = self.tau_b_arb_bc_std
            sign = self.sign_arb_bc_std
            matrix_det = self.matrix_det_arb_bc_std
        
        dU[:, :, :-1, :-1] = self._compute_U_from_U0_U1(lecs, U0, U1) + h * self.I
        
        for i in range(self.num_emulations):
            tau_pred = self._test_emulator(tau_expand[i], sign[i] * dU[i], 
                                           sol, h, matrix_det[i])
            tau[i] = self._fix_tau(tau_pred, self.emu_method_list[i])
        
        return tau
    
    def _alt_boundary_conditions(
        self,
        glockle: bool, 
        method: str,
        lecs_shape: int
    ) -> ArrayLike:
        """
        """
        self._emulation_method_list(method)
        tau = self._emulate_arbitrary_bc(lecs_shape, 
                                         self.emu_method_list, 
                                         self.is_coupled, glockle)
            
        if tau.shape[0] == 1:
            tau = squeeze(tau, axis=0)
            
        return tau
    
    def _emulate_arbitrary_bc(
        self,
        lecs_shape: int,
        method_list: list,
        is_coupled: bool,
        glockle: bool = None
    ) -> ArrayLike:
        """
        """
        len_k = self.len_k
        len_basis = self.len_basis
        num_emulations = self.num_emulations
        
        if is_coupled:
            tau = zeros((self.num_emulations, 3, len_k))
            tau_b_arb_bc = zeros((num_emulations, len_k, len_basis, 2, 2), dtype=complex)
            sign_arb_bc = zeros(num_emulations, dtype=complex)
            matrix_det_arb_bc = zeros(num_emulations, dtype=complex)
            U0_arb_bc = zeros((num_emulations, len_k, 4, 
                               len_basis, len_basis), dtype=complex)
            U1_arb_bc = zeros((num_emulations, len_k, 4, 
                               len_basis, len_basis, lecs_shape), dtype=complex)
        else:
            tau = zeros((self.num_emulations, len_k))
            tau_b_arb_bc = zeros((num_emulations, len_k, len_basis), dtype=complex)
            sign_arb_bc = zeros(num_emulations, dtype=complex)
            matrix_det_arb_bc = zeros(num_emulations, dtype=complex)
            U0_arb_bc = zeros((num_emulations, len_k, 
                               len_basis, len_basis), dtype=complex)
            U1_arb_bc = zeros((num_emulations, len_k, 
                               len_basis, len_basis, lecs_shape), dtype=complex)

        for m, emu_method in enumerate(method_list):
            if glockle:
                U0, U1 = self.U0_glockle, self.U1_glockle
            else:
                U0, U1 = self.U0_std, self.U1_std

            tau_b, matrix_det, factor, sign = self._get_emulation_method(self.tau_b, emu_method)

            if (isinstance(factor, ndarray) == True):
                
                if is_coupled:
                    U0, U1 = self._coupled_channel_normalization(len_k, U0, U1, factor)
                else:
                    U0 = einsum('ij, ijk, ik -> ijk', 
                                factor, U0, factor, optimize=True)
                    U1 = einsum('ij, ijkm, ik -> ijkm', 
                                factor, U1, factor, optimize=True)
            else:
                U0 = factor * U0 * factor
                U1 = factor * U1 * factor

            if is_coupled:
                U0 = swapaxes(np_sum(U0, axis=2), 2, 3)
                U1 = swapaxes(np_sum(U1, axis=2), 2, 3)
            
            U0_arb_bc[m] = U0
            U1_arb_bc[m] = U1
            tau_b_arb_bc[m] = tau_b
            sign_arb_bc[m] = sign
            matrix_det_arb_bc[m] = matrix_det
            
        if glockle:
            self.U0_arb_bc_glockle = U0_arb_bc
            self.U1_arb_bc_glockle = U1_arb_bc
            self.tau_b_arb_bc_glockle = tau_b_arb_bc
            self.sign_arb_bc_glockle = sign_arb_bc
            self.matrix_det_arb_bc_glockle = matrix_det_arb_bc
        else:
            self.U0_arb_bc_std = U0_arb_bc
            self.U1_arb_bc_std = U1_arb_bc
            self.tau_b_arb_bc_std = tau_b_arb_bc
            self.sign_arb_bc_std = sign_arb_bc
            self.matrix_det_arb_bc_std = matrix_det_arb_bc
            
        return tau_b_arb_bc
    
    def _coupled_channel_normalization(
        self,
        len_k: float,
        U0: ArrayLike, 
        U1: ArrayLike, 
        factor: Union[float, ArrayLike]
    ) -> Union[float, ArrayLike]:
        """
        """
        len_basis = self.len_basis
        factor = swapaxes(factor.reshape(len_k, 2 * len_basis, 2, order='F'), 1, 2)
        factor = stack((factor[:, 0, :len_basis], factor[:, 0, len_basis:], 
                        factor[:, 1, :len_basis], factor[:, 1, len_basis:]))
        U0_renorm = np.zeros_like(U0, dtype=complex)
        U1_renorm = np.zeros_like(U1, dtype=complex)

        for i in range(U0.shape[1]):
            U0_renorm[:, i, 0] = einsum('ij, ijk, ik -> ijk', 
                                        factor[i], U0[:, i, 0], factor[i], 
                                        optimize=True)
            U0_renorm[:, i, 1] = einsum('ij, ijk, ik -> ijk', 
                                        factor[i], U0[:, i, 1], factor[i], 
                                        optimize=True)
            U0_renorm[:, i, 2] = einsum('ij, ijk, ik -> ijk', 
                                        factor[i], U0[:, i, 2], factor[i], 
                                        optimize=True)
            U0_renorm[:, i, 3] = einsum('ij, ijk, ik -> ijk', 
                                        factor[i], U0[:, i, 3], factor[i], 
                                        optimize=True)

            U1_renorm[:, i, 0] = einsum('ij, ijkm, ik -> ijkm', 
                                        factor[i], U1[:, i, 0], factor[i], 
                                        optimize=True)
            U1_renorm[:, i, 1] = einsum('ij, ijkm, ik -> ijkm', 
                                        factor[i], U1[:, i, 1], factor[i], 
                                        optimize=True)
            U1_renorm[:, i, 2] = einsum('ij, ijkm, ik -> ijkm', 
                                        factor[i], U1[:, i, 2], factor[i], 
                                        optimize=True)
            U1_renorm[:, i, 3] = einsum('ij, ijkm, ik -> ijkm', 
                                        factor[i], U1[:, i, 3], factor[i], 
                                        optimize=True)        
        return U0_renorm, U1_renorm
    
#     def _solve_inv(
#         self, 
#         U_tilde: ArrayLike, 
#         tau: ArrayLike,
#         h: ArrayLike, 
#         matrix_det: bool
#     ) -> ArrayLike:
#         """
#         A choice for solving the U matrix and obtaining
#         the EC prediction using solve. 
#         This method uses the function np.linalg.solve.

#         Parameters
#         ----------
#         U : (basis size) x (basis size) array
#             Matrix used to make EC predictions.
#         tau : array
#             An array of the taus from the basis.
#         h : float
#             Nugget used to regulate the basis collinearity.
#         matrix_det : int
#             Determinant of matrix used for the boundary condition imposed.

#         Returns
#         -------
#         cj_tau - cj_U_cj : array
#             Variational prediction from the emulator.
#         """        
#         npad = ((0, 0), (0, 1), (0, 1))
#         b = pad(tau / self.k[:, None], pad_width=npad[0:2], mode='constant', constant_values=1)
#         A = pad(U_tilde + h * self.I, pad_width=npad, mode='constant', constant_values=1)
#         A[:, -1, -1] = 0
               
#         c_j = solve(A, b)[:, 0:self.len_basis]
# #         cj_tau = einsum('ij, ij -> i', c_j, tau, optimize=True)
# #         cj_U_cj = einsum('ij, ijk, ik -> i', c_j, U_tilde, c_j, optimize=True)
        
#         cj_tau = np_sum(c_j * tau, axis=1)
#         cj_U_cj = np_sum(c_j[:, :, None] * U_tilde * c_j[:, None, :], axis=(1, 2))
    
#         return cj_tau - matrix_det * cj_U_cj
        
#     def _leastsq_inv(
#         self, 
#         U_tilde: ArrayLike, 
#         tau: ArrayLike,
#         h: ArrayLike, 
#         matrix_det: bool
#     ) -> ArrayLike:
#         """
#         A choice for solving the U matrix and obtaining
#         the EC prediction using a least square solver. 
#         This method uses the function np.scipy.lstsq.

#         Parameters
#         ----------
#         U : (basis size) x (basis size) array
#             Matrix used to make EC predictions.
#         tau : array
#             An array of the taus from the basis.
#         h : ArrayLike
#             List of nugget used to regulate the basis collinearity.
#         matrix_det : int
#             Determinant of matrix used for the boundary condition imposed.

#         Returns
#         -------
#         cj_tau - cj_U_cj : array
#             Variational prediction from the emulator.
#         """
#         c_j = self.c_j
                
#         npad = ((0, 0), (0, 1), (0, 1))
#         b = pad(tau / self.k[:, None], pad_width=npad[0:2], mode='constant', constant_values=1)
#         A = pad(U_tilde, pad_width=npad, mode='constant', constant_values=1)
#         A[:, -1, -1] = 0
        
#         for i, (A_i, b_i) in enumerate(zip(A, b)):
#             c_j[i] = lstsq(A_i, b_i.T, cond=h)[0][0:self.len_basis]
            
# #         cj_tau = einsum('ij, ij -> i', c_j, tau, optimize=True)
# #         cj_U_cj = einsum('ij, ijk, ik -> i', c_j, U_tilde, c_j, optimize=True)

#         cj_tau = np_sum(c_j * tau, axis=1)
#         cj_U_cj = np_sum(c_j[:, :, None] * U_tilde * c_j[:, None, :], axis=(1, 2))
        
#         return cj_tau - matrix_det * cj_U_cj


    def _solve_inv(
        self, 
        U_tilde: ArrayLike, 
        tau: ArrayLike,
        h: ArrayLike, 
        matrix_det: bool
    ) -> ArrayLike:
        """
        A choice for solving the U matrix and obtaining
        the EC prediction using solve. 
        This method uses the function np.linalg.solve.

        Parameters
        ----------
        U : (basis size) x (basis size) array
            Matrix used to make EC predictions.
        tau : array
            An array of the taus from the basis.
        h : float
            Nugget used to regulate the basis collinearity.
        matrix_det : int
            Determinant of matrix used for the boundary condition imposed.

        Returns
        -------
        cj_tau - cj_U_cj : array
            Prediction from the emulator.
        """
        c_j = solve(U_tilde, tau)[:, 0:self.len_basis]
        cj_tau = np_sum(c_j * self.k[:, None] * tau[:, :-1], axis=1)
        cj_U_cj = np_sum(c_j[:, :, None] * U_tilde[:, :-1, :-1] * c_j[:, None, :], axis=(1, 2))
#         cj_tau = einsum('ij, ij -> i', c_j, tau[:, :-1], optimize=True)
#         cj_U_cj = einsum('ij, ijk, ik -> i', c_j, U_tilde[:, :-1, :-1], c_j, optimize=True)
    
        return cj_tau - matrix_det * cj_U_cj
    
    def _leastsq_inv(
        self, 
        U_tilde: ArrayLike, 
        tau: ArrayLike,
        h: ArrayLike, 
        matrix_det: bool
    ) -> ArrayLike:
        """
        A choice for solving the U matrix and obtaining
        the EC prediction using a least square solver. 
        This method uses the function np.scipy.lstsq.

        Parameters
        ----------
        U : (basis size) x (basis size) array
            Matrix used to make EC predictions.
        tau : array
            An array of the taus from the basis.
        h : ArrayLike
            List of nugget used to regulate the basis collinearity.
        matrix_det : int
            Determinant of matrix used for the boundary condition imposed.

        Returns
        -------
        cj_tau - cj_U_cj : array
            Variational prediction from the emulator.
        """
        c_j = self.c_j
        
#         c_j = lstsq_parallel(c_j, U_tilde, tau, self.len_basis, h)
        
        for i, (A_i, b_i) in enumerate(zip(U_tilde, tau)):
            c_j[i] = lstsq(A_i, b_i.T, cond=h)[0][0:self.len_basis]
            
        cj_tau = np_sum(c_j * self.k[:, None] * tau[:, :-1], axis=1)
        cj_U_cj = np_sum(c_j[:, :, None] * U_tilde[:, :-1, :-1] * c_j[:, None, :], axis=(1, 2))
#         cj_tau = einsum('ij, ij -> i', c_j, tau[:, :-1], optimize=True)
#         cj_U_cj = einsum('ij, ijk, ik -> i', c_j, U_tilde[:, :-1, :-1], c_j, optimize=True)
        
        return cj_tau - matrix_det * cj_U_cj
    
    def train_emulator(
        self, 
        V_b: ArrayLike,
        tau_b: ArrayLike,
        K_b: ArrayLike,
        glockle: bool
    ) -> None:
        """
        Emulator offline/training stage.

        Parameters
        ----------
        V_b : array
            Potentials used to build the basis.
        tau_b : array
            On-shell K values that correspond to build the basis.
        K_b : array
            K matrices used to build the basis.
        lecs : array
            An array containing the testing LECs.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
        
        Returns
        -------
        None
        """
        k = self.k
        len_k = self.len_k
        len_basis = self.len_basis
        is_coupled = self.is_coupled
        V0, V1 = self.V0, self.V1
        
        if is_coupled:
            U0 = zeros((len_k, 4, 4, len_basis, len_basis))
            U1 = zeros((len_k, 4, 4, len_basis, len_basis, V1.shape[-1]))
            
        else:
            U0 = zeros((len_k, len_basis, len_basis))
            U1 = zeros((len_k, len_basis, len_basis, V1.shape[-1]))
            
        V_b = swapaxes(V_b, 0, 1)
        
        for i, (k_i, K_b_i, V_b_i, tau_b_i) in enumerate(zip(k, K_b, V_b, tau_b)):
            
            U0[i], U1[i] = self._calculate_delta_U(k_i, K_b_i, V_b_i, 
                                                   tau_b_i, V0[i], V1[i], 
                                                   glockle, self.Sp[i])
        return U0, U1
    
    def _calculate_delta_U(
        self, 
        k0: int,
        K_b: ArrayLike, 
        V_b: ArrayLike, 
        tau_b: ArrayLike, 
        V0: ArrayLike,
        V1: ArrayLike,
        glockle: bool, 
        Sp: Optional[ArrayLike] = None,
    ) -> None:
        r"""
        Calculate the overlap matrix \Delta U_{ij}.

        Parameters
        ----------
        k0 : float
            Value of k at which we are predicting the phase shifts.
        K_b : array
            K matrix used to train the basis.
        V_b : array
            Potential used to train the basis.
        tau_b : array
            On-shell K values that correspond to the basis.
        V0 : array
            Parameter-independent part of potential.
        V1 : array
            Parameter-dependent part of potential.
        lecs : array
            An array containing the testing LECs.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
        Sp : array, optional (default=None)
            The vector matrix from the Glockle spline.
            
        Returns
        -------
        None
        """
        ps, ws = self.ps, self.ws

        if glockle:
            if not self.is_coupled:
                U0, U1 = self._delta_U_momentum_glockle(k0, ps, ws, Sp, K_b, 
                                                        V_b, tau_b, V0, V1,)
            if self.is_coupled:
                U0, U1 = self._delta_U_momentum_glockle_coupled(k0, ps, ws, Sp, K_b, 
                                                                V_b, tau_b, V0, V1)

        else:
            if not self.is_coupled:
                U0, U1 = self._delta_U_momentum(k0, ps, ws, K_b, V_b, 
                                                tau_b, V0, V1)
            if self.is_coupled:
                U0, U1 = self._delta_U_momentum_coupled(k0, ps, ws, K_b, V_b, 
                                                        tau_b, V0, V1)
                
        return U0, U1
            
    
    def _test_emulator(
        self, 
        tau_b: ArrayLike, 
        U: ArrayLike, 
        sol_type: str, 
        h: float,
        matrix_det: bool
    ) -> ArrayLike:
        """
        Sets up the online/testing stage of emulator.

        Parameters
        ----------
        tau_b : array
            On-shell K values that correspond to the basis.
        U : array
            Overlap matrix found in expression for variational prediction.
        sol : str
            Chooses method used to calculate the basis weights.
            Options: 'lstsq', 'solve', 'pinv'
        h : float
            Nugget used to regulate the basis collinearity.
        matrix_det : int
            Determinant of matrix used for the boundary condition imposed.
            
        Returns
        -------
        tau_var : array
            Variational prediction from the emulator.
        """                       
        if self.is_coupled:
            len_basis, tau_var = self.len_basis, self.tau_var
            
            U = stack((U[:, :len_basis, :len_basis], 
                       U[:, len_basis:, :len_basis], 
                       U[:, len_basis:, len_basis:]))
            
            tau = stack((tau_b[:, :, 0][:, :, 0], 
                         tau_b[:, :, 0][:, :, 1], 
                         tau_b[:, :, 1][:, :, 1]))
            
            for i, (tau_i, U_i) in enumerate(zip(tau, U)):
                tau_var[i] = self._get_emulated_tau(sol_type, tau_i, U_i, h, matrix_det)
        else:
            tau_var = self._get_emulated_tau(sol_type, tau_b, U, h, matrix_det)
            
        return tau_var
    
    def _get_emulated_tau(
        self, 
        sol_type: str,
        tau: ArrayLike, 
        U: ArrayLike,
        nugget: Union[float, ArrayLike],
        matrix_det: bool, 
        seed_rand: bool = False
    ) -> ArrayLike:
        """
        Applies the emulator to interpolate/extrapolate the 
        on-shell K for a specific scattering energy.

        Parameters
        ----------
        sol_type : str
            Chooses method used to calculate the basis weights.
            Options: 'lstsq', 'solve', 'pinv'
        tau : array
            Values of tau corresponding to the basis.
        U : array
            Overlap matrix found in expression for variational prediction.
        nugget : float or list
            Nugget used to regulate the basis collinearity.
            If None, a random value within some range is chosen.
        matrix_det : int
            Determinant of matrix used for the boundary condition imposed.
        seed_rand : boolean (default=False)
            Used to seed the random draw for the nugget. 
            If True, it seeds. If False, no seed.
            
        Returns
        -------
        tau_var : float
            Returns the variational tau calculated using EC.
        """
#         nugget = self._get_nugget(h, seed_rand)
            
        if sol_type == 'lstsq':
            tau_var = self._leastsq_inv(U, tau, nugget, matrix_det)
        elif sol_type == 'solve':
            tau_var = self._solve_inv(U, tau, nugget, matrix_det)
        else:
            raise Exception('Specify how to solve dU matrix!')
        return tau_var
    
    def _get_nugget(
        self, 
        h: Union[float, list], 
        seed_rand: bool
    ) -> ArrayLike:
        """
        Used to randomly choose nugget in some interval or, if an int is passed,
        fill an array with that value.

        Parameters
        ----------
        h_value : float or list
            Nugget used to regulate the basis collinearity.
        seed_rand : boolean
            Used to seed the random draw. 
            If True, it seeds. If False, no seed.
            
        Returns
        -------
        nugget : array
            An array of parameters used to regulate delta U tilde.
        """
        if (isinstance(h, list) == True and len(h) == 2):
            from numpy import random, log10
            h = log10(h)
            h.sort()
            
            if seed_rand:
                random.seed(self.rand_seed)
                
            nugget = 10**random.uniform(size=self.len_k, low=h[0].real, high=h[1].real)
            
            if (isinstance(h[0], complex) == True):
                nugget = nugget * 1j
            
        elif (isinstance(h, float) == True) or (isinstance(h, complex) == True):
            nugget = full(self.len_k, h)
            
        elif (h == None):
            nugget = full(self.len_k, 0)
            
        else:
            raise ValueError("Wrong entry for nugget!")
            
        return nugget
    
    def _delta_U_momentum(
        self,
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        tau_b: ArrayLike,
        V0: ArrayLike,
        V1: ArrayLike
    ) -> ArrayLike:
        r"""
        Calculates the kernel matrix \delta \tilde{U}_{ij} used for the emulator
        calculation in momentum space without Glockle interpolation for single channel.

        Parameters
        ----------
        k0 : float
            Value of k at which we are predicting the phase shifts.
        ps : array
            Mesh points.
        ws : array
            Mesh weights.
        K_b : array
            K matrix used to train the basis.
        V_b : array
            Potential used to train the basis.
        lecs : array
            An array containing the testing LECs.
        
        Returns
        -------
        U_ij : array
            Returns the kernel matrix.
        """
        len_basis = self.len_basis
        
        U0 = zeros((len_basis, len_basis))
        U1 = zeros((len_basis, len_basis, V1.shape[-1]))
        
        k = append(ps, k0)
        G0 = ps**2 * ws / (ps**2 - k0**2)
        G0_sum = np_sum(k0**2 * ws / (ps**2 - k0**2))
        
        Vb_bot, Vb_k0 = self._V_b_terms_non_glockle(k, G0, V_b)
        K_bot, K_k0 = self._K_b_terms_non_glockle(k, G0, K_b)
        
        V0_k0, V0_bot, V0 = self._get_V0_terms(k, ps, V_b, V0, Vb_k0, Vb_bot)
        V1_k0, V1_bot, V1 = self._get_V1_terms(k, ps, V_b, V1, Vb_k0, Vb_bot)
        
        V0_K_k0 = V0_k0 * K_k0
                
        if V1_k0.size == 1:
            if K_b.shape[0] == 1:
                V1_K_k0 = array([[K_k0 * V1_k0]])
            else:
                V1_K_k0 = K_k0[:, None] * V1_k0
        else:    
            V1_K_k0 = K_k0[:, None] * V1_k0[None, :]
            
        V0_K_bot = V0_bot * K_bot * G0
        wf_vec = G0 * K_bot
        
        if K_b.shape[0] == 1:
            V1_K_bot = G0[None, :, None] * (K_bot[None, :, None] * V1_bot)
            wf_vec = wf_vec[None,:]
            K_k0 = K_k0[None]
            K_bot = K_bot[None, :]
            V0_k0 = V0_k0[None]
            V0_K_k0 = V0_K_k0[None]
        else:
            V1_K_bot = G0[None, :, None] * (K_bot[:, :, None] * V1_bot)
        
        I0_4_11 = einsum('ij, ijk, lk -> li', 
                         conj(wf_vec), V0, wf_vec, optimize=True)
        I1_4_11 = swapaxes(swapaxes(swapaxes(
                         conj(wf_vec) @ V1[:-1, :-1], 0, 2) @ wf_vec.T, 0, 2), 0, 1)
        
        for i, (Ki_k0, Ki_bot) in enumerate(zip(K_k0, K_bot)):
            for j, (Kj_k0, Kj_bot) in enumerate(zip(K_k0, K_bot)):
                U0[i][j] = self._get_U0(k0, G0_sum, G0, V0_k0[j], V0_K_k0[j], V0_bot[j], \
                                        V0_K_bot[j], Ki_k0, Kj_k0, Ki_bot, I0_4_11[i][j])
                
                U1[i][j] = self._get_U1(k0, G0_sum, V1_k0, V1_K_k0[i], V1_K_k0[j], \
                                        V1_K_bot[i], V1_K_bot[j], Ki_k0, Kj_k0, I1_4_11[i][j])
        return U0, U1
    
    def _get_U0(
        self, 
        k0: float,
        G0_sum: float,
        G0: ArrayLike,
        V0_k0: ArrayLike,
        V0_K_k0: ArrayLike,
        V0_bot: ArrayLike,
        V0_K_bot: ArrayLike,
        Ki_k0: ArrayLike,
        Kj_k0: ArrayLike,
        Ki_bot: ArrayLike,
        I0_4_11: ArrayLike
    ) -> float:
        """
        Calculates the parameter-independent piece of the overlap matrix
        for the single channel calculation without Glockle interpolation.
        """
        I0_23 = np_sum(G0 * V0_bot * Ki_bot + V0_K_bot)
        I0_23_PV = (Ki_k0 * V0_k0 + V0_K_k0) * G0_sum
        I0_4_k02 = np_sum((G0 * Kj_k0 * Ki_bot * V0_bot + Ki_k0 * V0_K_bot)) * G0_sum
        I0_4 = I0_4_11 - I0_4_k02 + Ki_k0 * Kj_k0 * V0_k0 * G0_sum**2
        
        return V0_k0 + (2.0 / pi) * (I0_23 - I0_23_PV) / k0 + (2.0 / pi)**2 * I0_4 / k0**2
        
    def _get_U1(
        self, 
        k0: float, 
        G0_sum: float,
        V1_k0: ArrayLike,
        V1_Ki_k0: ArrayLike,
        V1_Kj_k0: ArrayLike,
        V1_Ki_bot: ArrayLike,
        V1_Kj_bot: ArrayLike,
        Ki_k0: ArrayLike,
        Kj_k0: ArrayLike,
        I1_4_11: ArrayLike
    ) -> float:
        """
        Calculates the parameter-dependent piece of the overlap matrix
        for the single channel calculation without Glockle interpolation.
        """
        I1_23 = np_sum(V1_Ki_bot + V1_Kj_bot, axis=0) - (V1_Ki_k0 + V1_Kj_k0) * G0_sum
        I1_4_k02 = np_sum((V1_Ki_bot * Kj_k0  + Ki_k0 * V1_Kj_bot), axis=0) * G0_sum
        I1_4 = I1_4_11 - I1_4_k02 + Ki_k0 * Kj_k0 * V1_k0 * G0_sum**2
        
        return V1_k0 + (2.0 / pi) * I1_23 / k0 + (2.0 / pi)**2 * I1_4 / k0**2
    
    def _V_b_terms_non_glockle(
        self,
        k: ArrayLike,
        ps: ArrayLike, 
        V_b: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Extracts the pieces needed from the potential basis for the single
        channel calculation without Glockle interpolation.
        """
        Vb_k0 = squeeze(V_b[:, ps.shape[0]:k.shape[0], ps.shape[0]:k.shape[0]])
        Vb_bot = squeeze(V_b[:, ps.shape[0]:k.shape[0], :ps.shape[0]])
        
        return Vb_bot, Vb_k0
    
    def _K_b_terms_non_glockle(
        self,
        k: ArrayLike,
        ps: ArrayLike,
        K_b: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Extracts the pieces needed from the K matrix basis for the single
        channel calculation without Glockle interpolation.
        """
        ### K-matrix basis ###
        Kb_bot = squeeze(K_b[:, :ps.shape[0]])
        Kb_k0 = squeeze(K_b[:, -1])
        
        return Kb_bot, Kb_k0
    
    def _get_V0_terms(
        self,
        k: ArrayLike,
        ps: ArrayLike, 
        V_b: ArrayLike,
        V0: ArrayLike,
        Vb_k0_all: ArrayLike,
        Vb_bot_all: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Extracts the pieces needed from the parameter-independent part of
        the potential for the single channel calculation without Glockle interpolation.
        """
        if self.is_coupled:
            V0 = V0[:len(ps), :len(ps)]
            
        V0_k0_all = squeeze(V0[ps.shape[0]:k.shape[0], ps.shape[0]:k.shape[0]])
        V0_bot_all = squeeze(V0[ps.shape[0]:k.shape[0], :ps.shape[0]])
        V0_k0 = V0_k0_all - Vb_k0_all
        V0_bot = V0_bot_all - Vb_bot_all
        V0_diff = (V0 - V_b)[:, :-1, :-1]
    
        return V0_k0, V0_bot, V0_diff
    
    def _get_V1_terms(
        self,
        k: ArrayLike,
        ps: ArrayLike, 
        V_b: ArrayLike,
        V1: ArrayLike,
        Vb_k0_all: ArrayLike,
        Vb_bot_all: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Extracts the pieces needed from the parameter-dependent part of
        the potential for the single channel calculation without Glockle interpolation.
        """
        if self.is_coupled:
            V1 = V1[:len(ps), :len(ps)]
        
        ### lecs-dependent potential ###
        V1 = moveaxis(V1, -1, 0)
        
        V1_k0_all = squeeze(V1[:, ps.shape[0]:k.shape[0], ps.shape[0]:k.shape[0]])
                
        V1_bot_all = V1[:, ps.shape[0]:k.shape[0], :ps.shape[0]]
        V1_bot_all = moveaxis(V1_bot_all, 0, -1)
        V1 = moveaxis(V1, 0, -1)
        
        return V1_k0_all, V1_bot_all, V1
    
    def _get_potential_K_coupled(
        self,
        V_b: ArrayLike,
        K_b: ArrayLike,
        V0: ArrayLike,
        V1: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Extracts the pieces needed from the potential operators, potential
        basis and K matrix basis for the coupled channel calculation
        without Glockle interpolation.
        """
        N = self.len_ps + 1
        
        V0_stack = stack((V0[:N, :N], V0[:N, N:], 
                          V0[N:, :N], V0[N:, N:]))
        V1_stack = stack((V1[:N, :N, :], V1[:N, N:, :], 
                          V1[N:, :N, :], V1[N:, N:, :]))
        Vb_stack = stack((V_b[:, :N, :N], V_b[:, :N, N:], 
                          V_b[:, N:, :N], V_b[:, N:, N:]))
        Kb_stack = stack((K_b[:, :N, :N], K_b[:, :N, N:], 
                          K_b[:, N:, :N], K_b[:, N:, N:]))
        
        return V0_stack, V1_stack, Vb_stack, Kb_stack
    
    def _delta_U_momentum_coupled(
        self,
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        tau_b: ArrayLike,
        V0: ArrayLike,
        V1: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the kernel matrix \delta \tilde{U}_{ij} used for the emulator
        calculation in momentum space without Glockle interpolation for coupled channels.

        Parameters
        ----------
        k0 : float
            Value of k at which we are predicting the phase shifts.
        ps : array
            Mesh points.
        ws : array
            Mesh weights.
        K_b : array
            K matrix used to train the basis.
        V_b : array
            Potential used to train the basis.
        lecs : array
            An array containing the testing LECs.
        
        Returns
        -------
        U_ij : array
            Returns the kernel matrix.
        """
        len_basis = self.len_basis        
        
        G0 = ps**2 * ws / (ps**2 - k0**2)
        G0_sum = np_sum(k0**2 * ws / (ps**2 - k0**2))
        V0_stack, V1_stack, Vb_stack, Kb_stack = self._get_potential_K_coupled(V_b, K_b, V0, V1)
        len_lecs = V1.shape[-1]

        U0_I00_00 = zeros((len_basis, len_basis))
        U0_I00_02 = zeros((len_basis, len_basis))
        U0_I00_20 = zeros((len_basis, len_basis))
        U0_I00_22 = zeros((len_basis, len_basis))
        U1_I00_00 = zeros((len_basis, len_basis, len_lecs))
        U1_I00_02 = zeros((len_basis, len_basis, len_lecs))
        U1_I00_20 = zeros((len_basis, len_basis, len_lecs))
        U1_I00_22 = zeros((len_basis, len_basis, len_lecs))

        U0_I00 = zeros((len_basis, len_basis))
        U1_I00 = zeros((len_basis, len_basis, len_lecs))
        
        U0_I02_00 = zeros((len_basis, len_basis))
        U0_I02_02 = zeros((len_basis, len_basis))
        U0_I02_20 = zeros((len_basis, len_basis))
        U0_I02_22 = zeros((len_basis, len_basis))
        U1_I02_00 = zeros((len_basis, len_basis, len_lecs))
        U1_I02_02 = zeros((len_basis, len_basis, len_lecs))
        U1_I02_20 = zeros((len_basis, len_basis, len_lecs))
        U1_I02_22 = zeros((len_basis, len_basis, len_lecs))
    
        U0_I02 = zeros((len_basis, len_basis))
        U1_I02 = zeros((len_basis, len_basis, len_lecs))
        
        U0_I20_00 = zeros((len_basis, len_basis))
        U0_I20_02 = zeros((len_basis, len_basis))
        U0_I20_20 = zeros((len_basis, len_basis))
        U0_I20_22 = zeros((len_basis, len_basis))
        U1_I20_00 = zeros((len_basis, len_basis, len_lecs))
        U1_I20_02 = zeros((len_basis, len_basis, len_lecs))
        U1_I20_20 = zeros((len_basis, len_basis, len_lecs))
        U1_I20_22 = zeros((len_basis, len_basis, len_lecs))

        U0_I20 = zeros((len_basis, len_basis))
        U1_I20 = zeros((len_basis, len_basis, len_lecs))
        
        U0_I22_00 = zeros((len_basis, len_basis))
        U0_I22_02 = zeros((len_basis, len_basis))
        U0_I22_20 = zeros((len_basis, len_basis))
        U0_I22_22 = zeros((len_basis, len_basis))
        U1_I22_00 = zeros((len_basis, len_basis, len_lecs))
        U1_I22_02 = zeros((len_basis, len_basis, len_lecs))
        U1_I22_20 = zeros((len_basis, len_basis, len_lecs))
        U1_I22_22 = zeros((len_basis, len_basis, len_lecs))

        U0_I22 = zeros((len_basis, len_basis))
        U1_I22 = zeros((len_basis, len_basis, len_lecs))
        
        
        V0_00, V0_02 = V0_stack[0] - Vb_stack[0], V0_stack[1] - Vb_stack[1]
        V0_20, V0_22 = V0_stack[2] - Vb_stack[2], V0_stack[3] - Vb_stack[3]
        
        V1_00, V1_02 = V1_stack[0], V1_stack[1]
        V1_20, V1_22 = V1_stack[2], V1_stack[3]
        
        K00, K02 = Kb_stack[0], Kb_stack[1]
        K20, K22 = Kb_stack[2], Kb_stack[3]
        
        for i in range(K_b.shape[0]):
            for j in range(K_b.shape[0]):
                
                #00 integral
                U0_I00_00[i][j] = self._U0_integral(k0, G0_sum, G0, V0_00[j], K00[i], K00[j])
                U1_I00_00[i][j] = self._U1_integral(k0, G0_sum, G0, V1_00, K00[i], K00[j])
                U0_I00_02[i][j] = self._U0_integral_2_4(k0, G0_sum, G0, V0_02[j], K00[i], K02[j])
                U1_I00_02[i][j] = self._U1_integral_2_4(k0, G0_sum, G0, V1_02, K00[i], K02[j])
                U0_I00_20[i][j] = self._U0_integral_3_4(k0, G0_sum, G0, V0_20[j], K02[i], K00[j])
                U1_I00_20[i][j] = self._U1_integral_3_4(k0, G0_sum, G0, V1_20, K02[i], K00[j])
                U0_I00_22[i][j] = self._U0_integral_4(k0, G0_sum, G0, V0_22[j], K02[i], K02[j])
                U1_I00_22[i][j] = self._U1_integral_4(k0, G0_sum, G0, V1_22, K02[i], K02[j])
                
                U0_I00[i][j] = U0_I00_00[i][j] + U0_I00_02[i][j] + U0_I00_20[i][j] + U0_I00_22[i][j]
                U1_I00[i][j] = U1_I00_00[i][j] + U1_I00_02[i][j] + U1_I00_20[i][j] + U1_I00_22[i][j]
                
                #02 integral
                U0_I02_00[i][j] = self._U0_integral_2_4(k0, G0_sum, G0, V0_00[j], K00[i], K20[j])
                U1_I02_00[i][j] = self._U1_integral_2_4(k0, G0_sum, G0, V1_00, K00[i], K20[j])
                U0_I02_02[i][j] = self._U0_integral_1_mixing(k0, G0_sum, G0, V0_02[j], K00[i], K22[j])
                U1_I02_02[i][j] = self._U1_integral_1_mixing(k0, G0_sum, G0, V1_02, K00[i], K22[j])
                U0_I02_20[i][j] = self._U0_integral_4_mixing(k0, G0_sum, G0, V0_20[j], K02[i], K20[j])
                U1_I02_20[i][j] = self._U1_integral_4_mixing(k0, G0_sum, G0, V1_20, K02[i], K20[j])
                U0_I02_22[i][j] = self._U0_integral_3_4(k0, G0_sum, G0, V0_22[j], K02[i], K22[j])
                U1_I02_22[i][j] = self._U1_integral_3_4(k0, G0_sum, G0, V1_22, K02[i], K22[j])
                
                U0_I02[i][j] = U0_I02_00[i][j] + U0_I02_02[i][j] + U0_I02_20[i][j] + U0_I02_22[i][j]
                U1_I02[i][j] = U1_I02_00[i][j] + U1_I02_02[i][j] + U1_I02_20[i][j] + U1_I02_22[i][j]
                
                
                # 20 integral
                U0_I20_00[i][j] = self._U0_integral_3_4(k0, G0_sum, G0, V0_00[j], K20[i], K00[j])
                U1_I20_00[i][j] = self._U1_integral_3_4(k0, G0_sum, G0, V1_00, K20[i], K00[j])
                U0_I20_02[i][j] = self._U0_integral_4_mixing(k0, G0_sum, G0, V0_02[j], K20[i], K02[j])
                U1_I20_02[i][j] = self._U1_integral_4_mixing(k0, G0_sum, G0, V1_02, K20[i], K02[j])
                U0_I20_20[i][j] = self._U0_integral_1_mixing(k0, G0_sum, G0, V0_20[j], K22[i], K00[j])
                U1_I20_20[i][j] = self._U1_integral_1_mixing(k0, G0_sum, G0, V1_20, K22[i], K00[j])
                U0_I20_22[i][j] = self._U0_integral_2_4(k0, G0_sum, G0, V0_22[j], K22[i], K02[j])
                U1_I20_22[i][j] = self._U1_integral_2_4(k0, G0_sum, G0, V1_22, K22[i], K02[j])
                
                U0_I20[i][j] = U0_I20_00[i][j] + U0_I20_02[i][j] + U0_I20_20[i][j] + U0_I20_22[i][j]
                U1_I20[i][j] = U1_I20_00[i][j] + U1_I20_02[i][j] + U1_I20_20[i][j] + U1_I20_22[i][j]
                
                # 22 integral
                U0_I22_00[i][j] = self._U0_integral_4(k0, G0_sum, G0, V0_00[j], K20[i], K20[j])
                U1_I22_00[i][j] = self._U1_integral_4(k0, G0_sum, G0, V1_00, K20[i], K20[j])        
                U0_I22_02[i][j] = self._U0_integral_3_4(k0, G0_sum, G0, V0_02[j], K20[i], K22[j])
                U1_I22_02[i][j] = self._U1_integral_3_4(k0, G0_sum, G0, V1_02, K20[i], K22[j])
                U0_I22_20[i][j] = self._U0_integral_2_4(k0, G0_sum, G0, V0_20[j], K22[i], K20[j])
                U1_I22_20[i][j] = self._U1_integral_2_4(k0, G0_sum, G0, V1_20, K22[i], K20[j])
                U0_I22_22[i][j] = self._U0_integral(k0, G0_sum, G0, V0_22[j], K22[i], K22[j])
                U1_I22_22[i][j] = self._U1_integral(k0, G0_sum, G0, V1_22, K22[i], K22[j])
                
                U0_I22[i][j] = U0_I22_00[i][j] + U0_I22_02[i][j] + U0_I22_20[i][j] + U0_I22_22[i][j]
                U1_I22[i][j] = U1_I22_00[i][j] + U1_I22_02[i][j] + U1_I22_20[i][j] + U1_I22_22[i][j]
        
                
        U0_I00_stack = stack((U0_I00_00.T, U0_I00_20.T, U0_I00_02.T, U0_I00_22.T))
        U1_I00_stack = stack((swapaxes(U1_I00_00, 0, 1), swapaxes(U1_I00_20, 0, 1), 
                              swapaxes(U1_I00_02, 0, 1), swapaxes(U1_I00_22, 0, 1)))

        U0_I02_stack = stack((U0_I02_00.T, U0_I02_20.T, U0_I02_02.T, U0_I02_22.T))
        U1_I02_stack = stack((swapaxes(U1_I02_00, 0, 1), swapaxes(U1_I02_20, 0, 1), 
                              swapaxes(U1_I02_02, 0, 1), swapaxes(U1_I02_22, 0, 1)))

        U0_I20_stack = stack((U0_I20_00.T, U0_I20_20.T, U0_I20_02, U0_I20_22))
        U1_I20_stack = stack((swapaxes(U1_I20_00, 0, 1), swapaxes(U1_I20_20, 0, 1), 
                              swapaxes(U1_I20_02, 0, 1), swapaxes(U1_I20_22, 0, 1)))

        U0_I22_stack = stack((U0_I22_00.T, U0_I22_20.T, U0_I22_02.T, U0_I22_22.T))
        U1_I22_stack = stack((swapaxes(U1_I22_00, 0, 1), swapaxes(U1_I22_20, 0, 1), 
                              swapaxes(U1_I22_02, 0, 1), swapaxes(U1_I22_22, 0, 1)))
               
        U0 = stack((U0_I00_stack, U0_I20_stack, U0_I02_stack, U0_I22_stack))
        U1 = stack((U1_I00_stack, U1_I20_stack, U1_I02_stack, U1_I22_stack))
        
        return U0, U1
    
    def _U0_integral(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-independent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        V_bot, V_k0 = V[-1][:-1], V[-1][-1]
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]

        int23_sum = np_sum(G0 * V_bot * (K_j_bot + K_i_bot))

        int23_PV = V_k0 * (K_j_k0 + K_i_k0) * G0_sum

        int4_k02 = np_sum(G0 * G0_sum * (K_j_k0 * K_i_bot + K_i_k0 * K_j_bot) * V_bot)

        int4_k04 = K_i_k0 * K_j_k0 * G0_sum**2 * V_k0

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = np_sum(p_vec @ V[:-1, :-1] @ k_vec)

        U_value = V_k0 + (2.0 / pi) * (int23_sum - int23_PV) / k0 \
                    + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value
    
    
    def _U1_integral(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-dependent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        len_ps = self.len_ps
        k = len_ps + 1
        
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        V = moveaxis(V, -1, 0)
        V_k0 = squeeze(V[:, len_ps:k, len_ps:k])
        V_bot = moveaxis(V[:, len_ps:k, :len_ps], 0, -1)
        V = moveaxis(V, 0, -1)
        
        int23_sum = np_sum((G0[:, None] * V_bot * (K_j_bot[:, None] + K_i_bot[:, None])), axis=1)
        int23_PV = (V_k0 * (K_j_k0 + K_i_k0) * G0_sum)

        int4_k02 = np_sum(G0[:, None] * G0_sum * (K_j_k0 * K_i_bot[:, None] \
                                                  + K_i_k0 * K_j_bot[:, None]) * V_bot, axis=1)
        int4_k04 = (K_i_k0 * K_j_k0 * G0_sum**2 * V_k0)

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = swapaxes(p_vec @ V[:-1, :-1], 0, 1) @ k_vec

        U_value = V_k0 + (2.0 / pi) * (int23_sum - int23_PV) / k0 \
                    + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value  

    def _U0_integral_2_4(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-independent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        V_bot, V_k0 = V[-1][:-1], V[-1][-1]
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        int2_sum = np_sum(G0 * V_bot * K_j[-1][:-1])
        int2_PV = V_k0 * K_j[-1][-1] * G0_sum
        
        int4_k02 = np_sum(G0 * G0_sum * (K_i_k0 * K_j_bot * V_bot \
                          + K_j_k0 * K_i.T[-1][:-1] * V.T[-1][:-1]))

        int4_k04 = K_i_k0 * K_j_k0 * G0_sum**2 * V_k0

        p_vec, k_vec = K_i_bot * G0 , K_j_bot * G0
        int4_11 = np_sum(p_vec @ V[:-1, :-1] @ k_vec)

        U_value = (2.0 / pi) * (int2_sum - int2_PV) / k0 \
                  + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
        
        return U_value
    
    def _U1_integral_2_4(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-dependent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        len_ps = self.len_ps
        k = len_ps + 1
        
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        V = moveaxis(V, -1, 0)
        V_k0 = squeeze(V[:, len_ps:k, len_ps:k])
        V_bot = moveaxis(V[:, len_ps:k, :len_ps], 0, -1)
        V_side = reshape(moveaxis(V[:, :len_ps, len_ps:k], 0, -1), (V.shape[1] - 1, V.shape[0]))
        V = moveaxis(V, 0, -1)
        
        int2_sum = np_sum((G0[:, None] * V_bot * K_j_bot[:, None]), axis=1) 
        int2_PV = V_k0 * K_j_k0 * G0_sum
        
        int4_k02 = np_sum(G0[:, None] * G0_sum * (K_i_k0 * K_j_bot[:, None] * V_bot \
                                         + K_j_k0 * K_i_bot[:, None] * V_side), axis=1)
        int4_k04 = (K_i_k0 * K_j_k0 * G0_sum**2 * V_k0)

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = swapaxes(k_vec @ V[:-1, :-1], 0, 1) @ p_vec

        U_value = (2.0 / pi) * (int2_sum - int2_PV) / k0 \
                  + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
        
        return U_value

    def _U0_integral_3_4(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-independent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        V_bot, V_k0 = V[-1][:-1], V[-1][-1]
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        int3_sum = np_sum(G0 * V.T[-1][:-1] * K_i[-1][:-1])
        int3_PV = V_k0 * K_i_k0 * G0_sum
    
        int4_k02 = np_sum(G0 * G0_sum * (K_i_k0 * K_j.T[-1][:-1] * V_bot \
                          + K_j_k0 * K_i_bot * V.T[-1][:-1]))

        int4_k04 = K_i_k0 * K_j_k0 * G0_sum**2 * V_k0

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = np_sum(p_vec @ V[:-1, :-1] @ k_vec)

        U_value = (2.0 / pi) * (int3_sum - int3_PV) / k0 \
                  + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
        
        return U_value
    
    def _U1_integral_3_4(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-dependent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        len_ps = self.len_ps
        k = len_ps + 1
        
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        V = moveaxis(V, -1, 0)
        V_k0 = squeeze(V[:, len_ps:k, len_ps:k])
        V_bot = moveaxis(V[:, len_ps:k, :len_ps], 0, -1)
        V_side = reshape(moveaxis(V[:, :len_ps, len_ps:k], 0, -1), (V.shape[1] - 1, V.shape[0]))
        V = moveaxis(V, 0, -1)
        
        int3_sum = np_sum((G0[:, None] * V_side * K_i_bot[:, None]), axis=0)
        int3_PV = V_k0 * K_i_k0 * G0_sum
        
        int4_k02 = np_sum(G0[:, None] * G0_sum * (K_i_k0 * K_j_bot[:, None] * V_bot \
                                         + K_j_k0 * K_i_bot[:, None] * V_side), axis=1)

        int4_k04 = (K_i_k0 * K_j_k0 * G0_sum**2 * V_k0)

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = swapaxes(k_vec @ V[:-1, :-1], 0, 1) @ p_vec
        
        U_value = (2.0 / pi) * (int3_sum - int3_PV) / k0 \
                  + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
        
        return U_value
                          
    def _U0_integral_4(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-independent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        V_bot, V_k0 = V[-1][:-1], V[-1][-1]
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]

        int4_k02 = np_sum(G0 * G0_sum
                   * (K_j_k0 * K_i_bot + K_i_k0 * K_j_bot) * V_bot)

        int4_k04 = K_i_k0 * K_j_k0 * G0_sum**2 * V_k0

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = np_sum(p_vec @ V[:-1, :-1] @ k_vec)
        
        U_value = (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value

    def _U1_integral_4(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-dependent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation.
        """
        len_ps = self.len_ps
        k = len_ps + 1
        
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        V = moveaxis(V, -1, 0)
        V_k0 = squeeze(V[:, len_ps:k, len_ps:k])
        V_bot = moveaxis(V[:, len_ps:k, :len_ps], 0, -1)
        V_side = reshape(moveaxis(V[:, :len_ps, len_ps:k], 0, -1), (V.shape[1] - 1, V.shape[0]))
        V = moveaxis(V, 0, -1)
        
        int4_k02 = np_sum(G0[:, None] * G0_sum * ((K_j_k0 * K_i_bot[:, None] \
                          + K_i_k0 * K_j_bot[:, None]) * V_bot), axis=1)

        int4_k04 = (K_i_k0 * K_j_k0 * G0_sum**2 * V_k0)

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = swapaxes(k_vec @ V[:-1, :-1], 0, 1) @ p_vec
        
        U_value = (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value
    
    def _U0_integral_1_mixing(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-independent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation
        for mixing channels.
        """
        V_bot, V_k0 = V[-1][:-1], V[-1][-1]
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]

        int2_sum = np_sum(G0 * V_bot * K_j_bot)
        int3_sum = np_sum(G0 * V.T[-1][:-1] * K_i_bot)
        int23_sum = int2_sum + int3_sum

        int23_PV = V_k0 * (K_j_k0 + K_i_k0) * G0_sum 

        int4_k02 = np_sum(G0 * G0_sum * (K_i_k0 * K_j_bot * V_bot \
                          + K_j_k0 * K_i.T[-1][:-1] * V.T[-1][:-1]))

        int4_k04 = K_i_k0 * K_j_k0 * G0_sum**2 * V_k0

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = np_sum(p_vec @ V[:-1, :-1] @ k_vec)
        
        U_value = V_k0 + (2.0 / pi) * (int23_sum - int23_PV) / k0 \
                    + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value
    
    def _U1_integral_1_mixing(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-dependent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation
        for mixing channels.
        """
        len_ps = self.len_ps
        k = len_ps + 1
        
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        V = moveaxis(V, -1, 0)
        V_k0 = squeeze(V[:, len_ps:k, len_ps:k])
        V_bot = moveaxis(V[:, len_ps:k, :len_ps], 0, -1)
        V_side = reshape(moveaxis(V[:, :len_ps, len_ps:k], 0, -1), (V.shape[1] - 1, V.shape[0]))
        V = moveaxis(V, 0, -1)
        
        int2_sum = np_sum((G0[:, None] * V_bot * K_j_bot[:, None]), axis=1)
        int3_sum = np_sum((G0[:, None] * V_side * K_i_bot[:, None]), axis=0)
        int23_sum = int2_sum + int3_sum
        
        int23_PV = V_k0 * (K_i_k0 + K_j_k0) * G0_sum
        
        int4_k02 = np_sum(G0[:, None] * G0_sum * (K_i_k0 * K_j_bot[:, None] * V_bot \
                                         + K_j_k0 * K_i_bot[:, None] * V_side), axis=1)

        int4_k04 = (K_i_k0 * K_j_k0 * G0_sum**2 * V_k0)

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = swapaxes(k_vec @ V[:-1, :-1], 0, 1) @ p_vec
        
        U_value = V_k0 + (2.0 / pi) * (int23_sum - int23_PV) / k0 \
                    + (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
        
        return U_value
                              
    def _U0_integral_4_mixing(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-independent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation
        for mixing channels.
        """
        V_bot, V_k0 = V[-1][:-1], V[-1][-1]
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        int4_k02 = np_sum(G0 * G0_sum
                          * (K_i_k0 * K_j_bot * V_bot \
                             + K_j_k0 * K_i_bot * V.T[-1][:-1]))

        int4_k04 = K_i_k0 * K_j_k0 * G0_sum**2 * V_k0

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = np_sum(p_vec @ V[:-1, :-1] @ k_vec)

        U_value = (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value
    
    def _U1_integral_4_mixing(
        self,
        k0: float, 
        G0_sum: float,
        G0: ArrayLike,
        V: ArrayLike,
        K_i: ArrayLike, 
        K_j: ArrayLike
    ) -> float:
        """
        Used to calculate the parameter-dependent part of the overlap
        matrix for coupled channels calculation without Glockle interpolation
        for mixing channels.
        """
        len_ps = self.len_ps
        k = len_ps + 1
        
        K_i_k0, K_j_k0 = K_i[-1][-1], K_j[-1][-1]
        K_i_bot, K_j_bot = K_i[-1][:-1], K_j[-1][:-1]
        
        V = moveaxis(V, -1, 0)
        V_k0 = squeeze(V[:, len_ps:k, len_ps:k])
        V_bot = moveaxis(V[:, len_ps:k, :len_ps], 0, -1)
        V_side = reshape(moveaxis(V[:, :len_ps, len_ps:k], 0, -1), (V.shape[1] - 1, V.shape[0]))
        V = moveaxis(V, 0, -1)
        
        int4_k02 = np_sum(G0[:, None] * G0_sum * (K_i_k0 * K_j_bot[:, None] * V_bot \
                          + K_j_k0 * K_i_bot[:, None] * V_side), axis=1)

        int4_k04 = (K_i_k0 * K_j_k0 * G0_sum**2 * V_k0)

        p_vec, k_vec = G0 * K_i_bot, G0 * K_j_bot
        int4_11 = swapaxes(k_vec @ V[:-1, :-1], 0, 1) @ p_vec

        U_value = (2.0 / pi)**2 * (int4_11 - int4_k02 + int4_k04) / k0**2
                
        return U_value
    
    def _delta_U_momentum_glockle(
        self,
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        Sp: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        tau_b: ArrayLike,
        V0: ArrayLike, 
        V1: ArrayLike
    ) -> ArrayLike:
        r"""
        Method
        ------
        Calculates the kernel matrix \delta \tilde{U}_{ij} used for the emulator
        calculation in momentum space with Glockle interpolation for single channel.

        Parameters
        ----------
        k0 : float
            Value of k at which we are predicting the phase shifts.
        ps : array
            Mesh points.
        ws : array
            Mesh weights.
        Sp : array
            The vector matrix from the Glockle spline.
        K_b : array
            K matrix used to train the basis.
        V_b : array
            Potential used to train the basis.
        lecs : array
            An array containing the testing LECs.
        
        Returns
        -------
        U_ij : array
            Returns the kernel matrix.
        """
        len_basis = self.len_basis
        len_ps = self.len_ps
        
        spline = resize(Sp, (len_basis, len_ps))

        K_half_on = k0**2 * (K_b[:, -1]) * np_sum(ws / (ps**2 - k0**2))
        K_half_on_sp = einsum('ij, i -> ij', spline, K_half_on, optimize=True)
        k_vector = column_stack(spline + 2 / pi * ((K_b[:, :-1] * ps**2 * ws 
                                               / (ps**2 - k0**2) - K_half_on_sp)) / k0)
        
        U0 = einsum('ij, ijk, kl -> li', 
                    k_vector.T, (V0[:-1, :-1] - V_b[:, :-1, :-1]), k_vector, optimize=True)
        U1 = swapaxes(swapaxes(swapaxes(k_vector.T @ V1[:-1, :-1, :], 0, 2) @ k_vector, 0, 2), 0, 1)
        
        return U0, U1
    
    def _delta_U_momentum_glockle_coupled(
        self, 
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        Sp: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        tau_b: ArrayLike,
        V0: ArrayLike, 
        V1: ArrayLike,
    ) -> ArrayLike:
        r"""
        Method
        ------
        Calculates the kernel matrix \delta \tilde{U}_{ij} used for the emulator
        calculation in momentum space with Glockle interpolation for coupled channels.

        Parameters
        ----------
        k0 : float
            Value of k at which we are predicting the phase shifts.
        ps : array
            Mesh points.
        ws : array
            Mesh weights.
        Sp : array
            The vector matrix from the Glockle spline.
        K_b : array
            K matrix used to train the basis.
        V_b : array
            Potential used to train the basis.
        lecs : array
            An array containing the testing LECs.
        
        Returns
        -------
        U_ij : array
            Returns the kernel matrix.
        """
        len_basis = self.len_basis
        len_ps = self.len_ps

        greens_sum = np_sum(ws / (ps**2 - k0**2))
        ps_factor = ps**2 * ws / (ps**2 - k0**2)
        spline = resize(Sp, (len_basis, len_ps))
        
        K00, K02 = K_b[:, :len_ps + 1, :len_ps + 1], K_b[:, :len_ps + 1, len_ps + 1:]
        K20, K22 = K_b[:, len_ps + 1:, :len_ps + 1], K_b[:, len_ps + 1:, len_ps + 1:]
        
        K00_half_on = k0**2 * K00[:, -1, -1] * greens_sum
        K02_half_on = k0**2 * K02[:, -1, -1] * greens_sum
        K20_half_on = k0**2 * K20[:, -1, -1] * greens_sum
        K22_half_on = k0**2 * K22[:, -1, -1] * greens_sum
        
        spline_stack = stack((spline, spline, spline, spline))
        K00_stack = stack((K00_half_on, K02_half_on, K20_half_on, K22_half_on))
        K_half_on_spline = einsum('ijk, ij -> ijk', spline_stack, K00_stack, optimize=True)
        
        k00_vector = spline + 2 / pi * (K00[:, -1, :-1] * ps_factor - K_half_on_spline[0]) / k0
        k02_vector = 2 / pi * (K20[:, -1, :-1] * ps_factor - K_half_on_spline[1]) / k0
        k20_vector = 2 / pi * (K02[:, -1, :-1] * ps_factor - K_half_on_spline[2]) / k0
        k22_vector = spline + 2 / pi * (K22[:, -1, :-1] * ps_factor - K_half_on_spline[3]) / k0
        
        V_sub_0 = V0 - V_b
        V0_00 = V_sub_0[:, :len_ps, :len_ps]
        V0_02 = V_sub_0[:, :len_ps, len_ps + 1:-1]
        V0_20 = V_sub_0[:, len_ps + 1:-1, :len_ps]
        V0_22 = V_sub_0[:, len_ps + 1:-1, len_ps + 1:-1]
        
        V1_00 = V1[:len_ps, :len_ps, :]
        V1_02 = V1[:len_ps, len_ps + 1:-1, :]
        V1_20 = V1[len_ps + 1:-1, :len_ps, :]
        V1_22 = V1[len_ps + 1:-1, len_ps + 1:-1, :]
        
        V0_stack = stack((V0_00, V0_02, V0_20, V0_22))
        V1_stack = stack((V1_00, V1_02, V1_20, V1_22))

        k00_stack = stack((k00_vector, k00_vector, k20_vector, k20_vector))
        p00_stack = stack((k00_vector, k20_vector, k00_vector, k20_vector))
        U0_00 = swapaxes(einsum('ijk, ijkl, iml -> jim', 
                                k00_stack, V0_stack, p00_stack, optimize=True), 0, 1)
        U1_00 = einsum('ijk, iklm, iol -> ijom', 
                       k00_stack, V1_stack, p00_stack, optimize=True)
        
        p02_stack = stack((k02_vector, k22_vector, k02_vector, k22_vector))
        U0_02 = swapaxes(einsum('ijk, ijkl, iml -> jim', 
                                k00_stack, V0_stack, p02_stack, optimize=True), 0, 1)
        U1_02 = einsum('ijk, iklm, iol -> ijom', 
                       k00_stack, V1_stack, p02_stack, optimize=True)

        k20_stack = stack((k02_vector, k02_vector, k22_vector, k22_vector))
        p20_stack = stack((k00_vector, k20_vector, k00_vector, k20_vector))
        U0_20 = swapaxes(einsum('ijk, ijkl, iml -> jim', 
                                k20_stack, V0_stack, p20_stack, optimize=True), 0, 1)
        U1_20 = einsum('ijk, iklm, iol -> ijom', 
                       k20_stack, V1_stack, p20_stack, optimize=True)
        
        p22_stack = stack((k02_vector, k22_vector, k02_vector, k22_vector))
        U0_22 = swapaxes(einsum('ijk, ijkl, iml -> jim', 
                                k20_stack, V0_stack, p22_stack, optimize=True), 0, 1)
        U1_22 = einsum('ijk, iklm, iol -> ijom', 
                       k20_stack, V1_stack, p22_stack, optimize=True)

        U0 = stack((U0_00, U0_20, U0_02, U0_22))
        U1 = stack((U1_00, U1_20, U1_02, U1_22))
        
        return U0, U1
    
    
#     def _delta_U_momentum_glockle_coupled(
#         self, 
#         k0: float,
#         ps: ArrayLike,
#         ws: ArrayLike,
#         Sp: ArrayLike,
#         K_b: ArrayLike,
#         V_b: ArrayLike,
#         lecs: ArrayLike
#     ) -> ArrayLike:
#         r"""
#         Method
#         ------
#         Calculates the kernel matrix \delta \tilde{U}_{ij} used for EC
#         in momentum space with Glockle interpolation for coupled channels.

#         Input
#         -----
#         k0 : float
#             Value of k at which we are predicting the phase shifts.
#         ps : array
#             Mesh points.
#         ws : array
#             Mesh weights.
#         K_b : array
#             Wave functions used to train the basis. Calculated at r.
#         V_b : array
#             Potential used to train the basis. Calculated at r.
#         pot : (ps) x (ps) array
#             Potential used for EC prediction.
#         inf_map : bool
#             If true, then no contribution from the cutoff is necessary.
#             Default: True 
        
#         Output
#         ------
#         U_ij : array
#             Returns the kernel matrix.
#         """ 
#         len_basis = self.len_basis
#         len_ps = self.len_ps

#         greens_sum = np_sum(ws / (ps**2 - k0**2))
#         ps_factor = ps**2 * ws / (ps**2 - k0**2)
#         spline = resize(Sp[0][:len_ps], (len_basis, len_ps))
#         V_sub = self.V0 + self.V1 @ lecs - V_b

#         V00 = V_sub[:, :len_ps, :len_ps]
#         V02 = V_sub[:, :len_ps, len_ps:]
#         V20 = V_sub[:, len_ps:, :len_ps]
#         V22 = V_sub[:, len_ps:, len_ps:]
#         V_stack = stack((V00, V02, V20, V22))
        
#         K_b = swapaxes(K_b.T, 1, 2)
#         K00, K02 = K_b[0][:, :len_ps], K_b[0][:, len_ps:]
#         K20, K22 = K_b[1][:, :len_ps], K_b[1][:, len_ps:]
        
#         K00_half_on = k0**2 * spline[0] @ K00.T * greens_sum
#         K02_half_on = k0**2 * spline[0] @ K02.T * greens_sum
#         K20_half_on = k0**2 * spline[0] @ K20.T * greens_sum
#         K22_half_on = k0**2 * spline[0] @ K22.T * greens_sum
        
#         spline_stack = stack((spline, spline, spline, spline))
#         K00_stack = stack((K00_half_on, K02_half_on, K20_half_on, K22_half_on))
#         K_half_on_spline = einsum('ijk, ij -> ijk', spline_stack, K00_stack, optimize=True)
        
#         k00_vector = spline - 2 / pi * (K00 * ps_factor - K_half_on_spline[0])
#         k02_vector = - 2 / pi * (K20 * ps_factor - K_half_on_spline[1])
#         k20_vector = - 2 / pi * (K02 * ps_factor - K_half_on_spline[2])
#         k22_vector = spline - 2 / pi * (K22 * ps_factor - K_half_on_spline[3])
        
#         k00_stack = stack((k00_vector, k00_vector, k20_vector, k20_vector))
#         p00_stack = stack((k00_vector, k20_vector, k00_vector, k20_vector))
#         p02_stack = stack((k02_vector, k22_vector, k02_vector, k22_vector))
#         k20_stack = stack((k02_vector, k02_vector, k22_vector, k22_vector))
#         p20_stack = stack((k00_vector, k20_vector, k00_vector, k20_vector))
#         p22_stack = stack((k02_vector, k22_vector, k02_vector, k22_vector))
        
#         U00 = einsum('ijk, ijkl, iml -> ijm', k00_stack, V_stack, p00_stack, optimize=True)
#         U02 = einsum('ijk, ijkl, iml -> ijm', k00_stack, V_stack, p02_stack, optimize=True)
#         U20 = einsum('ijk, ijkl, iml -> ijm', k20_stack, V_stack, p20_stack, optimize=True)
#         U22 = einsum('ijk, ijkl, iml -> ijm', k20_stack, V_stack, p22_stack, optimize=True)        
        
#         U_ij = block([[np_sum(U00, axis=0), np_sum(U02, axis=0)], 
#                       [np_sum(U20, axis=0), np_sum(U22, axis=0)]]).T

#         return U_ij
    
    
