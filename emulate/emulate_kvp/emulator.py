"""
Simulator and KVP emulator for two-body scattering.
"""
import numpy as np
from numba import jit, prange
from numpy import sum as np_sum
from numpy import (
    reshape, append, pad, zeros, ndarray, 
    pi, conj, real, zeros_like,
    identity, column_stack, resize,
    block, array, stack, squeeze, tile,
    swapaxes, moveaxis, einsum, ones
)
from numpy.linalg import solve, det
from numpy.typing import ArrayLike
from typing import Union, Optional
from .utils import glockle_cubic_spline, compute_G0
from .kinematics import K_to_T, K_to_S, T_to_K, S_to_K

@jit(nopython=True, parallel=True, fastmath=True)
def lstsq_parallel(mat, a, b, len_b, h):
    """
    Solves a matrix equation using a least square solver (np.linalg.lstsq) with Numba.
    """
    for i in prange(a.shape[0]):
        mat[i] = np.linalg.lstsq(a[i], b[i].T, rcond=h)[0][:len_b]
    return mat
    
def fix_coupled_lecs(
    lecs: ArrayLike,
    wave: Optional[str] = None
) -> ArrayLike:
    """
    Function used to ''fix'' the 3S1-3D1 and 3P2-3F2 coupled constants.
    LECs passed in are non-redundant, but these two partial waves have
    two redundant LECs that are taken into account.

    Parameters
    ----------
    lecs : array
        Array of coupling constants.
    wave : str or None (default=None)
        Specifies which partial wave is being considered.
    
    Returns
    -------
    lecs : array
        An array of the low-energy couplings.
    """
    if '3S1' in wave:
        lecs = array([lecs[0], lecs[1], lecs[3], lecs[2], 
                      lecs[5], lecs[2], lecs[5], lecs[4]])
        
    elif '3P2' in wave:
        lecs = array([lecs[0], lecs[1], lecs[2], lecs[2], lecs[3]])
        
    return lecs
    

class KVP_emulator:
    """
    A class that can simulate or emulate two-body scattering observables 
    in momentum space via the Kohn variational principle.
    
    Parameters
    ----------
    k : array
        The k grid.
    ps : array
        The momentum grid in inverse fermi.
    ws : array
        The weights that corresponds to the momentum mesh.
    V0 : array
        Parameter-independent part of potential, in units of energy.
    V1 : array
        Parameter-dependent (linear) part of potential, in units of energy.
    wave : str
        Denotes the partial wave.
    is_coupled : boolean
        If True, coupled channel calculation takes place.
        If False, non-coupled channel calculation takes place.
    """
    def __init__(
        self, 
        k: ArrayLike, 
        ps: ArrayLike, 
        ws: ArrayLike,
        V0: ArrayLike, 
        V1: ArrayLike, 
        wave: str,
        is_coupled: bool = False
    ):
        G0 = compute_G0(k, ps, ws)
        Sp = glockle_cubic_spline(ps, k)
        len_k, len_ps = len(k), len(ps)
        
        if is_coupled:
            self.K0_var = zeros((3, len_k), dtype=complex)
            Id = identity(2 * (len_ps + 1), float)
            K = zeros((len_k, 2 * (len_ps + 1), 2 * (len_ps + 1)), float)
            G0_coup = zeros_like(K, float)
            
            for i in range(G0.shape[0]):
                G0_coup[i] = tile(G0[i], (2 * (len_ps + 1), 2))
            
            self.Id, self.K = Id, K
            self.G0 = G0_coup
            
        else:
            self.G0 = G0
            self.Id = identity(len_ps + 1)
            self.K = zeros((len_k, len_ps + 1), float)           
            
        self.k = k
        self.ps = ps
        self.ws = ws
        self.Sp = Sp
        
        self.V0 = V0
        self.V1 = V1
        self.wave = wave
        self.len_k = len_k
        self.len_ps = len_ps
        self.is_coupled = is_coupled
        
        self.mat_trans_K = array([[1, 0], [0, 1]])
        self.mat_trans_S = array([[-1j, 1], [-1j, -1]])
        self.mat_trans_T = array([[1, 0], [1j, 1]])
    
    def high_fidelity(
        self,
        params: ArrayLike
    ) -> ArrayLike:
        """
        Wrapper used for the high-fidelity (simulator) solution.

        Parameters
        ----------
        params : array
            Array of coupling constants.

        Returns
        -------
        K0 : array
            On-shell K.
        """
        if '3S1' in self.wave or '3P2' in self.wave:
            params = fix_coupled_lecs(params, self.wave)
        
        K0, _ = self.ls_eq_no_interpolate(self.V0 + self.V1 @ params)
            
        if self.is_coupled:
            K0 = K0.swapaxes(0, 1).reshape(2, 2 * self.len_k, order='F')
            K0 = reshape(K0, (4, self.len_k)).T
            
        return K0
        
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
        Id, G0, K = self.Id, self.G0, self.K
        
        if self.is_coupled:
            len_ps = self.len_ps 
            K = self.ls_eq_no_interpolate_coupled(Id, V, G0, K)
            K0 = array([[K[:, len_ps, len_ps], 
                         K[:, len_ps, 2 * len_ps + 1]], 
                        [K[:, 2 * len_ps + 1, len_ps], 
                         K[:, 2 * len_ps + 1, 2 * len_ps + 1]]]).T
            
        else:
            K = self.ls_eq_no_interpolate_uncoupled(Id, V, G0, K)
            K0 = K[:, -1]
        
        return K0, K
    
    def ls_eq_no_interpolate_uncoupled(self, Id, V, G0, K):
        """
        Solves the Lippmann-Schwinger equation for the non-coupled channels.

        Parameters
        ----------
        Id : array
            Identity matrix.
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
        for i in range(K.shape[0]):
            K[i] = solve(Id + G0[i] * V[i], V[i][:, -1])
        return -self.k[:, None] * K
    
    def ls_eq_no_interpolate_coupled(self, Id, V, G0, K):
        """
        Solves the Lippmann-Schwinger equation for the coupled channels.

        Parameters
        ----------
        Id : array
            Identity matrix.
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
        for i in range(K.shape[0]):
            K[i] = solve(Id + G0[i] * V[i], V[i])
        return -0.5 * self.k[:, None, None] * (K + swapaxes(K, 1, 2))
    
    def prediction(
        self,
        test_params: ArrayLike,
        glockle: bool, 
        sol: str,
        h: float = None, 
    ) -> ArrayLike:
        """
        Makes prediction of the on-shell scattering matrix using the emulator.

        Parameters
        ----------
        test_params : array
            An array containing the test_params for the testing point.
        glockle : boolean
            If True, chooses U0 and U1 calculated using Glockle spline method.
            If False, chooses U0 and U1 calculated using Standard method.
        sol : str
            Chooses method used to calculate the basis weights.
            Options: 'lstsq' and 'solve'
        h : float
            Nugget used to regulate the basis collinearity.
            
        Returns
        -------
        K0 : array
            Prediction from the emulator.
        """
        K0 = self.K0
        dU_mat = self.dU_mat
        K0_mat = self.K0_mat
        sign = self.sign_arb_bc
        mat_det = self.matrix_det_arb_bc
        
        if glockle:
            U0 = self.U0_arb_bc_glockle
            U1 = self.U1_arb_bc_glockle
        else:
            U0 = self.U0_arb_bc_std
            U1 = self.U1_arb_bc_std
            
        U = self.k[None, :, None, None] * self._compute_U_from_U0_U1(test_params, U0, U1)
            
        if self.is_coupled:
            len_basis = self.len_basis
            dU_mat[:, 0, :, :-1, :-1] = U[:, :, :len_basis, :len_basis]
            dU_mat[:, 1, :, :-1, :-1] = U[:, :, len_basis:, :len_basis]
            dU_mat[:, 2, :, :-1, :-1] = U[:, :, len_basis:, len_basis:]
            dU_mat[:, :, :, :-1, :-1] = sign[:, None, None, None, None] * dU_mat[:, :, :, :-1, :-1]
        else:
            dU_mat[:, :, :-1, :-1] = sign[:, None, None, None] * U
        
        for i in range(self.num_emulations):
            K0_pred = self._test_emulator(K0_mat[i], dU_mat[i], sol, h, mat_det[i])
            K0[i] = self._fix_K0(K0_pred, self.emu_method_list[i])
        
        return K0
    
    def _test_emulator(
        self, 
        K0_b: ArrayLike, 
        U: ArrayLike, 
        sol_type: str, 
        h: float,
        matrix_det: bool
    ) -> ArrayLike:
        """
        Sets up the online/testing stage of emulator.

        Parameters
        ----------
        K0_b : array
            On-shell K values that correspond to the basis.
        U : array
            Overlap matrix found in expression for variational prediction.
        sol_type : str
            Chooses method used to calculate the basis weights.
            Options: 'lstsq' and 'solve'
        h : float
            Nugget used to regulate the basis collinearity.
        matrix_det : int
            Determinant of matrix used for the boundary condition imposed.
            
        Returns
        -------
        K0_var : array
            Variational prediction from the emulator.
        """
        if self.is_coupled:
            K0_var = self.K0_var
            
            for i, (K0_i, U_i) in enumerate(zip(K0_b, U)):
                K0_var[i] = self._compute_emulated_K0(sol_type, U_i, K0_i, h, matrix_det)
        else:
            K0_var = self._compute_emulated_K0(sol_type, U, K0_b, h, matrix_det)
            
        return K0_var
    
    def _compute_emulated_K0(
        self, 
        sol_type: str, 
        dU: ArrayLike, 
        K0: ArrayLike,
        h: ArrayLike, 
        matrix_det: int
    ) -> ArrayLike:
        """
        A choice for solving the U matrix and obtaining the EC prediction.

        Parameters
        ----------
        sol_type : str
            Chooses method used to calculate the basis weights.
            Options: 'lstsq' and 'solve'
        dU : array
            Delta tilde U matrix used to make EC predictions.
        K0 : array
            An array of the K0s from the basis.
        h : float
            Nugget used to regulate the basis collinearity.
        matrix_det : int
            Determinant of matrix used for the boundary condition imposed.

        Returns
        -------
        cj_K0 - matrix_det * cj_U_cj : array
            Emulator prediction for K0.
        """
        len_basis = self.len_basis
        
        if sol_type == 'lstsq':
            c_j = self.c_j
            c_j = lstsq_parallel(c_j, dU, K0, len_basis, h)
                
        elif sol_type == 'solve':
            I_h = self.I_h
            dU = dU + I_h * h
            dU[:, -1, -1] = 0
            
            c_j = solve(dU, K0)[:, :len_basis]
            
        else:
            raise Exception('Specify how to solve dU matrix!')
            
        cj_K0 = np_sum(c_j * K0[:, :-1], axis=1)
        cj_U_cj = np_sum(c_j[:, :, None] * dU[:, :-1, :-1] * c_j[:, None, :], axis=(1, 2))
    
        return cj_K0 - matrix_det * cj_U_cj
        
    def create_basis(
        self,
        wave: str, 
        train_params: ArrayLike
    ) -> None:
        """
        Calculates the basis needed to train the emulator.

        Parameters
        ----------
        wave : str
            Specifies which partial wave is being considered.
            Example: '1S0', '3P1', etc.
        train_params : array
            An array containing the parameters used for the training points.

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
        
        len_basis = len(train_params)
        
        if is_coupled:
            K0_b = zeros((len_basis, len_k, 2, 2))
            K_b = zeros((len_basis, len_k, 2 * (len_ps + 1), 2 * (len_ps + 1)))
            V_b = zeros((len_basis, len_k, 2 * (len_ps + 1), 2 * (len_ps + 1)))

        else:
            K0_b = zeros((len_basis, len_k))
            K_b = zeros((len_basis, len_k, len_ps + 1))
            V_b = zeros((len_basis, len_k, len_ps + 1, len_ps + 1))
        
        for i, param_i in enumerate(train_params):
            if '3S1' in wave or '3P2' in wave:
                param_i = fix_coupled_lecs(param_i, wave)

            V_b[i] = V0 + V1 @ param_i
            K0_b[i], K_b[i] = self.ls_eq_no_interpolate(V_b[i])
            
        self.len_basis = len_basis
        self.I = resize(identity(len_basis), 
                        (self.len_k, len_basis, len_basis))
        return V_b, swapaxes(K0_b, 0, 1), moveaxis(K_b, 0, 1)
    
    def _compute_U_from_U0_U1(
        self, 
        params: ArrayLike, 
        U0: ArrayLike, 
        U1: ArrayLike
    ) -> ArrayLike:
        r"""
        Calculates \Delta \tilde{U}_{ij} matrix from U0 and U1.

        Parameters
        ----------
        params : array
            An array containing the parameters for the testing point.
        U0 : array
            Parameter-independent portion of \Delta \tilde{U}_{ij} matrix.
        U1 : array
            Parameter-dependent portion of \Delta \tilde{U}_{ij} matrix.

        Returns
        -------
        U : array
            Full \Delta \tilde{U}_{ij} matrix.
        """
        if '3S1' in self.wave or '3P2' in self.wave:
            params = fix_coupled_lecs(params, self.wave)
            
        if self.is_coupled:
            if len(U0.shape) == 5:
                U = block([[U0[:, :, 0] + U1[:, :, 0] @ params, 
                            U0[:, :, 1] + U1[:, :, 1] @ params], 
                           [U0[:, :, 2] + U1[:, :, 2] @ params, 
                            U0[:, :, 3] + U1[:, :, 3] @ params]])
            else:
                U = block([[U0[:, 0] + U1[:, 0] @ params, 
                            U0[:, 1] + U1[:, 1] @ params], 
                           [U0[:, 2] + U1[:, 2] @ params, 
                            U0[:, 3] + U1[:, 3] @ params]])
        else:
            U = U0 + U1 @ params
        
        return U
    
    def _get_emulation_method(
        self,
        L0_b: ArrayLike, 
        method: str
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """
        Transforms basis states into boundary condition we want to emulate.

        Parameters
        ----------
        L0_b : array
            Array of on-shell scattering matrices for basis expressed
            as K-matrices.
        method : str
            Specifies the boundary condition transforming to.
        
        Returns
        -------
        L0_b : array
            Transformed on-shell scattering matrices for basis.
        matrix_det : array
            Determinant associated with on-shell basis scattering matrices.
        factor : array
            Rescale factor associated with on-shell basis scattering matrices.
        sign : array
            Sign associated with on-shell basis scattering matrices.
        """
        if method == 'K':
            matrix_det = det(self.mat_trans_K)
            factor = 1
            sign = 1
        elif method == '1/K':
            trans_matrix = self.mat_trans_K
            trans_matrix[[0, 1]] = trans_matrix[[1, 0]]
            matrix_det = det(trans_matrix)
            L0_b = 1 / L0_b
            factor = L0_b
            sign = -1
        elif method == 'T':
            matrix_det = det(self.mat_trans_T).real
            factor = 1 / (1 - 1j * L0_b)
            L0_b = K_to_T(L0_b)
            sign = 1
        elif method == 'S':
            matrix_det = det(self.mat_trans_S)
            factor = 2 * 1j / (1 - 1j * L0_b)
            L0_b = K_to_S(L0_b)
            sign = 1 / (2 * 1j)
        else:
            raise ValueError('Specify boundary condition!')

        return L0_b, matrix_det, factor, sign
    
    def _fix_K0(
        self,
        L0_i: ArrayLike,
        method: str
    ) -> ArrayLike:
        """
        Outputs on-shell solutions in its K-matrix equivalent.

        Parameters
        ----------
        L0_i : array
            Array of on-shell scattering matrix solutions.
        method : str
            Specifies the boundary condition transforming to.
        
        Returns
        -------
        K0_fix.real : array
            On-shell K matrix equivalent of L0_i
        """
        if method == 'K':
            K0_fix = L0_i
        elif method == '1/K':
            K0_fix = 1 / (L0_i + 1e-15)
        elif method == 'T':
            K0_fix = T_to_K(L0_i)
        elif method == 'S':
            K0_fix = S_to_K(L0_i)
        else:
            raise ValueError('Specify boundary condition!')
    
        return K0_fix.real
    
    def train(
        self,
        train_params: ArrayLike,
        glockle: bool = False,
        method: str = 'all'
    ) -> None:
        """
        Wrapper for training emulator.

        Parameters
        ----------
        train_params : array
            Array of lecs used for training the emulator.
        glockle : boolean
            If True, U0 and U1 are calculated using Glockle spline method.
            If False, U0 and U1 are calculated using Standard method.
        method : str
            Specifies the boundary condition being used.
        
        Returns
        -------
        None
        """
        len_k = self.len_k
        len_basis = len(train_params)
        
        V_b, K0_b, K_b = self.create_basis(self.wave, train_params)
        U0, U1 = self.train_emulator(V_b, K0_b, K_b, glockle)
        self.K0_b = K0_b
        
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
        
        K0_b_all = self._arbitrary_boundary_conditions(glockle, method, U1.shape[-1])
        npad = ((0, 0), (0, 0), (0, 1))
        
        if self.is_coupled:
            K0 = zeros((self.num_emulations, 3, len_k))
            K0_b_1 = pad(K0_b_all[:, :, :, 0][:, :, :, 0], 
                          pad_width=npad, mode='constant', constant_values=1)
            K0_b_2 = pad(K0_b_all[:, :, :, 0][:, :, :, 1], 
                          pad_width=npad, mode='constant', constant_values=1)
            K0_b_3 = pad(K0_b_all[:, :, :, 1][:, :, :, 1], 
                          pad_width=npad, mode='constant', constant_values=1)
            K0_mat = swapaxes(stack((K0_b_1, K0_b_2, K0_b_3)), 0, 1)
            
            dU_mat = ones((self.num_emulations, len_k, 
                           2 * (len_basis + 1), 2 * (len_basis + 1)), dtype=complex)
            dU_mat[:, :, len_basis, len_basis] = 0
            dU_mat[:, :, -1, len_basis] = 0
            dU_mat[:, :, len_basis, -1] = 0
            dU_mat[:, :, -1, -1] = 0
            
            dU_mat = swapaxes(stack((dU_mat[:, :, :len_basis + 1, :len_basis + 1], 
                                     dU_mat[:, :, len_basis + 1:, :len_basis + 1], 
                                     dU_mat[:, :, len_basis + 1:, len_basis + 1:])), 0, 1)
            
        else:
            K0 = zeros((self.num_emulations, len_k))
            K0_mat = pad(K0_b_all, pad_width=npad, 
                          mode='constant', constant_values=1)
            dU_mat = ones((self.num_emulations, len_k, 
                           len_basis + 1, len_basis + 1), dtype=complex)
            dU_mat[:, :, -1, -1] = 0
           
        self.K0 = K0
        self.dU_mat = dU_mat
        self.K0_mat = K0_mat
        self.I_h = identity(dU_mat.shape[1])[None, :] 
        self.c_j = zeros((self.len_k, self.len_basis), dtype=complex)
        return None
    
    def train_emulator(
        self, 
        V_b: ArrayLike,
        K0_b: ArrayLike,
        K_b: ArrayLike,
        glockle: bool
    ) -> None:
        """
        Emulator offline/training stage.

        Parameters
        ----------
        V_b : array
            Potentials used to build the basis.
        K0_b : array
            On-shell K values that correspond to build the basis.
        K_b : array
            K matrices used to build the basis.
        glockle : boolean
            If True, U0 and U1 are calculated using Glockle spline method.
            If False, U0 and U1 are calculated using Standard method.
        
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
        
        for i, (k_i, K_b_i, V_b_i, K0_b_i) in enumerate(zip(k, K_b, V_b, K0_b)):
            
            U0[i], U1[i] = self._calculate_delta_U(k_i, K_b_i, V_b_i, 
                                                   K0_b_i, V0[i], V1[i], 
                                                   glockle, self.Sp[i])
        return U0, U1
    
    def _calculate_delta_U(
        self, 
        k0: int,
        K_b: ArrayLike, 
        V_b: ArrayLike, 
        K0_b: ArrayLike, 
        V0: ArrayLike,
        V1: ArrayLike,
        glockle: bool, 
        Sp: Optional[ArrayLike] = None,
    ) -> None:
        r"""
        Wrapper for calculating the overlap matrix \Delta \tilde {U}_{ij}.

        Parameters
        ----------
        k0 : float
            Value of k at which we are predicting the phase shifts.
        K_b : array
            K matrix used to train the basis.
        V_b : array
            Potential used to train the basis.
        K0_b : array
            On-shell K values that correspond to the basis.
        V0 : array
            Parameter-independent part of potential.
        V1 : array
            Parameter-dependent part of potential.
        glockle : boolean
            If True, U0 and U1 are calculated using Glockle spline method.
            If False, U0 and U1 are calculated using Standard method.
        Sp : array, optional (default=None)
            The spline vector used in the Glockle spline calculation.
            
        Returns
        -------
        U0 : array
            Parameter-independent part of \Delta \tilde {U}_{ij} matrix.
        U1 : array
            Parameter-dependent part of \Delta \tilde {U}_{ij} matrix.
        """
        ps, ws = self.ps, self.ws

        if glockle:
            if not self.is_coupled:
                U0, U1 = self._delta_U_momentum_glockle(k0, ps, ws, Sp, 
                                                        K_b, V_b, V0, V1,)
            else:
                U0, U1 = self._delta_U_momentum_glockle_coupled(k0, ps, ws, Sp, 
                                                                K_b, V_b, V0, V1)

        else:
            if not self.is_coupled:
                U0, U1 = self._delta_U_momentum(k0, ps, ws, K_b, V_b, K0_b, V0, V1)
            else:
                U0, U1 = self._delta_U_momentum_coupled(k0, ps, ws, K_b, V_b, K0_b, V0, V1)
                
        return U0, U1
    
    def _delta_U_momentum(
        self,
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        K0_b: ArrayLike,
        V0: ArrayLike,
        V1: ArrayLike
    ) -> ArrayLike:
        r"""
        Calculates the kernel matrix \Delta \tilde{U}_{ij} used for the emulator
        calculation in momentum space for the Standard method for single channel.

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
        K0_b : array
            On-shell K matrix used to train the basis.
        V0 : array
            Parameter-independent part of potential matrix.
        V1 : array
            Parameter-dependent part of potential matrix.
        
        Returns
        -------
        U0 : array
            Parameter-independent part of \Delta \tilde {U}_{ij} matrix.
        U1 : array
            Parameter-dependent part of \Delta \tilde {U}_{ij} matrix.
        """
        len_basis = self.len_basis
        
        U0 = zeros((len_basis, len_basis))
        U1 = zeros((len_basis, len_basis, V1.shape[-1]))
        
        k = append(ps, k0)
        G0 = ps**2 * ws / (ps**2 - k0**2)
        G0_sum = np_sum(k0**2 * ws / (ps**2 - k0**2))
        
        Vb_bot, Vb_k0 = self._V_b_terms_non_glockle(k, G0, V_b)
        
        K_bot = squeeze(K_b[:, :ps.shape[0]])
        K_k0 = squeeze(K_b[:, -1])
        
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
        for the single channel calculation for the Standard method.
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
        for the single channel calculation for the Standard method.
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
        channel calculation for the Standard method.
        """
        Vb_k0 = squeeze(V_b[:, ps.shape[0]:k.shape[0], ps.shape[0]:k.shape[0]])
        Vb_bot = squeeze(V_b[:, ps.shape[0]:k.shape[0], :ps.shape[0]])
        
        return Vb_bot, Vb_k0
    
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
        the potential for the single channel calculation for the Standard method.
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
        the potential for the single channel calculation for the Standard method.
        """
        if self.is_coupled:
            V1 = V1[:len(ps), :len(ps)]
        
        ### parameter-dependent potential ###
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
        for the Standard method.
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
        K0_b: ArrayLike,
        V0: ArrayLike,
        V1: ArrayLike
    ) -> ArrayLike:
        r"""
        Calculates the kernel matrix \delta \tilde{U}_{ij} used for the emulator
        calculation in momentum space for the Standard method for coupled channels.

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
        K0_b : array
            On-shell K matrix used to train the basis.
        V0 : array
            Parameter-independent part of potential matrix.
        V1 : array
            Parameter-dependent part of potential matrix.
        
        Returns
        -------
        U0 : array
            Parameter-independent part of \Delta \tilde {U}_{ij} matrix.
        U1 : array
            Parameter-dependent part of \Delta \tilde {U}_{ij} matrix.
        """
        len_basis = self.len_basis        
        
        G0 = ps**2 * ws / (ps**2 - k0**2)
        G0_sum = np_sum(k0**2 * ws / (ps**2 - k0**2))
        V0_stack, V1_stack, Vb_stack, Kb_stack = self._get_potential_K_coupled(V_b, K_b, V0, V1)
        len_params = V1.shape[-1]

        U0_I00_00 = zeros((len_basis, len_basis))
        U0_I00_02 = zeros((len_basis, len_basis))
        U0_I00_20 = zeros((len_basis, len_basis))
        U0_I00_22 = zeros((len_basis, len_basis))
        U1_I00_00 = zeros((len_basis, len_basis, len_params))
        U1_I00_02 = zeros((len_basis, len_basis, len_params))
        U1_I00_20 = zeros((len_basis, len_basis, len_params))
        U1_I00_22 = zeros((len_basis, len_basis, len_params))

        U0_I00 = zeros((len_basis, len_basis))
        U1_I00 = zeros((len_basis, len_basis, len_params))
        
        U0_I02_00 = zeros((len_basis, len_basis))
        U0_I02_02 = zeros((len_basis, len_basis))
        U0_I02_20 = zeros((len_basis, len_basis))
        U0_I02_22 = zeros((len_basis, len_basis))
        U1_I02_00 = zeros((len_basis, len_basis, len_params))
        U1_I02_02 = zeros((len_basis, len_basis, len_params))
        U1_I02_20 = zeros((len_basis, len_basis, len_params))
        U1_I02_22 = zeros((len_basis, len_basis, len_params))
    
        U0_I02 = zeros((len_basis, len_basis))
        U1_I02 = zeros((len_basis, len_basis, len_params))
        
        U0_I20_00 = zeros((len_basis, len_basis))
        U0_I20_02 = zeros((len_basis, len_basis))
        U0_I20_20 = zeros((len_basis, len_basis))
        U0_I20_22 = zeros((len_basis, len_basis))
        U1_I20_00 = zeros((len_basis, len_basis, len_params))
        U1_I20_02 = zeros((len_basis, len_basis, len_params))
        U1_I20_20 = zeros((len_basis, len_basis, len_params))
        U1_I20_22 = zeros((len_basis, len_basis, len_params))

        U0_I20 = zeros((len_basis, len_basis))
        U1_I20 = zeros((len_basis, len_basis, len_params))
        
        U0_I22_00 = zeros((len_basis, len_basis))
        U0_I22_02 = zeros((len_basis, len_basis))
        U0_I22_20 = zeros((len_basis, len_basis))
        U0_I22_22 = zeros((len_basis, len_basis))
        U1_I22_00 = zeros((len_basis, len_basis, len_params))
        U1_I22_02 = zeros((len_basis, len_basis, len_params))
        U1_I22_20 = zeros((len_basis, len_basis, len_params))
        U1_I22_22 = zeros((len_basis, len_basis, len_params))

        U0_I22 = zeros((len_basis, len_basis))
        U1_I22 = zeros((len_basis, len_basis, len_params))
        
        
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method integrals.
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
        matrix for coupled channels calculation for the Standard method
        for mixing channel integrals.
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
        matrix for coupled channels calculation for the Standard method
        for mixing channel integrals.
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
        matrix for coupled channels calculation for the Standard method
        for mixing channel integrals.
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
        matrix for coupled channels calculation for the Standard method
        for mixing channel intergrals.
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
    
    def _wave_function_momentum(
        self, 
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        Sp: ArrayLike,
        K0_b: ArrayLike,
        K_half_on_b: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the momentum-space wave function used for Glockle interpolated emulator.

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
        K0_b : array
            On-shell K matrix used to train the basis.
        K_half_on_b : array
            Half-on-shell K matrix used to train the basis.
        
        Returns
        -------
        phi : array
            Momentum-space wave function.
        """
        G0 = ps**2 * ws / (ps**2 - k0**2)
        K_G0_pv = k0**2 * K0_b * np_sum(ws / (ps**2 - k0**2))
        spline = resize(Sp, (self.len_basis, self.len_ps))

        if self.is_coupled:
            spline_stack = stack((spline, spline, spline, spline))
            K_G0 = einsum('ijk, ij -> ijk', spline_stack, K_G0_pv, optimize=True)
            spline_stack[1], spline_stack[2] = 0, 0
            phi = spline_stack + 2 / pi * (K_half_on_b * G0 - K_G0) / k0
            
        else:
            K_G0 = einsum('ij, i -> ij', spline, K_G0_pv, optimize=True)
            phi = column_stack(spline + 2 / pi * (K_half_on_b * G0 - K_G0) / k0)
        
        return phi
    
    def _delta_U_momentum_glockle(
        self,
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        Sp: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        V0: ArrayLike, 
        V1: ArrayLike
    ) -> ArrayLike:
        r"""
        Calculates the kernel matrix \Delta \tilde {U}_{ij} used for the emulator
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
        V0 : array
            Parameter-independent part of potential.
        V1 : array
            Parameter-dependent part of potential.
        
        Returns
        -------
        U0 : array
            Parameter-independent part of \Delta \tilde{U}_{ij} matrix.
        U1 : array
            Parameter-dependent part of \Delta \tilde{U}_{ij} matrix.
        """
        K_on_shell = K_b[:, -1]
        K_half_on_shell = K_b[:, :-1]
        V0_minus_V_b = V0[:-1, :-1] - V_b[:, :-1, :-1]
        
        wf_phi = self._wave_function_momentum(k0, ps, ws, Sp, K_on_shell, K_half_on_shell)
        
        U0 = einsum('ij, ijk, kl -> li', wf_phi.T, V0_minus_V_b, wf_phi, optimize=True)
        U1 = swapaxes(swapaxes(swapaxes(wf_phi.T @ V1[:-1, :-1, :], 0, 2) @ wf_phi, 0, 2), 0, 1)
        
        return U0, U1
    
    def _delta_U_momentum_glockle_coupled(
        self, 
        k0: float,
        ps: ArrayLike,
        ws: ArrayLike,
        Sp: ArrayLike,
        K_b: ArrayLike,
        V_b: ArrayLike,
        V0: ArrayLike, 
        V1: ArrayLike,
    ) -> ArrayLike:
        r"""
        Calculates the kernel matrix \delta \tilde {U}_{ij} used for the emulator
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
        V0 : array
            Parameter-independent part of potential.
        V1 : array
            Parameter-dependent part of potential.
        
        Returns
        -------
        U0 : array
            Parameter-independent part of \Delta \tilde {U}_{ij} matrix.
        U1 : array
            Parameter-dependent part of \Delta \tilde {U}_{ij} matrix.
        """
        len_basis = self.len_basis
        len_ps = self.len_ps
        
        K00, K01 = K_b[:, :len_ps + 1, :len_ps + 1], K_b[:, :len_ps + 1, len_ps + 1:]
        K10, K11 = K_b[:, len_ps + 1:, :len_ps + 1], K_b[:, len_ps + 1:, len_ps + 1:]
        
        K_on_shell = stack((K00[:, -1, -1], K10[:, -1, -1], 
                            K01[:, -1, -1], K11[:, -1, -1]))
        K_half_on_shell = stack((K00[:, -1, :-1], K10[:, -1, :-1], 
                                 K01[:, -1, :-1], K11[:, -1, :-1]))
        
        wf_phi = self._wave_function_momentum(k0, ps, ws, Sp, K_on_shell, K_half_on_shell)
        V0_stack, V1_stack = self._partition_potential_coupled_channels(V_b, V0, V1)

        wf_00_left = stack((wf_phi[0], wf_phi[0], wf_phi[2], wf_phi[2]))
        wf_10_left = stack((wf_phi[1], wf_phi[1], wf_phi[3], wf_phi[3]))
        wf_00_right = stack((wf_phi[0], wf_phi[2], wf_phi[0], wf_phi[2]))
        wf_01_right = stack((wf_phi[1], wf_phi[3], wf_phi[1], wf_phi[3]))
        wf_10_right = stack((wf_phi[0], wf_phi[2], wf_phi[0], wf_phi[2]))
        wf_11_right = stack((wf_phi[1], wf_phi[3], wf_phi[1], wf_phi[3]))
        
        wf_all_left = stack((wf_00_left, wf_10_left, wf_00_left, wf_10_left))
        wf_all_right = stack((wf_00_right, wf_10_right, wf_01_right, wf_11_right))
        V0_all = stack((V0_stack, V0_stack, V0_stack, V0_stack))
        V1_all = stack((V1_stack, V1_stack, V1_stack, V1_stack))
        
        U0 = swapaxes(einsum('aijk, aijkl, aiml -> ajim', 
                             wf_all_left, V0_all, wf_all_right, optimize=True), 1, 2)
        U1 = einsum('aijk, aiklm, aiol -> aijom', 
                    wf_all_left, V1_all, wf_all_right, optimize=True)
        
        return U0, U1
    
    def _partition_potential_coupled_channels(
        self,
        V_b: ArrayLike,
        V0: ArrayLike, 
        V1: ArrayLike
    ) -> ArrayLike:
        """
        Partitions the potential in blocks for the coupled channel calculation.

        Parameters
        ----------
        V_b : array
            Potential used to train the basis.
        V0 : array
            Parameter-independent part of potential.
        V1 : array
            Parameter-dependent part of potential.
        
        Returns
        -------
        V0_stacked : array
            Stacked Parameter-independent part of potential.
        V1_stacked : array
            Stacked Parameter-dependent part of potential.
        """
        len_basis = self.len_basis
        len_ps = self.len_ps
        
        V0_Vb_diff = V0 - V_b
        V0_00 = V0_Vb_diff[:, :len_ps, :len_ps]
        V0_01 = V0_Vb_diff[:, :len_ps, len_ps + 1:-1]
        V0_10 = V0_Vb_diff[:, len_ps + 1:-1, :len_ps]
        V0_11 = V0_Vb_diff[:, len_ps + 1:-1, len_ps + 1:-1]
        
        V1_00 = V1[:len_ps, :len_ps, :]
        V1_01 = V1[:len_ps, len_ps + 1:-1, :]
        V1_10 = V1[len_ps + 1:-1, :len_ps, :]
        V1_11 = V1[len_ps + 1:-1, len_ps + 1:-1, :]
        
        V0_stack = stack((V0_00, V0_01, V0_10, V0_11))
        V1_stack = stack((V1_00, V1_01, V1_10, V1_11))
        
        return V0_stack, V1_stack
    
    def _emulation_method_list(
        self,
        method: str
    ) -> list:
        """
        Gets a list of the boundary conditions being emulated.

        Parameters
        ----------
        method : str
            Specifies boundary conditions being used. 
        
        Returns
        -------
        None
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
    
    def _arbitrary_boundary_conditions(
        self,
        glockle: bool, 
        method: str,
        param_shape: int
    ) -> ArrayLike:
        """
        Wrapper for emulating arbitrary boundary conditions.

        Parameters
        ----------
        glockle : boolean
            If True, chooses U0 and U1 calculated using Glockle spline method.
            If False, chooses U0 and U1 calculated using Standard method.
        method : str
            Specifies boundary conditions being used.
        param_shape : int
            Total number of parameters used in partial wave.
        
        Returns
        -------
        K0 : array
            On-shell K-matrix predictions.
        """
        self._emulation_method_list(method)
        K0 = self._emulate_arbitrary_bc(param_shape, 
                                        self.emu_method_list, 
                                        self.is_coupled, glockle)
            
        return K0
    
    def _emulate_arbitrary_bc(
        self,
        param_shape: int,
        method_list: list,
        is_coupled: bool,
        glockle: bool = None
    ) -> ArrayLike:
        r"""
        Arbitrary boundary conditions calculation.
        Only calculation done is rescaling \delta \tilde {U}_{ij} and basis states.

        Parameters
        ----------
        param_shape : int
            Total number of parameters used in partial wave.
        method_list : list
            List of boundary conditions being used.
        is_coupled : boolean
            If True, coupled channel calculation takes place.
            If False, non-coupled channel calculation takes place.
        glockle : boolean
            If True, chooses U0 and U1 calculated using Glockle spline method.
            If False, chooses U0 and U1 calculated using Standard method.
        
        Returns
        -------
        K0_b_arb_bc : array
            On-shell K-matrix predictions for different boundary conditions.
        """
        len_k = self.len_k
        len_basis = self.len_basis
        num_emulations = self.num_emulations
        
        sign_arb_bc = zeros(num_emulations, dtype=complex)
        matrix_det_arb_bc = zeros(num_emulations, dtype=complex)
            
        if is_coupled:
            K0 = zeros((self.num_emulations, 3, len_k))
            K0_b_arb_bc = zeros((num_emulations, len_k, len_basis, 2, 2), dtype=complex)
            U0_arb_bc = zeros((num_emulations, len_k, 4, 
                               len_basis, len_basis), dtype=complex)
            U1_arb_bc = zeros((num_emulations, len_k, 4, 
                               len_basis, len_basis, param_shape), dtype=complex)
        else:
            K0 = zeros((self.num_emulations, len_k))
            K0_b_arb_bc = zeros((num_emulations, len_k, len_basis), dtype=complex)
            U0_arb_bc = zeros((num_emulations, len_k, 
                               len_basis, len_basis), dtype=complex)
            U1_arb_bc = zeros((num_emulations, len_k, 
                               len_basis, len_basis, param_shape), dtype=complex)

        for m, emu_method in enumerate(method_list):
            if glockle:
                U0, U1 = self.U0_glockle, self.U1_glockle
            else:
                U0, U1 = self.U0_std, self.U1_std

            K0_b, matrix_det, factor, sign = self._get_emulation_method(self.K0_b, emu_method)

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
            K0_b_arb_bc[m] = K0_b
            sign_arb_bc[m] = sign
            matrix_det_arb_bc[m] = matrix_det
            
        if glockle:
            self.U0_arb_bc_glockle = U0_arb_bc
            self.U1_arb_bc_glockle = U1_arb_bc
        else:
            self.U0_arb_bc_std = U0_arb_bc
            self.U1_arb_bc_std = U1_arb_bc

        self.sign_arb_bc = sign_arb_bc
        self.matrix_det_arb_bc = matrix_det_arb_bc
            
        return K0_b_arb_bc
    
    def _coupled_channel_normalization(
        self,
        len_k: float,
        U0: ArrayLike, 
        U1: ArrayLike, 
        factor: Union[float, ArrayLike]
    ) -> Union[float, ArrayLike]:
        r"""
        Arbitrary boundary conditions calculation for coupled channels.
        Only calculation done is rescaling \delta \tilde {U}_{ij} and basis states.

        Parameters
        ----------
        len_k : int
            Length of energy grid.
        U0 : array
            Parameter-independent part of \Delta \tilde{U}_{ij} matrix.
        U1 : array
            Parameter-dependent part of \Delta \tilde{U}_{ij} matrix.
        factor : array
            Rescale factor associated with on-shell basis scattering matrices.
        
        Returns
        -------
        U0_renorm : array
            Recaled Parameter-independent part of \Delta \tilde{U}_{ij} matrix.
        U1_renorm : array
            Recaled Parameter-dependent part of \Delta \tilde{U}_{ij} matrix.
        """
        len_basis = self.len_basis
        factor = swapaxes(factor.reshape(len_k, 2 * len_basis, 2, order='F'), 1, 2)
        factor = stack((factor[:, 0, :len_basis], factor[:, 0, len_basis:], 
                        factor[:, 1, :len_basis], factor[:, 1, len_basis:]))
        U0_renorm = zeros_like(U0, dtype=complex)
        U1_renorm = zeros_like(U1, dtype=complex)

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
    
    
