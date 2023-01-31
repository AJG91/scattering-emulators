from __future__ import annotations

from typing import Union, Optional, Dict
import numpy as np
import scipy as sp
from numpy.typing import ArrayLike
from numpy import pi

from .constants import hbar_c
from .types import BoundaryCondition
from .utils import (
    cubic_spline_matrix,
    fix_phases_continuity,
    ScatteringSystem,
)

def fix_output(len_k, K):
    K_new = K.swapaxes(0, 1).reshape(2, 2 * len_k, order='F')
    K_new = np.reshape(K_new, (4, len_k)).T
    return -K_new

def fix_coupled_lecs(
    lecs: ArrayLike,
) -> ArrayLike:
    """
    """
    if len(lecs) == 6:
        lecs = np.array([lecs[0], lecs[1], lecs[3], lecs[2], 
                      lecs[5], lecs[2], lecs[5], lecs[4]])
        
    if len(lecs) == 4:
        lecs = np.array([lecs[0], lecs[1], lecs[2], lecs[2], lecs[3]])
        
    return lecs


def greens_function_free_space(
    k: ArrayLike,
    dk: ArrayLike,
    q_cm: ArrayLike,
    spline: ArrayLike,
    boundary_condition: BoundaryCondition,
    k_cut: Optional[float] = None,
    is_compressed: bool = True,
) -> ArrayLike:
    r"""Computes the partial-wave Green's function for free-space scattering.

    This is not really G_0, because it contains the integration measure dk and k^2.
    It also includes the subtraction of the 0 integral for numerical stability of the principle value.
    The factor of 2*mu is not included either, it is instead expected to be a part of the potential and K.
    This is still convenient to treat as G_0, because this is how G_0 acts between two partial wave matrices.

    Parameters
    ----------
    k : shape = (n_k,)
        The momentum grid in inverse fermi
    dk : shape = (n_k,)
        The integration measure
    q_cm : shape = (n_q_cm,)
        The on-shell center-of-mass momenta
    spline : shape = (n_q_cm, n_k)
        The interpolation matrix that maps from k -> q_cm
    k_cut :
        The cutoff of the momentum grid. It will be chosen automatically if omitted.
    boundary_condition :
        Whether the Green's function represents incoming, outgoing, or standing boundary conditions.
    is_compressed :
        Whether the shape of the output should be compressed to shape (n_q_cm, n_k)
        or a matrix with shape (n_q_cm, n_k, n_k), with zeros everywhere except the diagonal.
        Defaults to True, which compresses the output.

    Returns
    -------
    G0 : shape = (n_q_cm, n_k) or (n_q_cm, n_k, n_k)
        The free-space Green's function.
    """
    n_k = len(k)
    n_q = len(q_cm)
    if k_cut is None:
        # this is usually good for Gauss-Legendre quadrature rules
        k_cut = k[-1] + 0.25 * (k[-1] - k[-2])

    if boundary_condition is BoundaryCondition.STANDING:
        bc_term = 0.0
        dtype = float
    else:
        if boundary_condition is BoundaryCondition.OUTGOING:
            sgn = +1
        elif boundary_condition is BoundaryCondition.INCOMING:
            sgn = -1
        else:
            raise ValueError(
                "Boundary condition must be standing, incoming, or outgoing"
            )
        bc_term = sgn * 1j * pi / 2.0
        dtype = "complex128"
    
    
    G0 = np.zeros((n_q, n_k + 1), dtype)

    for i, p in enumerate(q_cm):
        D = k ** 2 * dk / ( p ** 2 - k ** 2 )
        D_k0_sum = -p ** 2 * np.sum(dk / (p ** 2 - k ** 2))
        D_k0 = D_k0_sum
        G0[i] = np.append(D, D_k0)

    if is_compressed:
        return G0
    else:
        G0_stack = np.stack([np.diag(G0_i) for G0_i in G0], axis=0)
        return G0_stack


def t_matrix_outgoing_to_standing(T):
    r"""Converts the outgoing on-shell T matrix to its standing wave (principal value) form, aka the K or R matrix.

    Parameters
    ----------
    T :
        The outgoing (+ie) on-shell T matrix

    Returns
    -------
    reactance :
        The K (or R) matrix.
    """
    return np.real(T / (1 - 1j * T))


def t_matrix_incoming_to_standing(T):
    r"""Converts the incoming on-shell T matrix to its standing wave (principal value) form, aka the K or R matrix.

    Parameters
    ----------
    T :
        The incoming (-ie) on-shell T matrix

    Returns
    -------
    reactance :
        The K (or R) matrix.
    """
    return np.real(T / (1 + 1j * T))


def K_to_R(K, is_coupled=False):
    """Calculates the partial wave R matrix given the partial wave K matrix.

    This definition of R is like -2i * T, and is equal to S-1.

    Parameters
    ----------
    K : array, shape = (..., n, n)
        The K matrix
    is_coupled :

    Returns
    -------
    R : array, shape = (..., n, n)
    """
    if is_coupled:
        Id = np.identity(K.shape[-1])
        R = -2j * np.linalg.solve(Id + 1j * K, K)
    else:
        R = -2j * K / (1 + 1j * K)
    return R


def compute_total_cross_section(R, q_cm, j):
    from .constants import fm_to_sqrt_mb

    q = q_cm / fm_to_sqrt_mb
    j = j[:, None]
    return (
        -pi
        / (2 * q ** 2)
        * np.sum((2 * j + 1) * np.real(np.trace(R, axis1=-1, axis2=-2)), axis=0)
    )


class TwoBodyScattering:
    r"""A class that can either simulate or emulate two-body scattering observables via the reactance matrix.

    Depending on the context, the reactance matrix is denoted by K or R.

    Parameters
    ----------
    V0 : ArrayLike, shape = (n_k, n_k)
        The piece of the potential that does not depend of parameters, in units of fermi. This may require
        multiplying the standard momentum space potential (which is in MeV fm^3) by the 2 times the reduced
        mass of the system: 2 * mu / hbar**2.
    V1 : ArrayLike, shape = (n_k, n_k, n_p)
        The piece of the potential that is linear in the parameters p. When multiplied by p, then this
        is expected to be in units of fermi.
    k :
        The momentum mesh, likely created using some quadrature rules, in units of inverse fermi.
    dk :
        The integration measure, likely created using some quadrature rules, in units of inverse fermi.
    t_lab :
        The on-shell energies of interest for computing observables, in units of MeV.
    system :
        The system of particles involved in the collision: 'pp', 'np', 'nn', 'p-alpha',
        or an instance of the Isospin class.
    dwa_wfs :
        The wave functions used for the distorted wave approach.
        Must include 'f', 'g', 'df', 'dg', 'f0', 'g0', 'df0', and 'dg0'. The g's may have an additional negative
        sign compared to some conventions.
    """

    def __init__(
        self,
        V0: ArrayLike,
        V1: ArrayLike,
        k: ArrayLike,
        dk: ArrayLike,
        t_lab: ArrayLike,
        system: Union[str, ScatteringSystem],
        boundary_condition: BoundaryCondition,
        dwa_wfs: Dict[ArrayLike] = None,
        is_coupled=False,
        nugget=0,
    ):
        # Mass and isospin info
        system = ScatteringSystem(system)
        mu = system.reduced_mass
        inv_mass = hbar_c ** 2 / (2 * mu)

        # Momentum info
        # q_cm = t_lab_to_q_cm(t_lab, isospin)
        q_cm = system.t_lab_to_q_cm(t_lab=t_lab)
        n_k = len(k)
        n_q = len(q_cm)

#         Id = np.identity(n_k, float)

        # In Landau's QM text, it is recommended to create an (n_k+1, n_k+1) matrix where the
        # extra element holds the on-shell part. This works fine, but is annoying because a new
        # matrix must be created for every on-shell piece you want to compute. Instead, we will
        # only create one set of (n_k, n_k) matrices, then upon solving the LS equation for the
        # off-shell reactance matrix, we will interpolate to the on-shell piece via this spline
        # matrix, which is only computed once and stored.
        Sp = cubic_spline_matrix(k, q_cm)

        is_G0_compressed = True
        G0 = greens_function_free_space(
                k=k,
                dk=dk,
                q_cm=q_cm,
                spline=Sp,
                boundary_condition=boundary_condition,
                is_compressed=is_G0_compressed,
        )

        Id_coup = G0_coup = Sp_coup = None
        if is_coupled:
            Id_coup = np.identity(2 * (n_k + 1), float)
            G0_coup = np.zeros((n_q, 2 * (n_k + 1), 2 * (n_k + 1)), float)
#                 G0_coup[:, :n_k + 1] = G0_coup[:, n_k + 1:] = G0

            for i in range(G0.shape[0]):
                G0_coup[i] = np.tile(G0[i], (2 * (n_k + 1), 2))

        n_p = V1.shape[-1]
        # Store everything
        self.V0 = V0
        self.V1 = V1
        V0_sub = []
        V1_sub = []
        if is_coupled:
            for i in range(n_q):
                V0_sub.append(np.array([[V0[i, n_k, n_k], V0[i, n_k, 2 * n_k + 1]], 
                                        [V0[i, 2 * n_k + 1, n_k], V0[i, -1, -1]]]))
                V1_sub.append(
                    np.stack(
                        [np.array([[V1[i, n_k, n_k, p], V1[i, n_k, 2 * n_k + 1, p]], 
                                   [V1[i, 2 * n_k + 1, n_k, p], V1[i, -1, -1, p]]]) for p in range(n_p)],
                        axis=-1,
                    )
                )
                V1_sub[i][0][1] = V1_sub[i][1][0]
        else:
            for i in range(n_q):
                V0_sub.append(V0[:, -1, -1])
                V1_sub.append(
                    np.stack(
                        [V1[:, -1, -1, p] for p in range(n_p)], axis=-1
                    )
                )                    
        if is_coupled:
            Id = np.identity(2 * (n_k + 1), float)
            self.V0_sub = np.stack(V0_sub, axis=0)
            self.V1_sub = np.stack(V1_sub, axis=0)
            
        else:
            Id = np.identity(n_k + 1, float)
            self.V0_sub = V0_sub[0]
            self.V1_sub = V1_sub[0]
            
        self.inv_mass = inv_mass
        self.q_cm = q_cm
        self.n_k = n_k
        self.n_q = n_q
        self.n_p = n_p
        self.k = k
        self.dk = dk
        self.Id = Id
        self.G0 = G0
        self.Sp = Sp
        self.Id_coup = Id_coup
        self.G0_coup = G0_coup
        self.Sp_coup = Sp_coup
        self.boundary_condition = boundary_condition
        self.is_coupled = is_coupled
        self.dwa_wfs = dwa_wfs
        self.nugget = nugget

        if boundary_condition is BoundaryCondition.STANDING:
            lippmann_schwinger_dtype = float
        else:
            lippmann_schwinger_dtype = "complex128"
        self.lippmann_schwinger_dtype = lippmann_schwinger_dtype

        # Attributes that will be created during the call to `fit`
        self.p_train = None
        self.K_train = None
        self.K_on_shell_train = None
        self.phase_train = None
        self.m0_vec = None
        self.m1_vec = None
        self.M0 = None
        self.M1 = None

    def full_potential(self, p: ArrayLike) -> ArrayLike:
        r"""Returns the full-space potential in momentum space.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        V : shape = (n_k, n_k)
        """
        return self.V0 + self.V1 @ p

    def on_shell_potential(self, p: ArrayLike) -> ArrayLike:
        r"""Returns the potential interpolated to the on-shell momenta.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        V_sub : shape = (n_t_lab, n_t_lab)
        """
        return self.V0_sub + self.V1_sub @ p

    def m_vec(self, p: ArrayLike) -> ArrayLike:
        r"""Returns the on shell part of the m vector.

        It is a vector in the space of training indices, and is defined by

        .. math::
            K_i G_0 V + V G_0 K_i

        where i is the index of the training points.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        m_vec : shape = (n_train, n_t_lab)
        """
        return self.m0_vec + self.m1_vec @ p

    def M_mat(self, p) -> ArrayLike:
        r"""Returns the on shell part of the M matrix.

        It is a matrix in the space of training indices, and is defined by

        .. math::
            K_i G_0 K_j + K_j G_0 K_i - K_i G_0 V G_0 K_j - K_j G_0 V G_0 K_i

        where i and j are indices of the training points.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        M_mat : shape = (n_t_lab, n_train, n_train)
        """
        M = self.M0 + self.M1 @ p
        if self.nugget != 0:
            M = M + self.nugget * np.eye(M.shape[-1])
        return M
    
    def coefficients(self, p):
        r"""Returns the coefficients of the reactance matrix expansion.

        The linear combination of these coefficients and the reactance training matrices allows the
        emulation of the reactance at other parameter values.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends

        Returns
        -------
        coefficients
        """
        return np.linalg.solve(self.M_mat(p), self.m_vec(p))
    
    def predict(
        self,
        p: ArrayLike,
        return_phase: bool = False,
        full_space: bool = False,
        return_gradient=False,
    ) -> ArrayLike:
        """Returns the on-shell reactance matrix (or phase shifts) either via emulation or the full-space calculation.

        Parameters
        ----------
        p :
            The parameters upon which the potential depends
        return_phase :
            If True, this will return phase shifts (in degrees) rather than the K matrix. Defaults to False
        full_space :
            If True, this will compute the quantity using the full-space simulator, rather than the emulator.
            Defaults to False.
        return_gradient : bool
            Whether the gradient is to be returned along with the reactance or phase shifts. Defaults to False.

        Returns
        -------
        quantity : shape = (n_t_lab,)
            Either the on-shell reactance matrix or the phase shifts
        """
        if len(p) >= 4:
            p = fix_coupled_lecs(p)
            
        if full_space:
            out = self.reactance(p, shell="on", return_gradient=return_gradient)
            if return_gradient:
                K, dK = out
            else:
                K = out
                dK = None
        else:
            nugget = self.nugget
            V = self.on_shell_potential(p)
            m_vec = self.m_vec(p)
            M = self.M_mat(p)
            
#             Minv_m = np.linalg.solve(M, m_vec)
#             K = V + 0.5 * (m_vec * Minv_m).sum(axis=-1)
            
            M_h = M + self.nugget * np.eye(M.shape[-1])
            c = self.coefficients(p)
            
            if self.is_coupled:
                c_M_c = np.einsum("qabi,qabij,qabj->qab", c, M_h, c)
            else:
                c_M_c = np.einsum("qi,qij,qj->q", c, M_h, c)
                
            K = V + 0.5 * c_M_c

            dK = None
            if return_gradient:
                dMinv_m = np.sum(self.m1_vec * Minv_m[..., None], axis=1)
                dK = self.V1_sub + dMinv_m
                dK -= 0.5 * np.einsum("qi,qijp,qj->qp", Minv_m, self.M1, Minv_m)

            if self.is_coupled:
                q_cm = self.q_cm[:, None, None]
            else:
                q_cm = self.q_cm
            K *= q_cm * pi / 2
            if return_gradient:
                dK *= q_cm[..., None] * pi / 2

        if self.dwa_wfs is not None:
            from .utils import dwa

            K = dwa(
                K=K,
                f0=self.dwa_wfs["f0"],
                g0=self.dwa_wfs["g0"],
                df0=self.dwa_wfs["df0"],
                dg0=self.dwa_wfs["dg0"],
                f=self.dwa_wfs["f"],
                g=self.dwa_wfs["g"],
                df=self.dwa_wfs["df"],
                dg=self.dwa_wfs["dg"],
                coupled=False,
                dK=dK,
            )
                        
        if self.is_coupled:
            K = fix_output(K.shape[0], K)
        else:
            K = -K

        if return_gradient:
            return K, dK
        if return_phase:
            # TODO: Handle gradients?
            return self.phase_shifts(K)
        return K

    def fit(self, p_train: ArrayLike) -> TwoBodyScattering:
        """Train the reactance emulator.

        Parameters
        ----------
        p_train : shape = (n_train, n_p)
            The parameters of the potential at which to compute the reactance matrix with full fidelity.
            These reactance matrices will be stored and used to quickly emulate the reactance matrix
            at other parameter values via the `predict` method.

        Returns
        -------
        self
        """
        # Loop over training points and compute the reactance at each point.
        K_train = []
        for i, p in enumerate(p_train):
            if len(p) >= 4:
                p = fix_coupled_lecs(p)
            K_i = self.reactance(p, include_q=False, shell="half")
            K_train.append(K_i)
        K_train = np.stack(K_train, axis=-1)
        
        # These variables will behave differently depending on is_coupled.
        # Define the up front to make life easier later.
        is_coupled = self.is_coupled
        n_k = self.n_k
        
        if is_coupled:
            G0 = self.G0_coup
            Sp = self.Sp_coup
            q_cm = self.q_cm[:, None, None, None]
        else:
            G0 = self.G0
            Sp = self.Sp
            q_cm = self.q_cm[:, None]

        # This is just for convenience for checking the training points.
        # Put q_cm back in so that the phases can be extracted.

        if is_coupled:
            K_on_shell_train = q_cm * np.stack(
                [K_train[i, :, -1] for i in range(self.n_q)], axis=0
            )
            
        else:
            K_on_shell_train = q_cm * np.stack(
                [K_train[i, -1] for i in range(self.n_q)], axis=0
            )
        # This matrix product is needed multiple times going forward. Compute it once.
        if is_coupled:
            G0_new = np.zeros((self.n_q, 2, 2 * (n_k + 1)))

            G0_new[:, 0] = G0[:, n_k]
            G0_new[:, 1] = G0[:, 2 * n_k + 1]

            G0_K = G0_new[:, :, :, None] * K_train
        else:
            # G0_K shape = (n_q, n_k, n_train)
            G0_K = self.G0[..., None] * K_train

        # =========================
        # The m vector.
        # Calculate the on shell part of the operator: K_i G_0 V + V G_0 K_i
        # where i denotes the training point. This creates a vector indexed by i.
        # =========================
        # We only want the on shell part

        # m0_vec shape = (n_q, n_train) or (n_q, 2, 2, n_train)
    
        if is_coupled:
            V0_on_shell = np.stack((self.V0[:, n_k], self.V0[:, 2 * n_k + 1]), axis=1)
            V1_on_shell = np.stack((self.V1[:, n_k], self.V1[:, 2 * n_k + 1]), axis=1)
            m0_vec = np.stack([V0_on_shell[i] @ G0_K[i] for i in range(self.n_q)])
                  
            m1_vec = np.stack(
            [
                np.stack([V1_on_shell[i, :, :, p] @ G0_K[i] for i in range(self.n_q)])
                for p in range(self.n_p)
            ],
            axis=-1,
        )
            
        else:
            m0_vec = np.stack([self.V0[i][:, -1] @ G0_K[i] for i in range(self.n_q)])
            m1_vec = np.stack(
                [
                    np.stack([self.V1[i][:, -1, p] @ G0_K[i] for i in range(self.n_q)])
                    for p in range(self.n_p)
                ],
                axis=-1,
            )

        # Do the same thing, but loop over the parameter dimension and stack.
        # m1_vec shape = (n_q, n_train, n_params) or (n_q, 2, 2, n_train, n_params)
        if is_coupled:
            m0_vec += m0_vec.swapaxes(-2, -3)
            m1_vec += m1_vec.swapaxes(-3, -4)
        else:
            m0_vec *= 2
            m1_vec *= 2

        # =========================
        # The M matrix.
        # Calculate the on shell part of the operator:
        # K_i G_0 K_j + K_j G_0 K_i - K_i G_0 V G_0 K_j - K_j G_0 V G_0 K_i
        # where i, j denote the training points. This creates a matrix indexed by i and j.
        # =========================

        if is_coupled:
            # This is a matrix in the space of training points: (n_t_lab, 2, 2, n_train, n_train)
            M0 = K_train.swapaxes(-1, -2)[:, None, :, :, :] @ G0_K[:, :, None, :, :]
            M0 -= (
                G0_K.swapaxes(-1, -2)[:, :, None, :, :]
                @ self.V0[:, None, None, :, :]
                @ G0_K[:, None, :, :, :]
            )
        else:
            # This is a matrix in the space of training points: (n_t_lab, n_train, n_train)
            M0 = K_train.swapaxes(-1, -2) @ G0_K
            M0 -= G0_K.swapaxes(-1, -2) @ self.V0 @ G0_K
                
        M0 += M0.swapaxes(-1, -2)

        if is_coupled:
            M1 = np.stack(
                [
                    G0_K.swapaxes(-1, -2)[:, None, :, :, :]
                    @ self.V1[:, None, None, :, :, :][..., i]
                    @ G0_K[:, :, None, :, :]
                    for i in range(self.n_p)
                ],
                axis=-1,
            )
                
        else:
            M1 = np.stack(
                [
                    G0_K.swapaxes(-1, -2) @ self.V1[..., i] @ G0_K
                    for i in range(self.n_p)
                ],
                axis=-1,
            )
            
        M1 += M1.swapaxes(-2, -3)
        M1 *= -1

        # Store the emulator-specific objects
        self.m0_vec = m0_vec
        self.m1_vec = m1_vec
        # self.M_const = M_const
        self.M0 = M0
        self.M1 = M1

        # Store other objects for convenience and debugging
        self.p_train = p_train
        self.K_train = K_train
        self.K_on_shell_train = K_on_shell_train
        self.phase_train = np.stack(
            [
                self.phase_shifts(self.K_on_shell_train[..., i], fix=True)
                for i in range(len(p_train))
            ],
            axis=-1,
        )
        return self

    def _compute_reactance_no_q(self, V, G0, Id, Sp, K, shell, dV=None, dK=None):
        return_gradient = dV is not None and dK is not None
        is_coupled = self.is_coupled
        n_k = self.n_k
        
        for i in range(self.n_q):            
            if is_coupled:
                ket = np.linalg.solve(Id - V[i] * G0[i], V[i])
            else:
                ket = np.linalg.solve(Id - V[i] * G0[i], V[i][:, -1])
                
            if shell == "half":
                if is_coupled:
                    K[i] = np.array(([ket.T[n_k], ket.T[2 * n_k + 1]]))
                else:
                    K[i] = ket.T
            else:
                if is_coupled:
                    K[i] = np.array(([ket.T[n_k, n_k], 
                                      ket.T[n_k, 2 * n_k + 1]], 
                                     [ket.T[2 * n_k + 1, n_k], 
                                      ket.T[2 * n_k + 1, 2 * n_k + 1]]))
                else:
                    K[i] = ket[-1]
                    
                if return_gradient:
                    d_bra = np.linalg.solve(M.T, Sp[i])
                    d_ket = Sp[i] + G0[i] * ket
                    for a in range(self.n_p):
                        dK[i, ..., a] = d_bra @ dV[..., a] @ d_ket
        K *= pi / 2
        if return_gradient:
            # dK is modified in place so does not need to be returned
            dK *= pi / 2
        return K

    def _reactance_coupled(
        self, p: ArrayLike, include_q: bool = True, shell="on", return_gradient=False
    ):
        n_k = self.n_k
        if shell not in ["on", "half"]:
            raise ValueError("shell must be one of 'on' or 'half'.")
        V = self.full_potential(p)
        if shell == "half":
            K = np.zeros((self.n_q, 2, 2 * (n_k + 1)), self.lippmann_schwinger_dtype)
        else:
            K = np.zeros((self.n_q, 2, 2), self.lippmann_schwinger_dtype)

        dK = None
        dV = None
        if return_gradient:
            if shell == "half":
                raise ValueError("If return_gradient is True, then shell must be 'on'.")
            dK = np.zeros((self.n_q, 2, 2, self.n_p), self.lippmann_schwinger_dtype)
            dV = self.V1

        Id = self.Id_coup
        Sp = self.Sp_coup
        G0 = self.G0_coup
        # dK is modified in place, if applicable
        K = self._compute_reactance_no_q(
            V=V, G0=G0, Id=Id, Sp=Sp, K=K, shell=shell, dV=dV, dK=dK
        )
        if include_q:
            q_cm = self.q_cm[:, None, None]
            # For the emulator training matrices, q should not be included
            K *= q_cm
            if return_gradient:
                dK *= q_cm[..., None]
        return K

    def _reactance_uncoupled(
        self, p: ArrayLike, include_q: bool = True, shell="on", return_gradient=False
    ):
        n_k = self.n_k
        if shell not in ["on", "half"]:
            raise ValueError("shell must be one of 'on' or 'half'.")
        V = self.full_potential(p)
        if shell == "half":
            K = np.zeros((self.n_q, n_k + 1), self.lippmann_schwinger_dtype)
        else:
            K = np.zeros(self.n_q, self.lippmann_schwinger_dtype)

        dK = None
        dV = None
        if return_gradient:
            if shell == "half":
                raise ValueError("If return_gradient is True, then shell must be 'on'.")
            dK = np.zeros((self.n_q, self.n_p), self.lippmann_schwinger_dtype)
            dV = self.V1

        Id = self.Id
        Sp = self.Sp
        G0 = self.G0
        # dK is modified in place, if applicable
        K = self._compute_reactance_no_q(
            V=V, G0=G0, Id=Id, Sp=Sp, K=K, shell=shell, dV=dV, dK=dK
        )
        if include_q:
            if shell == "half":
                q_cm = self.q_cm[:, None]
            else:
                q_cm = self.q_cm
                if return_gradient:
                    dK *= q_cm[:, None]
            # For the emulator training matrices, q should not be included
            K *= q_cm

        if return_gradient:
            return K, dK
        return K

    def reactance(
        self, p: ArrayLike, include_q: bool = True, shell="on", return_gradient=False
    ):
        """Computes the reactance matrix by solving the Lippmann-Schwinger equation.

        Parameters
        ----------
        p :
            The parameters of the potential at which to compute the reactance matrix with full fidelity.
        include_q :
            Whether the K matrix should be multiplied by the center-of-mass momentum.
            This makes the matrix dimensionless and makes extracting
            phase shifts easier since it just involves an arc-tangent. Defaults to True.
        shell : str
            Whether the reactance matrix should be on-shell or half on-shell. Valid values are ['on', 'half'].

        Returns
        -------
        K :
            The reactance matrix. If shell == 'on', then shape = (n_t_lab,).
            If shell == 'half' then shape = (n_t_lab, n_k).
        """
        if self.is_coupled:
            return self._reactance_coupled(
                p=p, include_q=include_q, shell=shell, return_gradient=return_gradient
            )
        else:
            return self._reactance_uncoupled(
                p=p, include_q=include_q, shell=shell, return_gradient=return_gradient
            )

    def phase_shifts(self, K, fix=True):
        r"""Computes phase shifts in degrees given the solution to the LS equation.

        Parameters
        ----------
        K :
            The on-shell solution to the LS equation. Depending on the choice of boundary condition, this could
            represent the K matrix, or the incoming or outgoing T matrix.
        fix :
            Whether to try to make the phase shifts continuous, as opposed to jump by 180 degrees.
            Defaults to True.

        Returns
        -------
        phase_shifts
        """
        if self.boundary_condition is BoundaryCondition.OUTGOING:
            K = t_matrix_outgoing_to_standing(K)
        if self.boundary_condition is BoundaryCondition.INCOMING:
            K = t_matrix_incoming_to_standing(K)

        if self.is_coupled:
            # Some attempts to make the coupled phases continuous. It doesn't always work.

            def _fix_phases(delta_minus, delta_plus, epsilon, bar=True):
                delta_minus, delta_plus, epsilon = np.atleast_1d(
                    delta_minus, delta_plus, epsilon
                )
                d = delta_minus - delta_plus
                # d must be in range -pi/2 to pi/2 for some reason.
                offset = (d + np.pi / 2) // np.pi
                # Will not affect S since phases are only defined modulo pi
                dm = delta_minus - offset * np.pi
                if bar:
                    # epsilon must be in -pi/4 to pi/4
                    e_offset = (2 * epsilon + np.pi / 2) // np.pi
                    e = epsilon - e_offset * np.pi / 2
                else:
                    e_offset = (epsilon + np.pi / 2) // np.pi
                    e = epsilon - e_offset * np.pi
                # e[offset % 2 == 1] *= -1
                return dm, delta_plus, e

            def transform_phases(delta_minus, delta_plus, epsilon, to_bar=True):
                # delta_minus = delta_minus % np.pi - np.pi/2
                # delta_plus = delta_plus % np.pi - np.pi / 2
                # epsilon = (epsilon % np.pi) - np.pi / 2
                delta_minus, delta_plus, epsilon = np.atleast_1d(
                    delta_minus, delta_plus, epsilon
                )
                # delta_minus, delta_plus, epsilon = _fix_phases(
                #     delta_minus, delta_plus, epsilon, bar=not to_bar)
                d = delta_minus - delta_plus
                s = delta_minus + delta_plus
                # offset = (d + np.pi / 2) // np.pi
                offset = (s + np.pi / 2) // np.pi
                s -= offset * np.pi
                # s = s % np.pi
                if to_bar:
                    # dm, dp, and e are *bar* phase shifts
                    e = 0.5 * np.arcsin(np.sin(2 * epsilon) * np.sin(d))
                    # dm = 0.5 * (s + np.arcsin(np.tan(2 * e) / np.tan(2 * epsilon)))
                    diff = np.arcsin(np.tan(2 * e) / np.tan(2 * epsilon))
                else:
                    e = 0.5 * np.arctan(np.tan(2 * epsilon) / np.sin(d))
                    diff = np.arcsin(np.sin(2 * epsilon) / np.sin(2 * e))
                # dm -= offset * np.pi
                # dp = s - dm
                dm = 0.5 * (s + diff)
                dp = 0.5 * (s - diff)
                return dm, dp, e

            K00, K01, K11 = K[..., 0, 0], K[..., 0, 1], K[..., 1, 1]
            PT = self.q_cm * 0 + 1
            Epsilon = np.arctan(2.0 * K01 / (K00 - K11)) / 2.0
            rEpsilon = (K00 - K11) / np.cos(2.0 * Epsilon)
            Delta_a = -1.0 * np.arctan(PT[:] * (K00 + K11 + rEpsilon) / 2.0)
            Delta_b = -1.0 * np.arctan(PT[:] * (K00 + K11 - rEpsilon) / 2.0)

            Delta_a, Delta_b, Epsilon = transform_phases(
                Delta_a, Delta_b, Epsilon, to_bar=True
            )
            Delta_a, Delta_b, Epsilon = _fix_phases(Delta_a, Delta_b, Epsilon, bar=True)
            ps = np.stack([Delta_a, Delta_b, Epsilon], axis=0) * 180.0 / pi
        else:
            ps = np.arctan(-K) * 180.0 / pi
        if fix:
            ps = fix_phases_continuity(ps, is_radians=False)
        return ps


