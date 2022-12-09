from typing import Tuple
from numpy.typing import ArrayLike
import numpy as np
from .constants import hbar_c
from .constants import alpha
from .types import Isospin, ScatteringSystem


def spin_obs_errors(E, deg, obs_sim, obs_emu):
    len_E = len(E)
    len_deg = len(deg)
    N = obs_sim.shape[0]
    
    dsg_sim = np.zeros((N, len_E, len_deg))    
    dsg_emu = np.zeros((N, len_E, len_deg))
    D_sim = np.zeros((N, len_E, len_deg))
    D_emu = np.zeros((N, len_E, len_deg))
    A_sim = np.zeros((N, len_E, len_deg))
    A_emu = np.zeros((N, len_E, len_deg))
    Ay_sim = np.zeros((N, len_E, len_deg))
    Ay_emu = np.zeros((N, len_E, len_deg))
    Axx_sim = np.zeros((N, len_E, len_deg))
    Axx_emu = np.zeros((N, len_E, len_deg))
    Ayy_sim = np.zeros((N, len_E, len_deg))
    Ayy_emu = np.zeros((N, len_E, len_deg))
    
    for i in range(N):
        dsg_sim[i] = obs_sim[i]['DSG']
        dsg_emu[i] = obs_emu[i]['DSG']
        D_sim[i] = obs_sim[i]['D']
        D_emu[i] = obs_emu[i]['D']
        A_sim[i] = obs_sim[i]['A']
        A_emu[i] = obs_emu[i]['A']
        Ay_sim[i] = obs_sim[i]['PB']
        Ay_emu[i] = obs_emu[i]['PB']
        Axx_sim[i] = obs_sim[i]['AXX']
        Axx_emu[i] = obs_emu[i]['AXX']
        Ayy_sim[i] = obs_sim[i]['AYY']
        Ayy_emu[i] = obs_emu[i]['AYY']
    
    dsg_abs_err, dsg_rel_err = compute_errors(dsg_sim, dsg_emu)
    D_abs_err, D_rel_err = compute_errors(D_sim, D_emu)
    A_abs_err, A_rel_err = compute_errors(A_sim, A_emu)
    Ay_abs_err, Ay_rel_err = compute_errors(Ay_sim, Ay_emu)
    Axx_abs_err, Axx_rel_err = compute_errors(Axx_sim, Axx_emu)
    Ayy_abs_err, Ayy_rel_err = compute_errors(Ayy_sim, Ayy_emu)
    
#     return dsg_err, D_err, A_err, Ay_err, Axx_err, Ayy_err
    return dsg_rel_err, D_rel_err, Ay_rel_err, Ayy_rel_err

def compute_errors(exact, predict, obs=None):
    
    if obs != None:
        predict = predict[obs]
        exact = exact[obs]
    
    abs_error = abs(predict - exact)
    rel_error = abs((predict - exact) / exact)
    
    return abs_error, rel_error





def fine_structure_relativistic(q_cm, mu):
    # TODO: Check this. Need to find a reference.
    E = np.sqrt((hbar_c * q_cm) ** 2 + 4 * mu ** 2)
    return alpha * (E ** 2 + (q_cm * hbar_c) ** 2) / (2 * mu * E)


def sommerfeld_parameter(q_cm, mu, z1=1, z2=1, relativistic=False):
    if relativistic:
        fine_struct = fine_structure_relativistic(q_cm, mu)
        # TODO: Must check alpha relativistic definition
        raise NotImplementedError
    else:
        fine_struct = alpha
    return fine_struct * z1 * z2 * mu / (q_cm * hbar_c)


def cubic_spline_matrix(old_mesh, new_mesh):
    r"""Computes a cubic spline matrix that only references the input and output locations, not the y values.

    This is useful because it can be computed once up front and stored, so long as the meshes remain constant.
    This code was originally written by Kyle Wendt.

    Parameters
    ----------
    old_mesh :
        The points where the function is already computed
    new_mesh :
        The points to interpolate towards

    Returns
    -------
    S : shape = (n_new, n_old)
        An interpolation matrix that will compute `f_new = S @ f_old`

    Notes
    -----
    This uses a technique called quasi-interpolation. See Ref [1]_.

    References
    ----------
    .. [1] Glöckle, W., Hasberg, G. & Neghabian, A.R.
       Numerical treatment of few body equations in momentum space by the Spline method.
       Z Physik A 305, 217–221 (1982). https://doi.org/10.1007/BF01417437
    """
    from numpy import zeros

    n = len(old_mesh)

    # All notation follows from the reference in the docstring.
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

    imin = old_mesh.argmin()
    imax = old_mesh.argmax()
    xmin = old_mesh[imin]
    xmax = old_mesh[imax]
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


def leggauss_shifted(deg, a=-1, b=1):
    """Obtain the Gaussian quadrature points and weights when the limits of integration are [a, b]

    Parameters
    ----------
    deg : int
        The degree of the quadrature
    a : float
        The lower limit of integration. Defaults to -1, the standard value.
    b : float
        The upper limit of integration. Defaults to +1, the standard value.

    Returns
    -------
    x : The integration locations
    w : The weights
    """
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(deg)
    w *= (b - a) / 2.0
    x = ((b - a) * x + (b + a)) / 2.0
    return x, w


def compute_vkk(
    k: ArrayLike,
    s: int,
    j: int,
    isospin: Isospin,
    v_r: ArrayLike,
    radial_mesh_spec: Tuple[str] = ("0 3 7 14", "250 250 250 500"),
    is_coupled: bool = False,
    reshape: bool = False,
    *args,
    **kwargs,
):
    r"""
    Compute the partial wave Fourier transform of a local potential.
    In the Scattering basis, the transform takes place with respect to free scattering states:

    .. math::
        \left \langle r l m \middle | k l' m' \right\rangle = \delta_{l,l'} \delta_{m, m'} j_l(k r)

    where as the Fourier basis we include a factor of :math:`\imath^l`.

    .. math::
        \left\langle r l m \middle | k l' m' \right\rangle = \imath^l \delta_{l,l'} \delta_{m, m'} j_l(k r)

    For the scattering basis, the final transforms has the form:

    .. math::
        V_{l, l'}(k, k') = \frac{2}{\pi} \int r^2 dr j_{l}(k r) j_{l}(k r') V_{l, l'}(r)

    The Fourier basis picks up an extra phase :math:`\imath^{l'-l}` relative to the scattering basis.

    Parameters
    ----------
    k : array_like of floats
        Momentum "grid" the compute the potential on
    s : int
        The spin angular momentum
    j : int
        Total angular momentum
    isospin :
        The isospin
    v_r :
        The potential in MeV. If is_coupled, then it must be (2, 2, r) shaped.
    radial_mesh_spec : args to compound_mesh
        The transform will use

        .. code-block:: python

            r, dr = nn_scattering.utils.quadrature.compound_mesh(*radial_mesh_spec)

        to construct the integration quadrature.  If radial_mesh_spec is None we use:

        .. code-block:: python

            radial_mesh_spec = '0 3 7 14', '250 250 250 500'

        This generates growing Guass-Legendre meshes from :math:`r=0\,\rm{fm}` to :math:`14\,\rm{fm}`
        and then mesh from :math:`14\,\rm{fm}` to :math:`\infty`
    reshape : bool
        Reshape coupled channels from a 2 x 2 x len(kmesh) x len(kmesh) to a 2 * len(kmesh) x 2 * len(kmesh) array

    Returns
    -------

    """
    from numpy import (
        ascontiguousarray,
        atleast_1d,
        outer,
        pi,
        sqrt,
        squeeze,
        stack,
        swapaxes,
        zeros,
    )
    from scipy.special import spherical_jn

    from .quadrature import compound_mesh

    k = atleast_1d(k)

    # radial_mesh_spec = radial_mesh_spec or ("0 3 7 14", "250 250 250 500")
    r, dr = compound_mesh(*radial_mesh_spec)

    def spherical_jl(ell, x):
        if ell < 0:
            return zeros(x.shape, float)
        return spherical_jn(ell, x)

    if is_coupled:
        v_kk = zeros((2, 2, k.size, k.size))
        j_l = (
            sqrt(2 / pi)
            * (r * sqrt(dr))
            * stack([spherical_jl(ell, outer(k, r)) for ell in (j - 1, j + 1)])
        )
        v_rr = v_r(r, j - 1, s, j, isospin, *args, **kwargs)

        for a, l_b in enumerate((j - 1, j + 1)):
            for b, l_k in enumerate((j - 1, j + 1)):
                v_kk[a, b] = (j_l[a, :, None, :] * j_l[b, None, :, :]) @ v_rr[a, b]
        if reshape:
            v_kk = ascontiguousarray(
                swapaxes(v_kk, 1, 2).reshape((2 * k.size, 2 * k.size))
            )
    else:
        j_l = sqrt(2 / pi) * (r * sqrt(dr)) * spherical_jn(j, outer(k, r))
        v_rr = v_r(r, j, s, j, isospin, *args, **kwargs)
        v_kk = (j_l[:, None, :] * j_l[None, :, :]) @ v_rr
    return squeeze(v_kk)


def compute_momentum_potential_from_nonlocal(
    k: ArrayLike,
    j: int,
    v_r: ArrayLike,
    radial_mesh_spec: Tuple[str] = ("0 3 7 14", "250 250 250 500"),
):
    from numpy import (
        atleast_1d,
        outer,
        pi,
        sqrt,
    )
    from scipy.special import spherical_jn
    from .quadrature import compound_mesh

    k = atleast_1d(k)
    r, dr = compound_mesh(*radial_mesh_spec)
    # One r for integral, the other to convert F_ell(k r) --> r * k * j_ell(k r)
    j_l = sqrt(2 / pi) * (r * dr) * r * spherical_jn(j, outer(k, r))
    v_rr = v_r(r[:, None], r)
    v_kk = j_l @ v_rr @ j_l.T
    return v_kk


def fix_phases_continuity(phases, n0=None, is_radians=True):
    """Returns smooth phase shifts by removing jumps by multiples of pi.

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
    n[..., 1:] = (
        round((phases[..., 1:] - phases[..., :-1]) / factor).cumsum(-1) * factor
    )
    # Make the jumps be relative to the final value of the phase shift
    # i.e., don't adjust phases[..., -1]
    n -= n[..., [-1]]
    # Subtract away the jumps
    smooth_phases = phases.copy()
    smooth_phases[...] -= n
    if (
        n0 is not None
    ):  # If the initial (rather than final) value of phases is constrained
        # Now move the entire phase shift at once so it starts in the range (n0-1/2, n0+1/2) * pi.
        smooth_phases[...] -= (round(smooth_phases[..., 0] / factor) - n0) * factor
    return smooth_phases


def coulomb_wave_functions(r, q, mu, j_max, z1=1, z2=1, is_free=False):
    """ P-P Coulomb wave functions

    Parameters
    ----------
    r : array_like of floats
        radial coordinates in ..math`\\rm{fm}^`
    q : array_like of floats
        radial coordinates in ..math`\\rm{fm}^-1`
    mu
    j_max : positive integer
    z1 :
    z2 :
    is_free

    Returns
    -------
        f, g, f_p, g_p : array_like of floats
            Wave functions and their derivatives wrt x = q * r.  Each is an array of shape (l_max + 1, r.size, q.size)
    """
    from numpy import arange, array, float128, sqrt, stack, squeeze, zeros, zeros_like
    from numpy import atleast_1d, broadcast_to, broadcast_arrays

    from mpmath import coulombf, coulombg, mp

    r, q = map(atleast_1d, (r, q))
    r, q = broadcast_arrays(r, q)

    x = squeeze(r * q)
    shape = (j_max + 1,) + x.shape

    if is_free:
        eta = zeros_like(q)
    else:
        eta = sommerfeld_parameter(q, mu=mu, z1=z1, z2=z2)

    ell = arange(0, j_max + 1)
    ell = ell.reshape(ell.shape + (1,) * len(x.shape))

    lams = broadcast_to(1.0 * ell, shape)

    def s(L):
        return L / x + eta / L

    def r(L):
        return sqrt(1 + (eta / L) ** 2)

    def coulombf_(lam, eta, x):
        return array(
            [coulombf(l_, e_, x_) for l_, e_, x_ in zip(lam.flat, eta.flat, x.flat)]
        ).reshape(x.shape)

    def coulombg_(lam, eta, x):
        return array(
            [coulombg(l_, e_, x_) for l_, e_, x_ in zip(lam.flat, eta.flat, x.flat)]
        ).reshape(x.shape)

    f, g, fm, gm = [[None for _ in range(len(lams))] for _ in range(4)]
    for j, lam in enumerate(lams):
        f[j] = coulombf_(lam, eta, x)
        g[j] = coulombg_(lam, eta, x)
        if j == 0:
            fm[j] = coulombf_(lam + 1, eta, x)
            gm[j] = coulombg_(lam + 1, eta, x)
        else:
            fm[j] = coulombf_(lam - 1, eta, x)
            gm[j] = coulombg_(lam - 1, eta, x)

    f, g, fm, gm = map(array, (f, g, fm, gm))
    df, dg = (zeros(shape, float128) for _ in range(2))

    for j, lam in enumerate(lams):
        if j == 0:
            df[j] = s(lam + 1) * f[j] - r(lam + 1) * fm[j]
            dg[j] = s(lam + 1) * g[j] - r(lam + 1) * gm[j]
        else:
            df[j] = r(lam) * fm[j] - s(lam) * f[j]
            dg[j] = r(lam) * gm[j] - s(lam) * g[j]

    return array(stack([f, -g, df, -dg]), dtype=float)


def dwa(
    K, f0, g0, df0, dg0, f, g, df, dg, coupled=True, out=None, dK=None, dK_out=None
):
    from numpy.linalg import solve

    if out is None:
        out = np.zeros_like(K)
    if dK_out is None and dK is not None:
        dK_out = np.zeros_like(dK)
    dA = None
    # We do not need to multiply K by -1, since G minus sign convention takes that into account
    # Uses the fact that K, f0, and g0 are symmetric <- are they symmetric with magnetic moment?
    if coupled:
        A = np.swapaxes(solve(df0 + K @ dg0, f0 + K @ g0), -1, -2)
        if (
            dK is not None
        ):  # Compute this before overwriting `out`, just in case K == out
            dA = np.swapaxes(solve(df0 + K @ dg0, dK @ (g0 - A @ dg0)), -1, -2)
        # This returns K ~ -tan(delta)
        out[:] = -solve(g - A @ dg, f - A @ df)
        if dK is not None:
            # This must use the corrected K (called `out` here)
            dK_out[...] = solve(g - A @ dg, dA @ (df + dg @ out))
            return out, dK_out
    else:
        A = (f0 + K * g0) / (df0 + dg0 * K)
        if (
            dK is not None
        ):  # Compute this before overwriting `out`, just in case K == out
            dA = (g0 - A * dg0) / (df0 + dg0 * K) * dK
        out[:] = -(f - A * df) / (g - A * dg)
        if dK is not None:
            # This must use the corrected K (called `out` here)
            dK_out[...] = (df + dg * out) / (g - A * dg) * dA
            return out, dK_out
    return out


def read_hdf5_potential(filename):
    import h5py
    from .array_subclasses import NuclearPotentialArray

    with h5py.File(filename, "r") as file:
        # Extract placement info
        [ell, s, j, t] = file["waves_sing"][...].T
        [_, _, j_coup, _] = file["waves_coup"][:, 0, :].T
        k = file["k"][...]
        dk = file["dk"][...]
        nk = k.size
        lec_names = file["lec names"][...]
        nlecs = lec_names.size
        j_max = np.max(j)
        try:
            quadratic = file["quadratic"][...]
            has_quad = np.any(quadratic)
        except KeyError:
            quadratic = None
            has_quad = False

        # Create potentials
        v0 = NuclearPotentialArray((j_max + 1, 2 * nk, 2 * nk), dtype=float)
        vi = NuclearPotentialArray((j_max + 1, 2 * nk, 2 * nk, nlecs), dtype=float)
        vij = None

        pot_list = [v0, vi]
        pot_str_list = ["V0", "Vi"]
        if has_quad:
            vij = NuclearPotentialArray(
                (j_max + 1, 2 * nk, 2 * nk, nlecs, nlecs), dtype=float
            )
            pot_list.append(vij)
            pot_str_list.append("Vij")

        # Fill potentials appropriately
        _3P0 = (s == 1) & (ell == 1) & (j == 0)
        singlet = (s == 0) & (~_3P0)
        triplet = (s == 1) & (~_3P0)
        for v, v_str in zip(pot_list, pot_str_list):
            v["st"][j[singlet], :nk, :nk] = file[v_str + "_sing"][...][singlet]
            v["st"][j[triplet], nk:, nk:] = file[v_str + "_sing"][...][triplet]
            v["tt"][j[_3P0], nk:, nk:] = file[v_str + "_sing"][...][_3P0]
            v["tt"][j_coup] = file[v_str + "_coup"][...]
    return k, dk, v0, vi, vij, quadratic, lec_names


class PhaseShiftData:
    def __init__(self, filepath, model, isospin, load_all_channels=True):

        if not load_all_channels:
            raise NotImplementedError("Delayed Loading not implemented yet")

        self._isospin = str(Isospin(isospin))
        if self._isospin == "nn":
            raise ValueError("Only Example Data for pp and np channels exists")

        if load_all_channels:
            import h5py

            from numpy import arange, pi, zeros

            with h5py.File(filepath, "r") as file:
                if model not in file:
                    raise ValueError(f'model "{model}" not found in file: {filepath}')
                model = file[model]

                self.t_lab = model["T lab"][...]
                self.t_lab.setflags(write=False)
                j_max = model.attrs["j max"]

                deltas = zeros((5, j_max + 1, self.t_lab.size))

                deltas[0] = model[f"delta singlet {self._isospin}"][...]
                deltas[1] = model[f"delta triplet {self._isospin}"][...]
                deltas[2] = model[f"delta 1 {self._isospin}"][...]
                deltas[3] = model[f"delta 2 {self._isospin}"][...]
                deltas[4] = model[f"epsilon {self._isospin}"][...]

                if model.attrs["unit"] != "radians":
                    print("changing to rad")
                    deltas *= pi / 180

                self.deltas = deltas
                self.delta_singlet = deltas[0]
                self.delta_triplet = deltas[1]
                self.delta_1 = deltas[2]
                self.delta_2 = deltas[3]
                self.epsilon = deltas[4]
                self.js = arange(0, j_max + 1, dtype=int)
                for arr in (
                    self.t_lab,
                    self.delta_singlet,
                    self.delta_triplet,
                    self.delta_1,
                    self.delta_2,
                    self.epsilon,
                    self.js,
                ):
                    arr.setflags(write=False)
