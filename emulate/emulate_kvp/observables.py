"""
"""
from numpy import(
    pi, cos, sin, zeros, real, sqrt, 
    ndarray, array, conj, imag, einsum,
    identity, trace, reshape, atleast_2d,
    apply_along_axis, ones_like, arange
)
from numpy.linalg import solve
from numpy.typing import ArrayLike
from scipy.special import lpmn

from .kinematics import wigner_rotations
from .constants import mass_proton, mass_neutron, fm_to_sqrt_mb


def ell_matrix(ell_max):
    R"""Computes the 7x7 matrix of ells that rotate the set of R elements to M elements

    Treats the matrices for ell = 0 and ell = 1 as special,
    since certain matrix elements would diverge if naively implemented.
    These elements are set to zero.

    References
    ----------
    Stapp 1957, Table III
    MacGregor 1960

    There is a sign convention difference between the above references

    Parameters
    ----------
    ell_max : int
              The maximum (noninclusive) ell value for which the matrix is constructed.
              The matrix is created for all ells in [0, ell_max).

    Returns
    -------
    out : ndarray
          (7, 7, ell_max) shaped array of ells as described in the notes
    """
    ell = arange(ell_max)

    ell0 = ell[1:]
    ell01 = ell[2:]

    mat = zeros((7, 7, ell_max))

    # Mss expansion
    # mat[0, 0, :-1] = (2 * ell + 1) / 2.  # R_{LL}^{L00}
    mat[0, 0] = (2 * ell + 1) / 2.  # R_{LL}^{L00}
    mat[0, 0, -1] = 0.

    # Mst expansion
    # mat[6, 6, 1:-1] = - (2 * ell_st + 1) / np.sqrt(ell_st * (ell_st + 1))  # R_{ll}^{l10}
    mat[6, 6, 1:] = - (2 * ell0 + 1) / sqrt(ell0 * (ell0 + 1))  # R_{ll}^{l10}
    mat[6, 6, -1] = 0.

    # Group the rest by R's since they each behave differently at endpoints of sum

    # R_{LL}^{L+1,1,1}
    R_jlp1 = zeros((7, ell_max))
    R_jlp1[1] = ell + 2  # M11
    R_jlp1[2] = ell + 1  # M00
    R_jlp1[3] = - (ell + 2.) / (ell + 1.)  # M01
    R_jlp1[4] = 1  # M10
    R_jlp1[5] = 1. / (ell + 1)  # M1n1
    R_jlp1[:, -2:] = 0.
    mat[:, 1] = R_jlp1

    # R_{LL}^{L11}
    Rzero = zeros((7, ell_max - 1))
    Rzero[1] = 2 * ell0 + 1  # M11
    Rzero[3] = (2 * ell0 + 1.) / (ell0 * (ell0 + 1))  # M01
    Rzero[5] = - Rzero[3]  # M1n1
    Rzero[:, -1] = 0.
    mat[:, 2, 1:] = Rzero

    # R_{LL}^{L-1,1,1}
    R_jlm1 = zeros((7, ell_max - 1))
    R_jlm1[1] = ell0 - 1  # M11
    R_jlm1[2] = ell0  # M00
    R_jlm1[3] = (ell0 - 1) / ell0  # M01
    R_jlm1[4] = - 1  # M10
    R_jlm1[5] = 1. / ell0  # M1n1
    mat[:, 3, 1:] = R_jlm1

    # R_{L+2,L}^{L+1,1,1} Stapp version
    R_jlp1_pm = zeros((7, ell_max))
    R_jlp1_pm[1] = - sqrt((ell + 1) * (ell + 2))  # M11
    R_jlp1_pm[2] = - R_jlp1_pm[1]  # M00
    R_jlp1_pm[3] = sqrt((ell + 2) / (ell + 1))  # M01
    R_jlp1_pm[4] = R_jlp1_pm[3]  # M10
    R_jlp1_pm[5] = 1. / R_jlp1_pm[1]  # M1n1
    R_jlp1_pm[:, -2:] = 0.
    mat[:, 4] = R_jlp1_pm  # Stapp version
    # mat[:, 4] = - R_jlp1_pm  # MacGregor version

    # R_{L-2,L}^{L-1,1,1} Stapp version
    R_jlm1_mp = zeros((7, ell_max - 2))
    R_jlm1_mp[1] = - sqrt(ell01 * (ell01 - 1))  # M11
    R_jlm1_mp[2] = - R_jlm1_mp[1]  # M00
    R_jlm1_mp[3] = - sqrt((ell01 - 1) / ell01)  # M01
    R_jlm1_mp[4] = R_jlm1_mp[3]  # M10
    R_jlm1_mp[5] = 1 / R_jlm1_mp[1]  # M1n1
    mat[:, 5, 2:] = R_jlm1_mp  # Stapp version
    # mat[:, 5, 2:] = - R_jlm1_mp  # MacGregor version

    mat[1] /= 4
    mat[2] /= 2
    mat[3] /= -2 * sqrt(2)  # Forget why there's minus signs, but they work
    mat[4] /= -2 * sqrt(2)
    mat[5] /= 4
    mat[6] /= -2 * sqrt(2)

    return mat

def rvec_to_spin_scatt_matrix(j_max, x):
    """
    The expansion of the spin scattering matrix elements in terms of associated Legendre polynomials
    """
    mat = ell_matrix(j_max + 2).reshape(7, 7, j_max + 2, 1)  # .astype('complex128')
    mat = mat * ones_like(x)  # Broadcast over final dimension

    # Compute P_l(x), and P_l^1(x)
    [Pl0, Pl1, Pl2], _ = apply_along_axis(
        lambda xx, m, n: lpmn(m, n, xx[0]), axis=0, arr=atleast_2d(x), m=2, n=j_max + 1)

    # Now scale by factors
    mat[0] *= Pl0
    mat[1] *= Pl0
    mat[2] *= Pl0
    mat[3] *= Pl1
    mat[4] *= Pl1
    mat[5] *= Pl2
    mat[6] *= Pl1

    return mat
    
    
class Observables:
    """
    Creates an environment used to calculate the two-body scattering observables.

    Parameters
    ----------
    E : array
        The energy grid.
        If a single value is passed, an array is created for it.
    k : array
        The k values that correspond to the energy grid.
        If a single value is passed, an array is created for it.
    degrees : array
        The angle grid for the spin observables.  
    """
    def __init__(
        self, 
        E: ArrayLike, 
        k: ArrayLike, 
        degrees: ArrayLike
    ):
        if (isinstance(E, ndarray) == False):
            E = array([E])
            k = array([k])
        if (isinstance(degrees, ndarray) == False):
            degrees = array([degrees])
        
        k_coup = k[:, None, None]
        radians = degrees * pi / 180
        x = cos(radians)
        
        self.sin0 = self.fix_sine(x)
        self.alpha = wigner_rotations(k, x, mass_neutron, mass_proton)[0]
        
        self.E, self.degrees = E, degrees
        self.k, self.k_coup = k, k_coup
        self.x, self.radians = x, radians
        self.len_k = len(k)
        
    def R_to_R_vec(
        self, 
        j: int,
        V: ArrayLike,
        simulator,
        glockle: bool
    ) -> ArrayLike:
        """
        Converts the reactance matrix R to R vector for a specific partial wave.
        Used for the spin observables calculation.
        
        Parameters
        ----------
        j : int
            The momentum for which we are calculating Rvec
        V : array
            The potential at specific partial wave.
        simulator : instance
            A specific instance of the simulator.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
        
        Returns
        -------
        R / (1j * k0 / fm_to_sqrt_mb) : array
            The R vector at j.
        """
        if simulator.is_coupled:
            k0 = self.k_coup
        else:
            k0 = self.k
            
        R = self.get_R(j, V, simulator, glockle)
        return R / (1j * k0 / fm_to_sqrt_mb)        

    def get_R(
        self, 
        j: int,
        V: ArrayLike,
        simulator,
        glockle: bool
    ) -> ArrayLike:
        """
        Calculates the reactance matrix R.
        Used for the spin observables calculation.
        
        Parameters
        ----------
        j : int
            The momentum for which we are calculating Rvec
        V : array
            The potential.
        simulator : instance
            A specific instance of the simulator.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
            If True, use Glockle spline.
            If False, use Standard method.
        
        Returns
        -------
        R : array, shape = (..., n, n)
        """
        len_k = self.len_k
        is_coupled = simulator.is_coupled
        K = simulator.get_K0_exact(V, glockle)

        if glockle and is_coupled:
            K = reshape(K, (4, len_k)).T.reshape(len_k, 2, 2)
            
        return self.K_to_R(K, is_coupled)
    
    def K_to_R(
        self, 
        K: ArrayLike,
        is_coupled: bool
    ) -> ArrayLike:
        r"""
        Calculates the partial wave R matrix given the partial wave K matrix.
        This definition of R is like -2i * T, and is equal to S-1.

        Parameters
        ----------
        K : array, shape = (..., n, n)
            The K matrix
        is_coupled : boolean
            If True, calculates for coupled channels.
            If False, calculates for non-coupled channels.

        Returns
        -------
        R : array, shape = (..., n, n)
        """
        if is_coupled:
            I = identity(K.shape[-1])
            R = -2j * solve(I - 1j * K, K)
        else:
            R = -2j * K / (1 - 1j * K)
        return R
    
    def calculate_Rvec(
        self,
        j_values: list, 
        potentials: list, 
        simulators: list,
        glockle: bool
    ) -> ArrayLike:
        """
        Calculates the full R vector.
        Used for the spin observables calculation.
        
        Parameters
        ----------
        j : list[int]
            All the momenta for which we are calculating Rvec
        potentials : list[arrays]
            The potentials for all partial waves.
        simulator : list[instances]
            A list of simulators.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
            If True, use Glockle spline.
            If False, use Standard method.
        
        Returns
        -------
        Rvec : array
            The R vector.
        """
        Rvec = zeros((7, j_values[-1] + 2, self.len_k), dtype=complex)
        j_i = -1

        for j, V, sim in zip(j_values, potentials, simulators):
            R = self.R_to_R_vec(j, V, sim, glockle)
            
            if V.shape[0] == sim.N:        
                if j_i != j:
                    Rvec[0][j] = R
                else:
                    if j == 0 and V.shape[0] == sim.N:
                        Rvec[3][0] = R
                        Rvec[3][1] = R
                    else:
                        Rvec[2][j] = R
                j_i = j

            else:
                R = R.T
                Rvec[1][j - 1] = R[0][0] # jmom = 1
                Rvec[3][j + 1] = R[1][1] # jmom = 1
                Rvec[4][j - 1] = R[0][1] # jmom = 1
                Rvec[5][j + 1] = R[1][0] # jmom = 1

        return Rvec

    def partial_wave_cross_section(
        self, 
        j: int,
        V: ArrayLike,
        simulator,
        glockle: bool
    ) -> ArrayLike:
        """
        The total cross section for a specific partial wave j.
        
        Parameters
        ----------
        j : int
            The momentum for which we are calculating the cross section.
        V : array
            The potential at specific partial wave.
        simulator : instance
            A specific instance of the simulator.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
            If True, use Glockle spline.
            If False, use Standard method.
        
        Returns
        -------
        (2 * j + 1) * real(R) : array
            The total cross section at j.
        """   
        R = self.get_R(j, V, simulator, glockle)
        
        if simulator.is_coupled:
            R = trace(R, axis1=-1, axis2=-2)
            
        return (2 * j + 1) * real(R)

    def total_cross_section(
        self,
        j_values: list, 
        potentials: list, 
        simulators: list,
        glockle: bool = True
    ) -> ArrayLike:
        """
        Calculates the total cross section.

        Parameters
        ----------
        j_values : list[int]
            List of angular momentum corresponding to each potential.
        potentials : list[array]
            List of potentials corresponding to each angular momentum.
        simulators : list[instances]
            List of simulators instances that depending on if the channel is
            uncoupled or coupled.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
            If True, use Glockle spline.
            If False, use Standard method.

        Returns
        -------
        pi / (2 * (k0 / fm_to_sqrt_mb)**2) * sigma_j : array
            An array of the total cross section.
        """
        k0 = self.k
        sigma_j = 0

        for j0, V0, sim in zip(j_values, potentials, simulators):
            sigma_j += self.partial_wave_cross_section(j0, V0, sim, glockle)

        return pi / (2 * (k0 / fm_to_sqrt_mb)**2) * sigma_j

    def spin_observables(
        self,
        j_values: list, 
        potentials: list, 
        simulators: list,
        glockle: bool = True
    ) -> ArrayLike:
        """
        Calculates the spin observables and maps to dictionary.

        Parameters
        ----------
        j_values : list[int]
            List of angular momentum corresponding to each potential.
        potentials : list[array]
            List of potentials corresponding to each angular momentum.
        simulators : list[instances]
            List of simulators instances that depending on if the channel is
            uncoupled or coupled.
        glockle : boolean
            Specifies if we are using the Glockle spline or the std. method.
            If True, use Glockle spline.
            If False, use Standard method.

        Returns
        -------
        dict : dict
            A dictionary with the spin observables.
        """
        k0 = self.k
        alpha, sin0 = self.alpha, self.sin0
        x, radians = self.x, self.radians
        Rvec = self.calculate_Rvec(j_values, potentials, simulators, glockle)

        Lmat = rvec_to_spin_scatt_matrix(j_values[-1], x)
        amp = einsum('mnLT, nLQ -> mQT', Lmat, Rvec)

        self.saclay_parameters(sin0, amp)

        dsg = self.compute_dsg()
        Ay = self.compute_Ay(dsg)
        A = self.compute_A(sin0, alpha, dsg)
        D = self.compute_D(dsg)
        Axx = self.compute_Axx(sin0, dsg)
        Ayy = self.compute_Ayy(dsg)

        return {
            'DSG': dsg, 'PB': Ay, 'D': D, 
            'AXX': Axx, 'AYY': Ayy, 'A': A
            }    
    
    def compute_dsg(self):
        """
        Calculates the unpolarized differential cross section from the Saclay parameters.
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        return (
                0.5 * (abs(a) ** 2 + abs(b) ** 2 + abs(c) ** 2 
                       + abs(d) ** 2 + abs(e) ** 2 + abs(f) ** 2)
               )
    
    def compute_Ay(self, dsg):
        """
        Calculates the analyzing power A_y, (beam polarization PB), from the Saclay parameters.
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        return real(conj(a) * e + conj(b) * f) / dsg
    
    
    def compute_A(self, sin0, alpha, dsg):
        """
        Calculates the spin-flip amplitude A from the Saclay parameters.
        """
        radians = self.radians
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        return (- real(conj(a) * b - conj(e) * f) * sin(alpha + radians / 2)
             + real(conj(c) * d) * sin(alpha - radians / 2)
             - imag(conj(b) * e + conj(a) * f) * cos(alpha + radians / 2)
            ) / dsg
    
    def compute_D(self, dsg):
        """
        Calculates the depolarization parameter D from the Saclay parameters.
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        return (
                0.5 * (abs(a) ** 2 + abs(b) ** 2 - abs(c) ** 2 
                       - abs(d) ** 2 + abs(e) ** 2 + abs(f) ** 2) / dsg
        )
    
    def compute_Axx(self, sin0, dsg):
        """
        Calculates the spin-correlation amplitude A_xx from the Saclay parameters.
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        return (real(conj(a) * d) * self.x + real(conj(b) * c) - imag(conj(d) * e) * sin0) / dsg
    
    def compute_Ayy(self, dsg):
        """
        Calculates the spin-correlation amplitude A_yy from the Saclay parameters.
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f
        return (
                0.5 * (abs(a) ** 2 - abs(b) ** 2 - abs(c) ** 2 
                       + abs(d) ** 2 + abs(e) ** 2 - abs(f) ** 2) / dsg
        )
            
    def saclay_parameters(self, sin, amp):
        """
        Calculates the Saclay parameters from the amplitudes.
        """
        amp_ss = amp[0]
        amp_11 = amp[1]
        amp_00 = amp[2]
        amp_01 = amp[3]
        amp_10 = amp[4]
        amp_1n1 = amp[5]
        amp_st = amp[6]

        self.a = 0.5 * (amp_11 + amp_00 - amp_1n1)
        self.b = 0.5 * (amp_11 + amp_ss + amp_1n1)
        self.c = 0.5 * (amp_11 - amp_ss + amp_1n1)
        self.d = - 1 / (sqrt(2) * sin) * (amp_10 + amp_01)
        self.e = 1j / sqrt(2) * (amp_10 - amp_01)
        self.f = 1j * sqrt(2) * amp_st

        return None

    def fix_sine(self, x):
        """
        Ensures the sine function does not give a 0.
        """
        forward = x == 1
        backward = x == -1

        x_safe = x.copy()
        x_safe[forward] = 1 - 1e-15
        x_safe[backward] = -1 + 1e-15
        sin0 = sqrt(1 - x_safe ** 2)

        return sin0
