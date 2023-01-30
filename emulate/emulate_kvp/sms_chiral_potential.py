
"""
This file contains a class for the semilocal momentum space regularized chiral
two-nucleon potential up to N4LO+ given by Reinert et al. (arXiv:1711.08821). 
This was converted to python from E. Epelbaum's Fortran code used to produce 
results for the arXiv reference.
"""
from scipy.special import erfc
from numpy import (
    ndarray, array, asarray,
    pi, cos, sqrt, exp, log,
    append, zeros, hstack, add, 
    concatenate, reshape, zeros_like
)
from numpy.typing import ArrayLike
from typing import Union, Optional

from .constants import (
    Mn_GeV, axial_coupling, hbar_c_GeV, 
    pion_decay_constant, mass_neutral_pion, 
    mass_charged_pion, mass_pion
)        

def potential_info(cutoff, chiral_order):
    if (cutoff == 1):
        value = str(400)
    elif (cutoff == 2):
        value = str(450)
    elif (cutoff == 3):
        value = str(500)
    elif (cutoff == 4):
        value = str(550)
    else:
        raiseValueError("Incorrect value for cutoff!")

    if (chiral_order == 0):
        order = 'LO'
    elif (chiral_order == 1):
        order = 'N1LO'
    elif (chiral_order == 2):
        order = 'N2LO'
    elif (chiral_order == 3):
        order = 'N3LO'
    elif (chiral_order == 4):
        order = 'N4LO'
    elif (chiral_order == 5):
        order = 'N4LO+'
    else:
        raiseValueError("Incorrect value for chiral order!")
        
    return value, order


class SMSChiralPotential:    
    """
    Creates an environment for the SMS momentum space regularized chiral 
    two-nucleon potential up to N4LO+.
    
    The routine to generate the momentum-space partial-wave matrix elements is
    get_chiral_potential(nodes, weights, jmax). See below for more information.
    
    The potentials at all chiral orders are designed for the standard
    non-relativistic-like Lippmann-Schwinger equation with the relativistic
    relation between the cms energy and momentum. 
    
    The spectral and contact potential operators are outputted separately in order 
    to facilitate the calculation of the SMS chiral potential with different coupling 
    constants. Varying the LECs amounts to scaling the contact terms portion of the 
    potential since the spectral part is independent of the LECs. By separating out 
    the contributions, the potential only needs to be calculated once.
    
    More information can be found in the following papers:
    Reinert, et al., Eur. Phys. J. A 54, 86 (2018) (arXiv: 1711.08821)
    Epelbaum, et al., Nucl. Phys. A 747, 362 (2005) (arXiv: nucl-th/0405048)
    

    Input
    -----
    chiral_order : int
        Order of the EFT expansion.
        0 = LO, 1 = NLO, 2 = N2LO, 3 = N3LO, 4 = N4LO, 5 = N4LO+
        (N4LO+ takes into account the 3F3, 1F3, 3F3, 3F4 contact terms)
    interaction : char, len = 2
        Type of interaction. 
        Currently only takes in: "np" (neutron-proton)
    cutoff : int
        The momentum-space cutoff lambda.
        1 = 0.40 GeV, 2 = 0.45 GeV, 3 = 0.50 GeV, 4 = 0.55 GeV
    """
    
    def __init__(
        self, 
        chiral_order: int, 
        interaction: str, 
        cutoff: int
    ):
        n_iso = 4
        n_spin = 4
        n_legendre = 64
        n_grid = 200
        
        grid_cutoff = 15.0
        n_j, j_reg1, j_reg2, j_f = 64, 0.01, 0.1, 1.0
        n_mu, mu_reg1, mu_reg2, mu_f = 128, 0.2, 0.5, 2.5
        
        cutoff_array = array([0.4, 0.45, 0.5, 0.55])
        cutoff_value = cutoff_array[cutoff - 1]
        spline = zeros( (4, n_iso * n_spin, n_grid) )
        
        self.q_grid = self._initialize_grid(grid_cutoff, n_grid)
        self.leg_nodes, self.leg_weights = self._gauss_legendre_mesh(-1.0, 1.0, n_legendre)
        
        # Used for calculating the spectral terms
        # n_mu is the total number of mass points, mu_reg1 is the cutoff of the first region,
        # mu_reg2 is the cutoff of the second region, and mu_f up to where we are integrating
        mass_nodes, mass_weights = self._transformed_mesh(n_mu/2, n_mu/2, mu_reg1, mu_reg2, mu_f)
        mass_nodes = add(mass_nodes, 2.0 * mass_pion).tolist()
        
        # Used for calculating the J1 and J2 integrals that appear in the spectral terms
        # n_j is the total number of points, j_reg1 is the cutoff of the first region,
        # j_reg2 is the cutoff of the second region, and j_f up to where we are integrating
        self.j_nodes, self.j_weights = self._transformed_mesh(n_j/2, n_j/2, j_reg1, j_reg2, j_f)
        
        self.mass_nodes = mass_nodes
        self.mass_weights = mass_weights
        self.grid_cutoff = grid_cutoff
        self.n_iso = n_iso
        self.n_spin = n_spin
        self.n_grid = n_grid
        self.n_legendre = n_legendre
        self.chiral_order = chiral_order
        self.interaction = interaction
        self.cutoff = cutoff
        self.cutoff_value = cutoff_value
        self.spline = spline
        
        self.info()
    
    def get_chiral_potential(
        self, 
        nodes: ArrayLike, 
        jmax: int
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""      
        Calculates the spectral and contact terms matrix elements of 
        the sms chiral potential at their corresponding momenta.

        Parameters
        ----------
        nodes : ndarray
            Mesh points where the potential is being calculated.
        jmax : int
            The maximum momentum value that the potential is calculated at.
            All values of j before jmax are also calculated.
            Ex: jmax = 2 corresponds to calculating the potential for jmax = 0,
            jmax = 1, and jmax = 2.

        Returns
        -------
        spectral_operators : ndarray array, shape = (jmax + 1, 6, len(nodes), len(nodes))
            Spectral contribution to the potential.
        contact_operators : ndarray array, shape = (jmax + 1, 12, len(nodes), len(nodes))
            Contact contribution to the potential.
        """
        leg_nodes, n_legendre, q_grid = self.leg_nodes, self.n_legendre, self.q_grid
        n_grid, n_iso, n_spin = self.n_grid, self.n_iso, self.n_spin
        
        len_k = len(nodes)
        spectral_operators = zeros( (jmax + 1, 6, len_k, len_k) )
        contact_operators = zeros( (jmax + 1, 12, len_k, len_k) )
        
        nodes_GeV = hbar_c_GeV * nodes
        spectral_functions = self._spectral_function_terms()
        spectral_grid = self._long_range_interaction(spectral_functions)
        spectral_grid = spectral_grid.reshape((n_grid, n_iso * n_spin), order='F')
        self._create_spline(q_grid, spectral_grid, n_grid, n_iso * n_spin)
        
        for k in range(jmax + 1):
            print("jmom =", k)
            
            if (k >= n_legendre / 2 - 5):
                raise Exception("Increase the number of n_legendre")
            elif (k < 0):
                raise Exception("Wrong value of jmom")
                
            self.ll_val, self.ll2_val = float(k * (k + 1)), float(2 * k + 1)
            self.j_mom, self.jmom_p1 = float(k), float(k + 1)
            
            pl = [self._legendre_polynomials(i, k) for i in leg_nodes]
            plp = [self._legendre_polynomials(i, k + 1) for i in leg_nodes]
            plm = [self._legendre_polynomials(i, k - 1) for i in leg_nodes if (k >= 1)]
            
            for i, p in enumerate(nodes_GeV):
                for j, q in enumerate(nodes_GeV):
                    spectral_terms = 0.5 * pi * self._get_spectral_terms(p, q, k, pl, plp, plm)
                    contact_terms = 0.5 * pi * self._get_contact_terms(p, q, k)
                                        
                    for m, pot_spectral in enumerate(spectral_terms):
                        spectral_operators[k, m, i, j] = pot_spectral
                        
                    for n, pot_contact in enumerate(contact_terms):
                        contact_operators[k, n, i, j] = pot_contact
        
        spectral_operators[abs(spectral_operators) <= 1e-15] = 1e-15
        contact_operators[abs(contact_operators) <= 1e-15] = 1e-15
        
        return spectral_operators, contact_operators
    
    
    def get_half_on_shell(
        self, 
        nodes: ArrayLike, 
        jmax: int
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""      
        Calculates the spectral and contact terms matrix elements of 
        the sms chiral potential at their corresponding momenta.

        Parameters
        ----------
        nodes : ndarray
            Mesh points where the potential is being calculated.
        jmax : int
            The maximum momentum value that the potential is calculated at.
            All values of j before jmax are also calculated.
            Ex: jmax = 2 corresponds to calculating the potential for jmax = 0,
            jmax = 1, and jmax = 2.

        Returns
        -------
        spectral_operators : ndarray array, shape = (jmax + 1, 6, len(nodes), len(nodes))
            Spectral contribution to the potential.
        contact_operators : ndarray array, shape = (jmax + 1, 12, len(nodes), len(nodes))
            Contact contribution to the potential.
        """
        leg_nodes, n_legendre, q_grid = self.leg_nodes, self.n_legendre, self.q_grid
        n_grid, n_iso, n_spin = self.n_grid, self.n_iso, self.n_spin
        
        len_k = len(nodes)
        spectral_operators = zeros( (jmax + 1, 6, len_k, len_k) )
        contact_operators = zeros( (jmax + 1, 12, len_k, len_k) )
        
        nodes_GeV = hbar_c_GeV * nodes
        nodes_end = nodes_GeV[-1]
        spectral_functions = self._spectral_function_terms()
        spectral_grid = self._long_range_interaction(spectral_functions)
        spectral_grid = spectral_grid.reshape((n_grid, n_iso * n_spin), order='F')
        self._create_spline(q_grid, spectral_grid, n_grid, n_iso * n_spin)
        
        for k in range(jmax + 1):
                
            self.ll_val, self.ll2_val = float(k * (k + 1)), float(2 * k + 1)
            self.j_mom, self.jmom_p1 = float(k), float(k + 1)
            
            pl = [self._legendre_polynomials(i, k) for i in leg_nodes]
            plp = [self._legendre_polynomials(i, k + 1) for i in leg_nodes]
            plm = [self._legendre_polynomials(i, k - 1) for i in leg_nodes if (k >= 1)]
            
            for i, p in enumerate(nodes_GeV):
                spectral_terms = 0.5 * pi * self._get_spectral_terms(p, nodes_end, k, pl, plp, plm)
                contact_terms = 0.5 * pi * self._get_contact_terms(p, nodes_end, k)

                for m, pot_spectral in enumerate(spectral_terms):
                    spectral_operators[k, m, i, -1] = pot_spectral
                    
                for n, pot_contact in enumerate(contact_terms):
                    contact_operators[k, n, i, -1] = pot_contact
        
        spectral_operators[abs(spectral_operators) <= 1e-15] = 1e-15
        contact_operators[abs(contact_operators) <= 1e-15] = 1e-15
        
        return spectral_operators, contact_operators

        
    def _get_spectral_terms(
        self, 
        k: ArrayLike, 
        p: ArrayLike, 
        j: int, 
        pl: ArrayLike, 
        plp: ArrayLike, 
        plm: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the long-range interaction of the potential.

        Parameters
        ----------
        k : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        p : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        j : int
            The momentum value that the potential is calculated at.
        pl : ndarray
            Legendre polynomials.
        plp : ndarray
        plm : ndarray

        Returns
        -------
        potent : ndarray
            Matrix elements corresponding to the k and p input parameters for all 
            partial waves at the specified j value for the long-range interaction.
        """
        chiral_order, spline = self.chiral_order, self.spline
        n_grid, n_iso, n_spin = self.n_grid, self.n_iso, self.n_spin
        q_grid, leg_nodes, grid_cutoff = self.q_grid, self.leg_nodes, self.grid_cutoff
        
        V1T, V2T, V3T, V4T, W1T, W2T, W3T, W4T = ([] for _ in range(8))
        spectral, operators = zeros(6), zeros( (n_spin, n_iso) )
            
        kp, k2, p2 = 2.0 * k * p, k**2, p**2
        coeff = 0.5 * Mn_GeV * ( 1.0 / sqrt(Mn_GeV**2 + k2) + 1.0 / sqrt(Mn_GeV**2 + p2) )

        for n in leg_nodes:
            q = sqrt( k2 + p2 - kp * n )
            
            if (q <= q_grid[0]):
                q = q_grid[0]

            if (q >= grid_cutoff):
                operators = zeros( (n_spin, n_iso) )
            else:
                interpolated_spline = self._spline_interpolation(q, q_grid, spline)
                operators = interpolated_spline.reshape((n_spin, n_iso), order='F')

                if (chiral_order >= 3):
                    operators = coeff * operators

            V1T = append(arr=V1T, values=operators[0,0] + operators[0,1] - 0.25*operators[0,2])
            V2T = append(arr=V2T, values=operators[1,0] + operators[1,1] - 0.25*operators[1,2])
            V3T = append(arr=V3T, values=operators[2,0] + operators[2,1] - 0.25*operators[2,2])
            V4T = append(arr=V4T, values=operators[3,0] + operators[3,1] - 0.25*operators[3,2])
            W1T = append(arr=W1T, values=operators[0,0] - 3.0*operators[0,1] - 0.25*operators[0,2])
            W2T = append(arr=W2T, values=operators[1,0] - 3.0*operators[1,1] - 0.25*operators[1,2])
            W3T = append(arr=W3T, values=operators[2,0] - 3.0*operators[2,1] - 0.25*operators[2,2])
            W4T = append(arr=W4T, values=operators[3,0] - 3.0*operators[3,1] - 0.25*operators[3,2])

        VR1T, WR1T = self._partial_wave_decomp_C(pl, plp, plm, j, V1T, W1T)
        VR2T, WR2T = self._partial_wave_decomp_S(pl, plp, plm, j, V2T, W2T)
        VR3T, WR3T = self._partial_wave_decomp_T(k, p, pl, plp, plm, j, V3T, W3T)
        VR4T, WR4T = self._partial_wave_decomp_LS(k, p, pl, plp, plm, j, V4T, W4T)

        iso_t_potential = [v1 + v2 + v3 + v4 for v1, v2, v3, v4 in zip(VR1T, VR2T, VR3T, VR4T)]
        iso_s_potential = [w1 + w2 + w3 + w4 for w1, w2, w3, w4 in zip(WR1T, WR2T, WR3T, WR4T)]
        
        if (int( (0.5 * j) ) * 2 != j):
            spectral[0] = iso_s_potential[0]
            spectral[1] = iso_t_potential[1]
            spectral[2] = iso_s_potential[3]
            spectral[3] = iso_s_potential[5]
            spectral[4] = iso_s_potential[4]
            spectral[5] = iso_s_potential[2]
        else:
            spectral[0] = iso_t_potential[0]
            spectral[1] = iso_s_potential[1]
            spectral[2] = iso_t_potential[3]
            spectral[3] = iso_t_potential[5]
            spectral[4] = iso_t_potential[4]
            spectral[5] = iso_t_potential[2]
            
        return spectral
    
    def _get_contact_terms(
        self, 
        k: ArrayLike, 
        p: ArrayLike, 
        j: int
    ) -> ArrayLike:
        """
        Calculates the short-range interaction of the potential.

        Parameters
        ----------
        k : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        p : ndarray
            Second value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        j : int
            The momentum value that the potential is calculated at.

        Returns
        -------
        short_range : ndarray
            Outputs the matrix elements corresponding to the k and p input
            parameters for all partial waves at the specified j value for the
            short-range interaction.
        """
        chiral_order, cutoff_value = self.chiral_order, self.cutoff_value
        
        non_local_regulator = exp( - (k**2 + p**2) / cutoff_value**2 )
        
        if (chiral_order == 0):
            contacts_array = self._contacts_zeroth_order(k, p, j)
                                
        elif (chiral_order == 1) or (chiral_order == 2):
            contacts_array = self._contacts_first_second_order(k, p, j)
           
        elif (chiral_order == 3) or (chiral_order == 4):
            contacts_array = self._contacts_third_fourth_order(k, p, j)
                
        if (chiral_order == 5):
            contacts_array = self._contacts_fifth_order(k, p, j)
    
        short_range = 10000.0 / (2.0*pi)**3 * contacts_array * non_local_regulator
        return short_range
    
    def _contacts_zeroth_order(
        self, 
        k: float,
        p: float,
        j: int
    ) -> ArrayLike:
        """
        Calculates the matrix elements corresponding to the points k and p 
        for all partial waves at the specified j value at chiral order = 0. 

        Parameters
        ----------
        k : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        p : ndarray
            Second value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        j : int
            The momentum value that the potential is calculated at.

        Returns
        -------
        contacts : ndarray
            Contact matrix elements at a specified momentum.
        """
        p0, p1 = 0.0, 1.0
        
        if (j == 0):
            contacts = array([p1, p0, p0, p0, p0])
        elif (j == 1):
            contacts = array([p0, p0, p0, p0, p1, p0, p0, p0, p0, p0, p0, p0])
        else:
            contacts = array([p0, p0, p0, p0, p0, p0, p0])
        return contacts
    
    def _contacts_first_second_order(
        self, 
        k: float,
        p: float,
        j: int
    ) -> ArrayLike:
        """
        Calculates the matrix elements corresponding to the points k and p 
        for all partial waves at the specified j value at chiral order = 2 and 3. 

        Parameters
        ----------
        k : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        p : ndarray
            Second value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        j : int
            The momentum value that the potential is calculated at.

        Returns
        -------
        contacts : ndarray
            Contact matrix elements at a specified momentum.
        """
        p0, p1 = 0.0, 1.0
        kp, k2, p2 = k * p, k**2, p**2
        k2_p2 = k**2 + p**2
        
        if (j == 0):
            contacts = array([p1, k2_p2, p0, kp, p0])
        elif (j == 1):      
            contacts = array([kp, p0, kp, p0, p1, k2_p2, p0, p2, p0, k2, p0, p0])
        elif (j == 2):
            contacts = array([p0, p0, kp, p0, p0, p0, p0])
        else:
            contacts = array([p0, p0, p0])
        return contacts
    
    def _contacts_third_fourth_order(
        self, 
        k: float,
        p: float,
        j: int
    ) -> ArrayLike:
        """
        Calculates the matrix elements corresponding to the points k and p 
        for all partial waves at the specified j value at chiral order = 3 and 4. 

        Parameters
        ----------
        k : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        p : ndarray
            Second value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        j : int
            The momentum value that the potential is calculated at.

        Returns
        -------
        contacts : ndarray
            Contact matrix elements at a specified momentum.
        """
        p0, p1 = 0.0, 1.0
        kp, k2, p2 = k * p, k**2, p**2
        k2p2, k2_p2, kp_k2_p2 = (k * p)**2, k**2 + p**2, k * p * (k**2 + p**2)
        kp3, pk3 = k * p**3, p * k**3
        
        if (j == 0):
            contacts = array([p1, k2_p2, k2p2, kp, kp_k2_p2])
        elif (j == 1):
            contacts = array([kp, kp_k2_p2, kp, kp_k2_p2, p1,  
                              k2_p2, k2p2, p2, k2p2, k2, k2p2, k2p2])
        elif (j == 2):
            contacts = array([k2p2, k2p2, kp, kp_k2_p2, kp3, pk3, p0])
        elif (j == 3):
            contacts = array([p0, p0, k2p2])
        else:
            contacts = array([p0])
        return contacts
    
    def _contacts_fifth_order(
        self, 
        k: float,
        p: float,
        j: int
    ) -> ArrayLike:
        """
        Calculates the matrix elements corresponding to the points k and p 
        for all partial waves at the specified j value at chiral order = 5. 

        Parameters
        ----------
        k : ndarray
            First value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        p : ndarray
            Second value of momentum where the potential matrix element is
            being calculated. Ex: V(k,p)
        j : int
            The momentum value that the potential is calculated at.

        Returns
        -------
        contacts : ndarray
            Contact matrix elements at a specified momentum.
        """
        p0, p1 = 0.0, 1.0
        kp, k2, p2 = k * p, k**2, p**2
        k2p2, k2_p2, kp_k2_p2 = (k * p)**2, k**2 + p**2, k * p * (k**2 + p**2)
        kp3, pk3, k3p3 = k * p**3, p * k**3, (k * p)**3
        
        if (j == 0):
            contacts = array( [p1, k2_p2, k2p2, kp, kp_k2_p2] )
        elif (j == 1):
            contacts = array( [kp, kp_k2_p2, kp, kp_k2_p2, p1,  
                               k2_p2, k2p2, p2, k2p2, k2, k2p2, k2p2] )
        elif (j == 2):
            contacts = array( [k2p2, k2p2, kp, kp_k2_p2, kp3, pk3, k3p3] )
        elif (j == 3):
            contacts = array( [k3p3, k3p3, k2p2] )
        elif (j == 4):
            contacts = array( [k3p3] )
        else:
            contacts = array( [p0] )
        return contacts
    
    def _spline_interpolation(
        self, 
        x: float, 
        nodes: ArrayLike, 
        spline: ArrayLike
    ) -> ArrayLike:
        """
        Used for interpolating on the spline.        

        Parameters
        ----------
        x : float
            Value at which we are interpolating.
        nodes : ndarray
            Mesh usually generated from a quadrature such as Gaussian-Legendre.
        spline : ndarray, shape = (4, n_iso * n_spin, n_grid)
            Pre-defined spline array.

        Returns
        -------
        interpolated_spline : ndarray
            Returns the interpolated spline.
        """
        n_grid, n_iso, n_spin = self.n_grid, self.n_iso, self.n_spin
        
        n = n_grid
        m = n_iso * n_spin
        interpolated_spline = []
        
        if (x >= nodes[0]) and (x <= nodes[n - 1]):
            low, high = 0, n - 1
            
            while((high - low) > 1):
                mid = int(0.5 * (high + low))
                if (nodes[mid] > x):
                    high = mid
                else:
                    low = mid
                    
            dx = x - nodes[low]
            spl1, spl2, spl3, spl4 = spline[0, :, low], spline[1, :, low], \
                                     spline[2, :, low], spline[3, :, low]
            
            interpolated_spline = asarray([((a * dx + b) * dx + c) * dx + d \
                                           for a, b, c, d in zip(spl1, spl2, spl3, spl4)])
        elif (x < nodes[0]):
            dx = x - nodes[0]
            spl3, spl4 = spline[2, :, 0], spline[3, :, 0]
            
            interpolated_spline = asarray([a * dx + b for a, b in zip(spl3, spl4)])
        else:
            dx = x - nodes[n - 1]
            spl3, spl4 = spline[2, :, n - 1], spline[3, :, n - 1]
            interpolated_spline = asarray([a * dx + b for a, b in zip(spl3, spl4)])

        return interpolated_spline
    
    def _create_spline(
        self, 
        x: ArrayLike, 
        y: ArrayLike, 
        n: int, 
        m: int
    ) -> None:
        """
        Used for creating the spline used for interpolation.
        
        Parameters
        ----------
        x : ndarray
            x points for the spline.
        y : ndarray
            y points for the spline.
        n : float
            Grid size
        m : float

        Returns
        -------
        None
        """
        spline = self.spline
        
        u = zeros(n - 1)
        v = zeros(n - 1)
        b = zeros(n - 1)
        z = zeros(n - 1)

        v[0] = 2.0 * (x[2] - x[0])

        for i in range(1, n - 2):
            u[i] = (x[i + 1] - x[i]) / v[i-1]
            v[i] = 2.0 * (x[i + 2] - x[i]) - u[i] * (x[i + 1] - x[i])
            
        for j in range(m):
            for i in range(1, n - 1):
                b[i] = 3.0 * ((y[i + 1, j] - y[i, j]) / (x[i + 1] - x[i]) \
                            - (y[i, j] - y[i - 1, j]) / (x[i] - x[i - 1]))
                
            z[0] = b[1]
            for i in range(1, n - 2):
                z[i] = b[i + 1] - u[i] * z[i - 1]

            spline[0, j, n - 1] = 0.0
            spline[1, j, n - 1] = 0.0
            spline[3, j, n - 1] = y[n - 1, j]
        
            for i in range(n - 3, -1, -1):
                spline[1, j, i + 1] = (z[i] - (x[i + 2] - x[i + 1]) * spline[1, j, i + 2]) / v[i]
                spline[0, j, i + 1] = (spline[1, j, i + 2] - spline[1, j, i + 1]) \
                                            / (3.0 * (x[i + 2] - x[i + 1]))
                spline[2, j, i + 1] = (y[i + 2, j] - y[i + 1, j]) / (x[i + 2] - x[i + 1]) \
                                           - (spline[1, j, i + 2] \
                                           + 2.0 * spline[1, j, i + 1]) / 3.0 * (x[i + 2] - x[i + 1])
                spline[3, j, i + 1] = y[i + 1, j]

            spline[1, j, 0] = 0.0
            spline[0, j, 0] = (spline[1, j, 1] - spline[1, j, 0]) / (3.0 * (x[1] - x[0]))
            spline[2, j, 0] = (y[1, j] - y[0, j]) / (x[1] - x[0]) - (spline[1, j, 1] \
                                   + 2.0 * spline[1, j, 0]) / 3.0 * (x[1] - x[0])
            spline[3, j, 0] = y[0, j]
            spline[3, j, n - 1] = (3.0 * spline[0, j, n - 2] * (x[n - 1] - x[n - 2]) \
                                       + 2.0 * spline[1, j, n - 2]) * (x[n - 1] - x[n - 2]) \
                                       + spline[2, j, n - 2]
        return None
    
    def _long_range_interaction(
        self, 
        spectral_functions: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the integrals associated with the long-range interaction. 

        Parameters
        ----------
        spectral_functions : ndarray
            An array filled with the mass spectral functions eta and rho.

        Returns
        -------
        spectral_grid : ndarray
            The spectral piece of the potential used to create the spline.
        """
        chiral_order, cutoff_value = self.chiral_order, self.cutoff_value
        q_grid, mass_nodes, mass_weights = self.q_grid, self.mass_nodes, self.mass_weights
        n_grid, n_spin, n_iso = self.n_grid, self.n_spin, self.n_iso
        spectral_grid = zeros((n_grid, n_spin, n_iso))
        
        faktconvnorb = -2.0
        lambda_GeV = sqrt(2.0) * cutoff_value
        
        error_mass_nodes = erfc(mass_nodes / lambda_GeV)
        error_pion_mass = erfc(2.0 * mass_pion / lambda_GeV)
        error_neutral_pion_mass = erfc(mass_neutral_pion / cutoff_value)
        error_charged_pion_mass = erfc(mass_charged_pion / cutoff_value)
        
        V_OPE_coeff = - (axial_coupling / (2.0 * pion_decay_constant))**2\
        
        V4_C_coeff = 6.0 * axial_coupling**4 * mass_pion**5 / \
                     (512.0 * pi * Mn_GeV * pion_decay_constant**4)
        
        W4_C_coeff = 3.0 * axial_coupling**4 * mass_pion**5 / \
                     (128.0 * pi * Mn_GeV * pion_decay_constant**4)
        
        # Coefficient of the spin-spin term in my one-pion exchange potential operator
        # Specifically for the neutral pion
        cs0N = -V_OPE_coeff * (((cutoff_value * (cutoff_value**2 - 2.0 * mass_neutral_pion**2)) \
               / exp(mass_neutral_pion**2 / cutoff_value**2)) \
               + 2.0 * sqrt(pi) * error_neutral_pion_mass \
               * mass_neutral_pion**3) / (3.0 * cutoff_value**3)
        
        # Coefficient of the spin-spin term in my one-pion exchange potential operator
        # Specifically for the charged pion
        cs0C = -V_OPE_coeff * (((cutoff_value * (cutoff_value**2 - 2.0 * mass_charged_pion**2)) \
               / exp(mass_charged_pion**2 / cutoff_value**2)) \
               + 2.0 * sqrt(pi) * error_charged_pion_mass \
               * mass_charged_pion**3) / (3.0 * cutoff_value**3)
        
        for k, grid in enumerate(q_grid):
            if (chiral_order >= 1):
                for i, (ps, ws) in enumerate(zip(mass_nodes, mass_weights)):
                    regulator = exp( -(grid**2 + ps**2) / lambda_GeV**2)
                    exp_term = exp(ps**2 / lambda_GeV**2)
                    error = error_mass_nodes[i]
                    
                    c_functions = self._c_functions_nlo_n2lo(ps, lambda_GeV, exp_term, error)
                    coeff_array = self._spectral_integrals_nlo_n2lo(grid, ps, ws, c_functions)
                    
                    spectral_grid[k, 0, 1] += spectral_functions[0][i] \
                                               * coeff_array[0] * regulator
                    spectral_grid[k, 1, 0] += spectral_functions[1][i] \
                                               * coeff_array[1] * regulator
                    spectral_grid[k, 2, 0] += -spectral_functions[1][i] * coeff_array[2] * regulator
                    
                    if (chiral_order >= 2):
                        spectral_grid[k, 0, 0] += spectral_functions[3][i] \
                                                   * coeff_array[0] * regulator
                        spectral_grid[k, 1, 1] += spectral_functions[2][i] \
                                                   * coeff_array[1] * regulator
                        spectral_grid[k, 2, 1] += -spectral_functions[2][i] \
                                                  * coeff_array[2] * regulator
                        
                        if (chiral_order >= 3):
                            coeff_array = self._spectral_integrals_n3lo(grid, ps, ws, c_functions)
                            
                            spectral_grid[k, 0, 0] += spectral_functions[8][i] \
                                                       * coeff_array[0] * regulator
                            spectral_grid[k, 1, 0] += spectral_functions[10][i] \
                                                       * coeff_array[1] * regulator
                            spectral_grid[k, 2, 0] += -spectral_functions[10][i] \
                                                      * coeff_array[2] * regulator

                            spectral_grid[k, 0, 1] += spectral_functions[9][i] \
                                                       * coeff_array[3] * regulator
                            spectral_grid[k, 1, 1] += spectral_functions[11][i] \
                                                       * coeff_array[4] * regulator 
                            spectral_grid[k, 2, 1] += -spectral_functions[11][i] \
                                                      * coeff_array[5] * regulator 

                            spectral_grid[k, 3, 0] += -spectral_functions[12][i] \
                                                      * coeff_array[6] * regulator * faktconvnorb
                            spectral_grid[k, 3, 1] += -spectral_functions[13][i] \
                                                      * coeff_array[7] * regulator * faktconvnorb
                            
                            c_functions = self._c_functions_n3lo_n4lo(ps, lambda_GeV, exp_term, error)
                            coeff_array = self._spectral_integrals_n3lo_n4lo(grid, ps, ws, c_functions)
                            
                            spectral_grid[k, 0, 0] += -spectral_functions[6][i] \
                                                       * coeff_array[0] * regulator
                            spectral_grid[k, 1, 0] += -spectral_functions[7][i] \
                                                       * coeff_array[1] * regulator
                            spectral_grid[k, 2, 0] += spectral_functions[7][i] \
                                                       * coeff_array[2] * regulator
                            spectral_grid[k, 0, 1] += -spectral_functions[4][i] \
                                                       * coeff_array[3] * regulator
                            spectral_grid[k, 1, 1] += -spectral_functions[5][i] \
                                                       * coeff_array[4] * regulator
                            spectral_grid[k, 2, 1] += spectral_functions[5][i] \
                                                       * coeff_array[5] * regulator
                            spectral_grid[k, 0, 0] += -spectral_functions[14][i] \
                                                       * coeff_array[6] * regulator

                            spectral_grid[k, 0, 1] += -spectral_functions[15][i] \
                                                       * coeff_array[7] * regulator
                            spectral_grid[k, 1, 1] += -spectral_functions[16][i] \
                                                       * coeff_array[8] * regulator
                            spectral_grid[k, 2, 1] += spectral_functions[16][i] \
                                                       * coeff_array[9] * regulator
                            spectral_grid[k, 3, 0] += spectral_functions[17][i] \
                                                       * coeff_array[10] * regulator * faktconvnorb
                            spectral_grid[k, 3, 1] += spectral_functions[18][i] \
                                                       * coeff_array[11] * regulator * faktconvnorb
                    
                            if (chiral_order >= 4):
                                coeff_array = self._spectral_integrals_n4lo(grid, ps, ws, c_functions)
                                
                                spectral_grid[k, 0, 0] += -spectral_functions[19][i] \
                                                           * coeff_array[0] * regulator
                                spectral_grid[k, 1, 0] += -spectral_functions[21][i] \
                                                           * coeff_array[1] * regulator
                                spectral_grid[k, 2, 0] += spectral_functions[21][i] \
                                                           * coeff_array[2] * regulator
                                spectral_grid[k, 0, 1] += -spectral_functions[20][i] \
                                                           * coeff_array[3] * regulator
                                spectral_grid[k, 1, 1] += -spectral_functions[22][i] \
                                                           * coeff_array[4] * regulator
                                spectral_grid[k, 2, 1] += spectral_functions[22][i] \
                                                           * coeff_array[5] * regulator
                
            if (chiral_order >= 3):
                # Regularized expressions for the pole contribution to V4_C and W4_c
                c2c1 = (2.0 * V4_C_coeff * ((-2.0 * lambda_GeV * (lambda_GeV**2 \
                       + 2.0 * mass_pion**2)) / exp((4.0 * mass_pion**2) / lambda_GeV**2) \
                       + error_pion_mass * mass_pion * (5.0 * lambda_GeV**2 \
                       + 8.0 * mass_pion**2) * sqrt(pi))) / lambda_GeV**5
                
                c2c2 = (4.0 * V4_C_coeff * ((lambda_GeV * (lambda_GeV**2 \
                       + 4.0 * mass_pion**2)) / exp((4.0 * mass_pion**2) / lambda_GeV**2) \
                       - error_pion_mass * mass_pion * (3.0 * lambda_GeV**2 \
                       + 8.0 * mass_pion**2) * sqrt(pi))) / (3.0 * lambda_GeV**7)
                
                spectral_grid[k, 0, 0] += V4_C_coeff / (grid**2 + 4.0 * mass_pion**2) \
                                          * exp(-(grid**2 + 4.0 * mass_pion**2) / lambda_GeV**2) \
                                          + (c2c1 + c2c2 * grid**2) * exp(-grid**2 / lambda_GeV**2)
                
                c2c1 = (2.0 * W4_C_coeff * ((-2.0 * lambda_GeV * (lambda_GeV**2 + 2.0 * mass_pion**2)) \
                       / exp((4.0 * mass_pion**2) / lambda_GeV**2) + error_pion_mass * mass_pion \
                       * (5.0 * lambda_GeV**2 + 8.0 * mass_pion**2) * sqrt(pi))) / lambda_GeV**5
                
                c2c2 = (4.0 * W4_C_coeff * ((lambda_GeV * (lambda_GeV**2 \
                       + 4.0 * mass_pion**2)) / exp((4.0 * mass_pion**2) / lambda_GeV**2) \
                       - error_pion_mass * mass_pion * (3.0 * lambda_GeV**2 + 8.0 * mass_pion**2) \
                       * sqrt(pi))) / (3.0 * lambda_GeV**7)
                
                spectral_grid[k, 0, 1] += W4_C_coeff / (grid**2 + 4.0 * mass_pion**2) \
                                          * exp(-(grid**2 + 4.0 * mass_pion**2) / lambda_GeV**2) \
                                          + (c2c1 + c2c2 * grid**2) * exp(-grid**2 / lambda_GeV**2)
            
            charged_pion_propagator = exp(-(grid**2 + mass_charged_pion**2) / cutoff_value**2) \
                                      / (mass_charged_pion**2 + grid**2)
            neutral_pion_propagator = exp(-(grid**2 + mass_neutral_pion**2) / cutoff_value**2) \
                                      / (mass_neutral_pion**2 + grid**2)
            
            spectral_grid[k, 2, 1] += V_OPE_coeff * charged_pion_propagator
            spectral_grid[k, 2, 2] += 4.0 * V_OPE_coeff * \
                                      (neutral_pion_propagator - charged_pion_propagator)
            spectral_grid[k, 1, 1] += cs0C * exp(-grid**2 / cutoff_value**2)
            spectral_grid[k, 1, 2] += 4.0 * (cs0N - cs0C) * exp(-grid**2 / cutoff_value**2)
            
        return spectral_grid / (2.0 * pi)**3
    
    def _c_functions_nlo_n2lo(
        self, 
        mu: float, 
        lam: float, 
        exp: float, 
        error: float
    ) -> ArrayLike:
        """
        Calculates the C functions present in the spectral integrals at NLO and N2LO.
        These functions are determined by the required short-distance behavior of the
        coordinate space potential.

        Parameters
        ----------
        mu : float
        lam : float 
        exp : float 
        error : float

        Returns
        -------
        Array of the C functions needed for NLO and N2LO.
        """        
        c2c1 = (mu**2 * (lam**5 - 4.0 * lam**3 * mu**2 \
               - 2.0 * lam * mu**4 \
               + exp * error * mu**3 * (5.0 * lam**2 \
               + 2.0 * mu**2) * sqrt(pi))) / lam**5

        c2c2 = (-3.0 * lam**7 + 4.0 * lam**3 * mu**4 \
               + 4.0 * lam * mu**6 \
               - 2.0 * exp * error * mu**5 * (3.0 * lam**2 \
               + 2.0 * mu**2) * sqrt(pi)) / (3.0 * lam**7)

        c1t = -(15.0 * lam**7 - 6.0 * lam**5 * mu**2 \
               + 4.0 * lam**3 * mu**4 \
               - 8.0 * lam * mu**6 + 8.0 * exp * error * mu**7 \
               * sqrt(pi)) / (15.0 * lam**7)

        c2s1 = (2.0 * mu**2 * (lam**5 - 4.0 * lam**3 * mu**2 \
               - 2.0 * lam * mu**4 \
               + exp * error * mu**3 * (5.0 * lam**2 \
               + 2.0 * mu**2) * sqrt(pi))) / (3.0 * lam**5)

        c2s2 = (-15.0 *lam**7 + 2.0 * lam**5 * mu**2 \
               + 12.0 *lam**3 * mu**4 \
               + 16.0 *lam * mu**6 - 4.0 * exp * error * mu**5 \
               * (5.0 *lam**2 + 4.0 * mu**2) * sqrt(pi)) / (15.0 * lam**7)
        
        c2s1_2 = (-3.0 * lam**5 + 2.0 * lam**3 * mu**2 \
                 - 4.0 * lam * mu**4 + 4.0 * exp * error * mu**5 \
                 * sqrt(pi)) / (3.0 * lam**5)
        
        return array([c2c1, c2c2, c1t, c2s1, c2s2, c2s1_2])
    
    def _c_functions_n3lo_n4lo(
        self, 
        mu: float, 
        lam: float, 
        exp: float, 
        error: float
    ) -> ArrayLike:
        """
        Calculates the C functions present in the spectral integrals at N3LO and N4LO.
        These functions are determined by the required short-distance behavior of the
        coordinate space potential.

        Parameters
        ----------
        mu : float
        lam : float 
        exp : float 
        error : float

        Returns
        -------
        Array of the C functions needed for N3LO and N4LO.
        """
        c3c1 = -(mu**4 * (4.0 * lam**7 - 24.0 * lam**5 * mu**2 \
              - 26.0 * lam**3 * mu**4 - 4.0 * lam * mu**6 \
              + exp * error * mu**3 * (35.0 * lam**4 \
              + 28.0 * lam**2 * mu**2 \
              + 4.0 * mu**4) * sqrt(pi))) / (4.0 * lam**7)
        c3c2 = (mu**2 * (3.0 * lam**9 - 12.0 * lam**5 * mu**4 \
              - 22.0 * lam**3 * mu**6 - 4.0 * lam * mu**8 \
              + exp * error * mu**5 * (21.0 * lam**4 + 24.0 * lam**2 * mu**2 \
              + 4.0 * mu**4) * sqrt(pi))) / (3.0 * lam**9)
        c3c3 = (-15.0 * lam**11 + 8.0 * lam**5 * mu**6 \
              + 18.0 * lam**3 * mu**8 + 4.0 * lam*mu**10 - \
              exp * error * mu**7 * (15.0 * lam**4 + 20.0 * lam**2 * mu**2 \
              + 4.0 * mu**4) * sqrt(pi)) / (15.0 * lam**11)
        c2t1 = (mu**2 * (15.0 * lam**9 - 12.0 * lam**7 * mu**2 \
              + 12.0 * lam**5 * mu**4 - 32.0 * lam**3 * mu**6 \
              - 8.0 * lam * mu**8 + 4.0 * exp * error * mu**7 * (9.0 * lam**2 \
              + 2.0 * mu**2) * sqrt(pi))) / (15.0 * lam**9)
        c2t2 = (-105.0 * lam**11 + 12.0 * lam**7 * mu**4 \
              - 16.0 * lam**5 * mu**6 + 48.0 * lam**3 * mu**8 \
              + 16.0 * lam * mu**10 - 8.0 * exp * error * mu**9 * (7.0 * lam**2 \
              + 2.0 * mu**2) * sqrt(pi)) / (105.0 * lam**11)
        c3s1 = -(mu**4 * (4.0 * lam**7 - 24.0 * lam**5 * mu**2 \
              - 26.0 * lam**3 * mu**4 - 4.0 * lam * mu**6 \
              + exp * error * mu**3 * (35.0 * lam**4 + 28.0 * lam**2*mu**2 \
              + 4.0 * mu**4) *sqrt(pi))) / (6.0 * lam**7)
        c3s2 = (mu**2 * (15.0 * lam**9 - 4.0 * lam**7 * mu**2 \
              - 36.0 * lam**5 * mu**4 - 84.0 * lam**3 * mu**6 - \
              16.0 * lam * mu**8 + 2.0 * exp * error * mu**5 * (35.0 * lam**4 \
              + 46.0 * lam**2 * mu**2 + 8.0 * mu**4) * sqrt(pi))) \
              / (15.0 * lam**9)
        c3s3 = (-105.0 * lam**11 + 4.0 * lam**7 * mu**4 \
              + 32.0 * lam**5 * mu**6 + 100.0 * lam**3 * mu**8 \
              + 24.0 * lam * mu**10 - 2.0 * exp * error * mu**7 * (35.0 * lam**4 \
              + 56.0 * lam**2 * mu**2 + 12.0 * mu**4) * sqrt(pi)) \
              / (105.0 * lam**11)
        c2s1 = (3.0 * lam**7 * mu**2 - 4.0 * lam**5 * mu**4 \
               + 12.0 * lam**3 * mu**6 + 4.0 * lam * mu**8 \
               - 2.0 * exp * error * mu**7 * (7.0 * lam**2 + 2.0 * mu**2) \
               * sqrt(pi)) / (3.0 * lam**7)
        c2s2 = (-15.0 * lam**9 + 4.0 * lam**5 * mu**4 \
               - 16.0 * lam**3 * mu**6 - 8.0 * lam * mu**8 \
               + 4.0 * exp * error * mu**7 * (5.0 * lam**2 + 2.0 * mu**2) \
               * sqrt(pi)) / (15.0 * lam**9)
        
        return array([c3c1, c3c2, c3c3, c2t1, c2t2, c3s1, c3s2, c3s3, c2s1, c2s2])
    
    def _spectral_integrals_nlo_n2lo(
        self,  
        q: float,
        ps: float, 
        ws: float,  
        c_array: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the coefficients for the spectral integrals at NLO and N2LO.

        Parameters
        ----------
        q : float
        ps : float
            Mesh
        ws : float
            Weights
        c_array : ndarray

        Returns
        -------
        coeff_array : ndarray
            An array containing the coefficients for the spectral integrals.
        """
        c2c1 = c_array[0]
        c2c2 = c_array[1]
        c1t = c_array[2]
        c2s1 = c_array[3]
        c2s2 = c_array[4]
        
        fraction = q**4 / (ps**2 + q**2)
        factor = (2.0 / pi) * ws / ps**3
        
        coeff_1 = factor * (fraction + c2c1 + c2c2 * q**2)
        coeff_2 = factor * (fraction + c2s1 + c2s2 * q**2)
        coeff_3 = factor * (fraction / q**2 + c1t)
        
        coeff_array = array([coeff_1, coeff_2, coeff_3])
        
        return coeff_array
    
    def _spectral_integrals_n3lo(
        self, 
        q: float,
        ps: float, 
        ws: float,  
        c_array: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the coefficients for the spectral integrals at N3LO.

        Parameters
        ----------
        q : float
        ps : float
            Mesh
        ws : float
            Weights
        c_array : ndarray

        Returns
        -------
        coeff_array : ndarray
            An array containing the coefficients for the spectral integrals.
        """
        c3c1 = c_array[0]
        c3c2 = c_array[1]
        c2t1 = c_array[2]
        c3s1 = c_array[3]
        c3s2 = c_array[4]
        c2s1 = c_array[5]
        
        fraction = q**4 / (ps**2 + q**2)
        factor = (2.0 / pi) * ws / ps**3
        
        coeff_1 = factor * (fraction + c3c1 + c3c2 * q**2)
        coeff_2 = factor * (fraction + c3s1 + c3s2 * q**2)
        coeff_3 = factor * (fraction / q**2 + c2t1)
        
        coeff_4 = factor * (fraction + c3c1 + c3c2 * q**2)
        coeff_5 = factor * (fraction + c3s1 + c3s2 * q**2)
        coeff_6 = factor * (fraction / q**2  + c2t1)
        
        coeff_7 = factor * ps**2 * (fraction / q**2 + c2s1)
        coeff_8 = factor * ps**2 * (fraction / q**2 + c2s1)
        
        coeff_array = array([coeff_1, coeff_2, coeff_3, coeff_4, 
                             coeff_5, coeff_6, coeff_7, coeff_8])
        
        return coeff_array
    
    def _spectral_integrals_n3lo_n4lo(
        self, 
        q: float,
        ps: float, 
        ws: float,  
        c_array: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the coefficients for the spectral integrals at N3LO and N4LO.

        Parameters
        ----------
        q : float
        ps : float
            Mesh
        ws : float
            Weights
        c_array : ndarray

        Returns
        -------
        coeff_array : ndarray
            An array containing the coefficients for the spectral integrals.
        """
        c3c1 = c_array[0]
        c3c2 = c_array[1]
        c3c3 = c_array[2]
        c2t1 = c_array[3]
        c2t2 = c_array[4]
        c3s1 = c_array[5]
        c3s2 = c_array[6]
        c3s3 = c_array[7]
        c2s1 = c_array[8]
        c2s2 = c_array[9]
        
        fraction = q**6 / (ps**2 + q**2)
        factor = (2.0 / pi) * ws / ps**5

        coeff_1 = factor * (fraction + c3c1 + c3c2 * q**2 + c3c3 * q**4)
        coeff_2 = factor * (fraction + c3s1 + c3s2 * q**2 + c3s3 * q**4)
        coeff_3 = factor * (fraction / q**2 + c2t1 + c2t2 * q**2)
        coeff_4 = factor * (fraction + c3c1 + c3c2 * q**2 + c3c3 * q**4)
        coeff_5 = factor * (fraction + c3s1 + c3s2 * q**2 + c3s3 * q**4)
        coeff_6 = factor * (fraction / q**2 + c2t1 + c2t2 * q**2)
        coeff_7 = factor * (fraction + c3c1 + c3c2 * q**2 + c3c3 * q**4)

        coeff_8 = factor * (fraction + c3c1 + c3c2 * q**2 + c3c3 * q**4)
        coeff_9 = factor * (fraction + c3s1 + c3s2 * q**2 + c3s3 * q**4)
        coeff_10 = factor * (fraction / q**2 + c2t1 + c2t2 * q**2)
        coeff_11 = factor * ps**2 * (fraction / q**2 + c2s1 + c2s2 * q**2)
        coeff_12 = factor * ps**2 * (fraction / q**2 + c2s1 + c2s2 * q**2)
        
        coeff_array = array([coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6, 
                             coeff_7, coeff_8, coeff_9, coeff_10, coeff_11, coeff_12])
        
        return coeff_array
    
    def _spectral_integrals_n4lo(
        self, 
        q: float,
        ps: float, 
        ws: float,  
        c_array: ArrayLike
    ) -> ArrayLike:
        """
        Calculates the coefficients for the spectral integrals at N4LO.

        Parameters
        ----------
        q : float
        ps : float
            Mesh
        ws : float
            Weights
        c_array : ndarray

        Returns
        -------
        coeff_array : ndarray
            An array containing the coefficients for the spectral integrals.
        """
        c3c1 = c_array[0]
        c3c2 = c_array[1]
        c3c3 = c_array[2]
        c2t1 = c_array[3]
        c2t2 = c_array[4]
        c3s1 = c_array[5]
        c3s2 = c_array[6]
        c3s3 = c_array[7]
        
        fraction = q**6 / (ps**2 + q**2)
        factor = (2.0 / pi) * ws / ps**5
        
        coeff_1 = factor * (fraction + c3c1 + c3c2 * q**2 + c3c3 * q**4)
        coeff_2 = factor * (fraction + c3s1 + c3s2 * q**2 + c3s3 * q**4)
        coeff_3 = factor * (fraction / q**2 + c2t1 + c2t2 * q**2)
        coeff_4 = factor * (fraction + c3c1 + c3c2 * q**2 + c3c3 * q**4)
        coeff_5 = factor * (fraction + c3s1 + c3s2 * q**2 + c3s3 * q**4)
        coeff_6 = factor * (fraction / q**2 + c2t1 + c2t2 * q**2)
        
        coeff_array = array([coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6])
        
        return coeff_array
    
    def _spectral_function_terms(self) -> ArrayLike:
        """
        Calculates the eta and rho integrals for the long-range part of the potential.

        Parameters
        ----------
        None

        Returns
        -------
        eta_rho : ndarray
            The eta and rho terms in the spectral integrals.
        """
        chiral_order = self.chiral_order
        mass_nodes = self.mass_nodes
            
        c1_N2LO, c3_N2LO, c4_N2LO = -0.74, -3.61, 2.44
        c1_N3LO, c2_N3LO, c3_N3LO, c4_N3LO = -1.07, 3.20, -5.32, 3.56
        d1pd2_N3LO, d3_N3LO, d5_N3LO, d14md15_N3LO = 1.04, -0.48, 0.14, -1.90
        c1_N4LO, c2_N4LO, c3_N4LO, c4_N4LO = -1.10, 3.57, -5.54, 4.17
        d1pd2_N4LO, d3_N4LO, d5_N4LO, d14md15_N4LO = 6.18, -8.91, 0.86, -12.18
        e14_N4LO, e17_N4LO, e19_N4LO, e21_N4LO, e22_N4LO = 1.18, -0.18, 0.0, 0.0, 0.0
        
        rho_sq2, rho_cq3, rho_cq4, rho_sq4, rho_cq4_rel, \
        rho_sq4_rel, eta_cq2, eta_sq3, eta_cq4, eta_sq4, \
        eta_cq4_rel, eta_sq4_rel, rho_slq4_rel, rho_cq5_rel, \
        rho_slq5_rel, rho_cq5, rho_sq5, eta_slq4_rel, eta_cq5_rel, \
        eta_sq5_rel, eta_slq5_rel, eta_cq5, eta_sq5 = ([] for _ in range(23))
        
        if (chiral_order == 2):
            c1, c3, c4 = c1_N2LO, c3_N2LO, c4_N2LO
            
            c_terms = array( [c1, c3, c4] )
            
        elif (chiral_order == 3):
            c1, c2, c3, c4 = c1_N3LO, c2_N3LO, c3_N3LO, c4_N3LO
            d1pd2, d3, d5, d14md15 = d1pd2_N3LO, d3_N3LO, d5_N3LO, d14md15_N3LO
            
            c_terms = array( [c1, c2, c3, c4] )
            d_terms = array( [d1pd2, d3, d5, d14md15] )
            
        elif (chiral_order >= 4):
            c1, c2, c3, c4 = c1_N4LO, c2_N4LO, c3_N4LO, c4_N4LO
            d1pd2, d3, d5, d14md15 = d1pd2_N4LO, d3_N4LO, d5_N4LO, d14md15_N4LO
            e14, e17, e19, e21, e22 = e14_N4LO, e17_N4LO, e19_N4LO, e21_N4LO, e22_N4LO
            
            c_terms = array( [c1, c2, c3, c4] )
            d_terms = array( [d1pd2, d3, d5, d14md15] )
            e_terms = array( [e14, e17, e19, e21, e22] )
        
        for i, mu in enumerate(mass_nodes):
            if (chiral_order >= 1):
                one_loop = - pi / (2.0 * mu) * sqrt(mu**2 - 4.0 * mass_pion**2)
                eta, rho = self._eta_rho_terms_first_order(mu, one_loop)
                eta_cq2 = append(arr=eta_cq2, values=eta)
                rho_sq2 = append(arr=rho_sq2, values=rho)
                
                if (chiral_order >= 2):
                    eta, rho = self._eta_rho_terms_second_order(mu, c_terms)
                    eta_sq3 = append(arr=eta_sq3, values=eta)
                    rho_cq3 = append(arr=rho_cq3, values=rho)
                    
                    if (chiral_order >= 3):
                        mass_diff = 4.0*mass_pion**2 - mu**2
                        r_sq = 0.5*sqrt(mu**2 - 4.0*mass_pion**2)
                        mass_sq = mu**2 - 2.0*mass_pion**2
                        
                        eta, rho = self._eta_rho_terms_third_order(mu, one_loop, mass_diff, 
                                                                   r_sq, mass_sq, c_terms, d_terms)
                        
                        eta_sq4 = append(arr=eta_sq4, values=eta[0])
                        eta_cq4 = append(arr=eta_cq4, values=eta[1])
                        
                        rho_cq4 = append(arr=rho_cq4, values=rho[0])
                        rho_sq4 = append(arr=rho_sq4, values=rho[1])
                        
                        eta_rel, rho_rel = self._eta_rho_terms_third_order_rel(mu, one_loop, 
                                                                               mass_diff, c_terms)
                        
                        eta_cq4_rel = append(arr=eta_cq4_rel, values=eta_rel[0])
                        eta_sq4_rel = append(arr=eta_sq4_rel, values=eta_rel[1])
                        eta_slq4_rel = append(arr=eta_slq4_rel, values=eta_rel[2])
                        eta_cq5_rel = append(arr=eta_cq5_rel, values=eta_rel[3])
                        eta_sq5_rel = append(arr=eta_sq5_rel, values=eta_rel[4])
                        eta_slq5_rel = append(arr=eta_slq5_rel, values=eta_rel[5])
                        
                        rho_cq4_rel = append(arr=rho_cq4_rel, values=rho_rel[0])
                        rho_sq4_rel = append(arr=rho_sq4_rel, values=rho_rel[1])
                        rho_slq4_rel = append(arr=rho_slq4_rel, values=rho_rel[2])
                        rho_cq5_rel = append(arr=rho_cq5_rel, values=rho_rel[3])
                        rho_slq5_rel = append(arr=rho_slq5_rel, values=rho_rel[4])
                        
                        rho_cq4_rel[i] += 3.0 * axial_coupling**4 / (512.0 * pi * Mn_GeV \
                                          * pion_decay_constant**4) \
                                          * (2.0 * mass_pion**2 - mu**2)**2 * pi / (4.0 * mu)
                        eta_cq4_rel[i] += -axial_coupling**4 / (256.0 * pi * Mn_GeV \
                                          * pion_decay_constant**4) \
                                          * (2.0*mass_pion**2 - mu**2)**2 * pi / (4.0 * mu)
                        
                        rho_sq4_rel[i] += -(3.0 * axial_coupling**4 / (1024.0 * pi * Mn_GeV \
                                          * pion_decay_constant**4) \
                                          * (4.0 * mass_pion**2 - mu**2) \
                                          * pi / (4.0 * mu)) * mu**2
                        eta_sq4_rel[i] += (axial_coupling**4 / (512.0 * pi * Mn_GeV \
                                          * pion_decay_constant**4) \
                                          * (4.0 * mass_pion**2 - mu**2) \
                                          * pi / (4.0 * mu)) * mu**2
            
                        if (chiral_order >= 4):
                            eta, rho = self._eta_rho_terms_fourth_order(mu, r_sq, 
                                                                        c_terms, e_terms)
                        
                            eta_cq5 = append(arr=eta_cq5, values=eta[0])
                            eta_sq5 = append(arr=eta_sq5, values=eta[1])
                            rho_cq5 = append(arr=rho_cq5, values=rho[0])
                            rho_sq5 = append(arr=rho_sq5, values=rho[1])
                            
                            rho_cq5[i] += 2.0 * ((48.0 * axial_coupling**2 * mass_pion \
                                          * (mass_pion**2 - 2.0 * mu**2) \
                                          * r_sq**2 * (6.0 * (2.0 * c1 - c3) * mass_pion**2 \
                                          + 3.0 * c3 * mu**2 + 2.0 * c2 * r_sq**2) \
                                          + axial_coupling**4 * (1536.0 * (2.0 * c1 - c3) \
                                          * mass_pion**7 - 16.0 * (84.0 * c1 + 5.0 * c2 - 42.0 * c3) \
                                          * mass_pion**6 * mu - 192.0 * (6.0 * c1 - 7.0 * c3) \
                                          * mass_pion**5 * mu**2 + 8.0 * (84.0 * c1 \
                                          + 5.0 * c2 - 84.0 * c3) * mass_pion**4 * mu**3 \
                                          - 288.0 * c3 * mass_pion**3 * mu**4 + (-36.0 * c1 \
                                          - 5.0 * c2 + 186.0 * c3) * mass_pion**2 * mu**5 \
                                          - 9.0 * c3 * mu**7 - 8.0 * (64.0 * c2 * mass_pion**5 \
                                          + (-12.0 * c1 - 5.0 * c2 + 6.0 * c3) * mass_pion**4 \
                                          * mu - 40.0 * c2 * mass_pion**3 * mu**2 + (4.0 * c2 \
                                          - 3.0 * c3) * mass_pion**2 * mu**3 + c2 * mu**5) * r_sq**2) \
                                          + 12.0 * mu * r_sq**2 * (3.0 * (4.0 * c1 - c2 \
                                          - 2.0 * c3) * mass_pion**4 + 3.0 * c3 * mass_pion**2 * mu**2 \
                                          + 2.0 * ((12.0 * c1 + c2 - 6.0 * c3) * mass_pion**2 \
                                          + 3.0 * c3 * mu**2) * r_sq**2 + 8.0 * c2 * r_sq**4)) \
                                          / (24576.0 * pion_decay_constant**6 * mu * pi**2 * r_sq) \
                                          + (axial_coupling**2 * (2.0 * mass_pion**4 \
                                          - 5.0 * mass_pion**2 * mu**2 + 2.0 * mu**4) * r_sq \
                                          * (6.0 * (2.0 * c1 - c3) * mass_pion**2 + 3.0 * c3 * mu**2 \
                                          + 2.0 * c2  *r_sq**2) * log(1.0 + (4.0 * mass_pion) \
                                          / (-2.0 * mass_pion + mu))) / \
                                          (2048.0 * pion_decay_constant**6 * mu**2 * pi**2) \
                                          + (mass_pion**2 * (3.0 * (-4.0 * c1 + c2 + 2.0 * c3) \
                                          * mass_pion**4 - 3.0 * c3 * mass_pion**2 * mu**2 \
                                          + 2.0 * axial_coupling**4 * ((28.0 * c1 + 9.0 * c2 \
                                          - 14.0 * c3) * mass_pion**4 - 4.0 * c3 * mu**4 \
                                          + 4.0 * (-8.0 * c1 + c2 + 4.0 * c3) * mass_pion**2 \
                                          * r_sq**2 - mu**2 * ((16.0 * c1 + 4.0 * c2 \
                                          - 15.0 * c3) * mass_pion**2 + 8.0 * c3 * r_sq**2))) \
                                          * log((mu / 2.0 + r_sq) / (mu / 2.0 - r_sq))) / \
                                          (2048.0 * pion_decay_constant**6 * mu * pi**2))
                            

                            eta_sq5[i] += 2.0 * ((c4 * axial_coupling**2 * (mu \
                                          * (2.0 * axial_coupling**2 * mass_pion**2 \
                                          * (-2.0 * mass_pion + mu)**2 * (2.0 * mass_pion + mu) \
                                          - (4.0 * (-3.0 + 4.0 * axial_coupling**2) \
                                          * mass_pion**3 - 9.0 * axial_coupling**2 \
                                          * mass_pion**2 * mu + 3.0 * mass_pion * mu**2 \
                                          + axial_coupling**2 * mu**3) * r_sq**2 + 2.0 \
                                          * (2.0 * mass_pion + axial_coupling**2 * mu) * r_sq**4) \
                                          + r_sq**4 * (3.0 * (-4.0 * mass_pion**2 \
                                          + mu**2) - 4.0 * r_sq**2) \
                                          * log(1.0 + (4.0 * mass_pion) / (-2.0 * mass_pion \
                                          + mu)) + 3.0 * axial_coupling**2 * mass_pion**2 * mu \
                                          * (5.0 * mass_pion**2 - mu**2) * r_sq \
                                          * log(-((mu + 2.0 * r_sq) / (-mu + 2.0 * r_sq))))) \
                                          / (6144.0 * pion_decay_constant**6 * pi**2 * r_sq))
        if (chiral_order == 0):
            eta_rho = array([0.0])
            
        elif (chiral_order == 1):
            eta_rho = concatenate(([eta_cq2], [rho_sq2]))
            
        elif (chiral_order == 2):
            eta_rho = concatenate(([eta_cq2], [rho_sq2], [eta_sq3], [rho_cq3]))
            
        elif (chiral_order == 3):
            eta_rho = concatenate(([eta_cq2], [rho_sq2], [eta_sq3], [rho_cq3], [eta_cq4], \
                                   [eta_sq4], [rho_cq4], [rhosq4], [rhocq4rel], [etacq4rel], \
                                   [rho_sq4_rel], [eta_sq4_rel], [rho_slq4_rel], [eta_slq4_rel], \
                                   [rho_cq5_rel], [eta_cq5_rel], [eta_sq5_rel], [rho_slq5_rel], \
                                   [eta_slq5_rel]))
        
        elif (chiral_order >= 4):
            eta_rho = concatenate(([eta_cq2], [rho_sq2], [eta_sq3], [rho_cq3], [eta_cq4], \
                                   [eta_sq4], [rho_cq4], [rho_sq4], [rho_cq4_rel], [eta_cq4_rel], \
                                   [rho_sq4_rel], [eta_sq4_rel], [rho_slq4_rel], [eta_slq4_rel], \
                                   [rho_cq5_rel], [eta_cq5_rel], [eta_sq5_rel], [rho_slq5_rel], \
                                   [eta_slq5_rel], [rho_cq5], [eta_cq5], [rho_sq5], [eta_sq5]))
            
        else:
            raise Exception("Wrong value of chiral_order!")
            
        return eta_rho
    
    def _eta_rho_terms_first_order(
        self, 
        mass: float,
        one_loop: float
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the coefficients for the two-pion exchange potential terms at NLO.
        
        Parameters
        ----------
        mass : float
            Mass term where rho and eta are being evaluated at.
        one_loop : float
            One loop contribution.
        
        Returns
        -------
        eta, rho : float
            Values for eta and rho at NLO.
        """
        eta = - one_loop / (384.0 * pi**2 * pion_decay_constant**4) * \
                (4.0 * mass_pion**2 * (5.0 * axial_coupling**4 \
                - 4.0 * axial_coupling**2 - 1.0) - mass**2 * \
                (23.0 * axial_coupling**4 - 10.0 * axial_coupling**2 - 1.0) \
                + 48.0 * axial_coupling**4 * mass_pion**4\
                / (4.0 * mass_pion**2 - mass**2))

        rho = -3.0 * mass**2 * axial_coupling**4 \
              / (64.0 * pi**2 * pion_decay_constant**4) * one_loop
        return eta, rho
    
    def _eta_rho_terms_second_order(
        self,
        mass: float,
        c_coeff: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the coefficients for the two-pion exchange potential terms at N2LO.
        
        Parameters
        ----------
        mass : float
            Mass term where rho and eta are being evaluated at.
        c_coeff : ndarray
            Array of coefficients. Values depend on the EFT order.
        
        Returns
        -------
        eta, rho : float
            Values for eta and rho at N2LO.
        """
        c1_N2LO = c_coeff[0]
        c3_N2LO = c_coeff[2]
        c4_N2LO = c_coeff[3]
        
        eta = -mass**2 * axial_coupling**2 / (32.0 * pi * pion_decay_constant**4) \
               * c4_N2LO * (4.0 * mass_pion**2 - mass**2) * pi / (4.0 * mass)
        
        rho = -3.0 * axial_coupling**2 / (16.0 * pi * pion_decay_constant**4) \
               * (2.0 * mass_pion**2 * (2.0 * c1_N2LO - c3_N2LO) + mass**2 * c3_N2LO) \
               * (2.0 * mass_pion**2 - mass**2) * pi / (4.0 * mass)
                    
        return eta, rho
    
    def _eta_rho_terms_third_order(
        self,
        mass: float,
        one_loop: float,
        mass_diff: float,
        r_sq: float,
        mass_sq: float,
        c_coeff: ArrayLike,
        d_coeff: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the coefficients for the two-pion exchange potential terms at N3LO.
        
        Parameters
        ----------
        mass : float
            Mass term where rho and eta are being evaluated at.
        one_loop : float
            One loop contribution.
        mass_diff : float
            mass_diff = 4.0*mass_pion**2 - mass**2
        r_sq : float
            r_sq = 0.5*sqrt(mass**2 - 4.0*mass_pion**2)
        mass_sq : float
            mass_sq = mass**2 - 2.0*mass_pion**2
        c_coeff : ndarray
            Array of coefficients. Values depend on the EFT order.
        d_coeff : ndarray
            Array of coefficients. Values depend on the EFT order.
            
        Returns
        -------
        eta, rho : float
            Values for eta and rho at N3LO.
        """
        c1_N3LO = c_coeff[0]
        c2_N3LO = c_coeff[1]
        c3_N3LO = c_coeff[2]
        c4_N3LO = c_coeff[3]
        
        d1pd2_N3LO = d_coeff[0]
        d3_N3LO = d_coeff[1]
        d5_N3LO = d_coeff[2]
        d14md15_N3LO = d_coeff[3]
        
        j1, j2 = self._calculate_j_integrals(mass)
        
        rho_cq4 = 3.0 / (16.0 * pi**2 * pion_decay_constant**4) * one_loop \
                  * ((c2_N3LO / 6.0 * mass_diff + c3_N3LO * (2 * mass_pion**2 - mass**2) \
                  - 4.0 * c1_N3LO * mass_pion**2)**2 + c2_N3LO**2 / 45.0 * mass_diff**2) \
                  - 3.0 * axial_coupling**4 * (mass**2 - 2.0 * mass_pion**2) \
                  / (pi * mass * (4.0 * pion_decay_constant)**6) \
                  * ((mass_pion**2 - 2.0 * mass**2) * (2.0 * mass_pion \
                  + (2.0 * mass_pion**2 - mass**2) / (2.0 * mass) \
                  * log((mass + 2.0 * mass_pion) / (mass - 2.0 * mass_pion))) \
                  + 4.0 * axial_coupling**2 * mass_pion * (2.0 * mass_pion**2 - mass**2))
        
        eta_sq4 = (c4_N3LO**2 / (96.0 * pi**2 * pion_decay_constant**4) * mass_diff * one_loop \
                   - axial_coupling**4 * (mass**2 - 4.0 * mass_pion**2) \
                   / (mass**2 * pi * (4.0 * pion_decay_constant)**6) * ((mass_pion**2 \
                   - mass**2 / 4) * log((mass + 2.0 * mass_pion) / (mass - 2.0 * mass_pion)) \
                   + (1.0 + 2.0 * axial_coupling**2) * mass * mass_pion)) * mass**2
        
        rho_sq4 = -(axial_coupling**2 * r_sq**3 * mass / (8.0 * pion_decay_constant**4 * pi) \
                 * d14md15_N3LO - 2.0 * axial_coupling**6 * mass * r_sq**3 \
                 / (8.0 * pi * pion_decay_constant**2)**3 * (1.0 / 9.0 - j1 + j2))
        
        eta_cq4 = r_sq * mass_sq / (24.0 * pion_decay_constant**4 * mass * pi) \
                  * (2.0 * (axial_coupling**2 - 1.0) * r_sq**2 - 3.0 * axial_coupling**2 * mass_sq) \
                  * d1pd2_N3LO + r_sq**3 / (60.0 * pion_decay_constant**4 * mass * pi) \
                  * (6.0 * (axial_coupling**2 - 1.0) * r_sq**2 - 5.0 * axial_coupling**2 * mass_sq) \
                  * d3_N3LO - r_sq * mass_pion**2 / (6.0 * pion_decay_constant**4 * mass * pi) \
                  * (2.0 * (axial_coupling**2 - 1.0) * r_sq**2 - 3.0 * axial_coupling**2 * mass_sq) \
                  * d5_N3LO - 1.0 / (92160.0 * pion_decay_constant**6 * mass**2 * pi**3) * (-320.0 * \
                  (1.0 + 2.0 * axial_coupling**2)**2 * mass_pion**6 + 240.0 * (1.0 \
                  + 6.0 * axial_coupling**2 + 8.0 * axial_coupling**4) * mass_pion**4 * mass**2 \
                  - 60.0 * axial_coupling**2 * (8.0 + 15.0 * axial_coupling**2) * mass_pion**2 \
                  * mass**4 + (-4.0 + 29.0 * axial_coupling**2 + 122.0 * axial_coupling**4 \
                  + 3.0 * axial_coupling**6) * mass**6) * log( (2.0*r_sq + mass)\
                  / (2.0 * mass_pion)) - r_sq / (2700.0 * mass * (8.0 * pi \
                  * pion_decay_constant**2)**3) \
                  * (-16.0 * (171.0 + 2.0 * axial_coupling**2 * (1.0 + axial_coupling**2) \
                  * (327.0 + 49.0 * axial_coupling**2)) * mass_pion**4 + 4.0 \
                  * (-73.0 + 1748.0 * axial_coupling**2 + 2549.0 * axial_coupling**4 \
                  + 726.0*axial_coupling**6) * mass_pion**2 * mass**2 \
                  - (-64.0 + 389.0 * axial_coupling**2 + 1782.0 * axial_coupling**4 \
                  + 1093.0 * axial_coupling**6) * mass**4) + 2.0 * r_sq / (3.0 * mass \
                  *(8.0 * pi * pion_decay_constant**2)**3) * (axial_coupling**6 \
                  * mass_sq**2 * j1 - 2.0 * axial_coupling**4 * (2.0 * axial_coupling**2 - 1.0) \
                  * r_sq**2 * mass_sq * j2)
                    
        eta = array( [eta_sq4, eta_cq4] )
        rho = array( [rho_cq4, rho_sq4] )
        return eta, rho
    
    def _eta_rho_terms_third_order_rel(
        self,
        mass: float,
        one_loop: float,
        mass_diff: float,
        c_coeff: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the coefficients for the two-pion exchange potential terms at N3LO.
        
        Parameters
        ----------
        mass : float
            Mass term where rho and eta are being evaluated at.
        one_loop : float
            One loop contribution.
        mass_diff : float
            mass_diff = 4.0*mass_pion**2 - mass**2
        c_coeff : ndarray
            Array of coefficients. Values depend on the EFT order.
        
        Returns
        -------
        eta, rho : float
            Values for eta and rho at N3LO for rel terms.
        """
        c1_N3LO = c_coeff[0]
        c2_N3LO = c_coeff[1]
        c3_N3LO = c_coeff[2]
        c4_N3LO = c_coeff[3]
        
        rho_cq4_rel = 3.0 * axial_coupling**4 / (512.0 * pi * Mn_GeV * pion_decay_constant**4) \
                      * (-3.0 * (4.0 * mass_pion**4 - mass**4) * pi / (4.0 * mass))
        
        eta_cq4_rel = -axial_coupling**2 / (128.0 * pi * Mn_GeV * pion_decay_constant**4) \
                        * (4.0 * mass_pion**2 - 2.0 * mass**2 - axial_coupling**2 \
                        * (7.0 * mass_pion**2 - 4.5 * mass**2)) \
                        * (2.0 * mass_pion**2 - mass**2) * pi / (4.0 * mass)
        
        rho_sq4_rel = (9.0 * axial_coupling**4 / (512.0 * pi * Mn_GeV * pion_decay_constant**4) \
                       * (4.0 * mass_pion**2 - 1.5 * mass**2) * pi / (4.0 * mass)) * mass**2
                        
        eta_sq4_rel = (-axial_coupling**2 / (256 * pi * Mn_GeV * pion_decay_constant**4) \
                        * (8.0 * mass_pion**2 - 2.0 * mass**2 - axial_coupling**2 \
                        * (4.0 * mass_pion**2 - 1.5 * mass**2)) * pi / (4.0 * mass)) * mass**2
                        
        rho_slq4_rel = -3.0 * axial_coupling**4 / (64.0 * pi * Mn_GeV * pion_decay_constant**4) \
                       * (2.0 * mass_pion**2 - mass**2) * pi / (4.0 * mass)
                        
        eta_slq4_rel = -axial_coupling**2 * (1.0 - axial_coupling**2) / (64.0 * pi * Mn_GeV \
                       * pion_decay_constant**4) * (4.0 * mass_pion**2 - mass**2) * pi / (4.0 * mass)
        
        rho_cq5_rel = -axial_coupling**2 * one_loop / (32.0 * pi**2 * Mn_GeV \
                      * pion_decay_constant**4) * ((c2_N3LO - 6.0 * c3_N3LO) * mass**4 \
                      - 4.0 * (6.0 * c1_N3LO + c2_N3LO - 3.0 * c3_N3LO) \
                      * mass_pion**2 * mass**2 + 6.0 * (c2_N3LO - 2.0 * c3_N3LO) * mass_pion**4 \
                      + 24.0 * (2.0 * c1_N3LO + c3_N3LO) * mass_pion**6 \
                      / (4.0 * mass_pion**2 - mass**2))
                        
        eta_cq5_rel = c4_N3LO / (192.0 * pi**2 * Mn_GeV * pion_decay_constant**4) \
                      * mass**2 * mass_diff * one_loop + c4_N3LO * axial_coupling**2 \
                      / (192.0 * pi**2 * Mn_GeV * pion_decay_constant**4) * mass**2 \
                      * (8.0 * mass_pion**2 - 5.0 * mass**2) * one_loop

        eta_sq5_rel = (c4_N3LO / (192.0 * pi**2 * Mn_GeV * pion_decay_constant**4) \
                       * mass_diff * one_loop - c4_N3LO * axial_coupling**2 \
                       / (192.0 * pi**2 * Mn_GeV * pion_decay_constant**4) \
                       * (16.0 * mass_pion**2 - 7.0 * mass**2) * one_loop) * mass**2

        rho_slq5_rel = -c2_N3LO * axial_coupling**2 / (16.0 * pi**2 \
                        * Mn_GeV * pion_decay_constant**4) * mass_diff * one_loop

        eta_slq5_rel = c4_N3LO / (96.0 * pi**2 * Mn_GeV * pion_decay_constant**4) \
                       * mass_diff * one_loop + c4_N3LO * axial_coupling**2 / (96.0 * pi**2 \
                       * Mn_GeV * pion_decay_constant**4) * (8.0 * mass_pion**2 \
                       - 5.0 * mass**2) * one_loop
        
        eta = array( [eta_cq4_rel, eta_sq4_rel, eta_slq4_rel, 
                      eta_cq5_rel, eta_sq5_rel, eta_slq5_rel] )
        rho = array( [rho_cq4_rel, rho_sq4_rel, rho_slq4_rel, 
                      rho_cq5_rel, rho_slq5_rel] )
        
        return eta, rho
    
    def _eta_rho_terms_fourth_order(
        self,
        mass: float,
        r_sq: float,
        c_coeff: ArrayLike,
        e_coeff: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Calculates the coefficients for the two-pion exchange potential terms at N4LO.
        
        Parameters
        ----------
        mass : float
            Mass term where rho and eta are being evaluated at.
        r_sq : float
            r_sq = 0.5*sqrt(mass**2 - 4.0*mass_pion**2)
        c_coeff : ndarray
            Array of coefficients. Values depend on the EFT order.
        e_coeff : ndarray
            Array of coefficients. Values depend on the EFT order.
        
        Returns
        -------
        eta, rho : float
            Values for eta and rho at N4LO.
        """
        c1_N4LO = c_coeff[0]
        c2_N4LO = c_coeff[1]
        c3_N4LO = c_coeff[2]
        c4_N4LO = c_coeff[3]
        
        e14_N4LO = e_coeff[0]
        e17_N4LO = e_coeff[1]
        e19_N4LO = e_coeff[2]
        e21_N4LO = e_coeff[3]
        e22_N4LO = e_coeff[4]
        
        rho_cq5 = 2.0 * ((axial_coupling**2 * (2.0 * mass_pion**2 - mass**2) \
                  * (mass * (72.0 * c1_N4LO * (mass_pion**4 - 2.0 * mass_pion**2 * mass**2) \
                  + c2_N4LO * (-24.0 * mass_pion**4 + 53.0 * mass_pion**2 * mass**2 \
                  - 10.0 * mass**4) + 18.0 * (c3_N4LO * (-2.0 * mass_pion**4 \
                  + 5.0 * mass_pion**2 * mass**2 - 2.0 * mass**4) \
                  + 128.0 * pion_decay_constant**2 * (e14_N4LO * (-2.0 * mass_pion**2 + mass**2)**2 \
                  + mass_pion**2 * (4.0 * e19_N4LO * mass_pion**2 - 2.0 * e19_N4LO * mass**2 \
                  + e22_N4LO * mass**2)) * pi**2)) - 6.0 * (mass_pion**2 \
                  - 2.0 * mass**2) * (24.0 * c1_N4LO * mass_pion**2 + c2_N4LO \
                  * (-4.0 * mass_pion**2 + mass**2) + 6.0 * c3_N4LO \
                  * (-2.0 * mass_pion**2 + mass**2)) * r_sq \
                  * log((mass + 2.0*r_sq) / (mass - 2.0 * r_sq)))) \
                  / (24576.0 * pion_decay_constant**6 * mass**2* pi**2))
                            
        eta_cq5 = 2.0 * ((-((-3.0 * mass * r_sq**2 * (-3.0 * (-4.0 * c1_N4LO \
                  + c2_N4LO + c3_N4LO) * mass_pion**4 \
                  + 2.0 * (12.0 * c1_N4LO + c2_N4LO + c3_N4LO) * mass_pion**2 * r_sq**2 \
                  + 8.0 * (c2_N4LO + c3_N4LO) * r_sq**4) + (c3_N4LO - c4_N4LO) \
                  * axial_coupling**4 * (12.0 * mass_pion**2 * (2.0 * mass_pion \
                  - mass) * (-2.0 * mass_pion**2 + mass**2)**2 + (-192.0 * mass_pion**5 \
                  + 87.0 * mass_pion**4 * mass  + 96.0 * mass_pion**3 * mass**2 \
                  - 54.0 * mass_pion**2 * mass**3 + 6.0 * mass**5) \
                  * r_sq**2 - 2.0 * (16.0 * mass_pion**3 - 19.0 * mass_pion**2 * mass \
                  + 6.0 * mass**3) * r_sq**4 + 8.0 * mass * r_sq**6) \
                  + axial_coupling**2 * r_sq**2 * (48.0 * (-c3_N4LO \
                  + c4_N4LO) * mass_pion**3 * mass**2 + 32.0 * (c3_N4LO - c4_N4LO) \
                  * mass_pion**3 *(3.0 * mass_pion**2 + r_sq**2) - 3.0 * mass**3 \
                  * ((24.0 * c1_N4LO + 3.0 * c2_N4LO - 2.0 * c3_N4LO \
                  + 5.0 * c4_N4LO) * mass_pion**2 + 2.0 * (3.0 * c2_N4LO \
                  + 2.0*c3_N4LO + c4_N4LO) * r_sq**2) \
                  + mass * (3.0 * (60.0 * c1_N4LO + 3.0 * c2_N4LO - 8.0 * c3_N4LO \
                  + 11.0 * c4_N4LO) * mass_pion**4 \
                  + 2.0 * (36.0 * c1_N4LO + 21.0 * c2_N4LO + 8.0 * c3_N4LO \
                  + 13.0 * c4_N4LO) * mass_pion**2 * r_sq**2 \
                  + 8.0 * (3.0 * c2_N4LO + 2.0 * c3_N4LO + c4_N4LO) * r_sq**4))) / r_sq) \
                  + 3.0 * ((3.0 * (c2_N4LO + c3_N4LO) - 12.0 * c1_N4LO * (1.0 \
                  + 3.0 * axial_coupling**2) + axial_coupling**2 * (3.0 * c2_N4LO + c3_N4LO \
                  * (8.0 - 35.0 * axial_coupling**2) + 5.0 * c4_N4LO \
                  * (-1.0 + 7.0 * axial_coupling**2))) * mass_pion**6 \
                  + 3.0 * axial_coupling**2 * (8.0 * c1_N4LO - c2_N4LO - 2.0 * c3_N4LO \
                  + c4_N4LO + 10.0 * c3_N4LO * axial_coupling**2 \
                  - 10.0 * c4_N4LO * axial_coupling**2) * mass_pion**4 * mass**2 \
                  + 6.0 * (-c3_N4LO + c4_N4LO) * axial_coupling**4 * mass_pion**2 * mass**4) \
                  * log(-((mass + 2.0 * r_sq) / (-mass + 2.0 * r_sq)))) \
                  / (9216.0 * pion_decay_constant**6 * mass * pi**2))
                            
        rho_sq5 = 2.0 * (((c3_N4LO - c4_N4LO) * axial_coupling**4 * mass \
                  * (-2.0 * mass_pion**2 * (-2.0 * mass_pion + mass)**2 \
                  * (2.0 * mass_pion + mass) + (16.0 * mass_pion**3 - 9.0 * mass_pion**2 \
                  * mass + mass**3) * r_sq**2 - 2.0 * mass * r_sq**4 + 3.0 * mass_pion**2 \
                  * (-5.0 * mass_pion**2 + mass**2) * r_sq * log(-((mass + 2.0*r_sq) \
                  / (-mass + 2.0 * r_sq))))) / (2048.0 * pion_decay_constant**6 * pi**2 * r_sq))

        eta_sq5 = 2.0 * ((axial_coupling**2 * (4.0 * mass_pion**2 - mass**2) \
                  * (mass * (c4_N4LO * (12.0 * (2.0 + 9.0 * axial_coupling**2) \
                  * mass_pion**2 - 5.0 * mass**2) - 1152.0 * pion_decay_constant**2 \
                  * (2.0 * e17_N4LO * mass_pion**2 + 2.0 * e21_N4LO * mass_pion**2 \
                  - e17_N4LO * mass**2) * pi**2) + 24.0 * c4_N4LO * r_sq**3 \
                  * log((mass + 2.0 * r_sq) / (mass - 2.0 * r_sq)))) \
                  / (73728.0 * pion_decay_constant**6 * pi**2))
                    
        eta = array( [eta_cq5, eta_sq5] )
        rho = array( [rho_cq5, rho_sq5] )
        return eta, rho
        
    def _calculate_j_integrals(
        self, 
        mass: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Calculates the J integrals that appear in the spectral functions.
        
        r = 0.5 * sqrt( mass^2 - 4*mass_pion^2)
        J1 = int^1_0 dx ( mass^2/(r^2 * x^2) - (1 + mass^2/(r^2 * x^2))^(3/2)
                        * ln( (r * x + sqrt(mass^2 + r^2 * x^2)) \ (mass) ) )
        J2 = int^1_0 dx x^2 * ( mass^2/(r^2 * x^2) - (1 + mass^2/(r^2 * x^2))^(3/2)
                        * ln( (r * x + sqrt(mass^2 + r^2 * x^2)) \ (mass) ) )

        Parameters
        ----------
        mass : float
            The mass where we evaluate the J1 and J2 integrals.

        Returns
        -------
        j1 : float
            Value of the J1 integral at the input mass.
        j2 : float
            Value of the J2 integral at the input mass.
        """
        j_nodes, j_weights = self.j_nodes, self.j_weights
        
        j1, j2 = 0.0, 0.0
        r = 0.5 * sqrt(mass**2 - 4 * mass_pion**2)
        
        for x0, w0 in zip(j_nodes, j_weights):
            ratio = r * x0 / mass_pion
            
            if (ratio >= 0.001):
                j1 += (1 / ratio**2 - (1 + 1 / ratio**2)**1.5 \
                       * log(ratio + sqrt(1 + ratio**2))) * w0
                j2 += (1 / ratio**2 - (1 + 1 / ratio**2)**1.5 \
                       * log(ratio + sqrt(1 + ratio**2))) * w0 * x0**2
            else:
                j1 += (-4 / 3 - ratio**2 / 5 + 2 * ratio**4 / 35 - 8 * ratio**6 / 315) * w0
                j2 += (-4 / 3 - ratio**2 / 5 + 2 * ratio**4 / 35 - 8 * ratio**6 / 315) * w0 * x0**2
                
        return j1, j2
    
    def _partial_wave_decomp_C(
        self, 
        pl: ArrayLike, 
        plp: ArrayLike, 
        plm: ArrayLike, 
        j: int, 
        V_C: ArrayLike,
        W_C: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Partial wave decomposition for two-pion exchange potential terms V_C and W_C.

        Parameters
        ----------
        pl : ndarray
        plp : ndarray
        plm : ndarray
        j : int
        V_C : ndarray
        W_C : ndarray

        Returns
        -------
        V_pw : ndarray
            Partial wave projection of the V_C spectral portion of the potential.
        W_pw : ndarray
            Partial wave projection of the W_C spectral portion of the potential.
        """
        leg_nodes, leg_weights = self.leg_nodes, self.leg_weights
        V_pw, W_pw = zeros(6), zeros(6)
        
        if (j == 0):
            for V0, W0, ps, ws, l in zip(V_C, W_C, leg_nodes, leg_weights, pl):
                V_pw[0] += V0 * ws * l
                V_pw[2] += V0 * ps * ws * l
                
                W_pw[0] += W0 * ws * l
                W_pw[2] += W0 * ps * ws * l
        else:
            for V0, W0, ws, l, lp, lm in zip(V_C, W_C, leg_weights, pl, plp, plm):                
                V_pw[0] += V0 * ws * l
                V_pw[1] += V0 * ws * l
                V_pw[2] += V0 * ws * lp
                V_pw[3] += V0 * ws * lm
                
                W_pw[0] += W0 * ws * l
                W_pw[1] += W0 * ws * l
                W_pw[2] += W0 * ws * lp
                W_pw[3] += W0 * ws * lm
                
        return 2.0 * pi * V_pw, 2.0 * pi * W_pw
    
    def _partial_wave_decomp_S(
        self, 
        pl: ArrayLike, 
        plp: ArrayLike, 
        plm: ArrayLike, 
        j: int, 
        V_S: ArrayLike,
        W_S: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Partial wave decomposition for two-pion exchange potential terms V_S and W_S.

        Parameters
        ----------
        pl : ndarray
        plp : ndarray
        plm : ndarray
        j : int
        V_T : ndarray
        W_T : ndarray

        Returns
        -------
        V_pw : ndarray
            Partial wave projection of the V_S spectral portion of the potential.
        W_pw : ndarray
            Partial wave projection of the W_S spectral portion of the potential.
        """
        leg_nodes, leg_weights = self.leg_nodes, self.leg_weights
        V_pw, W_pw = zeros(6), zeros(6)
        
        if (j == 0):
            for V0, W0, ps, ws, l in zip(V_S, W_S, leg_nodes, leg_weights, pl):                
                V_pw[0] += -3.0 * V0 * ws * l
                V_pw[2] += V0 * ps * ws * l
                
                W_pw[0] += -3.0 * W0 * ws * l
                W_pw[2] += W0 * ps * ws * l
        else:
            for V0, W0, ws, l, lp, lm in zip(V_S, W_S, leg_weights, pl, plp, plm):
                V_pw[0] += -3.0 * V0 * ws * l
                V_pw[1] += V0 * ws * l
                V_pw[2] += V0 * ws * lp
                V_pw[3] += V0 * ws * lm
                
                W_pw[0] += -3.0 * W0 * ws * l
                W_pw[1] += W0 * ws * l
                W_pw[2] += W0 * ws * lp
                W_pw[3] += W0 * ws * lm
                
        return 2.0 * pi * V_pw, 2.0 * pi * W_pw
    
    def _partial_wave_decomp_T(
        self, 
        k: float, 
        p: float, 
        pl: ArrayLike, 
        plp: ArrayLike, 
        plm: ArrayLike, 
        j: int, 
        V_T: ArrayLike,
        W_T: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Partial wave decomposition for two-pion exchange potential terms V_T and W_T.

        Parameters
        ----------
        pl : ndarray
        plp : ndarray
        plm : ndarray
        j : int
        V_T : ndarray
        W_T : ndarray

        Returns
        -------
        V_pw : ndarray
            Partial wave projection of the V_T spectral portion of the potential.
        W_pw : ndarray
            Partial wave projection of the W_T spectral portion of the potential.
        """
        leg_nodes, leg_weights = self.leg_nodes, self.leg_weights
        j_mom, jmom_p1 = self.j_mom, self.jmom_p1
        ll_val, ll2_val = self.ll_val, self.ll2_val
        V_pw, W_pw = zeros(6), zeros(6)
        
        if (j == 0):
            kp, k2_p2 = 2.0 * k * p, k**2 + p**2 
                
            for V0, W0, ps, ws, l in zip(V_T, W_T, leg_nodes, leg_weights, pl):
                V_pw[0] += V0 * ws * l * (kp * ps - k2_p2)
                V_pw[2] += -V0 * ws * l * (k2_p2 * ps - kp)
                
                W_pw[0] += W0 * ws * l * (kp * ps - k2_p2)
                W_pw[2] += -W0 * ws * l * (k2_p2 * ps - kp)
        else:
            sqrt_ll = sqrt(ll_val)
            kp, k2_p2 = 2.0 * k * p, k**2 + p**2 
                
            for V0, W0, ps, ws, l, lp, lm in zip(V_T, W_T, leg_nodes, leg_weights, pl, plp, plm):
                V_pw[0] += -V0 * ws * l * (k2_p2 - kp * ps)
                V_pw[1] += V0 * ws * (k2_p2 * l - kp / ll2_val * (j_mom * lp + jmom_p1 * lm))
                V_pw[2] += -V0 * ws * (k2_p2 * lp - kp * l) / ll2_val
                V_pw[3] += V0 * ws * (k2_p2 * lm - kp * l) / ll2_val
                V_pw[4] += 2.0 * V0 * ws * sqrt_ll / ll2_val * (k**2 * lm + p**2 * lp - kp * l)  
                V_pw[5] += 2.0 * V0 * ws * sqrt_ll / ll2_val * (k**2 * lp + p**2 * lm - kp * l)
                
                W_pw[0] += -W0 * ws * l * (k2_p2 - kp * ps)
                W_pw[1] += W0 * ws * (k2_p2 * l - kp / ll2_val * (j_mom * lp + jmom_p1 * lm))
                W_pw[2] += -W0 * ws * (k2_p2 * lp - kp * l) / ll2_val
                W_pw[3] += W0 * ws * (k2_p2 * lm - kp * l) / ll2_val
                W_pw[4] += 2.0 * W0 * ws * sqrt_ll / ll2_val * (k**2 * lm + p**2 * lp - kp * l)  
                W_pw[5] += 2.0 * W0 * ws * sqrt_ll / ll2_val * (k**2 * lp + p**2 * lm - kp * l)
                
        return 2.0 * pi * V_pw, 2.0 * pi * W_pw
    
    def _partial_wave_decomp_LS(
        self, 
        k: float, 
        p: float, 
        pl: ArrayLike, 
        plp: ArrayLike, 
        plm: ArrayLike, 
        j: int, 
        V_LS: ArrayLike,
        W_LS: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        r"""
        Partial wave decomposition for two-pion exchange potential terms V_LS and W_LS.

        Parameters
        ----------
        pl : ndarray
        plp : ndarray
        plm : ndarray
        j : int
        V_LS : ndarray
        W_LS : ndarray

        Returns
        -------
        V_pw : ndarray
            Partial wave projection of the V_LS spectral portion of the potential.
        W_pw : ndarray
            Partial wave projection of the W_LS spectral portion of the potential.
        """
        leg_nodes, leg_weights, ll2_val = self.leg_nodes, self.leg_weights, self.ll2_val
        V_pw, W_pw = zeros(6), zeros(6)
        kp = k * p
        
        if (j == 0):
            for V0, W0, ps, ws, l in zip(V_LS, W_LS, leg_nodes, leg_weights, pl):
                V_pw[2] += V0 * kp * ws * l * (ps**2 - 1)
                W_pw[2] += W0 * kp * ws * l * (ps**2 - 1)
        else:
            for V0, W0, ps, ws, l, lp, lm in zip(V_LS, W_LS, leg_nodes, leg_weights, pl, plp, plm):
                V_pw[1] += V0 * kp * ws * (lp - lm) / ll2_val
                V_pw[2] += -V0 * kp * ws * (l - ps * lp)
                V_pw[3] += -V0 * kp * ws * (l - ps * lm)
                
                W_pw[1] += W0 * kp * ws * (lp - lm) / ll2_val
                W_pw[2] += -W0 * kp * ws * (l - ps * lp)
                W_pw[3] += -W0 * kp * ws * (l - ps * lm)
                
        return 2.0 * pi * V_pw, 2.0 * pi * W_pw
    
    def info(self) -> None:
        """
        Prints the order of the chiral expansion and at what cutoff.
        """
        chiral_order, cutoff = self.chiral_order, self.cutoff
        
        if (chiral_order == 0):
            print("Semilocal momentum-space chiral NN potential at LO [Q^0]")
        elif (chiral_order == 1):
            print("Semilocal momentum-space chiral NN potential at NLO [Q^2]")
        elif (chiral_order == 2):
            print("Semilocal momentum-space chiral NN potential at N2LO [Q^3]")
        elif (chiral_order == 3):
            print("Semilocal momentum-space chiral NN potential at N3LO [Q^4]")
        elif (chiral_order == 4):
            print("Semilocal momentum-space chiral NN potential at N4LO [Q^5]")
        elif (chiral_order == 5):
            print("Semilocal momentum-space chiral NN potential at N4LO [Q^5] + N5LO [Q^6]")
            print("Contacts in 3F2, 1F3, 3F3, 3F4")
        else:
            raise ValueError("Wrong value of chiral_order!")
            
        if (cutoff == 1):
            print("Cutoff value: lambda = 400 MeV")
        elif (cutoff == 2):
            print("Cutoff value: lambda = 450 MeV")
        elif (cutoff == 3):
            print("Cutoff value: lambda = 500 MeV")
        elif (cutoff == 4):
            print("Cutoff value: lambda = 550 MeV")
        else:
            raise ValueError("Wrong value of cutoff!")
        print("\n")
        return None
    
    def _initialize_grid(
        self,
        grid_cutoff: float,
        n_grid: int
    ) -> None:
        """
        Initializes the grid where the potential will be evaluated.

        Parameters
        ----------
        grid_cutoff : float
            The max value the grid can take.
        n_grid : int
            Total number of points in the grid.

        Returns
        -------
        None
        """
        q_grid = []
        mom_scale = 0.5
        power = 1.0
        n_grid_float = float(n_grid - 1.0)
        
        y_i = 1.0 / (mom_scale + 0.001)**power
        y_f = 1.0 / (mom_scale + grid_cutoff)**power
        
        for i in range(n_grid):
            val = 1.0 / (y_i + (y_f - y_i) * float(i) / n_grid_float)**(1 / power) - mom_scale
            q_grid = append(arr=q_grid, values=val)
        return q_grid
        
    def _legendre_polynomials(
        self, 
        x: ArrayLike, 
        n: int
    ) -> ArrayLike:
        """
        Generates an array of the Legendre polynomials.

        Parameters
        ----------
        x : ndarray
            Points where we are calculating the Legendre polynomials.
        n : int
            Value of angular momentum.

        Returns
        -------
        legendre : ndarray
            The Legendre polynomials.
        """
        pln = zeros(n + 1)
        pln[0] = 1.0
        
        if (n > 0):
            pln[1] = x

        if (n <= 1):
            legendre = pln[n]
        else:
            for k in range(n - 1):
                pln[k + 2] = ((2.0 * (k + 1) + 1.0) * x * pln[k + 1] \
                              - float(k + 1) * pln[k]) / (float(k + 2))
            legendre = pln[n]

        return legendre
    
    def _transformed_mesh(
        self, 
        p_in: float, 
        p_out: float, 
        p1: float, 
        p2: float, 
        p3: float
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Applies transformations to the Gaussian Legendre mesh/nodes.

        Parameters
        ----------
        p_in : float
            These points are transformed via the hyperbolic transformation
            x -> (1 + x) / (1 / p1 - (1 / p1 - 2 / p2) * x).
            They are mapped to interval (0, p2) where points p_in/2 are in 
            (0, p1) and points p_in/2 are in (p1, p2).
        p_out : float
            These points are transformed via the linear transformation
            x -> (p3 + p2) /2 + x * (p3 - p2)/2.
        p1 : float
            Number of points in region 1.
        p2 : float
            Number of points in region 2.
        p3 : float
            Total number of points in mesh

        Returns
        -------
        nodes : ndarray
            Transformed mesh from quadrature routine.
        weights : ndarray
            Transformed weights from quadrature routine.
        """
        p_in, p_out = int(p_in), int(p_out)
        tot_mom = p_in + p_out
        
        if (p_in > tot_mom) or (p_out > tot_mom):
            raise Exception("Error with p_in/p_out")
            
        n_in, w_in = self._gauss_legendre_mesh(-1.0, 1.0, p_in)
        nodes_in = [(1.0 + n) / (1 / p1 - (1 / p1 - 2 / p2) * n) for n in n_in]
        weights_in = [2.0 * (1 / p1 - 1 / p2) * w / (1 / p1 - (1 / p1 - 2 / p2) * n)**2 
                      for n, w in zip(n_in, w_in)]
            
        if (p_out != 0):
            n_out, w_out = self._gauss_legendre_mesh(-1.0, 1.0, p_out)
            nodes_out = [0.5 * ((p3 + p2) + (p3 - p2) * n) for n in n_out]
            weights_out = [0.5 * (p3 - p2) * w for w in w_out]
            
        transformed_nodes = hstack((nodes_in, nodes_out))
        transformed_weights = hstack((weights_in, weights_out))
        
        return transformed_nodes, transformed_weights
    
    def _gauss_legendre_mesh(
        self, 
        x0: float, 
        xf: float, 
        total_pts: Union[int, float]
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Generates Gaussian Legendre mesh/nodes.

        Parameters
        ----------
        x0 : float
            Initial point in interval (always -1 for Gauss-Legendre quadrature).
        xf : float
            Final point in interval (always 1 for Gauss-Legendre quadrature).
        total_pts : float
            Total number of points.

        Returns
        -------
        nodes : ndarray
            Mesh generated from x0 and xf.
        weights : ndarray
            Weights corresponding to the mesh.
        """
        total_pts = int(total_pts)
        z1, pp, tol = 0.0, 0.0, 3e-14
        m, slope, midpoint = int(0.5 * (total_pts + 1)), 0.5 * (x0 + xf), 0.5 * (xf - x0)
        nodes, weights = zeros(total_pts), zeros(total_pts)
        
        for i in range(m):
            x = cos(pi * (float(i + 1) - 0.25) / (float(total_pts) + 0.5))
            
            while (abs(x - z1) > tol):
                p1, p2 = 1.0, 0.0

                for j in range(total_pts):
                    p3 = p2
                    p2 = p1
                    p1 = ((2.0*(j + 1) - 1.0) * x * p2 - ((j + 1)-1.0) * p3) / (j + 1)

                pp = total_pts * (x * p1 - p2) / (x**2 - 1.0)
                z1 = x
                x = z1 - p1 / pp

            nodes[i] = slope - midpoint * x
            nodes[total_pts - i - 1] = slope + midpoint * x
            weights[i] = 2.0 * midpoint / ((1.0 - x**2) * pp**2)
            weights[total_pts - i - 1] = weights[i]
            
        return nodes, weights
            
            
            
            
            
        