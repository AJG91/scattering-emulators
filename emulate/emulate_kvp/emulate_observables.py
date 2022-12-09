"""
File containing the classes used to emulate/simulate the observables.
"""
from numpy import (
    zeros, identity, real, 
    trace, insert, pi, einsum
)
from numpy.linalg import solve
from numpy.typing import ArrayLike

from .constants import fm_to_sqrt_mb
from .observables import rvec_to_spin_scatt_matrix
from .utils import compute_mixed_S

def sampled_partial_wave(
    test_params: ArrayLike, 
    emulator, 
    solver, 
    nugget, 
    glockle, 
    emulate
) -> ArrayLike:
    """
    Samples a partial wave over different parameter sets.

    Parameters
    ----------
    test_params : array
        Sampled parameters array.
    emulator : class instance
        Specific emulator instance.
    solver : str
        Specifies how to solve system of equation for coefficients \beta.
    nugget : float
        Nugget for inverting singular matrices.
    glockle : boolean
        If True, use Glockle method emulation.
        If False, use Standard method emulation.
    emulate : boolean
        If True, use emulator.
        If False, use high-fidelity (simulator).
        
    Returns
    -------
    test_array : array
        On-shell K prediction for given parameters.
    """
    test_array = zeros((test_params.shape[0], emulator.len_k))
    
    if emulate:
        for i, params_i in enumerate(test_params):
            output = emulator.prediction(params_i, glockle=glockle, 
                                         sol=solver, h=nugget)
            if output.shape[0] == 3:
                test_array[i] = compute_mixed_S(output)
            else:
                test_array[i] = output
                
    else:
        for i, params_i in enumerate(test_params):
            test_array[i] = emulator.high_fidelity(params_i)
    
    return test_array

class EmulateCrossSection:
    """
    A class used to emulate the total cross section using the KVP emulator.
    
    Parameters
    ----------
    js : array
        Denotes the momentum.
    lecs : array
        Specifies the LECs used for the cross section calculation.
    emulator : instance
        A specific instance of the emulator.
    solver : str
        Picks the method used to solve the system of equations for the
        emulator prediction.
    """    
    def __init__(self, js, lecs, emulators, solver):
        self.js = js
        self.lecs = lecs
        self.emulators = emulators
        self.solver = solver
        self._cache = {}
    
    def partial_wave_cross_section(
        self, 
        len_k: int, 
        j: int, 
        K: ArrayLike, 
        is_coupled: bool
    ) -> ArrayLike:
        """
        The total cross section for a specific partial wave j.
        
        Parameters
        ----------
        len_k : int
            Length of energy grid.
        j : int
            The momentum for which we are calculating the cross section.
        K : array
            The K matrix for specific partial wave.
        is_coupled : boolean
            Specifies if in coupled or uncoupled channel.
        
        Returns
        -------
        (2 * j + 1) * real(R) : array
            The cross section at j.
        """
        if is_coupled:
            K = K.reshape(len_k, 2, 2)
            I = identity(K.shape[-1])
            R = -2j * solve(I - 1j * K, K)
            R = trace(R, axis1=-1, axis2=-2)
        else:
            R = -2j * K / (1 - 1j * K)
            
        return (2 * j + 1) * real(R)
        
    def emulate_cross_section(
        self, 
        lecs_test: ArrayLike, 
        pots: ArrayLike,
        nugget: float, 
        glockle: bool, 
        emulate: bool,
        remove_kohn: bool = True
    ) -> ArrayLike:
        """
        Calculates the total cross section over a set of parameters.
        
        Parameters
        ----------
        lecs_test : array
            Array of LECs used for cross section calculation.
        pots : array
            Array of potentials of the partial waves that will be used
            for the cross section calculation.
        nugget : float
            Nugget for inverting singular matrices.
        glockle : boolean
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        emulate : boolean
            If True, emulates cross section for given LECs.
            If False, simulates cross section for given LECs (high-fidelity solution).
        remove_kohn : boolean (default=True)
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        
        Returns
        -------
        pi / (2 * (k / fm_to_sqrt_mb)**2) * sigma : array
            The total cross section.
        """
        js = self.js
        emu = self.emulators
        lecs = self.lecs
        solver = self.solver
        
        sigma, len_i = 0, 0
        k, len_k = emu[0].k, emu[0].len_k
        
        L0_coupled = zeros((emu[0].num_emulations, len_k, 4))
        
        for i, (j_i, lecs_i, pot_i, emu_i) in enumerate(zip(js, lecs, pots, emu)):
            is_coupled = emu_i.is_coupled
            
            if lecs_i is None and i in self._cache:
                sigma += self._cache[i]
                continue
                
            if lecs_i is None:
                L0, _ = emu_i.ls_eq_no_interpolate(pot_i)
            else:
                lecs_pred = lecs_test[len_i:len_i + len(lecs_i)]
                len_i += len(lecs_i)
                
                if emulate:
                    L0 = emu_i.prediction(lecs_pred, glockle, solver, nugget)
                    
                    if is_coupled:
                        for i in range(L0.shape[0]):
                            L0_coupled[i] = insert(L0[i], 1, L0[i][1], axis=0).T
                        L0 = L0_coupled
                        
                    if remove_kohn:
                        L0 = compute_mixed_S(L0)
                else:         
                    L0 = emu_i.high_fidelity(lecs_pred)
                    
            sigma_j = self.partial_wave_cross_section(len_k, j_i, L0, is_coupled)
        
            if lecs_i is None:
                self._cache[i] = sigma_j
            sigma += sigma_j
            
        return pi / (2 * (k / fm_to_sqrt_mb)**2) * sigma
    
    def predict(
        self, 
        cs: ArrayLike, 
        lecs: ArrayLike, 
        pots: ArrayLike, 
        nugget: float, 
        glockle: bool, 
        emulate: bool, 
        remove_kohn: bool = True
    ) -> ArrayLike:
        """
        Emulator prediction of total cross section for different parameter sets.
        
        Parameters
        ----------
        cs : array
            Array that saves the emulator prediction.
        lecs : array
            Array of different combination of parameters used for cross section calcualtion.
        pots : array
            Array of potentials of the partial waves that will be used
            for the cross section calculation.
        nugget : float
            Nugget for inverting singular matrices.
        glockle : boolean
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        emulate : boolean
            If True, emulates cross section for given LECs.
            If False, simulates cross section for given LECs (high-fidelity solution).
        remove_kohn : boolean (default=True)
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        
        Returns
        -------
        cs : array
            Predictions of the total cross section for all sampled parameter sets.
        """
        for i, lecs_i in enumerate(lecs):
            cs[i] = self.emulate_cross_section(lecs_i, pots, nugget, 
                                               glockle, emulate, remove_kohn)

        return cs

class EmulateSpinObservables:
    """
    A class used to emulate spin observables using the KVP emulator.
    
    Parameters
    ----------
    js : array
        Denotes the momentum.
    lecs : array
        Specifies the LECs used for the cross section calculation.
    emulator : instance
        A specific instance of the emulator.
    solver : str
        Picks the method used to solve the system of equations for the
        emulator prediction.
    """    
    def __init__(self, js, lecs, emulators, solver):
        self.js = js
        self.lecs = lecs
        self._cache_uncoupled = zeros((len(js), emulators[0].len_k), dtype=complex)
        self._cache_coupled = zeros((len(js), emulators[0].len_k, 2, 2), dtype=complex)
        self.emulators = emulators
        self.solver = solver
        
    def emulate_spin_obs(
        self, 
        obs, 
        lecs_test: ArrayLike, 
        pots: ArrayLike,
        nugget: float, 
        glockle: bool, 
        emulate: bool,
        remove_kohn: bool = True
    ) -> ArrayLike:
        """
        Calculates the total cross section over a set of parameters.
        
        Parameters
        ----------
        obs : instance
            A specific instance of the the observable class.
        lecs_test : array
            Array of LECs used for cross section calculation.
        pots : array
            Array of potentials of the partial waves that will be used
            for the cross section calculation.
        nugget : float
            Nugget for inverting singular matrices.
        glockle : boolean
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        emulate : boolean
            If True, emulates cross section for given LECs.
            If False, simulates cross section for given LECs (high-fidelity solution).
        remove_kohn : boolean (default=True)
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        
        Returns
        -------
        observables : dict
            A dictionary of all the spin observables.
        """
        js = self.js
        emu = self.emulators
        lecs = self.lecs
        solver = self.solver
        
        len_i, j_0, idx = 0, -1, 0
        len_k = emu[0].len_k
        
        K0_coupled = zeros((emu[0].num_emulations, len_k, 4))
        Rvec = zeros((7, js[-1] + 2, len_k), dtype=complex)
        
        for i, (j_i, lecs_i, pot_i, emu_i) in enumerate(zip(js, lecs, pots, emu)):
            is_coupled = emu_i.is_coupled
            k = emu_i.k
                
            if lecs_i is None and emulate:
                if is_coupled:
                    R = self._cache_coupled[i]
                else:
                    R = self._cache_uncoupled[i]
            
            else:                    
                if lecs_i is None:
                    K0, _ = emu_i.ls_eq_no_interpolate(pot_i)
                else:
                    lecs_pred = lecs_test[len_i:len_i + len(lecs_i)]
                    len_i += len(lecs_i)

                    if emulate:
                        K0 = emu_i.prediction(lecs_pred, glockle, solver, nugget)
                    
                        if is_coupled:
                            for i in range(K0.shape[0]):
                                K0_coupled[i] = insert(K0[i], 1, K0[i][1], axis=0).T
                            K0 = K0_coupled

                        if remove_kohn:    
                            K0 = compute_mixed_S(K0)
                    else:
                        K0 = emu_i.high_fidelity(lecs_pred)
                if is_coupled:
                    k = k[:, None, None]
                    K0 = K0.reshape(len_k, 2, 2)
                    
                R = obs.K_to_R(K0, is_coupled) / (1j * k / fm_to_sqrt_mb)

            j_0, idx, Rvec = self.build_rvec(j_0, j_i, idx, R, Rvec, is_coupled)
                
            if lecs_i is None:
                if is_coupled:
                    self._cache_coupled[i] = R
                else:
                    self._cache_uncoupled[i] = R
                
        return self.get_observables(js, Rvec, obs)
    
    def build_rvec(
        self, 
        j_0: int, 
        j_i: int, 
        idx: int,
        R: ArrayLike, 
        Rvec: ArrayLike,
        is_coupled: bool
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Computes the R vector needed for the spin observables calculation.
        Fills the previously defined Rvec for specific partial wave.
        
        Parameters
        ----------
        j_0 : int
            Counter used to specify which part of Rvec to fill.
        j_i : int
            Specific partial wave where we are calculating the R vector.
        R : array
            Reactance matrix for specific partial wave.
        Rvec : array
            R vector.
        is_coupled : boolean
            If True, use coupled channels.
            If False, use non-coupled channels.
        
        Returns
        -------
        j_0 : int
            Counter used to specify which part of Rvec to fill.
        Rvec : array
            R vector.
        """
        if j_i == j_0:
            idx += 1
        else:
            idx = 0
        
        if not is_coupled:
            if j_0 != j_i:
                if j_i < 12:
                    Rvec[0][j_i] = R
            else:
                if j_i == 0:
                    Rvec[3][0] = R
                    Rvec[3][1] = R
                else:
                    if j_i < 5:
                        Rvec[2][j_i] = R
                    elif idx == 1 and j_i < 13:
                        Rvec[2][j_i] = R
                        
            if j_i == 6 and idx == 2:
                Rvec[1][5] = R
            elif j_i == 7 and idx == 2:
                Rvec[1][6] = R
            elif j_i == 8 and idx == 2:
                Rvec[1][7] = R
            elif j_i == 9 and idx == 2:
                Rvec[1][8] = R
            elif j_i == 10 and idx == 2:
                Rvec[1][9] = R
            elif j_i == 11 and idx == 2:
                Rvec[1][10] = R
            elif j_i == 12 and idx == 2:
                Rvec[1][11] = R
            elif j_i == 12 and idx == 0:
                Rvec[0][12] = R
            elif j_i == 5 and idx == 5:
                Rvec[3][6] = R
            elif j_i == 6 and idx == 5:
                Rvec[3][7] = R
            elif j_i == 7 and idx == 5:
                Rvec[1][4] = R
                Rvec[3][8] = R
            elif j_i == 8 and idx == 5:
                Rvec[3][9] = R
            elif j_i == 9 and idx == 5:
                Rvec[3][10] = R
            elif j_i == 10 and idx == 5:
                Rvec[3][11] = R
            elif j_i == 11 and idx == 5:
                Rvec[3][12] = R
            elif j_i == 12 and idx == 5:
                Rvec[3][13] = R
            elif j_i == 6 and idx == 1:
                Rvec[4][4] = R
                Rvec[5][6] = R
            elif j_i == 7 and idx == 1:
                Rvec[4][5] = R
                Rvec[5][7] = R
            elif j_i == 8 and idx == 1:
                Rvec[4][6] = R
                Rvec[5][8] = R
            elif j_i == 9 and idx == 1:
                Rvec[4][7] = R
                Rvec[5][9] = R
            elif j_i == 10 and idx == 1:
                Rvec[4][8] = R
                Rvec[5][10] = R
            elif j_i == 11 and idx == 1:
                Rvec[4][9] = R
                Rvec[5][11] = R
            elif j_i == 12 and idx == 1:
                Rvec[4][10] = R
                Rvec[5][12] = R
            elif j_i == 13 and idx == 1:
                Rvec[4][11] = R
                Rvec[5][13] = R
                
            j_0 = j_i

        else:
            R = R.T
            Rvec[1][j_i - 1] = R[0][0] # jmom = 1
            Rvec[3][j_i + 1] = R[1][1] # jmom = 1
            Rvec[4][j_i - 1] = R[0][1] # jmom = 1
            Rvec[5][j_i + 1] = R[1][0] # jmom = 1
    
        return j_0, idx, Rvec
    
    def get_observables(
        self, 
        js: list, 
        Rvec: ArrayLike, 
        obs
    ) -> dict:
        """
        Computes the spin observables given an R vector.
        
        Parameters
        ----------
        js : list
            List of all the momenta (partial waves).
        Rvec : array
            Filled R vector.
        obs : instance
            A specific instance of the the observable class.
        
        Returns
        -------
        observables : dict
            A dictionary of all the spin observables.
        """
        sin0 = obs.sin0
        Lmat = rvec_to_spin_scatt_matrix(js[-1], obs.x)
        obs_list = ['DSG', 'PB', 'D', 'AXX', 'AYY', 'A']
        
        amp = einsum('mnLT, nLQ -> mQT', Lmat, Rvec)
        obs.saclay_parameters(sin0, amp)

        dsg = obs.compute_dsg()
        Ay = obs.compute_Ay(dsg)
        A = obs.compute_A(sin0, obs.alpha, dsg)
        D = obs.compute_D(dsg)
        Axx = obs.compute_Axx(sin0, dsg)
        Ayy = obs.compute_Ayy(dsg)
    
        return dict(zip(obs_list, [dsg, Ay, D, Axx, Ayy, A]))

    def predict(
        self,
        spin_obs: ArrayLike, 
        obs, 
        lecs: ArrayLike, 
        pots: ArrayLike, 
        nugget: float, 
        glockle: bool, 
        emulate: bool, 
        remove_kohn: bool = True
    ) -> ArrayLike:
        """
        Emulator prediction of spin observables for different parameter sets.
        
        Parameters
        ----------
        spin_obs : array
            Array that saves the emulator prediction.
        obs : instance
            A specific instance of the the observable class.
        lecs : array
            Array of different combination of parameters used for cross section calcualtion.
        pots : array
            Array of potentials of the partial waves that will be used
            for the cross section calculation.
        nugget : float
            Nugget for inverting singular matrices.
        glockle : boolean
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        emulate : boolean
            If True, emulates cross section for given LECs.
            If False, simulates cross section for given LECs (high-fidelity solution).
        remove_kohn : boolean (default=True)
            If True, use Glockle method emulation.
            If False, use Standard method emulation.
        
        Returns
        -------
        spin_obs : dict
            Predictions of the spin observables for all sampled parameter sets.
        """
        for i, lecs_i in enumerate(lecs):
            spin_obs[i] = self.emulate_spin_obs(obs, lecs_i, pots, nugget, 
                                                glockle, emulate, remove_kohn)

        return spin_obs
    
    
    
    
    
    