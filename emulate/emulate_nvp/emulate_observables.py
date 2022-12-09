"""
"""
from numpy import (
    array, real, trace, zeros, 
    insert, pi, einsum, stack
)
from numpy.typing import ArrayLike
from emulate_kvp.observables import rvec_to_spin_scatt_matrix
from emulate_kvp.constants import fm_to_sqrt_mb
from .physics import K_to_R

def partial_wave_emulator(
    test_lecs, 
    emulator, 
    full_space
) -> ArrayLike:
    """
    Samples a partial wave over different parameter sets.

    Parameters
    ----------
    test_lecs : array
        Sampled LECs array.
    emulator : class instance
        Specific emulator instance.
    full_space : boolean
        If True, use high-fidelity (simulator).
        If False, use emulator.
        
    Returns
    -------
    test_array: array
        On-shell K prediction for given LECs.
    """
    test_array = zeros((test_lecs.shape[0], emulator.n_q))
    
    for i, lecs_i in enumerate(test_lecs):
        test_array[i] = emulator.predict(lecs_i, full_space=full_space)
    return test_array

class ObservablesEmulator:
    """
    A class used to emulate the observables using the NVP emulator.
    
    Parameters
    ----------
    emulator : instance
        A specific instance of the emulator.
    lecs : array
        Specifies the LECs used for the cross section calculation.
    js : array
        Denotes the momentum.
    obs : instance
        A specific instance of the the observable class.
    """   
    def __init__(self, emulators, lecs, j, obs):
        self.emulators = emulators
        self.lecs = lecs
        self.j = j
        self._cache_cs = {}
        self._cache_so = {}
        self.obs = obs

    def _predict_cs(
        self, 
        p: int, 
        full_space: bool = False
    ) -> ArrayLike:
        """
        The total cross section emulation.
        
        Parameters
        ----------
        p : array
            Array of LECs used for cross section calculation.
        full_space : boolean (default=False)
            If True, use high-fidelity (simulator).
            If False, use emulator.
        
        Returns
        -------
        -pi / (2 * q_cm ** 2) * sigma : array
            The total cross section.
        """
        js = self.j
        lecs = self.lecs
        emulators = self.emulators

        sigma = 0
        len_i = 0
        q_cm = None
        
        for i, (j_i, lecs_i, emu_i) in enumerate(zip(js, lecs, emulators)):
            q_cm = emu_i.q_cm
            q_cm = q_cm / fm_to_sqrt_mb
            
            if lecs_i is None and i in self._cache_cs:
                sigma += self._cache_cs[i]   
                continue
            
            if lecs_i is None:
                K = emu_i.predict(array([0]), full_space=True, return_phase=False)
            else:
                lec_predict = p[len_i:len_i + len(lecs_i)]
                len_i += len(lecs_i)
                K = emu_i.predict(lec_predict, full_space=full_space, return_phase=False)

            is_coupled = emu_i.is_coupled
            
            if is_coupled:
                K = K.reshape(len(q_cm), 2, 2)
                
            R = K_to_R(K, is_coupled)
            if is_coupled:
                R = trace(R, axis1=-1, axis2=-2)

            sigma_i = (2 * j_i + 1) * real(R)
            if lecs_i is None:
                self._cache_cs[i] = sigma_i
            sigma += sigma_i

        return -pi / (2 * q_cm ** 2) * sigma
    
    
    def _predict_spin_obs(
        self, 
        p: ArrayLike, 
        full_space: bool = False
    ) -> dict:
        """
        The total cross section emulation.
        
        Parameters
        ----------
        p : array
            Array of LECs used for cross section calculation.
        full_space : boolean (default=False)
            If True, use high-fidelity (simulator).
            If False, use emulator.
        
        Returns
        -------
        observables : dict
            A dictionary of all the spin observables.
        """
        js = self.j
        lecs = self.lecs
        emulators = self.emulators
        n_q = emulators[0].n_q
        Rvec = zeros((7, js[-1] + 2, n_q), dtype=complex)

        sigma = 0
        len_i = 0
        j_0 = -1
        q_cm = None
        
        for i, (j_i, lecs_i, emu_i) in enumerate(zip(js, lecs, emulators)):
            q_cm = emu_i.q_cm
            q_cm = q_cm / fm_to_sqrt_mb
            is_coupled = emu_i.is_coupled
            
            if lecs_i is None and i in self._cache_so:
                K = self._cache_so[i]
            else:
                if lecs_i is None:
                    K = emu_i.predict(array([0]),
                                      full_space=True, 
                                      return_phase=False)
                else:
                    lec_predict = p[len_i:len_i + len(lecs_i)]
                    len_i += len(lecs_i)
                    K = emu_i.predict(lec_predict, 
                                      full_space=full_space, 
                                      return_phase=False)
            if is_coupled:
                K = K.reshape(len(q_cm), 2, 2)
                q_cm = q_cm[:, None, None]
                
            Rvec_i = K_to_R(K, is_coupled) / (1j * pi * q_cm / fm_to_sqrt_mb)

            if is_coupled == False:
                if j_0 != j_i:
                    Rvec[0][j_i] = Rvec_i
                else:
                    if j_i == 0 and is_coupled == False:
                        Rvec[3][0] = Rvec_i
                        Rvec[3][1] = Rvec_i
                    else:
                        Rvec[2][j_i] = Rvec_i
                j_0 = j_i

            else:
                Rvec_i = Rvec_i.T
                Rvec[1][j_i - 1] = Rvec_i[0][0] # jmom = 1
                Rvec[3][j_i + 1] = Rvec_i[1][1] # jmom = 1
                Rvec[4][j_i - 1] = Rvec_i[0][1] # jmom = 1
                Rvec[5][j_i + 1] = Rvec_i[1][0] # jmom = 1
                
            if lecs_i is None:
                self._cache_so[i] = K
                
        return self._get_observables(js, Rvec)
    
    def _get_observables(
        self, 
        js: list, 
        Rvec: ArrayLike
    ) -> dict:
        """
        Computes the spin observables given an R vector.
        
        Parameters
        ----------
        js : list
            List of all the momenta (partial waves).
        Rvec : array
            Filled R vector.
        
        Returns
        -------
        observables : dict
            A dictionary of all the spin observables.
        """
        obs = self.obs
        Lmat = rvec_to_spin_scatt_matrix(js[-1], obs.x)
        amp = einsum('mnLT, nLQ -> mQT', Lmat, Rvec)
        
        sin0 = obs.sin0
        obs.saclay_parameters(sin0, amp)
        
        dsg = obs.compute_dsg()
        Ay = obs.compute_Ay(dsg)
        A = obs.compute_A(sin0, obs.alpha, dsg)
        D = obs.compute_D(dsg)
        Axx = obs.compute_Axx(sin0, dsg)
        Ayy = obs.compute_Ayy(dsg)
    
        return {
            'DSG': dsg, 'PB': Ay, 'D': D, 
            'AXX': Axx, 'AYY': Ayy, 'A': A
            }

    def predict(
        self, 
        p: ArrayLike, 
        spin_obs: bool = False, 
        full_space: bool = False, 
        out = None
    ) -> ArrayLike:
        """
        Emulator prediction of observables for different parameter sets.
        
        Parameters
        ----------
        p : array
            Array of different combination of parameters used for cross section calcualtion.
        spin_obs : boolean (default=False)
            If True, calculating spin observables.
            If False, calculating total cross section.
        full_space : boolean (default=False)
            If True, use high-fidelity (simulator).
            If False, use emulator.
        out : [None, array]
            If out is not None, output the array passed in filled with the prediction.
        
        Returns
        -------
        pred : [dict, array]
            Predictions of the observables for all sampled parameter sets.
        """
        
        if spin_obs == False:
            if p.ndim == 1:
                pred = self._predict_cs(p, full_space=full_space)
            elif p.ndim == 2:
                pred = stack(
                    [self._predict_cs(p_i, full_space=full_space) for p_i in p],
                    axis=0,
                )
            else:
                raise ValueError("p must be 1d or 2d")
        else:
            if p.ndim == 1:
                pred = self._predict_spin_obs(p, full_space=full_space)
            elif p.ndim == 2:
                pred = stack(
                    [self._predict_spin_obs(p_i, full_space=full_space) for p_i in p],
                    axis=0,
                )
            else:
                raise ValueError("p must be 1d or 2d")

        if out is not None:
            out[:] = pred
        return pred
    
    
    
    
