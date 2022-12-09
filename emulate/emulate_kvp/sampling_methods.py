"""
"""
from numpy import (append, vstack, array, 
                   zeros, reshape, ceil)
from typing import Union, Optional
from numpy.typing import ArrayLike
from smt.sampling_methods import LHS

def random_sample(
    parameters: ArrayLike, 
    lower_bound: Union[int, float], 
    upper_bound: Union[int, float], 
    num_test: int, 
    rng: int = None
) -> ArrayLike:
    """
    Used for choosing a uniform, randomized set of parameters 
    and basis elements for parameter sampling calculations.    
    """
    rng = np.random.default_rng(rng)
    return rng.uniform(
        lower_bound, upper_bound,
        (num_test, len(parameters))
    )

def LHS_setup(
    parameters: ArrayLike, 
    vary: float, 
    num_basis: Union[int, float], 
    num_test: Union[int, float], 
    inc_param: bool = False, 
    fix_seed: int = None
) -> tuple[ArrayLike, ArrayLike]:
    """
    Setup used for implementing Latin-Hypercube sampling. Used for choosing a 
    randomized set of parameters and basis elements for parameter
    sampling calculations.

    Parameters
    ----------
    parameters : array
        Provides an array of the "true" points we are going to use to sample.
    vary : float
        Provides the variation in the pts (by percent) to create our 
        LHC sampling space.
    num_basis : int
        Number of basis points in the Latin-Hypercube.
    num_test : int
        Number of test points in the Latin-Hypercube.
    inc_param : boolean (default=False)
        Used to test accuracy of emulator by including the testing point
        in the training points basis.
    fix_seed : int (default=None)
        Used to fix the random state for reproducability.
        
    Returns
    -------
    basis : array
        Sampled constants used for training.
    test : array
        Sampled constants used for testing.
    """
    if (isinstance(num_basis, float) == True):
        num_basis = int(ceil(num_basis))
        
    if (isinstance(num_test, float) == True):
        num_test = int(ceil(num_test))
        
    if (isinstance(vary, float) == True):
        dp_i = []
        for p_i in parameters:
            dp_i = append(arr=dp_i, values=vary * p_i)
    else:
        dp_i = vary
        
    basis, test = sample_LHS(pts=parameters, 
                             pts_vary=dp_i, 
                             basis_size=num_basis, 
                             test_size=num_test, 
                             seed=fix_seed)
    
    if inc_param:
        values = []

        for p_i in parameters:
            values = append(arr=values, values=p_i)
        basis = basis[:-1]    
        basis = vstack((basis, values))
        
    return basis, test

def sample_LHS(
    pts: ArrayLike, 
    pts_vary: ArrayLike, 
    basis_size: int, 
    test_size: int, 
    seed: int = None
) -> tuple[ArrayLike, ArrayLike]:
    """
    Samples a parameter space using Latin-Hypercube sampling.

    Parameters
    ----------
    pts : array
        Provides an array of the "true" points used for sampling.
    pts_vary : array
        Provides the variation in the pts to create our LHS sampling space.
    basis_size : int
        Number of basis points in the Latin-Hypercube.
    test_size : int
        Number of test points in the Latin-Hypercube.
    seed : int (default=None)
        Seeds the randomized state.
        
    Returns
    -------
    basis : array
        Sampled constants used for training.
    test : array
        Sampled constants used for testing.
    """
    sample_lim = []
    
    if (isinstance(pts_vary, list) == False):
        for vals, dvals in zip(pts, pts_vary):
            sample_lim = append(arr=sample_lim, values=[vals - abs(dvals), vals + dvals])
            
    elif (isinstance(pts_vary, list) == True):
        for vals in pts:
            sample_lim = append(arr=sample_lim, values=[pts_vary[0], pts_vary[1]])
            
    else:
        raise ValueError("Wrong type for pts_vary")

    sample_lim = reshape(sample_lim, (len(pts), 2))
    
    if seed is None:
        LHC_sampling_basis = LHS(xlimits=sample_lim)
        LHC_sampling_test = LHS(xlimits=sample_lim)
    else:
        LHC_sampling_basis = LHS(xlimits=sample_lim, random_state=seed)
        LHC_sampling_test = LHS(xlimits=sample_lim, random_state=seed)

    basis = LHC_sampling_basis(basis_size)
    test = LHC_sampling_test(test_size)
    
    return basis, test








