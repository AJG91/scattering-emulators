# Scattering Emulators

This repository contains all code and data necessary to generate the results in
*Wave function-based emulation for nucleon-nucleon scattering in momentum space* ([arXiv:2301.05093][arXiv]).
It extends the coordinate-space [Kohn variational principle (KVP) emulator][Furnstahl] to momentum-space (including coupled channels) with arbitrary boundary conditions, which enable the mitigation of spurious singularities known as [Kohn anomalies][Drischler].
It also provides comparisons with the [Newton's variational principle (NVP) emulator][Melendez] for selected partial waves and NN observables using the semilocal momentum-space (SMS) regularized chiral potential at N4LO+.


## Getting Started

* This project relies on `python=3.9`. It was not tested with different versions.
  To view the entire list of required packages, see `environment.yml`.
* Clone the repository to your local machine.
* Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate scattering-emulators-env`
* Install the `emulate_kvp` and `emulate_nvp` packages in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `emulate/`).


## Example

The main class for the KVP-based emulator is `KVP_emulator`, which implements both the standard Lippmann-Schwinger solver
and the KVP emulator.
The code snippet below shows how it should be used:
```python
from emulate import KVP_emulator

# Setup
V0, V1 = ...  # The parameter independent piece, and the linear piece of the potential
ps, ws = ...   # The momentum and integration measure in units of inverse fm, corresponding to the potential mesh
E = ...   # The lab energy in MeV

# Initialize object. Only handles linear potentials: V = V0 + V1 @ lecs
emu = KVP_emulator(k=k, ps=ps, ws=ws, V0=V0, 
                   V1=V1, wave=wave, is_coupled=False) 
# The argument wave controls the partial wave being trained.
# For coupled channels, is_coupled=True.

# Train the emulator
emu.train(train_params=basis, glockle=True, method=emu_method)  # basis = (n_b, n_a)
# If glockle=True, the Glockle spline method is used in the calculation. 
# If glockle=False, the Standard method is used.
# The method argument controls the boundary conditions used by the emulator.
# Boundary conditions for emulator: 'K', '1/K', 'T', 'all'

# Predict phase shifts at validation parameter values using the simulator and the emulator
emu_pred = emu.prediction(test_params=lecs, glockle=False, sol=solver, h=nugget)  # Emulator
sim = emu.high_fidelity(params=lecs)  # No emulator
```

The main class for the NVP-based emulator is `NVP_emulator`, which implements both the standard Lippmann-Schwinger solver
and the NVP emulator.
The code snippet below shows how it should be used (with the same setup as above):
```python
from emulate import TwoBodyScattering as NVP_emulator

# Initialize object. Only handles linear potentials: V = V0 + V1 @ p
scatt = NVP_emulator(V0=V0, V1=V1, k=ps, dk=ws,
                     t_lab=E, system="np")

# Train the emulator
scatt.fit(basis)  # basis = (n_b, n_a)

# Predict phase shifts at validation parameter values using the simulator and the emulator
phase_pred_valid = scatt.predict(lecs, return_phase=True)                   # Emulator
phase_full_valid = scatt.predict(lecs, return_phase=True, full_space=True)  # No emulator
# If full_space=True, the simulator is used.
# If full_space=False, the emulator is used.
```

## Citing this work

Please cite this work as follows:

```bibtex
@article{Garcia:2023slj,
    author = "Garcia, A. J. and Drischler, C. and Furnstahl, R. J. and Melendez, J. A. and Zhang, Xilin",
    title = "{Wave function-based emulation for nucleon-nucleon scattering in momentum space}",
    eprint = "2301.05093",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "1",
    year = "2023"
}
```

[arxiv]: https://arxiv.org/abs/2301.05093
[Furnstahl]: https://www.sciencedirect.com/science/article/pii/S0370269320305220
[Drischler]: https://www.sciencedirect.com/science/article/pii/S0370269321007176
[Melendez]: https://www.sciencedirect.com/science/article/pii/S0370269321005487
