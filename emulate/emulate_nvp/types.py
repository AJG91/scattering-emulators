from enum import Enum
from typing import NamedTuple

from .constants import mass_proton, mass_neutron, mass_alpha

from .kinematics import compute_reduced_mass
from .kinematics import t_lab_to_t_cm
from .kinematics import t_lab_to_q_cm_beam_and_target
from .kinematics import t_cm_to_t_lab


class Isospin(Enum):
    PP = -1
    PN = 0
    NP = 0
    NN = +1

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
            if value == "pp":
                return cls.PP
            if value == "nn":
                return cls.NN
            if value == "pn":
                return cls.PN
            if value == "np":
                return cls.NP
        return super()._missing_(value)

    def __int__(self):
        return self.value

    @property
    def mass(self):
        from .constants import mass_proton, mass_neutron, mass_nucleon

        if self == Isospin.PP:
            return mass_proton
        if self == Isospin.PN:
            return mass_nucleon
        if self == Isospin.NN:
            return mass_neutron
        return NotImplemented

    def __str__(self):
        if self == Isospin.PP:
            return "pp"
        if self == Isospin.NN:
            return "nn"
        return "np"


class ScatteringSystem(Enum):
    PP = "pp"
    PN = "np"
    NP = "np"
    NN = "nn"
    P_ALPHA = "p-alpha"

    @property
    def mass(self):
        return 2.0 * self.reduced_mass

    @property
    def reduced_mass(self):
        return compute_reduced_mass(self.mass_beam, self.mass_target)

    @property
    def mass_beam(self):
        if self is self.PP:
            return mass_proton
        if self is self.PN:
            return mass_neutron
        if self is self.NN:
            return mass_neutron
        if self is self.P_ALPHA:
            return mass_alpha
        return NotImplemented

    @property
    def mass_target(self):
        if self is self.PP:
            return mass_proton
        if self is self.PN:
            return mass_proton
        if self is self.NN:
            return mass_neutron
        if self is self.P_ALPHA:
            return mass_proton
        return NotImplemented

    def t_lab_to_t_cm(self, t_lab):
        return t_lab_to_t_cm(
            t_lab=t_lab, mass_beam=self.mass_beam, mass_target=self.mass_target
        )

    def t_cm_to_t_lab(self, t_cm):
        return t_cm_to_t_lab(
            t_cm=t_cm, mass_beam=self.mass_beam, mass_target=self.mass_target
        )

    def t_lab_to_q_cm(self, t_lab):
        return t_lab_to_q_cm_beam_and_target(
            t_lab, mass_beam=self.mass_beam, mass_target=self.mass_target
        )


class BoundaryCondition(Enum):
    INCOMING = -1  # -ie
    OUTGOING = +1  # +ie
    STANDING = +0  # principal value
