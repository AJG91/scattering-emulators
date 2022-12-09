"""
This file contains nuclear physics constants needed.
"""
from numpy import sqrt

MeV_to_GeV = 1e3 # Conversion factor between MeV and GeV
fm_to_sqrt_mb = sqrt(10)  # 1 fm**2 == 10 mb

mass_proton = 938.27208816  # [mass_proton] =  MeV/c^2
mass_neutron = 939.56542052  # [mass_neutron] =  MeV/c^2
Mn_MeV = 2.0 * (mass_neutron * mass_proton) / (mass_neutron + mass_proton) # [Mn_MeV] =  MeV/c^2
Mn_GeV = Mn_MeV / MeV_to_GeV # [Mn_GeV] =  GeV/c^2

hbar_c = 197.3269631 # [hbar*c] = MeV - fm
hbar_c_GeV = hbar_c / MeV_to_GeV # [hbar * c] = GeV - fm
GeV_to_fm = MeV_to_GeV / hbar_c # [GeV_to_fm] = 1 / fm

hbarsq_over_Mn =  hbar_c**2 / Mn_MeV # [hbar^2 / Mn] = 1 / (MeV^2 - fm)
hbarsq_over_Mn_GeV = hbarsq_over_Mn / MeV_to_GeV # [hbar^2 / Mn] = 1 / (GeV^2 - fm)

V_factor_RME = hbar_c_GeV * Mn_GeV  # [V_factor] = (GeV^2 - fm)

axial_coupling = 1.29 # Axial charge coupling constant of the nucleon
pion_decay_constant = 0.0924 # [GeV]
mass_neutral_pion = 0.1349768 # [GeV/c]
mass_charged_pion = 0.13957039 # [GeV/c]
mass_pion = (mass_neutral_pion + 2.0 * mass_charged_pion) / 3 # [GeV/c]








