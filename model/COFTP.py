# TP-COF simplest model using hcb
import numpy as np

from k_model import *
from hcb import *

r_lv,k_lv = lattice_vector(37.58990, 3.63765)

mo_mode = 'u'
core_orbs = []
core_orbs.append(CoreOrb(0, -1.191, 0.252))
core_orbs.append(CoreOrb(1, -0.670, 0.205))

link_orbs = []
link_orbs.append(LinkOrb(-1.357, 0.296))

v_cl = np.array([[0.307], [0.314]])

core_orb_num,orb_num,r_H = model_Hamiltonian(core_orbs, link_orbs, v_cl)

k_dos(k_dos_num, k_dos_sigma, r_H, core_orb_num)

k_band(k_band_num, k_high_symm, k_lv, r_H, core_orb_num)

k0 = np.array([0.0, 0.0, 0.5])
k1 = np.array([0.0, 0.0, 0.0])
dk = 0.001
nk = 20
# effective mass of [0.0, 0.0, 0.5] -> [0.0, 0.0, 0.0], i.e., interlayer
print(k_effmass(k_lv, k0, k1, dk, nk, r_H, mo_mode, 2))
# effective mass of [0.0, 0.0, 0.5] -> [0.5, 0.5, 0.5], i.e., intralayer
k1 = np.array([0.5, 0.5, 0.5])
print(k_effmass(k_lv, k0, k1, dk, nk, r_H, mo_mode, 2))