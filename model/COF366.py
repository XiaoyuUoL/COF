# # TP-COF simplest model for COF-366
import numpy as np

from k_model import *
from sql import *

r_lv,k_lv = lattice_vector(25.4167, 6.1885)

mo_mode = 'u'
core_orbs = []
core_orbs.append(CoreOrb(1, -2.261, 0.001))

link_orbs = []
link_orbs.append(LinkOrb(-2.648, 0.008))

v_cl = np.array([[0.130]])

core_orb_num,orb_num,r_H = model_Hamiltonian(core_orbs, link_orbs, v_cl)
k_dos(k_dos_num, k_dos_sigma, r_H, core_orb_num)
k_band(k_band_num, k_high_symm, k_lv, r_H, core_orb_num)