'''
model parameter set for sql COF
'''

import numpy as np

# crystal information
def lattice_vector(r_a, r_c):
    r_lv = np.zeros((3, 3), dtype=float)
    r_lv[0, 0] = r_a
    r_lv[1, 1] = r_a
    r_lv[2, 2] = r_c

    k_lv = np.zeros((3, 3), dtype=float)
    V = np.dot(r_lv[0, :], np.cross(r_lv[1, :], r_lv[2, :]))
    k_lv[0, :] = np.cross(r_lv[1, :], r_lv[2, :]) / V
    k_lv[1, :] = np.cross(r_lv[2, :], r_lv[0, :]) / V
    k_lv[2, :] = np.cross(r_lv[0, :], r_lv[1, :]) / V
    k_lv *= 2. * np.pi

    return r_lv, k_lv

k_dos_num = 20
k_dos_sigma = 0.05

# k-paths between high symmetry points for band structure calculation
k_band_num = 100
k_high_symm = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                        [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
                        [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                        [[0.0, 0.0, 0.5], [0.0, 0.5, 0.5]],
                        [[0.0, 0.5, 0.5], [0.5, 0.5, 0.5]],
                        [[0.5, 0.5, 0.5], [0.0, 0.0, 0.5]],
                        [[0.0, 0.5, 0.0], [0.0, 0.5, 0.5]],
                        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.5]]], dtype=float)

# unit cell information
core_num = 1
link_num = 2

# general set of core (4-member ring) orbitals in sql COF
# orbital degenerate
core_degen_num = np.array([1, 2, 1], dtype=int)
# phase of orbital (for coupling calculation)
core_phase = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.5, 1.0, 1.5],
                       [0.0, 1.0, 2.0, 3.0]], dtype=float) * np.pi

class CoreOrb:
    '''information of core orbital information'''
    def __init__(self, orb_type, orb_e, orb_vpi):
        # orbital type: three types for 4-member ring for sql
        self.type = orb_type
        # orbital energies
        self.energy = orb_e
        # pi-pi coupling (interlayer)
        self.vpi = orb_vpi

class LinkOrb:
    '''link orbital information'''
    def __init__(self, orb_e, orb_vpi):
        # orbital energies
        self.energy = orb_e
        # pi-pi coupling (interlayer)
        self.vpi = orb_vpi

def model_Hamiltonian(core_orbs, link_orbs, v_cl):
    '''return real-space model Hamiltonian for k-space electronic structure'''
    core_orb_num = len(core_orbs)
    link_orb_num = len(link_orbs)

    if core_orb_num != np.shape(v_cl)[0] or link_orb_num != np.shape(v_cl)[1]:
        print('model_Hamiltonian: dimension mismatch between orbs and v_cl')
        exit()

    orb_num = 0
    for core_orb in core_orbs:
        orb_num += core_degen_num[core_orb.type] * core_num
    orb_num += link_orb_num * link_num

    # real-space Hamiltonian
    # rH[i][j]: [da, db, dc, hij]
    rH = []
    for i in np.arange(orb_num):
        h = [[]]
        for j in np.arange(orb_num):
            h.append([])
        rH.append(h)

    hi = 0
    for core_orb in core_orbs:
        for i in np.arange(core_degen_num[core_orb.type]):
            # core orbital energy term
            rH[hi][hi].append([ 0,  0,  0, core_orb.energy])
            # core pi-pi coupling term
            rH[hi][hi].append([ 0,  0, -1, core_orb.vpi])
            hi += 1
    hi_core = hi

    for link_orb in link_orbs:
        for i in np.arange(link_num):
            # link orbital energy term
            rH[hi][hi].append([ 0,  0,  0, link_orb.energy])
            # link pi-pi coupling term
            rH[hi][hi].append([ 0,  0, -1, link_orb.vpi])
            hi += 1

    # bond core-link coupling term
    hi = 0
    for i,core_orb in enumerate(core_orbs):
        hj = hi_core
        for j,link_orb in enumerate(link_orbs):
            if core_degen_num[core_orb.type] == 1:
                v = np.cos(core_phase[core_orb.type]) * v_cl[i, j]
                rH[hi][hj].append([ 0,  0,  0, v[0]])
                rH[hi][hj + 1].append([ 0,  0,  0, v[1]])
                rH[hi][hj].append([-1,  0,  0, v[2]])
                rH[hi][hj + 1].append([ 0, -1,  0, v[3]])
                hj += link_num
            else:
                v = np.cos(core_phase[core_orb.type]) * v_cl[i, j]
                rH[hi][hj].append([ 0,  0,  0, v[0]])
                rH[hi][hj + 1].append([ 0,  0,  0, v[1]])
                rH[hi][hj].append([-1,  0,  0, v[2]])
                rH[hi][hj + 1].append([ 0, -1,  0, v[3]])
                v = np.sin(core_phase[core_orb.type]) * v_cl[i, j]
                rH[hi + 1][hj].append([ 0,  0,  0, v[0]])
                rH[hi + 1][hj + 1].append([ 0,  0,  0, v[1]])
                rH[hi + 1][hj].append([-1,  0,  0, v[2]])
                rH[hi + 1][hj + 1].append([ 0, -1,  0, v[3]])
                hj += link_num
        hi += core_degen_num[core_orb.type] * core_num

    # for all coupling terms:
    # rH[i][j][:3] = -rH[j][i][:3], rH[i][j][4] = rH[j][i][4]
    for i in np.arange(orb_num):
        tmp = []
        for h in rH[i][i]:
            if h[0] != 0 or h[1] != 0 or h[2] != 0:
                tmp.append([-h[0], -h[1], -h[2], h[3]])
        rH[i][i] += tmp
        for j in np.arange(i):
            for h in rH[j][i]:
                rH[i][j].append([-h[0], -h[1], -h[2], h[3]])

    return hi_core, orb_num, rH