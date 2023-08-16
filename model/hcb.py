'''
model parameter set for hcb COF
'''

import numpy as np

# crystal information
def lattice_vector(r_a, r_c):
    r_lv = np.zeros((3, 3), dtype=float)
    r_lv[0, 0] = r_a
    r_lv[1, 0] = r_a * np.cos(2. / 3. * np.pi)
    r_lv[1, 1] = r_a * np.sin(2. / 3. * np.pi)
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
                        [[0.5, 0.0, 0.0], [1./3., 1./3., 0.0]],
                        [[1./3., 1./3., 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                        [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]],
                        [[0.5, 0.0, 0.5], [1./3., 1./3., 0.5]],
                        [[1./3., 1./3., 0.5], [0.0, 0.0, 0.5]],
                        [[0.5, 0.0, 0.5], [0.5, 0.0, 0.0]],
                        [[1./3., 1./3., 0.0], [1./3., 1./3., 0.5]]], dtype=float)

# unit cell information
core_num = 2
link_num = 3

# general set of core (4-member ring) orbitals in sql COF
# orbital degenerate
core_degen_num = np.array([1, 2], dtype=int)
# phase of orbital (for coupling calculation)
core_phase = np.array([[0.0, 0.0, 0.0],
                       [0., 2./3., 4./3.]], dtype=float) * np.pi

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
    core_orb_num0 = len(core_orbs)
    link_orb_num0 = len(link_orbs)

    if core_orb_num0 != np.shape(v_cl)[0] or link_orb_num0 != np.shape(v_cl)[1]:
        print('model_Hamiltonian: dimension mismatch between orbs and v_cl')
        exit()

    core_orb_num = 0
    for core_orb in core_orbs:
        core_orb_num += core_degen_num[core_orb.type] * core_num
    link_orb_num = link_orb_num0 * link_num
    orb_num = core_orb_num + link_orb_num

    # real-space Hamiltonian
    # r_H[i][j]: [da, db, dc, hij]
    r_H = []
    for i in np.arange(orb_num):
        h = [[]]
        for j in np.arange(orb_num):
            h.append([])
        r_H.append(h)

    hi = 0
    for core_orb in core_orbs:
        for i in np.arange(core_num):
            for j in np.arange(core_degen_num[core_orb.type]):
                # core orbital energy term
                r_H[hi][hi].append([ 0,  0,  0, core_orb.energy])
                # core pi-pi coupling term
                r_H[hi][hi].append([ 0,  0, -1, core_orb.vpi])
                hi += 1

    for link_orb in link_orbs:
        for i in np.arange(link_num):
            # link orbital energy term
            r_H[hi][hi].append([ 0,  0,  0, link_orb.energy])
            # link pi-pi coupling term
            r_H[hi][hi].append([ 0,  0, -1, link_orb.vpi])
            hi += 1

    # bond core-link coupling term
    hi = 0
    for i,core_orb in enumerate(core_orbs):
        hj = core_orb_num
        for j,link_orb in enumerate(link_orbs):
            if core_degen_num[core_orb.type] == 1:
                v = np.cos(core_phase[core_orb.type]) * v_cl[i, j]
                r_H[hi][hj].append([ 0,  0,  0, v[0]])
                r_H[hi][hj + 1].append([ 0,  0,  0, v[1]])
                r_H[hi][hj + 2].append([ 0,  0,  0, v[2]])
                v = np.cos(core_phase[core_orb.type] + np.pi) * v_cl[i, j]
                r_H[hi + 1][hj].append([ 0,  0,  0, v[0]])
                r_H[hi + 1][hj + 1].append([ 0, -1,  0, v[1]])
                r_H[hi + 1][hj + 2].append([ 1,  0,  0, v[2]])
                hj += link_num
            else:
                v = np.cos(core_phase[core_orb.type]) * v_cl[i, j]
                r_H[hi][hj].append([ 0,  0,  0, v[0]])
                r_H[hi][hj + 1].append([ 0,  0,  0, v[1]])
                r_H[hi][hj + 2].append([ 0,  0,  0, v[2]])
                v = np.sin(core_phase[core_orb.type]) * v_cl[i, j]
                r_H[hi + 1][hj].append([ 0,  0,  0, v[0]])
                r_H[hi + 1][hj + 1].append([ 0,  0,  0, v[1]])
                r_H[hi + 1][hj + 2].append([ 0,  0,  0, v[2]])
                v = np.cos(core_phase[core_orb.type] + np.pi) * v_cl[i, j]
                r_H[hi + 2][hj].append([ 0,  0,  0, v[0]])
                r_H[hi + 2][hj + 1].append([ 0, -1,  0, v[1]])
                r_H[hi + 2][hj + 2].append([ 1,  0,  0, v[2]])
                v = np.sin(core_phase[core_orb.type] + np.pi) * v_cl[i, j]
                r_H[hi + 3][hj].append([ 0,  0,  0, v[0]])
                r_H[hi + 3][hj + 1].append([ 0, -1,  0, v[1]])
                r_H[hi + 3][hj + 2].append([ 1,  0,  0, v[2]])
                hj += link_num
        hi += core_degen_num[core_orb.type] * core_num

    # for all coupling terms:
    # r_H[i][j][:3] = -r_H[j][i][:3], r_H[i][j][4] = r_H[j][i][4]
    for i in np.arange(orb_num):
        tmp = []
        for h in r_H[i][i]:
            if h[0] != 0 or h[1] != 0 or h[2] != 0:
                tmp.append([-h[0], -h[1], -h[2], h[3]])
        r_H[i][i] += tmp
        for j in np.arange(i):
            for h in r_H[j][i]:
                r_H[i][j].append([-h[0], -h[1], -h[2], h[3]])

    return core_orb_num, orb_num, r_H