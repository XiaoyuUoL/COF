# model k-space calculations
import numpy as np
import os

def project(c, core_orb_num):
    orb_num = np.shape(c)[0]
    csq = (c * c.conjugate()).real
    result = np.zeros((2, orb_num), dtype=float)
    # cores
    result[0, :] = np.sum(csq[:core_orb_num, :], axis=0)
    # links
    result[1, :] = np.sum(csq[core_orb_num:, :], axis=0)

    return result

# distance in k-space
def k_dist(k0, k1, k_lv):
    dk = np.dot(k0, k_lv) - np.dot(k1, k_lv)
    return np.sqrt(np.sum(dk * dk))

# return Hamiltonian in k-space
def k_Hamiltonian(k, r_H):
    orb_num = len(r_H)
    k_H = np.zeros((orb_num, orb_num), dtype=complex)
    for i in np.arange(orb_num):
        for j in np.arange(orb_num):
            for h in r_H[i][j]:
                k_H[i, j] += np.exp(2.j * np.pi * np.dot(k, h[:3])) * h[3]

    return k_H

# return density of state (including pDoS)
def k_dos(k_dos_num, k_dos_sigma, r_H, core_orb_num):
    # multi-centre weighted-Gaussian broaden
    def Gaussian(x, sigma, mu, weight):
        x_num = len(x)
        result = np.zeros(x_num, dtype = float)
        for i in np.arange(x_num):
            dx = x[i] - mu
            result[i] = np.sum(weight * np.exp(-0.5 * (dx / sigma) ** 2))
        result /=  sigma * np.sqrt(2 * np.pi)

        return result

    def DoS(e, sigma, normalize=1.0):
        de = sigma / 10.0
        e_val = np.arange(min(e) - 5.0 * sigma, max(e) + 5.0 * sigma, de)
    
        dos_val = Gaussian(e_val, sigma, e, 1.0 / normalize)
    
        return np.array([e_val, dos_val]).T

    def pDoS(e, weight, sigma, normalize=1.0):
        de = sigma / 10.0
        e_val = np.arange(min(e) - 5.0 * sigma, max(e) + 5.0 * sigma, de)
        pdos_val = np.zeros((len(weight), len(e_val)), dtype=float)
        result = np.array(e_val)
        for i in np.arange(len(weight)):
            pdos_val[i, :] = Gaussian(e_val, sigma, e, weight[i, :] / normalize)
            result = np.append(result, pdos_val[i, :])
        result = np.append(result, np.sum(pdos_val, axis=0))

        return np.reshape(result, (len(weight) + 2, -1)).T

    e = np.array([], dtype=float)
    w = np.array([[], []], dtype=float)
    for ia in np.arange(k_dos_num):
        ka = -0.5 + ia / (1.0 * k_dos_num)
        for ib in np.arange(k_dos_num):
            kb = -0.5 + ib / (1.0 * k_dos_num)
            for ic in np.arange(k_dos_num):
                kc = -0.5 + ic / (1.0 * k_dos_num)
                k_H = k_Hamiltonian([ka, kb, kc], r_H)
                ke,kC = np.linalg.eigh(k_H)
                e = np.append(e, ke)
                w = np.append(w, project(kC, core_orb_num), axis=1)

    np.savetxt('DoS.dat', DoS(e, k_dos_sigma, k_dos_num ** 3))
    np.savetxt('pDoS.dat', pDoS(e, w, k_dos_sigma, k_dos_num ** 3))

# return band structure besed on High Symmetry points provided
def k_band(k_band_num, k_high_symm, k_V, r_H, core_orb_num):
    k_points = []
    kl = []
    for i in np.arange(len(k_high_symm)):
        dk = k_high_symm[i, 1] - k_high_symm[i, 0]
        for j in np.arange(k_band_num + 1):
            k_point = k_high_symm[i, 0] + j / k_band_num * dk
            k_points.append(k_point)
            if (i == 0 and j == 0):
                kl.append(0.0)
            elif (j == 0):
                kl.append(kl[-1])
            else:
                kl.append(k_dist(k_points[-1], k_points[-2], k_V) + kl[-1])

    if (os.path.isfile('bands.dat')):
        os.system('rm bands.dat')

    emin = None
    emax = None
    orb_num = len(r_H)
    fout = []
    for i in np.arange(orb_num):
        fout.append(open('band-{:d}.dat'.format(i), 'w'))
    for dk,k_point in zip(kl,k_points):
        k_H = k_Hamiltonian(k_point, r_H)
        ke,kC = np.linalg.eigh(k_H)
        if (emin == None or emin > min(ke)):
            emin = min(ke)
        if (emax == None or emax < max(ke)):
            emax = max(ke)
        w = project(kC, core_orb_num)
        for i in np.arange(orb_num):
            result_str = '{:14.7f}{:14.7f}{:14.7f}{:14.7f}{:14.7f}'.format(dk,
                k_point[0], k_point[1], k_point[2], ke[i])
            for j in np.arange(len(w)):
                result_str += '{:14.7f}'.format(w[j, i])
            fout[i].writelines('{}\n'.format(result_str))
    for i in np.arange(orb_num):
        fout[i].writelines('\n')
        fout[i].close()
        os.system('cat band-{:d}.dat >> bands.dat'.format(i))
        os.system('rm band-{}.dat'.format(i))

    fout = open('highsymm.dat', 'w')
    de = emax - emin
    for i in np.arange(len(k_high_symm)):
        k = kl[i * (k_band_num + 1)]
        fout.writelines('{:14.7f}{:14.7f}\n'.format(k, emin - 0.05 * de))
        fout.writelines('{:14.7f}{:14.7f}\n'.format(k, emax + 0.05 * de))
        fout.writelines('\n')
    k = kl[-1]
    fout.writelines('{:14.7f}{:14.7f}\n'.format(k, emin - 0.05 * de))
    fout.writelines('{:14.7f}{:14.7f}\n'.format(k, emax + 0.05 * de))
    fout.writelines('\n')
    fout.close()

## return effective mass based on a begin and end k-point
# k0: begin k-point (k-point of VBM for hole or CBM for electron)
# k1: end k-point (i.e., effective mass through k0-k path)
# dk: interval of |k| for effective mass calculation (unit: A^-1)
# nk: number of k-points for effective mass calculation
# n: highest order of polynomial fitting
def EffMass(k_V, k0, k1, dk, nk, r_H, mo_mode='u', n=2):
    vk = (k1 - k0) / k_dist(k0, k1, k_V)
    k = []
    e = []
    for i in np.arange(nk):
        # a.u. of 1/length
        k.append(i * dk * 0.52917721090380)
        k_point = k0 + vk * dk * i
        k_H = k_Hamiltonian(k_point, r_H)
        ke,kC = np.linalg.eigh(k_H)
        # VBM
        if (mo_mode == 'o'):
            # a.u. of energy
            e.append(max(ke) / 27.21138624598853)
        # CBM
        elif (mo_mode == 'u'):
            # a.u. of energy
            e.append(min(ke) / 27.21138624598853)
        else:
            print('EffMass: please use "u" or "o" for mo_mode')
            exit()
    
    np.savetxt('test.dat', np.array([k, e]).T)
    coeffs = np.polyfit(k, e, n)

    # mass=hbar/(d2E/dk2), unit: me
    mass = 0.5 / coeffs[-3]
    return mass