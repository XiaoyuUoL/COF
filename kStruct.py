# Electronic structure in 3D k-space
import numpy as np
import os

import input

np.set_printoptions(precision=3, suppress=True)

# solve M * C = S * C * e (return 'C' and 'e')
def GeneralEigen(S, M):
    # calculate the power of symmetry matrix
    def MatrixPower(M, x):
        e,V = np.linalg.eigh(M)
        return np.matmul(V, np.matmul(np.diag(np.power(e, x)), V.conj().T))

    Smh = MatrixPower(S, -0.5)
    Mp = np.matmul(Smh, np.matmul(M, Smh))

    e,Cp = np.linalg.eigh(Mp)
    C = np.matmul(Smh, Cp)

    return e, C

# multi-centre weighted-Gaussian broaden
def Gaussian(x, sigma, mu, weight):
    xNum = len(x)
    result = np.zeros(xNum, dtype = float)
    for i in np.arange(xNum):
        result[i] = np.sum(weight * np.exp(-(mu - x[i]) ** 2 / (2. * sigma ** 2)))
    result /=  sigma * np.sqrt(2 * np.pi)

    return result

def Project(C):
    Csq = (C * C.conjugate()).real
    result = np.zeros((2, input.TMON), dtype=float)
    # cores
    result[0, :] = np.sum(Csq[:input.CoreTMON, :], axis=0)
    # links
    result[1, :] = np.sum(Csq[input.CoreTMON:, :], axis=0)

    return result

def DoS(e, sigma, normalize=1.0):
    de = sigma / 10.0
    eVal = np.arange(e.min() - 5.0 * sigma, e.max() + 5.0 * sigma, de)

    DoSVal = Gaussian(eVal, sigma, e, 1.0 / normalize)

    return np.array([eVal, DoSVal]).T

def pDoS(e, weight, sigma, normalize=1.0):
    de = sigma / 10.0
    eVal = np.arange(e.min() - 5.0 * sigma, e.max() + 5.0 * sigma, de)
    pDoSVal = np.zeros((len(weight), len(eVal)), dtype=float)
    result = np.array(eVal)
    for i in np.arange(len(weight)):
        pDoSVal[i, :] = Gaussian(eVal, sigma, e, weight[i, :] / normalize)
        result = np.append(result, pDoSVal[i, :])
    result = np.append(result, np.sum(pDoSVal, axis=0))

    return np.reshape(result, (len(weight) + 2, -1)).T

# distance in k-space
def kDist(k0, k1):
    dk = np.dot(k0, input.kV) - np.dot(k1, input.kV)
    return np.sqrt(np.sum(dk * dk))

# return matrix in k-space
def kOverlap(k, rS):
    if (len(rS) != input.TMON):
        print('error in kOverlap: mismatch of length between TMON and rS')
        exit()

    kS = np.zeros((input.TMON, input.TMON), dtype=complex)
    for i in np.arange(input.TMON):
        for j in np.arange(input.TMON):
            for s in rS[i][j]:
                kS[i, j] += np.exp(2.j * np.pi * np.dot(k, s[:3])) * s[3]

    return kS

# return Hamiltonian in k-space
def kHamiltonian(k, rH):
    if (len(rH) != input.TMON):
        print('error in kHamiltonian: mismatch of length between TMON and rH')
        exit()

    kH = np.zeros((input.TMON, input.TMON), dtype=complex)
    for i in np.arange(input.TMON):
        for j in np.arange(input.TMON):
            for h in rH[i][j]:
                kH[i, j] += np.exp(2.j * np.pi * np.dot(k, h[:3])) * h[3]

    return kH

# return density of state (including pDoS)
def kDoS(kNum, rH, rS=None):
    if (len(rH) != input.TMON):
        print('error in kDoS: mismatch of length between TMON and rH')
        exit()

    e = np.array([], dtype=float)
    w = np.array([[], []], dtype=float)
    for ia in np.arange(kNum):
        ka = -0.5 + ia / (1.0 * kNum)
        for ib in np.arange(kNum):
            kb = -0.5 + ib / (1.0 * kNum)
            for ic in np.arange(kNum):
                kc = -0.5 + ic / (1.0 * kNum)
                kH = kHamiltonian([ka, kb, kc], rH)
                if (rS == None):
                    ke,kC = np.linalg.eigh(kH)
                else:
                    kS = kOverlap([ka, kb, kc], rS)
                    ke,kC = GeneralEigen(kS, kH)
                e = np.append(e, ke)
                w = np.append(w, Project(kC), axis=1)

    np.savetxt('{}/DoS.dat'.format(input.WorkDir), DoS(e, input.sigma, kNum ** 3))
    np.savetxt('{}/pDoS.dat'.format(input.WorkDir), pDoS(e, w, input.sigma, kNum ** 3))

# return band structure besed on High Symmetry points provided
def kBand(kHighSymm, kNum, rH, rS=None):
    if (len(rH) != input.TMON):
        print('error in kBand: mismatch of length between TMON and rH')
        exit()

    kPoints = []
    for i in np.arange(len(kHighSymm) - 1):
        for j in np.arange(kNum):
            kPoints.append(kHighSymm[i] + j / kNum * (kHighSymm[i + 1] - kHighSymm[i]))
    kPoints.append(kHighSymm[-1])
    kl = np.zeros(len(kPoints))
    for i in np.arange(len(kPoints) - 1):
        kl[i + 1] = kDist(kPoints[i + 1], kPoints[i]) + kl[i]

    if (os.path.isfile('bands.dat')):
        os.system('rm bands.dat')

    emin = None
    emax = None
    fout = []
    for i in np.arange(input.TMON):
        fout.append(open('band-{}.dat'.format(i), 'w'))
    for dk,kPoint in zip(kl,kPoints):
        kH = kHamiltonian(kPoint, rH)
        if (rS == None):
            ke,kC = np.linalg.eigh(kH)
        else:
            kS = kOverlap(kPoint, rS)
            ke,kC = GeneralEigen(kS, kH)
        if (emin == None or emin > min(ke)):
            emin = min(ke)
        if (emax == None or emax < max(ke)):
            emax = max(ke)
        w = Project(kC)
        for i in np.arange(input.TMON):
            ResultStr = '{:14.7f}{:14.7f}{:14.7f}{:14.7f}{:14.7f}'.format(dk, kPoint[0], kPoint[1], kPoint[2], ke[i])
            for j in np.arange(len(w)):
                ResultStr += '{:14.7f}'.format(w[j, i])
            fout[i].writelines('{}\n'.format(ResultStr))
    for i in np.arange(input.TMON):
        fout[i].writelines('\n')
        fout[i].close()
        os.system('cat band-{}.dat >> {}/bands.dat'.format(i, input.WorkDir))
        os.system('rm band-{}.dat'.format(i))

    fout = open('{}/highsymm.dat'.format(input.WorkDir), 'w')
    for i in np.arange(len(kHighSymm)):
        de = emax - emin
        fout.writelines('{:14.7f}{:14.7f}\n'.format(kl[i * kNum], emin - 0.05 * de))
        fout.writelines('{:14.7f}{:14.7f}\n'.format(kl[i * kNum], emax + 0.05 * de))
        fout.writelines('\n')
    fout.close()