import numpy as np

## script to localize MOs [Ben Amor, Nadia, et al. "Local Orbitals in Quantum Chemistry." Basis Sets in Computational Chemistry. Cham: Springer International Publishing, 2021. 41-101.]

# Foster–Boys [J. M. Foster, S. F. Boys. Rev. Mod. Phys. 1960, 32, 300.]

# Edmiston–Ruedenberg [C. Edmiston, K. Ruedenberg. Rev. Mod. Phys. 1963, 35, 457.]

# Pipek–Mezey [J. Pipek, P. G. J. Mezey, Chem. Phys. 1989, 90, 4916-4926.]
# AOS: AO overlap matrix; MOC: coeff of MOs to localize; MolIdx: AO index for different molecules
def PipekMezey(AOS, MOC, MolIdx):
    # calculate <psi0|Operator|psi1>
    def MatrixVal(psi0, psi1, Operator):
        return np.dot(psi0, np.matmul(Operator, psi1))

    MONum = len(MOC[0])
    MolNum = len(MolIdx)
    MOIdx = np.arange(MONum)
    np.random.shuffle(MOIdx)
    print(MOIdx)
    for i0 in np.arange(MONum):
        s = MOC[:, MOIdx[i0]]
        for i1 in np.arange(i0 + 1, MONum):
            t = MOC[:, MOIdx[i1]]
            Ast = 0.0
            Bst = 0.0
            for j in np.arange(MolNum):
                PA = np.zeros_like(AOS)
                PA[:, MolIdx[j, 0]:MolIdx[j, 1]] += 0.5 * AOS[:, MolIdx[j, 0]:MolIdx[j, 1]]
                PA[MolIdx[j, 0]:MolIdx[j, 1], :] += 0.5 * AOS[MolIdx[j, 0]:MolIdx[j, 1], :]
                ss = MatrixVal(s, s, PA)
                st = MatrixVal(s, t, PA)
                tt = MatrixVal(t, t, PA)
                Ast += st ** 2. - 0.25 * (ss - tt) ** 2.
                Bst += st * (ss - tt)
            #print(Ast, Bst)
            if (Bst == 0. and Ast < 0.):
                continue
            elif (Bst >= 0):
                Gamma = 0.25 * np.arccos(-Ast / np.sqrt(Ast ** 2. + Bst ** 2.))
                sp = np.cos(Gamma) * s + np.sin(Gamma) * t
                tp = -np.sin(Gamma) * s + np.cos(Gamma) * t
            else:
                Gamma = 0.25 * (2. * np.pi - np.arccos(-Ast / np.sqrt(Ast ** 2. + Bst ** 2.)))
                sp = np.sin(Gamma) * s - np.cos(Gamma) * t
                tp = np.cos(Gamma) * s + np.sin(Gamma) * t

            MOC[:, MOIdx[i0]] = sp
            MOC[:, MOIdx[i1]] = tp

# Natural Localized Molecular Orbitals

# Cholesky Decomposition

# Fock matrix block diagonalization (https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00887)
# AOS/AOF: overlap/Fock matrix under atomic basis; BlockBasis: basis number of each block (1D array)
# mode: POD/POD2L
def BlockDiagFock(AOS, AOF, BlockBasis, mode='POD'):
    # calculate the power of symmetry matrix
    def MatrixPower(M, x):
        e,v = np.linalg.eigh(M)
        return np.matmul(v, np.matmul(np.diag(np.power(e, x)), v.T))

    BasisNumber = np.sum(BlockBasis)
    if(AOF.shape[0] != BasisNumber):
        print("dismatch between Fock matrix and block basis")
        exit(1)

    if (mode.lower() == 'pod'):
        Smh = MatrixPower(AOS, -0.50)
        AOFp = np.matmul(Smh, np.matmul(AOF, Smh))
        Cp = np.zeros((BasisNumber, BasisNumber), dtype=float)
        bBegin = 0
        for basis in BlockBasis:
            bEnd = bBegin + basis
            e,Cp[bBegin:bEnd, bBegin:bEnd] = np.linalg.eigh(AOFp[bBegin:bEnd, bBegin:bEnd])
            bBegin = bEnd

        Fad = np.matmul(Cp.T, np.matmul(AOFp, Cp))
        C = np.matmul(Smh, Cp)
    elif (mode.lower() == 'pod2l'):
        Cp = np.zeros((BasisNumber, BasisNumber), dtype=float)
        bBegin = 0
        for basis in BlockBasis:
            bEnd = bBegin + basis
            Smh = MatrixPower(AOS[bBegin:bEnd, bBegin:bEnd], -0.50)
            AOFp = np.matmul(Smh, np.matmul(AOF[bBegin:bEnd, bBegin:bEnd], Smh))
            e,C = np.linalg.eigh(AOFp)
            Cp[bBegin:bEnd, bBegin:bEnd] = np.matmul(Smh, C)
            bBegin = bEnd

        Fadp = np.matmul(Cp.T, np.matmul(AOF, Cp))
        S = np.matmul(Cp.T, np.matmul(AOS, Cp))
        Smh = MatrixPower(S, -0.5)
        Fad = np.matmul(Smh, np.matmul(Fadp, Smh))
        C = np.matmul(Smh, Cp)
    else:
        print('BlockDiagFock: please use POD or POD2L for mode')
        exit()

    return Fad, C