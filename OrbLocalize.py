import numpy as np

## MOs localization
# Foster–Boys
# [J. M. Foster, S. F. Boys. Rev. Mod. Phys. 1960, 32, 300.]

# Edmiston–Ruedenberg
# [C. Edmiston, K. Ruedenberg. Rev. Mod. Phys. 1963, 35, 457.]

# Pipek–Mezey
# [J. Pipek, P. G. J. Mezey, Chem. Phys. 1989, 90, 4916-4926.]
# AOS: AO overlap matrix
# MOC: coeff of MOs to localize
# MolIdx: AO index for different molecules
def PipekMezey(AOS, MOC, MolIdx, sweeps=100):
    # calculate <psi0|Operator|psi1>
    def MatrixVal(psi0, psi1, Operator):
        return np.dot(psi0, np.matmul(Operator, psi1))
    
    # consecutive two-by-two rotations of orbital pairs (1 sweep)
    def PM():
        AONum,MONum = np.shape(MOC)
        MolNum = len(MolIdx)
        MOIdx = np.arange(MONum)
        np.random.shuffle(MOIdx)
        for i0 in np.arange(MONum):
            s = MOC[:, MOIdx[i0]]
            for i1 in np.arange(i0 + 1, MONum):
                t = MOC[:, MOIdx[i1]]
                Ast = 0.0
                Bst = 0.0
                for j in np.arange(MolNum):
                    ao0 = MolIdx[j, 0]
                    ao1 = MolIdx[j, 1]
                #for j in np.arange(AONum):
                #    ao0 = j
                #    ao1 = j + 1
                    ss = 0.5 * (MatrixVal(s[ao0:ao1], s, AOS[ao0:ao1, :]) +
                                MatrixVal(s, s[ao0:ao1], AOS[:, ao0:ao1]))
                    st = 0.5 * (MatrixVal(s[ao0:ao1], t, AOS[ao0:ao1, :]) +
                                MatrixVal(s, t[ao0:ao1], AOS[:, ao0:ao1]))
                    tt = 0.5 * (MatrixVal(t[ao0:ao1], t, AOS[ao0:ao1, :]) +
                                MatrixVal(t, t[ao0:ao1], AOS[:, ao0:ao1]))
                    Ast += st ** 2. - 0.25 * (ss - tt) ** 2.
                    Bst += st * (ss - tt)
                sin = Bst / np.sqrt(Ast ** 2. + Bst ** 2.)
                cos = -Ast / np.sqrt(Ast ** 2. + Bst ** 2.)
                if (abs(sin) < 1.e-5 and cos < 0.):
                    continue
                elif (sin >= 0):
                    Gamma = 0.25 * np.arccos(cos)
                    sp = np.cos(Gamma) * s + np.sin(Gamma) * t
                    tp = -np.sin(Gamma) * s + np.cos(Gamma) * t
                else:
                    Gamma = 0.25 * (2. * np.pi - np.arccos(cos))
                    sp = np.sin(Gamma) * s - np.cos(Gamma) * t
                    tp = np.cos(Gamma) * s + np.sin(Gamma) * t
    
                MOC[:, MOIdx[i0]] = sp
                MOC[:, MOIdx[i1]] = tp
    
    for i in np.arange(sweeps):
        print(i)
        MOCtmp = np.copy(MOC)
        PM()
        Overlap = np.matmul(MOC.T, np.matmul(AOS, MOCtmp))
        if (abs(abs(np.prod(np.diag(Overlap))) - 1.0) < 1.e-5):
            return

# Natural Localized Molecular Orbitals

# Cholesky Decomposition

# Fock matrix block diagonalization
# (https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00887)
# AOS/AOF: overlap/Fock matrix under atomic basis
# BlockBasis: basis number of each block (1D array)
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
        b0 = 0
        for basis in BlockBasis:
            b1 = b0 + basis
            e,Cp[b0:b1, b0:b1] = np.linalg.eigh(AOFp[b0:b1, b0:b1])
            b0 = b1
        Fad = np.matmul(Cp.T, np.matmul(AOFp, Cp))
        C = np.matmul(Smh, Cp)

    elif (mode.lower() == 'pod2l'):
        Cp = np.zeros((BasisNumber, BasisNumber), dtype=float)
        b0 = 0
        for basis in BlockBasis:
            b1 = b0 + basis
            Smh = MatrixPower(AOS[b0:b1, b0:b1], -0.50)
            AOFp = np.matmul(Smh, np.matmul(AOF[b0:b1, b0:b1], Smh))
            e,C = np.linalg.eigh(AOFp)
            Cp[b0:b1, b0:b1] = np.matmul(Smh, C)
            b0 = b1

        Fadp = np.matmul(Cp.T, np.matmul(AOF, Cp))
        S = np.matmul(Cp.T, np.matmul(AOS, Cp))
        Smh = MatrixPower(S, -0.5)
        Fad = np.matmul(Smh, np.matmul(Fadp, Smh))
        C = np.matmul(Smh, Cp)
    else:
        print('BlockDiagFock: please use POD or POD2L for mode')
        exit()

    return Fad, C