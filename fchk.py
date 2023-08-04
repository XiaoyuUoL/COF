import numpy as np

## return MO orbital information from fchk file ('fchkfile')
# 'MOE': energy, 'MOC': coeff, 'AOS': overlap and 'iHOMO': index of HOMO
# only for closed-shell systems
def ReadFchkOrb(fchkfile):
    # calculate the power of symmetry matrix
    def MatrixPower(M, x):
        e,v = np.linalg.eigh(M)
        return np.matmul(v, np.matmul(np.diag(np.power(e, x)), v.T))

    AON = None  # Number of AO basis
    MON = None  # Number of MOs
    iHOMO = None  # index of HOMO
    MOE = []  # Energy of MOs (unit: Hartree)
    MOC = []  # Coeff of MOs

    fin = open('{}.fchk'.format(fchkfile), 'r')
    line = fin.readline()
    while (len(line) != 0):
        words = line.rstrip().split()
        # read number of basis functions
        if(len(words) > 4 and words[:4] == ['Number', 'of', 'basis', 'functions']):
            AON = int(words[-1])
        elif(len(words) > 4 and words[:4] == ['Number', 'of', 'alpha', 'electrons']):
            iHOMO = int(words[-1])
        # read MO energies
        elif(len(words) > 3 and words[:3] == ['Alpha', 'Orbital', 'Energies']):
            MON = int(words[-1])
            for i in np.arange(int((MON - 1) / 5) + 1):
                MOE += fin.readline().rstrip().split()
        # read MO coefficients
        elif(len(words) > 3 and words[:3] == ['Alpha', 'MO', 'coefficients']):
            for i in np.arange(int((MON * AON - 1) / 5) + 1):
                MOC += fin.readline().rstrip().split()
        line = fin.readline()
    fin.close()

    if (AON == None or MON == None or len(MOE) == 0 or len(MOC) == 0):
        print('no orbital information in {}.fchk'.format(fchkfile))
        exit()

    MOE = np.array(MOE, dtype=float)
    MOC = np.reshape(np.array(MOC, dtype=float), (MON, AON)).T
    AOS = MatrixPower(np.matmul(MOC, MOC.T), -1.0)

    return MOE, MOC, AOS, iHOMO

## write orbital into fchk file ('fchkfile') based on original one ('fchkfile0')
def WriteFchkOrb(fchkfile0, fchkfile, MOE, MOC):
    # Number of AO basis
    AON = np.shape(MOC)[0]
    # Number of MOs
    MON = np.shape(MOC)[1]
    if (MON != len(MOE)):
        print('WriteFchkOrb: number of MOs does not match between MOE and MOC')
        exit()
    MON0 = None

    fin = open('{}.fchk'.format(fchkfile0), 'r')
    fout = open('{}.fchk'.format(fchkfile), 'w')
    line = fin.readline()
    while (len(line) != 0):
        words = line.rstrip().split()
        # read number of basis functions
        if(len(words) > 4 and words[:4] == ['Number', 'of', 'basis', 'functions']):
            if (AON != int(words[-1])):
                print('WriteFchkOrb: number of AOs does not match between MOC and {}.fchk'.format(fchkfile0))
                exit()
            fout.writelines(line)
        # read MO energies
        elif(len(words) > 3 and words[:3] == ['Alpha', 'Orbital', 'Energies']):
            MON0 = int(words[-1])
            for i in np.arange(int((MON0 - 1) / 5) + 1):
                fin.readline()
            
            fout.writelines('{}{:>12d}\n'.format(line[:-13], MON))
            index = 0
            for i in np.arange(int((MON - 1) / 5) + 1):
                line = []
                for j in np.arange(5):
                    if (index < MON):
                        line += '{:16.8E}'.format(MOE[index])
                        index += 1
                    else:
                        break
                line += '\n'
                fout.writelines(line)
        # read MO coefficients
        elif(len(words) > 3 and words[:3] == ['Alpha', 'MO', 'coefficients']):
            for i in np.arange(int((MON0 * AON - 1) / 5) + 1):
                fin.readline()
            
            fout.writelines('{}{:>12d}\n'.format(line[:-13], MON * AON))
            index = 0
            for i in np.arange(int((MON * AON - 1) / 5) + 1):
                line = []
                for j in np.arange(5):
                    if (index < MON * AON):
                        indexAO = index % AON
                        indexMO = index // AON
                        line += '{:16.8E}'.format(MOC[indexAO, indexMO])
                        index += 1
                    else:
                        break
                line += '\n'
                fout.writelines(line)
        else:
            fout.writelines(line)
        line = fin.readline()

    fin.close()
    fout.close()