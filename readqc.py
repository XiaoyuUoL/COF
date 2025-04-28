import numpy as np

import input

## return MO orbital information from fchk file ('fchkfile')
# 'moe': energy, 'moc': coeff, 'aos': overlap and 'aof': Fock matrix
# only for closed-shell systems
def ReadFchk(fchkfile, if_cluster):
    # calculate the power of symmetry matrix
    def MatrixPower(M, x):
        e,v = np.linalg.eigh(M)
        return v @ np.diag(np.power(e, x)) @ v.T

    aon = None  # Number of AO basis
    mon = None  # Number of MOs
    iHOMO = None  # index of HOMO
    moe = []  # Energy of MOs (unit: Hartree)
    moc = []  # Coeff of MOs

    fin = open('{}.fchk'.format(fchkfile), 'r')
    line = fin.readline()
    while (len(line) != 0):
        words = line.rstrip().split()
        # read number of basis functions
        if(len(words) > 4 and words[:4] == ['Number', 'of', 'basis', 'functions']):
            aon = int(words[-1])
        elif(len(words) > 4 and words[:4] == ['Number', 'of', 'alpha', 'electrons']):
            iHOMO = int(words[-1])
        # read MO energies
        elif(len(words) > 3 and words[:3] == ['Alpha', 'Orbital', 'Energies']):
            mon = int(words[-1])
            for i in np.arange(int((mon - 1) / 5) + 1):
                moe += fin.readline().rstrip().split()
        # read MO coefficients
        elif(len(words) > 3 and words[:3] == ['Alpha', 'MO', 'coefficients']):
            for i in np.arange(int((mon * aon - 1) / 5) + 1):
                moc += fin.readline().rstrip().split()
        line = fin.readline()
    fin.close()

    if (aon == None or mon == None or len(moe) == 0 or len(moc) == 0):
        print('no orbital information in {}.fchk'.format(fchkfile))
        exit()

    moe = np.array(moe, dtype=float)
    moc = np.reshape(np.array(moc, dtype=float), (mon, aon)).T

    if if_cluster:
        aos = MatrixPower(np.matmul(moc, moc.T), -1.0)
        aof = aos @ moc @ np.diag(moe) @ moc.T @ aos
        return aos, aof
    else:
        return moe[:iHOMO], moc[:, :iHOMO], moe[iHOMO:], moc[:, :iHOMO]

## write orbital into fchk file ('fchkfile') based on original one ('fchkfile0')
def WriteFchk(fchkfile0, fchkfile, moe, moc):
    # Number of AO basis
    aon = np.shape(moc)[0]
    # Number of MOs
    mon = np.shape(moc)[1]
    if (mon != len(moe)):
        print('WriteFchkOrb: number of MOs does not match between moe and moc')
        exit()
    mon0 = None

    fin = open('{}.fchk'.format(fchkfile0), 'r')
    fout = open('{}.fchk'.format(fchkfile), 'w')
    line = fin.readline()
    while (len(line) != 0):
        words = line.rstrip().split()
        # read number of basis functions
        if(len(words) > 4 and words[:4] == ['Number', 'of', 'basis', 'functions']):
            if (aon != int(words[-1])):
                print('WriteFchkOrb: number of aos does not match between moc and {}.fchk'.format(fchkfile0))
                exit()
            fout.writelines(line)
        # read MO energies
        elif(len(words) > 3 and words[:3] == ['Alpha', 'Orbital', 'Energies']):
            mon0 = int(words[-1])
            for i in np.arange(int((mon0 - 1) / 5) + 1):
                fin.readline()

            fout.writelines('{}{:>12d}\n'.format(line[:-13], mon))
            index = 0
            for i in np.arange(int((mon - 1) / 5) + 1):
                line = []
                for j in np.arange(5):
                    if (index < mon):
                        line += '{:16.8E}'.format(moe[index])
                        index += 1
                    else:
                        break
                line += '\n'
                fout.writelines(line)
        # read MO coefficients
        elif(len(words) > 3 and words[:3] == ['Alpha', 'MO', 'coefficients']):
            for i in np.arange(int((mon0 * aon - 1) / 5) + 1):
                fin.readline()

            fout.writelines('{}{:>12d}\n'.format(line[:-13], mon * aon))
            index = 0
            for i in np.arange(int((mon * aon - 1) / 5) + 1):
                line = []
                for j in np.arange(5):
                    if (index < mon * aon):
                        indexAO = index % aon
                        indexMO = index // aon
                        line += '{:16.8E}'.format(moc[indexAO, indexMO])
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

## return MO orbital information from pyscf npy file ('npyfile')
def ReadNpy(npyfile, if_cluster):
    data = np.load('{}.npy'.format(npyfile), allow_pickle=True).item()

    if if_cluster:
        return data['aos'], data['aof']
    else:
        return data['occe'], data['occc'], data['vire'], data['virc']

def ReadQC(filename, if_cluster=False):
    if input.Package == 'g16':
        return ReadFchk(filename, if_cluster)
    elif input.Package == 'pyscf':
        return ReadNpy(filename, if_cluster)