## create g16/pyscf input files of cores/links/clusters and corresponding submission
import numpy as np
import os

import input

def XYZRead(FileName):
    Name = []
    Coord = []
    fin = open('{}/{}.xyz'.format(input.WorkDir, FileName))
    AtomNumber = int(fin.readline().rstrip())
    fin.readline()
    for i in np.arange(AtomNumber):
        data = fin.readline().rstrip().split()
        Name.append(data[0])
        Coord.append([data[1], data[2], data[3]])
    fin.close()

    Coord = np.array(Coord, dtype=float)
    return Name, Coord

# read xyz information for cores and links
CoreName = []
CoreCoord = []
for i in np.arange(input.CoreNum):
    name,coord = XYZRead('c{}'.format(i))
    CoreName.append(name)
    CoreCoord.append(coord)
LinkName = []
LinkCoord = []
for i in np.arange(input.LinkNum):
    name,coord = XYZRead('l{}'.format(i))
    LinkName.append(name)
    LinkCoord.append(coord)

def prepare():
    def XYZWrite(FileName, Name, Coord):
        AtomNumber = len(Name)
        if len(Coord) != AtomNumber:
            print('error in XYZWrite: mismatch of length between Name and Coord')
            exit()

        fout = open('{}/{}.xyz'.format(input.QCDir, FileName), 'w')
        fout.writelines('{:d}\n'.format(AtomNumber))
        fout.writelines('{}\n'.format(FileName))
        for i in np.arange(AtomNumber):
            fout.writelines('{}{:14.7f}{:14.7f}{:14.7f}\n'.format(
                Name[i], Coord[i, 0], Coord[i, 1], Coord[i, 2]))
        fout.close()

    def GJFWrite(FileName, Name, Coord):
        AtomNumber = len(Name)
        if len(Coord) != AtomNumber:
            print('error in GJFWrite: mismatch of length between Name and Coord')
            exit()

        # gjf file
        fout = open('{}/{}.gjf'.format(input.QCDir, FileName), 'w')
        fout.writelines('%nprocshared={:d}\n'.format(input.ProcNum))
        fout.writelines('%mem={:d}MW\n'.format(input.Memory))
        fout.writelines('%chk={}.chk\n'.format(FileName))
        fout.writelines('#p {} {} nosymm\n'.format(input.Func, input.Basis))
        fout.writelines('\n')
        fout.writelines('{}\n'.format(FileName))
        fout.writelines('\n')
        fout.writelines('0 1\n')
        for i in np.arange(AtomNumber):
            fout.writelines('{}{:14.7f}{:14.7f}{:14.7f}\n'.format(
                Name[i], Coord[i, 0], Coord[i, 1], Coord[i, 2]))
        fout.writelines('\n')
        fout.writelines('\n')
        fout.close()

        # running file (based on g16.sh)
        os.system('cp {}/g16.sh {}/{}.sh'.format(input.WorkDir, input.QCDir, FileName))
        os.system('echo "g16 {}.gjf" >> {}/{}.sh'.format(FileName, input.QCDir, FileName))
        os.system('echo "formchk {}.chk" >> {}/{}.sh'.format(FileName, input.QCDir, FileName))

        # command for running QC calculation
        os.system('echo "./{}.sh" >> {}/run.sh'.format(FileName, input.QCDir))

    def RunPySCF(FileName, aon_h=0, if_cluster=False, s_threshold=0.2):
        from pyscf import gto, dft, lo

        mol = gto.Mole()
        mol.atom = '{}/{}.xyz'.format(input.QCDir, FileName)
        mol.basis = input.Basis
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = input.Func
        mf.kernel()
        aof = mf.get_fock()

        if if_cluster:
            aos = mol.intor('int1e_ovlp')
            # save overlap and Fock matrix of cluster
            NpyFile = '{}/{}.npy'.format(input.QCDir, FileName)
            np.save(NpyFile, {'aos': aos, 'aof': aof})
        else:
            aos = mol.intor('int1e_ovlp')[-aon_h:, -aon_h:]

            # create local occupied MO using IBO
            moc_occ = mf.mo_coeff[:, :mol.nelec[0]]
            iboc0 = lo.ibo.ibo(mol, moc_occ, locmethod='IBO')
            s = np.diag(iboc0[-aon_h:, :].T @ aos @ iboc0[-aon_h:, :])
            iboc = iboc0[:, s < s_threshold]
            ibof = iboc.T @ aof @ iboc
            iboe,v = np.linalg.eigh(ibof)
            iboc = iboc @ v
            for i in range(iboc.shape[1]):
                print('{} occ {}: '.format(FileName, i), iboe[i])
            #    CubeFile = '{}/cube/{}-occ-{}.cube'.format(input.QCDir, FileName, i)
            #    tools.cubegen.orbital(mol, CubeFile, iboc[:,i])

            # create local unoccupied MO using LIVVO
            moc_vir = mf.mo_coeff[:, mol.nelec[0]:]
            livvoc0 = lo.vvo.livvo(mol, moc_occ, moc_vir)
            s = np.diag(livvoc0[-aon_h:, :].T @ aos @ livvoc0[-aon_h:, :])
            livvoc = livvoc0[:, s < s_threshold]
            livvof = livvoc.T @ aof @ livvoc
            livvoe,v = np.linalg.eigh(livvof)
            livvoc = livvoc @ v
            for i in range(livvoc.shape[1]):
                print('{} vir {}: '.format(FileName, i), livvoe[i])
            #    CubeFile = '{}/cube/{}-vir-{}.cube'.format(input.QCDir, FileName, i)
            #    tools.cubegen.orbital(mol, CubeFile, livvoc[:,i])

            # save occ/unoccupied orbital info
            NpyFile = '{}/{}.npy'.format(input.QCDir, FileName)
            np.save(NpyFile, {'occe': iboe, 'occc': iboc, 'vire': livvoe, 'virc': livvoc})

    # build cluster based on cluster index
    def ClusterBuild(ClusterIdx):
        IdxCore = []
        IdxLink = []
        for index in ClusterIdx:
            if index[0] == 'c':
                IdxCore.append(index[1:])
            else:
                IdxLink.append(index[1:])

        name = []
        coord = []
        for index in ClusterIdx:
            if index[0] == 'c':
                ic = index[1]
                pbcc = index[2:]
                name += CoreName[ic]
                dxyz = np.dot(input.rV.T, pbcc)
                coord += list(CoreCoord[ic] + dxyz)
                for il in np.arange(input.LinkNum):
                    for connect,pbc in zip(input.Connect[ic][il],input.PBC[ic][il]):
                        pbcl = list(np.array(pbcc) + np.array(pbc))
                        if [il] + pbcl not in IdxLink:
                            name += 'H'
                            coordc = CoreCoord[ic][connect[0]]
                            dxyzl = np.dot(input.rV.T, pbc)
                            coordl = LinkCoord[il][connect[1]] + dxyzl
                            BondVect = coordl - coordc
                            BondLength = np.sqrt(np.sum(BondVect * BondVect))
                            XHL = input.XHLength[CoreName[ic][connect[0]].lower()]
                            coordH = coordc + XHL / BondLength * BondVect + dxyz
                            coord += [list(coordH)]
            else:
                il = index[1]
                pbcl = index[2:]
                name += LinkName[il]
                dxyz = np.dot(input.rV.T, pbcl)
                coord += list(LinkCoord[il] + dxyz)
                nh = 0
                for ic in np.arange(input.CoreNum):
                    for connect,pbc in zip(input.Connect[ic][il],input.PBC[ic][il]):
                        pbcc = list(np.array(pbcl) - np.array(pbc))
                        if [ic] + pbcc not in IdxCore:
                            name += 'H'
                            dxyzc = np.dot(input.rV.T, pbc)
                            coordc = CoreCoord[ic][connect[0]] - dxyzc
                            coordl = LinkCoord[il][connect[1]]
                            BondVect = coordc - coordl
                            BondLength = np.sqrt(np.sum(BondVect * BondVect))
                            XHL = input.XHLength[LinkName[il][connect[1]].lower()]
                            coordH = coordl + XHL / BondLength * BondVect + dxyz
                            coord += [list(coordH)]

        coord = np.array(coord, dtype=float)

        return name, coord#, NCL, NH

    # prepare input files for QC calculation
    if not os.path.isdir(input.QCDir):
        os.system('mkdir {}'.format(input.QCDir))
    else:
        os.system('rm -rf {}/*'.format(input.QCDir))

    if os.path.isfile('{}/run.sh'.format(input.QCDir)):
        os.system('rm {}/run.sh'.format(input.QCDir))

    if input.Package == 'g16':
        # write xyz information for cores(-H) and links-(H)
        for i in np.arange(input.CoreNum):
            CoreIdx = [['c', i, 0, 0, 0]]
            name,coord = ClusterBuild(CoreIdx)
            XYZWrite('c{}-H'.format(i), name, coord)
            GJFWrite('c{}-H'.format(i), name, coord)

        for i in np.arange(input.LinkNum):
            LinkIdx = [['l', i, 0, 0, 0]]
            name,coord = ClusterBuild(LinkIdx)
            XYZWrite('l{}-H'.format(i), name, coord)
            GJFWrite('l{}-H'.format(i), name, coord)

        '''
        # write xyz information for cores(-links) and links(-cores)
        for i in np.arange(input.CoreNum):
            ClusterIdx = [['c', i, 0, 0, 0]]
            for j in np.arange(input.LinkNum):
                for pbc in input.PBC[i][j]:
                    ClusterIdx.append(['l', j] + pbc)
            name,coord,NCL,NH = ClusterBuild(ClusterIdx)
            XYZWrite('c{}-l'.format(i), name, coord)
            GJFWrite('c{}-l'.format(i), name, coord)
        
        for i in np.arange(input.LinkNum):
            ClusterIdx = [['l', i, 0, 0, 0]]
            for j in np.arange(input.CoreNum):
                for pbc in input.PBC[j][i]:
                    ClusterIdx.append(['c', j] + list(-np.array(pbc)))
            name,coord,NCL,NH = ClusterBuild(ClusterIdx)
            XYZWrite('l{}-c'.format(i), name, coord)
            GJFWrite('l{}-c'.format(i), name, coord)
        '''

        # write xyz information for clusters
        for i,indices in enumerate(input.ClusterIdx):
            name,coord = ClusterBuild(indices)
            XYZWrite('cluster{}-H'.format(i), name, coord)
            GJFWrite('cluster{}-H'.format(i), name, coord)

        os.system('chmod 755 {}/run.sh'.format(input.QCDir))

    elif input.Package == 'pyscf':
        if not os.path.isdir('{}/cube'.format(input.QCDir)):
            os.system('mkdir {}/cube'.format(input.QCDir))
        else:
            os.system('rm -rf {}/cube/*'.format(input.QCDir))

        # run pyscf calculation for cores(-H) and links-(H)
        for i in np.arange(input.CoreNum):
            CoreIdx = [['c', i, 0, 0, 0]]
            name,coord = ClusterBuild(CoreIdx)
            XYZWrite('c{}-H'.format(i), name, coord)
            RunPySCF('c{}-H'.format(i), aon_h=input.HAON * input.CoreNH[i])

        for i in np.arange(input.LinkNum):
            LinkIdx = [['l', i, 0, 0, 0]]
            name,coord = ClusterBuild(LinkIdx)
            XYZWrite('l{}-H'.format(i), name, coord)
            RunPySCF('l{}-H'.format(i), aon_h=input.HAON * input.LinkNH[i])

        # run pyscf calculation for clusters
        for i,indices in enumerate(input.ClusterIdx):
            name,coord = ClusterBuild(indices)
            XYZWrite('cluster{}-H'.format(i), name, coord)
            RunPySCF('cluster{}-H'.format(i), if_cluster=True)

    else:
        print('error in input.py: unknown package')
        exit()

for i,indices in enumerate(input.ClusterIdx):
    IdxCore = []
    IdxLink = []
    for index in indices:
        if index[0] == 'c':
            IdxCore.append(index[1:])
        else:
            IdxLink.append(index[1:])
    NCL = [0, 0]
    NH = []
    for index in indices:
        if index[0] == 'c':
            NCL[0] += 1
            ic = index[1]
            pbcc = index[2:]
            nh = 0
            for il in np.arange(input.LinkNum):
                for connect,pbc in zip(input.Connect[ic][il],input.PBC[ic][il]):
                    pbcl = list(np.array(pbcc) + np.array(pbc))
                    if [il] + pbcl not in IdxLink:
                        nh += 1
        else:
            NCL[1] += 1
            il = index[1]
            pbcl = index[2:]
            nh = 0
            for ic in np.arange(input.CoreNum):
                for connect,pbc in zip(input.Connect[ic][il],input.PBC[ic][il]):
                    pbcc = list(np.array(pbcl) - np.array(pbc))
                    if [ic] + pbcc not in IdxCore:
                        nh += 1
        NH.append(nh)
    input.ClusterNCL.append(NCL)
    input.ClusterNH.append(NH)