## create g16 input files of cores/links/clusters and corresponding submission
import numpy as np
import os

import input

def XYZRead(XYZFile):
    Name = []
    Coord = []
    fin = open('{}/{}.xyz'.format(input.WorkDir, XYZFile))
    AtomNumber = int(fin.readline().rstrip())
    fin.readline()
    for i in np.arange(AtomNumber):
        data = fin.readline().rstrip().split()
        Name.append(data[0])
        Coord.append([data[1], data[2], data[3]])
    fin.close()

    Coord = np.array(Coord, dtype=float)
    return Name, Coord

def XYZWrite(XYZFile, Name, Coord):
    AtomNumber = len(Name)
    if (len(Coord) != AtomNumber):
        print('error in XYZWrite: mismatch of length between Name and Coord')
        exit()

    fout = open('{}/{}.xyz'.format(input.WorkDir, XYZFile), 'w')
    fout.writelines('{:d}\n'.format(AtomNumber))
    fout.writelines('{}\n'.format(XYZFile))
    for i in np.arange(AtomNumber):
        fout.writelines('{}{:14.7f}{:14.7f}{:14.7f}\n'.format(
            Name[i], Coord[i, 0], Coord[i, 1], Coord[i, 2]))
    fout.close()

def GJFWrite(GJFFile, Name, Coord):
    AtomNumber = len(Name)
    if (len(Coord) != AtomNumber):
        print('error in GJFWrite: mismatch of length between Name and Coord')
        exit()

    # gjf file
    fout = open('{}/{}.gjf'.format(input.WorkDir, GJFFile), 'w')
    fout.writelines('%nprocshared={:d}\n'.format(input.ProcNum))
    fout.writelines('%mem={:d}MW\n'.format(input.Memory))
    fout.writelines('%chk={}.chk\n'.format(GJFFile))
    fout.writelines('#p {} {} nosymm\n'.format(input.Func, input.Basis))
    fout.writelines('\n')
    fout.writelines('{}\n'.format(GJFFile))
    fout.writelines('\n')
    fout.writelines('0 1\n')
    for i in np.arange(AtomNumber):
        fout.writelines('{}{:14.7f}{:14.7f}{:14.7f}\n'.format(
            Name[i], Coord[i, 0], Coord[i, 1], Coord[i, 2]))
    fout.writelines('\n')
    fout.writelines('\n')
    fout.close()

    # sbatch file
    fout = open('{}/{}.sh'.format(input.WorkDir, GJFFile), 'w')
    fout.writelines('#!/bin/bash -l\n')
    fout.writelines('\n')
    for key,value in input.SBATCH.items():
        fout.writelines('#SBATCH -{} {}\n'.format(key, value))
    fout.writelines('\n')
    fout.writelines('module load apps/gaussian/16\n')
    fout.writelines('\n')
    fout.writelines('g16 {}.gjf\n'.format(GJFFile))
    fout.writelines('formchk {}.chk\n'.format(GJFFile))
    fout.close()

    os.system('echo "sbatch {}.sh" >> {}/submit.sh'.format(
        GJFFile, input.WorkDir))

# build cluster based on cluster index
def ClusterBuild(ClusterIdx):
    IdxCore = []
    IdxLink = []
    for index in ClusterIdx:
        if (index[0] == 'c'):
            IdxCore.append(index[1:])
        else:
            IdxLink.append(index[1:])

    name = []
    coord = []
    NCL = [0, 0]
    NH = []
    for index in ClusterIdx:
        if (index[0] == 'c'):
            NCL[0] += 1
            ic = index[1]
            pbcc = index[2:]
            name += CoreName[ic]
            dxyz = np.dot(input.rV.T, pbcc)
            coord += list(CoreCoord[ic] + dxyz)
            nh = 0
            for il in np.arange(input.LinkNum):
                for connect,pbc in zip(input.Connect[ic][il],input.PBC[ic][il]):
                    pbcl = list(np.array(pbcc) + np.array(pbc))
                    if ([il] + pbcl not in IdxLink):
                        nh += 1
                        name += 'H'
                        coordc = CoreCoord[ic][connect[0]]
                        dxyzl = np.dot(input.rV.T, pbc)
                        coordl = LinkCoord[il][connect[1]] + dxyzl
                        BondVect = coordl - coordc
                        BondLength = np.sqrt(np.sum(BondVect * BondVect))
                        XHL = input.XHLength[LinkName[il][connect[1]].lower()]
                        coordH = coordc + XHL / BondLength * BondVect + dxyz
                        coord += [list(coordH)]
        else:
            NCL[1] += 1
            il = index[1]
            pbcl = index[2:]
            name += LinkName[il]
            dxyz = np.dot(input.rV.T, pbcl)
            coord += list(LinkCoord[il] + dxyz)
            nh = 0
            for ic in np.arange(input.CoreNum):
                for connect,pbc in zip(input.Connect[ic][il],input.PBC[ic][il]):
                    pbcc = list(np.array(pbcl) - np.array(pbc))
                    if ([ic] + pbcc not in IdxCore):
                        nh += 1
                        name += 'H'
                        dxyzc = np.dot(input.rV.T, pbc)
                        coordc = CoreCoord[ic][connect[0]] - dxyzc
                        coordl = LinkCoord[il][connect[1]]
                        BondVect = coordc - coordl
                        BondLength = np.sqrt(np.sum(BondVect * BondVect))
                        XHL = input.XHLength[CoreName[ic][connect[0]].lower()]
                        coordH = coordl + XHL / BondLength * BondVect + dxyz
                        coord += [list(coordH)]
        NH.append(nh)

    coord = np.array(coord, dtype=float)

    return name, coord, NCL, NH

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

if (os.path.isfile('{}/submit.sh'.format(input.WorkDir))):
    os.system('rm {}/submit.sh'.format(input.WorkDir))

# write xyz information for cores(-H) and links-(H)
for i in np.arange(input.CoreNum):
    CoreIdx = [['c', i, 0, 0, 0]]
    name,coord,NCL,NH = ClusterBuild(CoreIdx)
    XYZWrite('c{}-H'.format(i), name, coord)
    GJFWrite('c{}-H'.format(i), name, coord)

for i in np.arange(input.LinkNum):
    LinkIdx = [['l', i, 0, 0, 0]]
    name,coord,NCL,NH = ClusterBuild(LinkIdx)
    XYZWrite('l{}-H'.format(i), name, coord)
    GJFWrite('l{}-H'.format(i), name, coord)

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

# write xyz information for clusters
for i,indices in enumerate(input.ClusterIdx):
    name,coord,NCL,NH = ClusterBuild(indices)
    XYZWrite('cluster{}-H'.format(i), name, coord)
    GJFWrite('cluster{}-H'.format(i), name, coord)
    input.ClusterNCL.append(NCL)
    input.ClusterNH.append(NH)
#print(input.ClusterNCL)
#print(input.ClusterNH)
#print()

os.system('chmod 755 {}/submit.sh'.format(input.WorkDir))