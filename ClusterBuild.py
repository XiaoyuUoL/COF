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
    name = []
    coord = []
    NCL = [0, 0]
    NH = []
    for index in ClusterIdx:
        if (index[0] == 'c'):
            NCL[0] += 1
            name += CoreName[index[1]]
            dxyz = np.dot(input.rV.T, index[2:])
            coord += list(CoreCoord[index[1]] + dxyz)
            nh = input.CoreNH[index[1]]
            mark = [True] * input.LinkNum
            for indexp in ClusterIdx:
                if (indexp[0] == 'l'):
                    pbc0 = list(input.PBC[index[1], indexp[1]])
                    pbc1 = list(np.array(indexp[2:]) - np.array(index[2:]))
                    if (pbc0 == pbc1):
                        mark[indexp[1]] = False
            for i in np.arange(input.LinkNum):
                if (mark[i] and CoreNameH[index[1]][i] != ''):
                    name += CoreNameH[index[1]][i]
                    coord.append(CoreCoordH[index[1]][i] + dxyz)
                    nh -= 1
        else:
            NCL[1] += 1
            name += LinkName[index[1]]
            dxyz = np.dot(input.rV.T, index[2:])
            coord += list(LinkCoord[index[1]] + dxyz)
            mark = [True] * input.CoreNum
            nh = input.LinkNH[index[1]]
            for indexp in ClusterIdx:
                if (indexp[0] == 'c'):
                    pbc0 = list(input.PBC[indexp[1], index[1]])
                    pbc1 = list(np.array(index[2:]) - np.array(indexp[2:]))
                    if (pbc0 == pbc1):
                        mark[indexp[1]] = False
            for i in np.arange(input.CoreNum):
                if (mark[i] and LinkNameH[index[1]][i] != ''):
                    name += LinkNameH[index[1]][i]
                    coord.append(LinkCoordH[index[1]][i] + dxyz)
                    nh -= 1
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
CoreNameH = []
CoreCoordH = []
input.CoreNH = []
for i in np.arange(input.CoreNum):
    name = CoreName[i][:]
    coord = CoreCoord[i][:]
    nameH = [''] * input.LinkNum
    coordH = [0] * input.LinkNum
    nh = 0
    # add H based on connection info
    for j in np.arange(input.LinkNum):
        connect = input.Connect[i, j]
        PBC = np.dot(input.rV.T, input.PBC[i, j])
        if (connect[0] == -1):
            nameH[j] = ''
            coordH[j] = []
            continue
        nameH[j] = 'H'
        name.append(nameH[j])
        BondVect = LinkCoord[j][connect[1]] + PBC - coord[connect[0]]
        BondLength = np.sqrt(np.sum(BondVect * BondVect))
        XHL = input.XHLength[LinkName[j][connect[1]].lower()]
        coordH[j] = coord[connect[0]] + XHL / BondLength * BondVect
        coord = np.append(coord, [coordH[j]], axis=0)
        nh += 1
    XYZWrite('c{}-H'.format(i), name, coord)
    GJFWrite('c{}-H'.format(i), name, coord)
    CoreNameH.append(nameH)
    CoreCoordH.append(coordH)
    input.CoreNH.append(nh)

LinkNameH = []
LinkCoordH = []
input.LinkNH = []
for i in np.arange(input.LinkNum):
    name = LinkName[i][:]
    coord = LinkCoord[i][:]
    nameH = [''] * input.CoreNum
    coordH = [0] * input.CoreNum
    nh = 0
    # add H based on connection info
    for j in np.arange(input.CoreNum):
        connect = input.Connect[j, i]
        PBC = np.dot(input.rV.T, input.PBC[j, i])
        if (connect[0] == -1):
            nameH[j] = ''
            coordH[j] = []
            continue
        nameH[j] = 'H'
        name.append(nameH[j])
        BondVect = CoreCoord[j][connect[0]] - coord[connect[1]] - PBC
        BondLength = np.sqrt(np.sum(BondVect * BondVect))
        XHL = input.XHLength[CoreName[j][connect[0]].lower()]
        coordH[j] = coord[connect[1]] + XHL / BondLength * BondVect
        coord = np.append(coord, [coordH[j]], axis=0)
        nh += 1
    XYZWrite('l{}-H'.format(i), name, coord)
    GJFWrite('l{}-H'.format(i), name, coord)
    LinkNameH.append(nameH)
    LinkCoordH.append(coordH)
    input.LinkNH.append(nh)

# write xyz information for cores(-links) and links(-cores)
for i in np.arange(input.CoreNum):
    ClusterIdx = [['c', i, 0, 0, 0]]
    for j in np.arange(input.LinkNum):
        if (input.Connect[i, j, 0] != -1):
            ClusterIdx.append(['l', j] + list(input.PBC[i, j]))
    name,coord,NCL,NH = ClusterBuild(ClusterIdx)
    XYZWrite('c{}-l'.format(i), name, coord)
    GJFWrite('c{}-l'.format(i), name, coord)

for i in np.arange(input.LinkNum):
    ClusterIdx = [['l', i, 0, 0, 0]]
    for j in np.arange(input.CoreNum):
        if (input.Connect[j, i, 1] != -1):
            ClusterIdx.append(['c', j] + list(-input.PBC[j, i]))
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
#print(input.CoreNH)
#print(input.LinkNH)
#print(input.ClusterNH)
#print()

os.system('chmod 755 {}/submit.sh'.format(input.WorkDir))