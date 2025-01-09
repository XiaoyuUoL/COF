import numpy as np
import os

## Work folder
WorkDir = 'example'

## crystal parameter
# real space lattice vector
if (os.path.isfile('{}/CONTCAR'.format(WorkDir))):
    fin = open('{}/CONTCAR'.format(WorkDir))
    fin.readline()
    scale = float(fin.readline())
    rV = np.zeros((3, 3), dtype=float)
    for i in np.arange(3):
        data = fin.readline().rstrip().split()
        rV[i, :] = np.array(data, dtype=float)
    rV *= scale
    fin.close()
elif (os.path.isfile('{}/POSCAR'.format(WorkDir))):
    fin = open('{}/POSCAR'.format(WorkDir))
    fin.readline()
    scale = float(fin.readline())
    rV = np.zeros((3, 3), dtype=float)
    for i in np.arange(3):
        data = fin.readline().rstrip().split()
        rV[i, :] = np.array(data, dtype=float)
    rV *= scale
    fin.close()
else:
    rV = np.array([[37.5899009704999969, 0.0000000000000000, 0.0000000000000000],
                   [-18.7949504852000011, 32.5538091661999971, 0.0000000000000000],
                   [0.0000000000000000, 0.0000000000000000, 3.6376500130000000]], dtype=float)

# k-space lattice vector
kV = np.zeros((3, 3), dtype=float)
V = np.dot(rV[0, :], np.cross(rV[1, :], rV[2, :]))
kV[0, :] = np.cross(rV[1, :], rV[2, :]) / V
kV[1, :] = np.cross(rV[2, :], rV[0, :]) / V
kV[2, :] = np.cross(rV[0, :], rV[1, :]) / V
kV *= 2. * np.pi

## decomposition of COF into cores and links (in unit cell)
# number of cores and links in unit cell
CoreNum = 2
LinkNum = 3

# connect points (atoms) between core and link
# (consider there are only connects between core and link)
Connect = []
for i in np.arange(CoreNum):
    connect = []
    for j in np.arange(LinkNum):
        connect.append([])
    Connect.append(connect)

Connect[0][0].append([14, 0])
Connect[0][1].append([12, 0])
Connect[0][2].append([13, 0])
Connect[1][0].append([14, 1])
Connect[1][1].append([12, 28])
Connect[1][2].append([13, 27])

PBC = []
for i in np.arange(CoreNum):
    pbc = []
    for j in np.arange(LinkNum):
        tmp = []
        for connect in Connect[i][j]:
            tmp.append([0, 0, 0])
        pbc.append(tmp)
    PBC.append(pbc)

PBC[1][1][0] = [ 0, -1,  0]
PBC[1][2][0] = [ 1,  0,  0]

# number of H atoms in core/link
CoreNH = [0] * CoreNum
LinkNH = [0] * LinkNum

# indices of clusters [core or link, fragment index, a index, b index, c index]
ClusterIdx = []
# consider all connected core-link dimer
for i in np.arange(CoreNum):
    CoreIdx = ['c', i, 0, 0, 0]
    for j in np.arange(LinkNum):
        if (PBC[i][j] != []):
            for pbc in PBC[i][j]:
                LinkIdx = ['l', j] + list(pbc)
                ClusterIdx.append([CoreIdx, LinkIdx])
                CoreNH[i] += 1
                LinkNH[j] += 1

# indices of clusters [core or link, fragment index, a index, b index, c index]
ClusterIdx.append([['c', 0, 0, 0, 0], ['c', 0, 0, 0, 1]])
ClusterIdx.append([['c', 1, 0, 0, 0], ['c', 1, 0, 0, 1]])
ClusterIdx.append([['l', 0, 0, 0, 0], ['l', 0, 0, 0, 1]])
ClusterIdx.append([['l', 1, 0, 0, 0], ['l', 1, 0, 0, 1]])
ClusterIdx.append([['l', 2, 0, 0, 0], ['l', 2, 0, 0, 1]])

# number of cores/links in each cluster
ClusterNCL = []
# number of H atoms in each fragment in each cluster
ClusterNH = []

# X-H (single-) bond length (Angstrom)
XHLength = {'c': 1.07, 'n': 1.00, 'o': 0.96}

## electronic structure information
# g16 calculation options
ProcNum = 8
Memory = 9500
Func = 'b3lyp'
Basis = '6-31g*'

# sbatch options
SBATCH = {'N': '{:d}'.format(1),
          'n': '{:d}'.format(ProcNum),
          't': '72:00:00',
          'p': 'troisi'}

# number of basis function of H element (6-31g* basis set)
HAON = 2

# local FMOs for k-space calculation
MOMode = 'u'  # 'u' for CBM, 'o' for VBM and 'a' for all
CoreMON = 3  # number of FMOs of cores
LinkMON = 1  # number of FMOs of links
if (MOMode == 'u' or MOMode == 'o'):
    CoreTMON = CoreMON * CoreNum
    LinkTMON = LinkMON * LinkNum
else:
    CoreTMON = CoreMON * CoreNum * 2
    LinkTMON = LinkMON * LinkNum * 2
TMON = CoreTMON + LinkTMON

# indices of FMOs for k-space calculation
MONIdx = {}
imon = 0
for i in np.arange(CoreNum):
    key = '{}{}'.format('c', i)
    if (MOMode == 'u' or MOMode == 'o'):
        MONIdx[key] = np.arange(TMON)[imon:imon+CoreMON]
        imon += CoreMON
    else:
        MONIdx[key] = np.arange(TMON)[imon:imon+CoreMON*2]
        imon += CoreMON * 2
for i in np.arange(LinkNum):
    key = '{}{}'.format('l', i)
    if (MOMode == 'u' or MOMode == 'o'):
        MONIdx[key] = np.arange(TMON)[imon:imon+LinkMON]
        imon += LinkMON
    else:
        MONIdx[key] = np.arange(TMON)[imon:imon+LinkMON*2]
        imon += LinkMON * 2

# parameters for DoS calculation
kDoSNum = 15  # sampling of k-space
sigma = 0.01  # brodening (eV)

# parameters for band structure calculation
if (os.path.isfile('{}/KPOINTS'.format(WorkDir))):
    fin = open('{}/KPOINTS'.format(WorkDir))
    fin.readline()
    # sampling of k-path
    kBandNum = int(fin.readline())
    fin.readline()
    fin.readline()
    kHighSymm = []
    line = fin.readline()
    while (len(line) != 0):
        data = line.rstrip().split()
        if (data != ['']):
            kBegin = np.array(data[:3], dtype=float)
            kEnd = np.array(fin.readline().rstrip().split()[:3], dtype=float)
            fin.readline()
        else:
            break
        line = fin.readline()
        kHighSymm.append([kBegin, kEnd])
    fin.close()
    # high symmetry points
    kHighSymm = np.array(kHighSymm)
else:
    kBandNum = 100
    kHighSymm = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                          [[0.5, 0.0, 0.0], [1./3., 1./3., 0.0]],
                          [[1./3., 1./3., 0.0], [0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                          [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]],
                          [[0.5, 0.0, 0.5], [1./3., 1./3., 0.5]],
                          [[1./3., 1./3., 0.5], [0.0, 0.0, 0.5]],
                          [[0.5, 0.0, 0.5], [0.5, 0.0, 0.0]],
                          [[1./3., 1./3., 0.0], [1./3., 1./3., 0.5]]], dtype=float)
