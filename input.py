import numpy as np
import os

## Work folder
WorkDir = 'test'

## crystal parameter
# real space lattice vector
if os.path.isfile('{}/CONTCAR'.format(WorkDir)):
    fin = open('{}/CONTCAR'.format(WorkDir))
    fin.readline()
    scale = float(fin.readline())
    rV = np.zeros((3, 3), dtype=float)
    for i in np.arange(3):
        data = fin.readline().rstrip().split()
        rV[i, :] = np.array(data, dtype=float)
    rV *= scale
    fin.close()
elif os.path.isfile('{}/POSCAR'.format(WorkDir)):
    fin = open('{}/POSCAR'.format(WorkDir))
    fin.readline()
    scale = float(fin.readline())
    rV = np.zeros((3, 3), dtype=float)
    for i in np.arange(3):
        data = fin.readline().rstrip().split()
        rV[i, :] = np.array(data, dtype=float)
    rV *= scale
    fin.close()
#else:
#    rV = np.array([[14.5740003585999993,0.0000000000000000,0.0000000000000000],
#                   [-7.2870001792999997,12.6214545453000007,0.0000000000000000],
#                   [0.0000000000000000,0.0000000000000000,3.4040000439000000]], dtype=float)

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
Connect[0][0].append([0, 8])
Connect[0][1].append([4, 0])
Connect[0][2].append([3, 0])
Connect[1][0].append([4, 0])
Connect[1][1].append([3, 7])
Connect[1][2].append([0, 8])

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
        if PBC[i][j] != []:
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
Package = 'pyscf' # 'pyscf' or 'g16'
ProcNum = 8
Memory = 9500
Func = 'b3lyp'
Basis = '6-31g*'

# number of basis function of H element 
HAON = 2 # 6-31g* basis set

# define working (QC) directory
QCDir = ''
if Package == 'g16':
    if not os.path.isfile('{}/g16.sh'.format(WorkDir)):
        print('error: g16.sh is not found for submitting g16 job')
        exit()
    QCDir = WorkDir + '/g16'
elif Package == 'pyscf':
    QCDir = WorkDir + '/pyscf'
else:
    print('error in input.py: unknown package')
    exit()

# local FMOs for k-space calculation
MOMode = 'o'  # 'u' for CBM, 'o' for VBM
CoreMON = 6  # number of FMOs of cores
LinkMON = 8  # number of FMOs of links
CoreTMON = CoreMON * CoreNum
LinkTMON = LinkMON * LinkNum
TMON = CoreTMON + LinkTMON

# indices of FMOs for k-space calculation
MONIdx = {}
imon = 0
for i in np.arange(CoreNum):
    key = '{}{}'.format('c', i)
    MONIdx[key] = np.arange(TMON)[imon:imon+CoreMON]
    imon += CoreMON
for i in np.arange(LinkNum):
    key = '{}{}'.format('l', i)
    MONIdx[key] = np.arange(TMON)[imon:imon+LinkMON]
    imon += LinkMON

# parameters for DoS calculation
kDoSNum = 15  # sampling of k-space
sigma = 0.01  # brodening (eV)

# parameters for band structure calculation
if os.path.isfile('{}/KPOINTS'.format(WorkDir)):
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
        if data != ['']:
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
#else:
#    kBandNum = 20
#    kHighSymm = np.array([[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
#                          [[0.0, 0.5, 0.0], [0.0, 0.5, 0.5]],
#                          [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]],
#                          [[0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
#                          [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5]],
#                          [[-0.5, 0.0, 0.5], [-0.5, 0.5, 0.5]],
#                          [[-0.5, 0.5, 0.5], [0.0, 0.5, 0.0]],
#                          [[0.0, 0.5, 0.0], [-0.5, 0.5, 0.0]],
#                          [[-0.5, 0.5, 0.0], [-0.5, 0.0, 0.0]],
#                          [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=float)