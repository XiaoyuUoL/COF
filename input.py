import numpy as np
import os

## Work folder
WorkDir = 'COF-366/1/Fragment0'

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
    rV = np.array([[38.9482440969893489, 0.0118690278767904, -0.2128707946153858],
                   [-0.0498951423867981, 33.1187748841537868, 0.0384065281382595],
                   [-0.3440320652054656, -0.0399050908279700, 8.9554762976636670]], dtype=float)

# k-space lattice vector
kV = np.zeros((3, 3), dtype=float)
V = np.dot(rV[0, :], np.cross(rV[1, :], rV[2, :]))
kV[0, :] = np.cross(rV[1, :], rV[2, :]) / V
kV[1, :] = np.cross(rV[2, :], rV[0, :]) / V
kV[2, :] = np.cross(rV[0, :], rV[1, :]) / V
kV *= 2. * np.pi

## decomposition of COF into cores and links (in unit cell)
# number of cores and links in unit cell
CoreNum = 4
LinkNum = 8

# connect points (atoms) between core and link
# (consider there are only connects between core and link)
Connect = np.full((CoreNum, LinkNum, 2), -1, dtype=int)
# Fragment0
if (WorkDir[-1] == '0'):
    Connect[0, 0] = [1, 8]
    Connect[0, 1] = [15, 19]
    Connect[0, 2] = [30, 8]
    Connect[0, 3] = [16, 19]
    Connect[1, 0] = [29, 20]
    Connect[1, 1] = [2, 8]
    Connect[1, 2] = [28, 19]
    Connect[1, 3] = [3, 8]
    Connect[2, 4] = [1, 18]
    Connect[2, 5] = [16, 8]
    Connect[2, 6] = [29, 18]
    Connect[2, 7] = [15, 8]
    Connect[3, 4] = [2, 8]
    Connect[3, 5] = [12, 18]
    Connect[3, 6] = [3, 8]
    Connect[3, 7] = [13, 18]
# Fragment1
else:
    Connect[0, 0] = [5, 8]
    Connect[0, 1] = [54, 8]
    Connect[0, 2] = [70, 7]
    Connect[0, 3] = [59, 8]
    Connect[1, 0] = [41, 9]
    Connect[1, 1] = [14, 7]
    Connect[1, 2] = [40, 8]
    Connect[1, 3] = [15, 7]
    Connect[2, 4] = [7, 15]
    Connect[2, 5] = [59, 14]
    Connect[2, 6] = [70, 15]
    Connect[2, 7] = [55, 14]
    Connect[3, 4] = [14, 14]
    Connect[3, 5] = [36, 15]
    Connect[3, 6] = [15, 14]
    Connect[3, 7] = [37, 15]

PBC =  np.zeros((CoreNum, LinkNum, 3), dtype=int)
PBC[0, 1] = [-1,  0,  0]
PBC[0, 2] = [-1, -1,  0]
PBC[0, 3] = [ 0, -1,  0]
PBC[2, 5] = [-1,  0,  0]
PBC[2, 6] = [-1, -1,  0]
PBC[2, 7] = [ 0, -1,  0]

# number of H atoms in core/link
CoreNH = []
LinkNH = []

# indices of clusters [core or link, fragment index, a index, b index, c index]
ClusterIdx = []
# consider all connected core-link dimer
for i in np.arange(CoreNum):
    CoreIdx = ['c', i, 0, 0, 0]
    for j in np.arange(LinkNum):
        if (Connect[i, j, 0] != -1):
            LinkIdx = ['l', j] + list(PBC[i, j])
            ClusterIdx.append([CoreIdx, LinkIdx])

# indices of clusters [core or link, fragment index, a index, b index, c index]
ClusterIdx.append([['c', 0, 0, 0, 0], ['c', 2, 0, 0, -1]])
ClusterIdx.append([['c', 0, 0, 0, 0], ['c', 2, 0, 0,  0]])
ClusterIdx.append([['c', 1, 0, 0, 0], ['c', 3, 0, 0, -1]])
ClusterIdx.append([['c', 1, 0, 0, 0], ['c', 3, 0, 0,  0]])
ClusterIdx.append([['l', 0, 0, 0, 0], ['l', 4, 0, 0, -1]])
ClusterIdx.append([['l', 0, 0, 0, 0], ['l', 4, 0, 0,  0]])
ClusterIdx.append([['l', 1, 0, 0, 0], ['l', 5, 0, 0, -1]])
ClusterIdx.append([['l', 1, 0, 0, 0], ['l', 5, 0, 0,  0]])
ClusterIdx.append([['l', 2, 0, 0, 0], ['l', 6, 0, 0, -1]])
ClusterIdx.append([['l', 2, 0, 0, 0], ['l', 6, 0, 0,  0]])
ClusterIdx.append([['l', 3, 0, 0, 0], ['l', 7, 0, 0, -1]])
ClusterIdx.append([['l', 3, 0, 0, 0], ['l', 7, 0, 0,  0]])

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
CoreMON = 2  # number of FMOs of cores
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
                          [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
                          [[0.5, 0.5, 0.0], [0.0, 0.5, 0.0]],
                          [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                          [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]],
                          [[0.5, 0.0, 0.5], [0.5, 0.5, 0.5]],
                          [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5]],
                          [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]]], dtype=float)