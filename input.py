import numpy as np
import os

## Work folder
WorkDir = '1/Fragment0'

## crystal parameter
# real space lattice vector
if (os.path.isfile('CONTCAR')):
    fin = open('CONTCAR')
    fin.readline()
    scale = float(fin.readline())
    rV = np.zeros((3, 3), dtype=float)
    for i in np.arange(3):
        data = fin.readline().rstrip().split()
        rV[i, :] = np.array(data, dtype=float)
    rV *= scale
    fin.close()
elif (os.path.isfile('POSCAR')):
    fin = open('POSCAR')
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
kV[0, :] = 2. * np.pi / np.dot(rV[0, :], np.cross(rV[1, :], rV[2, :])) * np.cross(rV[1, :], rV[2, :])
kV[1, :] = 2. * np.pi / np.dot(rV[1, :], np.cross(rV[2, :], rV[0, :])) * np.cross(rV[2, :], rV[0, :])
kV[2, :] = 2. * np.pi / np.dot(rV[2, :], np.cross(rV[0, :], rV[1, :])) * np.cross(rV[0, :], rV[1, :])

## decomposition of COF into cores and links (in unit cell)
# number of cores and links in unit cell
CoreNum = 4
LinkNum = 8

# connect points (atoms) between core and link (consider there are only connects between core and link)
# if there is no connect between Core i and Link j, connect[i][j] = [-1, -1]
#Connect = np.array([[[1, 8], [15, 19], [30, 8], [16, 19], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
#                   [[29, 20], [2, 8], [28, 19], [3, 8], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
#                   [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 18], [16, 8], [29, 18], [15, 8]],  
#                   [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [2, 8], [12, 18], [3, 8], [13, 18]]], dtype=int)
#Connect = np.array([[[5, 8], [54, 8], [70, 7], [59, 8], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
#                   [[41, 9], [14, 7], [40, 8], [15, 7], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
#                   [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [7, 15], [59, 14], [70, 15], [55, 14]],
#                   [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [14, 14], [36, 15], [15, 14], [37, 15]]], dtype=int)
Connect = np.full((CoreNum, LinkNum, 2), -1, dtype=int)
# Fragment0
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
## Fragment1
#Connect[0, 0] = [5, 8]
#Connect[0, 1] = [54, 8]
#Connect[0, 2] = [70, 7]
#Connect[0, 3] = [59, 8]
#Connect[1, 0] = [41, 9]
#Connect[1, 1] = [14, 7]
#Connect[1, 2] = [40, 8]
#Connect[1, 3] = [15, 7]
#Connect[2, 4] = [7, 15]
#Connect[2, 5] = [59, 14]
#Connect[2, 6] = [70, 15]
#Connect[2, 7] = [55, 14]
#Connect[3, 4] = [14, 14]
#Connect[3, 5] = [36, 15]
#Connect[3, 6] = [15, 14]
#Connect[3, 7] = [37, 15]

#PBC = np.array([[[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[0, 0, 0], [ 0, 0, 0], [ 0,  0, 0], [0,  0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [ 0, 0, 0], [ 0,  0, 0], [0,  0, 0]]], dtype=int)
PBC =  np.zeros((CoreNum, LinkNum, 3), dtype=int)
PBC[0, 1] = [-1,  0,  0]
PBC[0, 2] = [-1, -1,  0]
PBC[0, 3] = [ 0, -1,  0]
PBC[2, 5] = [-1,  0,  0]
PBC[2, 6] = [-1, -1,  0]
PBC[2, 7] = [ 0, -1,  0]

# indices of clusters [core or link, fragment index, a index, b index, c index]
ClusterIdx = [[['c', 0, 0, 0, 0], ['l', 0, 0, 0, 0], ['l', 1, -1, 0, 0], ['l', 2, -1, -1, 0], ['l', 3, 0, -1, 0]],
              [['c', 1, 0, 0, 0], ['l', 0, 0, 0, 0], ['l', 1,  0, 0, 0], ['l', 2,  0,  0, 0], ['l', 3, 0,  0, 0]],
              [['c', 2, 0, 0, 0], ['l', 4, 0, 0, 0], ['l', 5, -1, 0, 0], ['l', 6, -1, -1, 0], ['l', 7, 0, -1, 0]],
              [['c', 3, 0, 0, 0], ['l', 4, 0, 0, 0], ['l', 5,  0, 0, 0], ['l', 6,  0,  0, 0], ['l', 7, 0,  0, 0]],
              [['c', 0, 0, 0, 0], ['c', 2, 0, 0, -1]],
              [['c', 0, 0, 0, 0], ['c', 2, 0, 0,  0]],
              [['c', 1, 0, 0, 0], ['c', 3, 0, 0, -1]],
              [['c', 1, 0, 0, 0], ['c', 3, 0, 0,  0]],
              [['l', 0, 0, 0, 0], ['l', 4, 0, 0, -1]],
              [['l', 0, 0, 0, 0], ['l', 4, 0, 0,  0]],
              [['l', 1, 0, 0, 0], ['l', 5, 0, 0, -1]],
              [['l', 1, 0, 0, 0], ['l', 5, 0, 0,  0]],
              [['l', 2, 0, 0, 0], ['l', 6, 0, 0, -1]],
              [['l', 2, 0, 0, 0], ['l', 6, 0, 0,  0]],
              [['l', 3, 0, 0, 0], ['l', 7, 0, 0, -1]],
              [['l', 3, 0, 0, 0], ['l', 7, 0, 0,  0]]]

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
CoreMON = 2  # number of FMOs of cores
LinkMON = 1  # number of FMOs of links
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
if (os.path.isfile('KPOINTS')):
    fin = open('KPOINTS')
    fin.readline()
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
    kHighSymm = np.array(kHighSymm)
else:
    kBandNum = 100  # sampling of k-path
    kHighSymm = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                          [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
                          [[0.5, 0.5, 0.0], [0.0, 0.5, 0.0]],
                          [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                          [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5]],
                          [[0.5, 0.0, 0.5], [0.5, 0.5, 0.5]],
                          [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5]],
                          [[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]]], dtype=float)  # high symmetry points