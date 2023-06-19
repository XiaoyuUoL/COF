import numpy as np
import os

## Work folder
WorkDir = 'COF-701/Fragment1'

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
    rV = np.array([[ 30.0705509185999986, 0.0000000000000000, 0.0000000000000000],
                   [-14.7504731939999996, 26.0116246147000005, 0.0000000000000000],
                   [-2.5083044823999998, -3.8658725720999998, 5.5791483006000000]], dtype=float)

# k-space lattice vector
kV = np.zeros((3, 3), dtype=float)
kV[0, :] = 2. * np.pi / np.dot(rV[0, :], np.cross(rV[1, :], rV[2, :])) * np.cross(rV[1, :], rV[2, :])
kV[1, :] = 2. * np.pi / np.dot(rV[1, :], np.cross(rV[2, :], rV[0, :])) * np.cross(rV[2, :], rV[0, :])
kV[2, :] = 2. * np.pi / np.dot(rV[2, :], np.cross(rV[0, :], rV[1, :])) * np.cross(rV[0, :], rV[1, :])

## decomposition of COF into cores and links (in unit cell)
# number of cores and links in unit cell
CoreNum = 4
LinkNum = 6

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
if (WorkDir[-1] == '0'):
    Connect[0, 0] = [2, 14]
    Connect[0, 1] = [13, 14]
    Connect[0, 2] = [15, 14]
    Connect[1, 0] = [8, 8]
    Connect[1, 1] = [6, 8]
    Connect[1, 2] = [7, 8]
    Connect[2, 3] = [2, 14]
    Connect[2, 4] = [15, 14]
    Connect[2, 5] = [13, 14]
    Connect[3, 3] = [7, 8]
    Connect[3, 4] = [8, 8]
    Connect[3, 5] = [6, 8]
# Fragment1
else:
    Connect[0, 0] = [0, 21]
    Connect[0, 1] = [3, 21]
    Connect[0, 2] = [4, 21]
    Connect[1, 0] = [2, 13]
    Connect[1, 1] = [0, 13]
    Connect[1, 2] = [1, 13]
    Connect[2, 3] = [0, 21]
    Connect[2, 4] = [4, 21]
    Connect[2, 5] = [3, 21]
    Connect[3, 3] = [1, 13]
    Connect[3, 4] = [2, 13]
    Connect[3, 5] = [0, 13]

#PBC = np.array([[[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[0, 0, 0], [ 0, 0, 0], [ 0,  0, 0], [0,  0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]],
#                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [ 0, 0, 0], [ 0,  0, 0], [0,  0, 0]]], dtype=int)
PBC =  np.zeros((CoreNum, LinkNum, 3), dtype=int)
PBC[0, 1] = [-1,  0,  0]
PBC[0, 2] = [-1, -1,  0]
PBC[2, 4] = [-1, -1,  0]
PBC[2, 5] = [ 0, -1,  0]

# indices of clusters [core or link, fragment index, a index, b index, c index]
ClusterIdx = [[['c', 0, 0, 0, 0], ['l', 0, 0, 0, 0], ['l', 1, -1,  0, 0], ['l', 2, -1, -1, 0]],
              [['c', 1, 0, 0, 0], ['l', 0, 0, 0, 0], ['l', 1,  0,  0, 0], ['l', 2,  0,  0, 0]],
              [['c', 2, 0, 0, 0], ['l', 3, 0, 0, 0], ['l', 4, -1, -1, 0], ['l', 5,  0, -1, 0]],
              [['c', 3, 0, 0, 0], ['l', 3, 0, 0, 0], ['l', 4,  0,  0, 0], ['l', 5,  0,  0, 0]],
              [['c', 0, 0, 0, 0], ['c', 2, 0, 0, -1]],
              [['c', 0, 0, 0, 0], ['c', 2, 0, 0,  0]],]

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
MOMode = 'a'  # 'u' for CBM, 'o' for VBM and 'a' for all
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
                          [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                          [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
                          [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                          [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]], dtype=float)  # high symmetry points