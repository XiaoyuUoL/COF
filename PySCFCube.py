import numpy as np
from pyscf import gto, tools
import sys

import input

# create cube files for core and link
# SysName should be 'cx-H' or 'lx-H'
# i.e. python PySCFCube cx-H
SysName = sys.argv[1]
mol = gto.Mole()
mol.atom = '{}/pyscf/{}.xyz'.format(input.WorkDir, SysName)
mol.basis = input.Basis
mol.build()

data = np.load('{}/pyscf/{}.npy'.format(input.WorkDir, SysName), allow_pickle=True).item()

occc = data['occc']
for i in range(occc.shape[1]):
    CubeFile = '{}/pyscf/cube/{}-occ-{}.cube'.format(input.WorkDir, SysName, i)
    tools.cubegen.orbital(mol, CubeFile, occc[:,i])

virc = data['virc']
for i in range(virc.shape[1]):
    CubeFile = '{}/pyscf/cube/{}-vir-{}.cube'.format(input.WorkDir, SysName, i)
    tools.cubegen.orbital(mol, CubeFile, virc[:,i])