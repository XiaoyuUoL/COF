import numpy as np

import OrbLocalize
import fchk

np.set_printoptions(precision=3, suppress=True)

MOE,MOC,AOS,IHOMO = fchk.ReadFchkOrb('COF-701/Fragment0/cluster0-H')
tmp = np.matmul(AOS, MOC)
AOF = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
fchk.WriteFchkOrb('COF-701/Fragment0/cluster0-H', 'COF-701/Fragment0/test0', MOE, MOC)

MolIdx = np.array([[0, 196], [196, 394]])

for i in np.arange(10):
    OrbLocalize.PipekMezey(AOS, MOC[:, :IHOMO], MolIdx)
    MOC0 = np.zeros_like(MOC[:, :IHOMO])
    MOC0[:196, :] = MOC[:196, :IHOMO]
    print(np.diag(np.matmul(MOC0.T, np.matmul(AOS, MOC0))))
    MOC1 = np.zeros_like(MOC[:, :IHOMO])
    MOC1[196:, :] = MOC[196:, :IHOMO]
    print(np.diag(np.matmul(MOC1.T, np.matmul(AOS, MOC1))))
    print()
fchk.WriteFchkOrb('COF-701/Fragment0/cluster0-H', 'COF-701/Fragment0/test', MOE, MOC)