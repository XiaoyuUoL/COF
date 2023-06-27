import numpy as np

import OrbLocalize
import fchk

np.set_printoptions(precision=3, suppress=True)

WorkDir = 'COF-701/Fragment0'
FileIn = '{}/cluster0-H'.format(WorkDir)
FileOut = '{}/test'.format(WorkDir)

MOE,MOC,AOS,IHOMO = fchk.ReadFchkOrb(FileIn)
tmp = np.matmul(AOS, MOC)
AOF = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
MolIdx = np.array([[0, 196], [196, 394]])

OrbLocalize.PipekMezey(AOS, MOC[:, IHOMO:IHOMO+50], MolIdx)
MOC0 = np.zeros_like(MOC[:, IHOMO:IHOMO+50])
MOC0[:196, :] = MOC[:196, IHOMO:IHOMO+50]
print(np.diag(np.matmul(MOC0.T, np.matmul(AOS, MOC0))))
MOC1 = np.zeros_like(MOC[:, IHOMO:IHOMO+50])
MOC1[196:, :] = MOC[196:, IHOMO:IHOMO+50]
print(np.diag(np.matmul(MOC1.T, np.matmul(AOS, MOC1))))
print()
fchk.WriteFchkOrb(FileIn, FileOut, MOE, MOC)