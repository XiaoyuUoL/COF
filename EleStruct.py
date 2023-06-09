import numpy as np

import fchk
import input

# local orbitals information from gas phase calculation
def LocalMO(MOs='u'):
    # overlap matrix
    def MOS(MOC0, MOC1, AOS):
        return np.matmul(MOC0.T, np.matmul(AOS, MOC1))
    
    # Fock matrix
    def MOF(MOC0, MOC1, AOF):
        return np.matmul(MOC0.T, np.matmul(AOF, MOC1))
    
    # calculate the power of symmetry matrix
    def MatrixPower(M, x):
        e,V = np.linalg.eigh(M)
        return np.matmul(V, np.matmul(np.diag(np.power(e, x)), V.T))

    # (real space) Overlap (local orbital) from DFT gas phase calculation
    rS = []
    for i0 in np.arange(input.TMON):
        s = [[]]
        for i1 in np.arange(input.TMON):
            s.append([])
        rS.append(s)
    
    # (real space) Hamiltonian (local orbital) from DFT gas phase calculation
    rH0 = []
    for i0 in np.arange(input.TMON):
        h0 = [[]]
        for i1 in np.arange(input.TMON):
            h0.append([])
        rH0.append(h0)

    # (real space) Hamiltonian (local ortho orbital) from DFT gas phase calculation
    rH = []
    for i0 in np.arange(input.TMON):
        h = [[]]
        for i1 in np.arange(input.TMON):
            h.append([])
        rH.append(h)

    #CoreMOe = []
    CoreMOC = []
    for i in np.arange(input.CoreNum):
        moe,moc,aos,ihomo = fchk.ReadFchkOrb('{}/c{}-H'.format(input.WorkDir, i))
        moe *= 27.2113
        #print(moe[ihomo:ihomo+input.CoreMON])
        #print(moe[ihomo-input.CoreMON:ihomo])
        #print()
        indices = input.MONIdx['{}{}'.format('c', i)]
        for i in np.arange(input.CoreMON):
            idx = indices[i]
            rS[idx][idx].append([ 0, 0, 0, 1.])
            if (MOs == 'u'):  # unoccupied FMOs
                rH0[idx][idx].append([ 0, 0, 0, moe[ihomo+i]])  # using energy of core units
                rH[idx][idx].append([ 0, 0, 0, moe[ihomo+i]])  # using energy of core units
            else:  # occupied FMOs
                rH0[idx][idx].append([ 0, 0, 0, moe[ihomo+i-input.CoreMON]])  # using energy of core units
                rH[idx][idx].append([ 0, 0, 0, moe[ihomo+i-input.CoreMON]])  # using energy of core units
        if (MOs == 'u'):  # unoccupied FMOs
            #CoreMOe.append(moe[ihomo:ihomo+input.CoreMON])
            CoreMOC.append(moc[:, ihomo:ihomo+input.CoreMON])
        else:  # occupied FMOs
            #CoreMOe.append(moe[ihomo-input.CoreMON:ihomo])
            CoreMOC.append(moc[:, ihomo-input.CoreMON:ihomo])
    
    #LinkMOe = []
    LinkMOC = []
    for i in np.arange(input.LinkNum):
        moe,moc,aos,ihomo = fchk.ReadFchkOrb('{}/l{}-H'.format(input.WorkDir, i))
        moe *= 27.2113
        #print(moe[ihomo:ihomo+input.LinkMON])
        #print(moe[ihomo-input.LinkMON:ihomo])
        #print()
        indices = input.MONIdx['{}{}'.format('l', i)]
        for i in np.arange(input.LinkMON):
            idx = indices[i]
            rS[idx][idx].append([ 0, 0, 0, 1.])
            if (MOs == 'u'):  # unoccupied FMOs
                rH0[idx][idx].append([ 0, 0, 0, moe[ihomo+i]])  # using energy of link units
                rH[idx][idx].append([ 0, 0, 0, moe[ihomo+i]])  # using energy of link units
            else:  # occupied FMOs
                rH0[idx][idx].append([ 0, 0, 0, moe[ihomo+i-input.LinkMON]])  # using energy of link units
                rH[idx][idx].append([ 0, 0, 0, moe[ihomo+i-input.LinkMON]])  # using energy of link units
        if (MOs == 'u'):  # unoccupied FMOs
            #LinkMOe.append(moe[ihomo:ihomo+input.LinkMON])
            LinkMOC.append(moc[:, ihomo:ihomo+input.LinkMON])
        else:  # occupied FMOs
            #LinkMOe.append(moe[ihomo-input.LinkMON:ihomo])
            LinkMOC.append(moc[:, ihomo-input.LinkMON:ihomo])
    
    # cluster orbitals information
    for i,indices in enumerate(input.ClusterIdx):
        MOE,MOC,AOS,iHOMO = fchk.ReadFchkOrb('{}/cluster{}-H'.format(input.WorkDir, i))
        AON = np.shape(MOC)[0]
        tmp = np.matmul(AOS, MOC)
        AOF = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
        ClusterMON = np.dot([input.CoreMON, input.LinkMON], input.ClusterNCL[i])
        LOrbMOC = np.zeros((AON, ClusterMON), dtype=float)  # local orbital
        iao0 = 0
        imo0 = 0
        imo = []
        for j,index in enumerate(indices):
            if (index[0] == 'c'):
                aon = np.shape(CoreMOC[index[1]])[0] - input.HAON * input.ClusterNH[i][j]
                iao1 = iao0 + aon
                imo1 = imo0 + input.CoreMON
                LOrbMOC[iao0:iao1, imo0:imo1] = CoreMOC[index[1]][:aon, :]
                imo.append(np.arange(ClusterMON)[imo0:imo1])
            else:
                aon = np.shape(LinkMOC[index[1]])[0] - input.HAON * input.ClusterNH[i][j]
                iao1 = iao0 + aon
                imo1 = imo0 + input.LinkMON
                LOrbMOC[iao0:iao1, imo0:imo1] = LinkMOC[index[1]][:aon, :]
                imo.append(np.arange(ClusterMON)[imo0:imo1])
            iao0 = iao1
            imo0 = imo1
        LOrbMOS = MOS(LOrbMOC, LOrbMOC, AOS)
        LOrbMOF = MOF(LOrbMOC, LOrbMOC, AOF) * 27.2113  # unit: eV
        Smh = MatrixPower(LOrbMOS, -0.5)
        LOOrbMOF = np.matmul(Smh, np.matmul(LOrbMOF, Smh))  # unit: eV, local orthonormalized orbital
        #print(LOrbMOS)
        #print(LOrbMOF)
        #print(LOOrbMOF)
        #print()
        # cosider all coupling terms in dimer clusters 
        if (len(indices) == 2):
            imo0 = input.MONIdx['{}{}'.format(indices[0][0], indices[0][1])]
            imo1 = input.MONIdx['{}{}'.format(indices[1][0], indices[1][1])]
            pbc = np.array(indices[1][2:]) - np.array(indices[0][2:])
            for i0,idx0 in enumerate(imo0):
                for i1,idx1 in enumerate(imo1):
                    mos = LOrbMOS[imo[0][i0], imo[1][i1]]
                    mof0 = LOrbMOF[imo[0][i0], imo[1][i1]]
                    mof = LOOrbMOF[imo[0][i0], imo[1][i1]]
                    rS[idx0][idx1].append([ pbc[0],  pbc[1],  pbc[2], mos])
                    rS[idx1][idx0].append([-pbc[0], -pbc[1], -pbc[2], mos])
                    rH0[idx0][idx1].append([ pbc[0],  pbc[1],  pbc[2], mof0])
                    rH0[idx1][idx0].append([-pbc[0], -pbc[1], -pbc[2], mof0])
                    rH[idx0][idx1].append([ pbc[0],  pbc[1],  pbc[2], mof])
                    rH[idx1][idx0].append([-pbc[0], -pbc[1], -pbc[2], mof])
        # consider only 'c'-'l' coupling terms in larger clusters
        else:
            for i0,index0 in enumerate(indices):
                for i1,index1 in enumerate(indices):
                    if (index0[0] == 'c' and index1[0] == 'l'):
                        imo0 = input.MONIdx['{}{}'.format(index0[0], index0[1])]
                        imo1 = input.MONIdx['{}{}'.format(index1[0], index1[1])]
                        pbc = np.array(index1[2:]) - np.array(index0[2:])
                        for j0,idx0 in enumerate(imo0):
                            for j1,idx1 in enumerate(imo1):
                                mos = LOrbMOS[imo[i0][j0], imo[i1][j1]]
                                mof0 = LOrbMOF[imo[i0][j0], imo[i1][j1]]
                                mof = LOOrbMOF[imo[i0][j0], imo[i1][j1]]
                                rS[idx0][idx1].append([ pbc[0],  pbc[1],  pbc[2], mos])
                                rS[idx1][idx0].append([-pbc[0], -pbc[1], -pbc[2], mos])
                                rH0[idx0][idx1].append([ pbc[0],  pbc[1],  pbc[2], mof0])
                                rH0[idx1][idx0].append([-pbc[0], -pbc[1], -pbc[2], mof0])
                                rH[idx0][idx1].append([ pbc[0],  pbc[1],  pbc[2], mof])
                                rH[idx1][idx0].append([-pbc[0], -pbc[1], -pbc[2], mof])
    
    #for i0 in np.arange(input.TMON):
    #    if(rH[i0][i0] != []):
    #        for h in rH[i0][i0]:
    #            print(i0, i0, h)
    #print()
    #
    #for i0 in np.arange(input.TMON):
    #    for i1 in np.arange(input.TMON):
    #        if (i0 >= i1):
    #            continue
    #        if(rH[i0][i1] != []):
    #            for h in rH[i0][i1]:
    #                print(i0, i1, h)
    #exit()

    return rS, rH0, rH