import numpy as np

import fchk
import input

# local orbitals information from gas phase calculation
def LocalMO():
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

    # (real space) Hamiltonian (local orbital) from gas phase calculation
    rH0 = []
    for i0 in np.arange(input.TMON):
        h0 = [[]]
        for i1 in np.arange(input.TMON):
            h0.append([])
        rH0.append(h0)

    # (real space) Hamiltonian (local ortho orbital) from gas phase calculation
    rH = []
    for i0 in np.arange(input.TMON):
        h = [[]]
        for i1 in np.arange(input.TMON):
            h.append([])
        rH.append(h)

    CoreMOC = []
    for i in np.arange(input.CoreNum):
        moe,moc,aos,ihomo = fchk.ReadFchkOrb('{}/c{}-H'.format(
            input.WorkDir, i))
        moe *= 27.2113
        #print(moe[ihomo:ihomo+input.CoreMON])
        #print(moe[ihomo-input.CoreMON:ihomo])
        indices = input.MONIdx['{}{}'.format('c', i)] 
        # using orbital energy of core units
        if (input.MOMode == 'a'):  # all
            for i in np.arange(input.CoreMON * 2):
                idx = indices[i]
                rS[idx][idx].append([0, 0, 0, 1.])
                rH0[idx][idx].append([0, 0, 0, moe[ihomo+i-input.CoreMON]])
                rH[idx][idx].append([0, 0, 0, moe[ihomo+i-input.CoreMON]])
        else:
            for i in np.arange(input.CoreMON):
                idx = indices[i]
                rS[idx][idx].append([0, 0, 0, 1.])
                if (input.MOMode == 'u'):  # unoccupied FMOs
                    rH0[idx][idx].append([0, 0, 0, moe[ihomo+i]])
                    rH[idx][idx].append([0, 0, 0, moe[ihomo+i]])
                else:  # occupied FMOs
                    rH0[idx][idx].append([0, 0, 0, moe[ihomo+i-input.CoreMON]])
                    rH[idx][idx].append([0, 0, 0, moe[ihomo+i-input.CoreMON]])
        if (input.MOMode == 'u'):  # unoccupied FMOs
            CoreMOC.append(moc[:, ihomo:ihomo+input.CoreMON])
        elif (input.MOMode == 'o'):  # occupied FMOs
            CoreMOC.append(moc[:, ihomo-input.CoreMON:ihomo])
        else:  # all
            CoreMOC.append(moc[:, ihomo-input.CoreMON:ihomo+input.CoreMON])
    #print()

    ## try to use orbital energy of core in core-links cluster
    #for i in np.arange(input.CoreNum):
    #    MOE,MOC,AOS,iHOMO = fchk.ReadFchkOrb('{}/c{}-l'.format(
    #        input.WorkDir, i))
    #    AON = np.shape(MOC)[0]
    #    tmp = np.matmul(AOS, MOC)
    #    AOF = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
    #    MON = input.CoreMON
    #    LOrbMOC = np.zeros((AON, MON), dtype=float)
    #    aon = np.shape(CoreMOC[i])[0] - input.HAON * input.CoreNH[i]
    #    for j in np.arange(MON):
    #        LOrbMOC[:aon, j] = CoreMOC[i][:aon, j]
    #    LOrbMOF = MOF(LOrbMOC, LOrbMOC, AOF) * 27.2113
    #    indices = input.MONIdx['{}{}'.format('c', i)]
    #    ## using orbital energy of local orbital in core-links cluster
    #    if (input.MOMode == 'a'):  # all
    #        for i in np.arange(input.CoreMON * 2):
    #            idx = indices[i]
    #            rS[idx][idx].append([0, 0, 0, 1.])
    #            rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #            rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #    else:
    #        for i in np.arange(input.CoreMON):
    #            idx = indices[i]
    #            rS[idx][idx].append([0, 0, 0, 1.])
    #            if (input.MOMode == 'u'):  # unoccupied FMOs
    #                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #            else:  # occupied FMOs
    #                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #    print(LOrbMOF)
    #print()

    LinkMOC = []
    for i in np.arange(input.LinkNum):
        moe,moc,aos,ihomo = fchk.ReadFchkOrb('{}/l{}-H'.format(
            input.WorkDir, i))
        moe *= 27.2113
        #print(moe[ihomo:ihomo+input.LinkMON])
        #print(moe[ihomo-input.LinkMON:ihomo])
        indices = input.MONIdx['{}{}'.format('l', i)]
        # using energy of link units
        if (input.MOMode == 'a'):  # all
            for i in np.arange(input.LinkMON * 2):
                idx = indices[i]
                rS[idx][idx].append([0, 0, 0, 1.])
                rH0[idx][idx].append([0, 0, 0, moe[ihomo+i-input.LinkMON]])
                rH[idx][idx].append([0, 0, 0, moe[ihomo+i-input.LinkMON]])
        else:
            for i in np.arange(input.LinkMON):
                idx = indices[i]
                rS[idx][idx].append([0, 0, 0, 1.])
                if (input.MOMode == 'u'):  # unoccupied FMOs
                    rH0[idx][idx].append([0, 0, 0, moe[ihomo+i]])
                    rH[idx][idx].append([0, 0, 0, moe[ihomo+i]])
                else:  # occupied FMOs
                    rH0[idx][idx].append([0, 0, 0, moe[ihomo+i-input.LinkMON]])
                    rH[idx][idx].append([0, 0, 0, moe[ihomo+i-input.LinkMON]])
        if (input.MOMode == 'u'):  # unoccupied FMOs
            LinkMOC.append(moc[:, ihomo:ihomo+input.LinkMON])
        elif (input.MOMode == 'o'):  # occupied FMOs
            LinkMOC.append(moc[:, ihomo-input.LinkMON:ihomo])
        else:  # all
            LinkMOC.append(moc[:, ihomo-input.LinkMON:ihomo+input.LinkMON])
    #print()

    ## try to use energy of link in link-cores cluster
    #for i in np.arange(input.LinkNum):
    #    MOE,MOC,AOS,iHOMO = fchk.ReadFchkOrb('{}/l{}-c'.format(
    #        input.WorkDir, i))
    #    AON = np.shape(MOC)[0]
    #    tmp = np.matmul(AOS, MOC)
    #    AOF = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
    #    MON = input.LinkMON
    #    LOrbMOC = np.zeros((AON, MON), dtype=float)
    #    aon = np.shape(LinkMOC[i])[0] - input.HAON * input.LinkNH[i]
    #    for j in np.arange(MON):
    #        LOrbMOC[:aon, j] = LinkMOC[i][:aon, j]
    #    LOrbMOF = MOF(LOrbMOC, LOrbMOC, AOF) * 27.2113
    #    indices = input.MONIdx['{}{}'.format('l', i)]
    #    # using orbital energy of local orbital in link-cores cluster
    #    if (input.MOMode == 'a'):  # all
    #        for i in np.arange(input.LinkMON * 2):
    #            idx = indices[i]
    #            rS[idx][idx].append([0, 0, 0, 1.])
    #            rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #            rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #    else:
    #        for i in np.arange(input.LinkMON):
    #            idx = indices[i]
    #            rS[idx][idx].append([0, 0, 0, 1.])
    #            if (input.MOMode == 'u'):  # unoccupied FMOs
    #                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #            else:  # occupied FMOs
    #                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
    #    print(LOrbMOF)
    #print()

    # cluster orbitals information
    for i,indices in enumerate(input.ClusterIdx):
        MOE,MOC,AOS,iHOMO = fchk.ReadFchkOrb('{}/cluster{}-H'.format(
            input.WorkDir, i))
        AON = np.shape(MOC)[0]
        tmp = np.matmul(AOS, MOC)
        AOF = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
        ClusterMON = np.dot([input.CoreMON, input.LinkMON], input.ClusterNCL[i])
        if (input.MOMode == 'a'):
            ClusterMON *= 2
        LOrbMOC = np.zeros((AON, ClusterMON), dtype=float)  # local orbital
        iao0 = 0
        imo0 = 0
        imo = []
        for j,index in enumerate(indices):
            if (index[0] == 'c'):
                aon0 = np.shape(CoreMOC[index[1]])[0]
                aon = aon0 - input.HAON * input.CoreNH[index[1]]
                iao1 = iao0 + aon
                if (input.MOMode == 'u' or input.MOMode == 'o'):
                    imo1 = imo0 + input.CoreMON
                else:
                    imo1 = imo0 + input.CoreMON * 2
                LOrbMOC[iao0:iao1, imo0:imo1] = CoreMOC[index[1]][:aon, :]
                imo.append(np.arange(ClusterMON)[imo0:imo1])
                iao0 += aon0 - input.HAON * input.ClusterNH[i][j]
            else:
                ano0 = np.shape(LinkMOC[index[1]])[0]
                aon = ano0 - input.HAON * input.LinkNH[index[1]]
                iao1 = iao0 + aon
                if (input.MOMode == 'u' or input.MOMode == 'o'):
                    imo1 = imo0 + input.LinkMON
                else:
                    imo1 = imo0 + input.LinkMON * 2
                LOrbMOC[iao0:iao1, imo0:imo1] = LinkMOC[index[1]][:aon, :]
                imo.append(np.arange(ClusterMON)[imo0:imo1])
                iao0 += ano0 - input.HAON * input.ClusterNH[i][j]
            imo0 = imo1
        LOrbMOS = MOS(LOrbMOC, LOrbMOC, AOS)
        LOrbMOF = MOF(LOrbMOC, LOrbMOC, AOF) * 27.2113  # unit: eV
        Smh = MatrixPower(LOrbMOS, -0.5)
        # local orthonormalized orbital
        LOOrbMOF = np.matmul(Smh, np.matmul(LOrbMOF, Smh))  # unit: eV
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
                    rS[idx0][idx1].append(list(pbc) + [mos])
                    rS[idx1][idx0].append(list(-pbc) + [mos])
                    rH0[idx0][idx1].append(list(pbc) + [mof0])
                    rH0[idx1][idx0].append(list(-pbc) + [mof0])
                    rH[idx0][idx1].append(list(pbc) + [mof])
                    rH[idx1][idx0].append(list(-pbc) + [mof])
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
                                rS[idx0][idx1].append(list(pbc) + [mos])
                                rS[idx1][idx0].append(list(-pbc) + [mos])
                                rH0[idx0][idx1].append(list(pbc) + [mof0])
                                rH0[idx1][idx0].append(list(-pbc) + [mof0])
                                rH[idx0][idx1].append(list(pbc) + [mof])
                                rH[idx1][idx0].append(list(-pbc) + [mof])

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