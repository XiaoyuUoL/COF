import numpy as np

import input
import readqc

# local orbitals information from gas phase calculation
def LocalMO():
    # overlap matrix
    def MOS(moc0, moc1, aos):
        return moc0.T @ aos @ moc1

    # Fock matrix
    def MOF(moc0, moc1, aof):
        return moc0.T @aof @ moc1

    # calculate the power of symmetry matrix
    def MatrixPower(M, x):
        e,v = np.linalg.eigh(M)
        return v @ np.diag(np.power(e, x)) @ v.T

    # (real space) Overlap (local orbital) from DFT gas phase calculation
    rS = []
    for i0 in np.arange(input.TMON):
        s = [[]]
        for i1 in np.arange(input.TMON - 1):
            s.append([])
        rS.append(s)

    # (real space) Hamiltonian (local orbital) from gas phase calculation
    rH0 = []
    for i0 in np.arange(input.TMON):
        h0 = [[]]
        for i1 in np.arange(input.TMON - 1):
            h0.append([])
        rH0.append(h0)

    # (real space) Hamiltonian (local ortho orbital) from gas phase calculation
    rH = []
    for i0 in np.arange(input.TMON):
        h = [[]]
        for i1 in np.arange(input.TMON - 1):
            h.append([])
        rH.append(h)

    CoreMOC = []
    for i in np.arange(input.CoreNum):
        occe,occc,vire,virc = readqc.ReadQC('{}/c{}-H'.format(input.QCDir, i))
        occe *= 27.2113
        vire *= 27.2113
        indices = input.MONIdx['{}{}'.format('c', i)]
        # using orbital energy of core units
        for i in np.arange(input.CoreMON):
            idx = indices[i]
            rS[idx][idx].append([0, 0, 0, 1.])
            if input.MOMode == 'o':  # occupied FMOs
                rH0[idx][idx].append([0, 0, 0, occe[i-input.CoreMON]])
                rH[idx][idx].append([0, 0, 0, occe[i-input.CoreMON]])
            elif input.MOMode == 'u':  # unoccupied FMOs
                rH0[idx][idx].append([0, 0, 0, vire[i]])
                rH[idx][idx].append([0, 0, 0, vire[i]])
            else:
                print('MOMode should be "o" or "u"')
                exit(0)
        if input.MOMode == 'o':  # occupied FMOs
            CoreMOC.append(occc[:, -input.CoreMON:])
        elif input.MOMode == 'u':  # unoccupied FMOs
            CoreMOC.append(virc[:, :input.CoreMON])
        else:
            print('MOMode should be "o" or "u"')
            exit(0)

    '''
    # try to use orbital energy of core in core-links cluster
    for i in np.arange(input.CoreNum):
        MOE,MOC,aos,iHOMO = readqc.ReadFchkOrb('{}/c{}-l'.format(input.QCDir, i))
        AON = np.shape(MOC)[0]
        tmp = np.matmul(aos, MOC)
        aof = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
        MON = input.CoreMON
        LOrbMOC = np.zeros((AON, MON), dtype=float)
        aon = np.shape(CoreMOC[i])[0] - input.HAON * input.CoreNH[i]
        for j in np.arange(MON):
            LOrbMOC[:aon, j] = CoreMOC[i][:aon, j]
        LOrbMOF = MOF(LOrbMOC, LOrbMOC, aof) * 27.2113
        indices = input.MONIdx['{}{}'.format('c', i)]
        ## using orbital energy of local orbital in core-links cluster
        for i in np.arange(input.CoreMON):
            idx = indices[i]
            rS[idx][idx].append([0, 0, 0, 1.])
            if input.MOMode == 'u':  # unoccupied FMOs
                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
            else:  # occupied FMOs
                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
        print(LOrbMOF)
    print()
    '''

    LinkMOC = []
    for i in np.arange(input.LinkNum):
        occe,occc,vire,virc = readqc.ReadQC('{}/l{}-H'.format(input.QCDir, i))
        occe *= 27.2113
        vire *= 27.2113
        indices = input.MONIdx['{}{}'.format('l', i)]
        # using energy of link units
        for i in np.arange(input.LinkMON):
            idx = indices[i]
            rS[idx][idx].append([0, 0, 0, 1.])
            if input.MOMode == 'o':  # occupied FMOs
                rH0[idx][idx].append([0, 0, 0, occe[i-input.LinkMON]])
                rH[idx][idx].append([0, 0, 0, occe[i-input.LinkMON]])
            elif input.MOMode == 'u':  # unoccupied FMOs
                rH0[idx][idx].append([0, 0, 0, vire[i]])
                rH[idx][idx].append([0, 0, 0, vire[i]])
            else:
               print('MOMode should be "o" or "u"')
               exit(0)
        if input.MOMode == 'o':  # occupied FMOs
            LinkMOC.append(occc[:, -input.LinkMON:])
        elif input.MOMode == 'u':  # unoccupied FMOs
            LinkMOC.append(virc[:, :input.LinkMON])
        else:
            print('MOMode should be "o" or "u"')
            exit(0)

    '''
    # try to use energy of link in link-cores cluster
    for i in np.arange(input.LinkNum):
        MOE,MOC,aos,iHOMO = fchk.ReadFchkOrb('{}/l{}-c'.format(
            input.QCDir, i))
        AON = np.shape(MOC)[0]
        tmp = np.matmul(aos, MOC)
        aof = np.matmul(tmp, np.matmul(np.diag(MOE), tmp.T))
        MON = input.LinkMON
        LOrbMOC = np.zeros((AON, MON), dtype=float)
        aon = np.shape(LinkMOC[i])[0] - input.HAON * input.LinkNH[i]
        for j in np.arange(MON):
            LOrbMOC[:aon, j] = LinkMOC[i][:aon, j]
        LOrbMOF = MOF(LOrbMOC, LOrbMOC, aof) * 27.2113
        indices = input.MONIdx['{}{}'.format('l', i)]
        # using orbital energy of local orbital in link-cores cluster
        if input.MOMode == 'a':  # all
            for i in np.arange(input.LinkMON * 2):
                idx = indices[i]
                rS[idx][idx].append([0, 0, 0, 1.])
                rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
                rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
        else:
            for i in np.arange(input.LinkMON):
                idx = indices[i]
                rS[idx][idx].append([0, 0, 0, 1.])
                if input.MOMode == 'u':  # unoccupied FMOs
                    rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
                    rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
                else:  # occupied FMOs
                    rH0[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
                    rH[idx][idx].append([0, 0, 0, LOrbMOF[i, i]])
        print(LOrbMOF)
    print()
    '''

    # cluster orbitals information
    for i,indices in enumerate(input.ClusterIdx):
        aos,aof = readqc.ReadQC('{}/cluster{}-H'.format(input.QCDir, i), True)
        AON = np.shape(aos)[0]
        aof *= 27.2113  # unit: eV
        ClusterMON = np.dot([input.CoreMON, input.LinkMON], input.ClusterNCL[i])
        LOrbMOC = np.zeros((AON, ClusterMON), dtype=float)  # local orbital
        iao0 = 0
        imo0 = 0
        imo = []
        for j,index in enumerate(indices):
            if index[0] == 'c':
                aon0 = np.shape(CoreMOC[index[1]])[0]
                aon = aon0 - input.HAON * input.CoreNH[index[1]]
                iao1 = iao0 + aon
                imo1 = imo0 + input.CoreMON
                LOrbMOC[iao0:iao1, imo0:imo1] = CoreMOC[index[1]][:aon, :]
                imo.append(np.arange(ClusterMON)[imo0:imo1])
            else:
                ano0 = np.shape(LinkMOC[index[1]])[0]
                aon = ano0 - input.HAON * input.LinkNH[index[1]]
                iao1 = iao0 + aon
                imo1 = imo0 + input.LinkMON
                LOrbMOC[iao0:iao1, imo0:imo1] = LinkMOC[index[1]][:aon, :]
                imo.append(np.arange(ClusterMON)[imo0:imo1])
            iao0 = iao1 + input.HAON * input.ClusterNH[i][j]
            imo0 = imo1
        LOrbMOS = MOS(LOrbMOC, LOrbMOC, aos)
        LOrbMOF = MOF(LOrbMOC, LOrbMOC, aof)
        Smh = MatrixPower(LOrbMOS, -0.5)
        # local orthonormalized orbital
        LOOrbMOF = np.matmul(Smh, np.matmul(LOrbMOF, Smh))  # unit: eV
        # cosider all coupling terms in dimer clusters 
        if len(indices) == 2:
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
                    if index0[0] == 'c' and index1[0] == 'l':
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
    #    if rH[i0][i0] != []:
    #        for h in rH[i0][i0]:
    #            print(i0, i0, h)
    #print()
    #
    #for i0 in np.arange(input.TMON):
    #    for i1 in np.arange(input.TMON):
    #        if i0 >= i1:
    #            continue
    #        if rH[i0][i1] != []:
    #            for h in rH[i0][i1]:
    #                print(i0, i1, h)
    #exit()

    return rS, rH0, rH