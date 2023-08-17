import sys

import ClusterBuild
import EleStruct
import input
import kStruct

# build core-H/link-H/cluster
if (sys.argv[1] == '0'):
    pass
    ########## find a way to run g16 calculation
    #os.system('./{}/submit.sh'.format(input.WorkDir))
# calculate DoS and band structure
else:
    rS,rH0,rH = EleStruct.LocalMO()

    ## k-space calculations
    # DoS calculation
    kStruct.kDoS(input.kDoSNum, rH)
    #kStruct.kDoS(input.kDoSNum, rH0, rS=rS)

    # band structure calculation
    kStruct.kBand(input.kHighSymm, input.kBandNum, rH)
    #kStruct.kBand(input.kHighSymm, input.kBandNum, rH0, rS=rS)