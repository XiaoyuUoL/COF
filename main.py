import ClusterBuild
import EleStruct
import input
import kStruct

########## find a way to run g16 calculation
os.system('./{}/submit.sh'.format(input.WorkDir))

rS,rH0,rH = EleStruct.LocalMO('u')

## k-space calculations
# DoS calculation
kStruct.kDoS(input.kDoSNum, rH)
#kStruct.kDoS(input.kDoSNum, rH0)
#kStruct.kDoS(input.kDoSNum, rH0, rS=rS)

# band structure calculation
kStruct.kBand(input.kHighSymm, input.kBandNum, rH)
#kStruct.kBand(input.kHighSymm, input.kBandNum, rH0)
#kStruct.kBand(input.kHighSymm, input.kBandNum, rH0, rS=rS)