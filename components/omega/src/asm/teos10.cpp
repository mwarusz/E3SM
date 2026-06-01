#include "DataTypes.h"
#include "Eos.h"
#include "OmegaKokkos.h"

using namespace OMEGA;

void teos10(const Teos10Eos &ComputeSpecVolTeos10, int ICell,
            const TeamMember &Team, const Array2DReal &SpecVol,
            const Array2DReal &ConservTemp, const Array2DReal &AbsSalinity,
            const Array2DReal &Pressure, int KDisp) {
   ComputeSpecVolTeos10(SpecVol, Team, ICell, ConservTemp, AbsSalinity,
                        Pressure, KDisp);
}
