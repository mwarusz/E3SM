#include "DataTypes.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void tracer_adv_cell(const TracerHorzAdvOnCell &LocTracerHorzAdv, int L,
                     int ICell, const TeamMember &Team,
                     const Array3DReal &TracerTend) {
   LocTracerHorzAdv(Team, TracerTend, L, ICell);
}
