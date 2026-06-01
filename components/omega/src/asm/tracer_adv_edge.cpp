#include "DataTypes.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void tracer_adv_edge(const TracerHorzAdvOnCell &LocTracerHorzAdv, int L,
                     int IEdge, const TeamMember &Team,
                     const Array3DReal &TracerArray,
                     const Array2DReal &FluxPseudoThickEdge,
                     const Array2DReal &NormalVelocity) {
   LocTracerHorzAdv(Team, L, IEdge, TracerArray, FluxPseudoThickEdge,
                    NormalVelocity);
}
