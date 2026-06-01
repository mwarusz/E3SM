#include "DataTypes.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void tracer_diff(const TracerDiffOnCell &LocTracerDiffusion, int L, int ICell,
                 const TeamMember &Team, const Array3DReal &LocTracerTend,
                 const Array3DReal &TracerArray,
                 const Array2DReal &MeanPseudoThickEdge) {
   LocTracerDiffusion(Team, LocTracerTend, L, ICell, TracerArray,
                      MeanPseudoThickEdge);
}
