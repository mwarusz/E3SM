#include "DataTypes.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void ke_grad(const KEGradOnEdge &LocKEGrad, int IEdge, const TeamMember &Team,
             const Array2DReal &LocNormalVelocityTend,
             const Array2DReal &KECell) {
   LocKEGrad(Team, LocNormalVelocityTend, IEdge, KECell);
}
