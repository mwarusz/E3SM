#include "DataTypes.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void thick_adv(const PseudoThicknessFluxDivOnCell &LocThicknessFluxDiv,
               int ICell, const TeamMember &Team,
               const Array2DReal &LocPseudoThicknessTend,
               const Array2DReal &ThickFluxEdge,
               const Array2DReal &NormalVelEdge) {
   LocThicknessFluxDiv(Team, LocPseudoThicknessTend, ICell, ThickFluxEdge,
                       NormalVelEdge);
}
