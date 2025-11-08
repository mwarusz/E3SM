#ifndef OMEGA_AUX_KINETIC_H
#define OMEGA_AUX_KINETIC_H

#include "DataTypes.h"
#include "HorzMesh.h"
#include "OmegaKokkos.h"
#include "VertCoord.h"

#include <string>

namespace OMEGA {

class KineticAuxVars {
 public:
   Array2DReal KineticEnergyCell;
   Array2DReal VelocityDivCell;

   KineticAuxVars(const std::string &AuxStateSuffix, const HorzMesh *Mesh,
                  const VertCoord *VCoord);

   KOKKOS_FUNCTION void
   computeVarsOnCell(const TeamMember &Team, int ICell,
                     const Array2DReal &NormalVelEdge) const {

      Scratch1DReal KineticEnergyCellTmp(Team.team_scratch(0), NVertLayers);
      Scratch1DReal VelocityDivCellTmp(Team.team_scratch(0), NVertLayers);

      const int KMin   = MinLayerCell(ICell);
      const int KMax   = MaxLayerCell(ICell);
      const int KRange = vertRange(KMin, KMax);

      const Real InvAreaCell = 1._Real / AreaCell(ICell);

      parallelForInner(
          Team, KRange, INNER_LAMBDA(int KOff) {
             const I4 K              = KMin + KOff;
             KineticEnergyCellTmp(K) = 0;
             VelocityDivCellTmp(K)   = 0;
          });

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const int JEdge     = EdgesOnCell(ICell, J);
         const Real AreaEdge = 0.5_Real * DvEdge(JEdge) * DcEdge(JEdge);
         parallelForInner(
             Team, KRange, INNER_LAMBDA(int KOff) {
                const I4 K = KMin + KOff;
                KineticEnergyCellTmp(K) += AreaEdge * 0.5_Real * InvAreaCell *
                                           NormalVelEdge(JEdge, K) *
                                           NormalVelEdge(JEdge, K);
                VelocityDivCellTmp(K) -= DvEdge(JEdge) * InvAreaCell *
                                         EdgeSignOnCell(ICell, J) *
                                         NormalVelEdge(JEdge, K);
             });
      }
      parallelForInner(
          Team, KRange, INNER_LAMBDA(int KOff) {
             const I4 K                  = KMin + KOff;
             KineticEnergyCell(ICell, K) = KineticEnergyCellTmp(K);
             VelocityDivCell(ICell, K)   = VelocityDivCellTmp(K);
          });
   }

   void registerFields(const std::string &AuxGroupName,
                       const std::string &MeshName) const;
   void unregisterFields() const;

 private:
   I4 NVertLayers;
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DReal EdgeSignOnCell;
   Array1DReal DcEdge;
   Array1DReal DvEdge;
   Array1DReal AreaCell;
   Array1DI4 MinLayerCell;
   Array1DI4 MaxLayerCell;
};

} // namespace OMEGA
#endif
