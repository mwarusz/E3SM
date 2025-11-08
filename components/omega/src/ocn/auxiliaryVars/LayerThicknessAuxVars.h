#ifndef OMEGA_AUX_THICKNESS_H
#define OMEGA_AUX_THICKNESS_H

#include "DataTypes.h"
#include "HorzMesh.h"
#include "OmegaKokkos.h"
#include "VertCoord.h"

#include <string>

namespace OMEGA {

enum class FluxThickEdgeOption { Center, Upwind };

class LayerThicknessAuxVars {
 public:
   Array2DReal FluxLayerThickEdge;
   Array2DReal MeanLayerThickEdge;
   Array2DReal SshCell;

   FluxThickEdgeOption FluxThickEdgeChoice;

   LayerThicknessAuxVars(const std::string &AuxStateSuffix,
                         const HorzMesh *Mesh, const VertCoord *VCoord);

   KOKKOS_FUNCTION void
   computeVarsOnEdge(const TeamMember &Team, int IEdge,
                     const Array2DReal &LayerThickCell,
                     const Array2DReal &NormalVelEdge) const {
      const int KMin   = MinLayerEdgeBot(IEdge);
      const int KMax   = MaxLayerEdgeTop(IEdge);
      const int KRange = vertRange(KMin, KMax);

      const int JCell0 = CellsOnEdge(IEdge, 0);
      const int JCell1 = CellsOnEdge(IEdge, 1);

      parallelForInner(
          Team, KRange, INNER_LAMBDA(int KOff) {
             const I4 K = KMin + KOff;
             MeanLayerThickEdge(IEdge, K) =
                 0.5_Real *
                 (LayerThickCell(JCell0, K) + LayerThickCell(JCell1, K));
          });

      switch (FluxThickEdgeChoice) {
      case FluxThickEdgeOption::Center:
         parallelForInner(
             Team, KRange, INNER_LAMBDA(int KOff) {
                const I4 K = KMin + KOff;
                FluxLayerThickEdge(IEdge, K) =
                    0.5_Real *
                    (LayerThickCell(JCell0, K) + LayerThickCell(JCell1, K));
             });
         break;
      case FluxThickEdgeOption::Upwind:
         parallelForInner(
             Team, KRange, INNER_LAMBDA(int KOff) {
                const I4 K = KMin + KOff;
                if (NormalVelEdge(IEdge, K) > 0) {
                   FluxLayerThickEdge(IEdge, K) = LayerThickCell(JCell0, K);
                } else if (NormalVelEdge(IEdge, K) < 0) {
                   FluxLayerThickEdge(IEdge, K) = LayerThickCell(JCell1, K);
                } else {
                   FluxLayerThickEdge(IEdge, K) = Kokkos::max(
                       LayerThickCell(JCell0, K), LayerThickCell(JCell1, K));
                }
             });
         break;
      }
   }

   KOKKOS_FUNCTION void
   computeVarsOnCells(const TeamMember &Team, int ICell,
                      const Array2DReal &LayerThickCell) const {

      // Temporary for stacked shallow water
      const int KMin   = MinLayerCell(ICell);
      const int KMax   = MaxLayerCell(ICell);
      const int KRange = vertRange(KMin, KMax);

      parallelForInner(
          Team, KRange, INNER_LAMBDA(int KOff) {
             const I4 K        = KMin + KOff;
             SshCell(ICell, K) = LayerThickCell(ICell, K) - BottomDepth(ICell);
          });

      /*
      Real TotalThickness = 0.0;
      for (int K = 0; K < NVertLayers; K++) {
         TotalThickness += LayerThickCell(ICell, K);
      }

      SshCell(ICell) = TotalThickness - BottomDepth(ICell);
      */
   }

   void registerFields(const std::string &AuxGroupName,
                       const std::string &MeshName) const;
   void unregisterFields() const;

 private:
   Array2DI4 CellsOnEdge;
   Array1DReal BottomDepth;
   Array1DI4 MinLayerEdgeBot;
   Array1DI4 MaxLayerEdgeTop;
   Array1DI4 MinLayerCell;
   Array1DI4 MaxLayerCell;
};

} // namespace OMEGA
#endif
