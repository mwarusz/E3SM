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

   template <class FC>
   KOKKOS_FUNCTION void
   computeVarsOnEdge(int IEdge, int KStart, const Array2DReal &LayerThickCell,
                     const Array2DReal &NormalVelEdge, FC FullChunk) const {

      const int KLen =
          FullChunk ? VecLength : MaxLayerEdgeTop(IEdge) - KStart + 1;

      const int JCell0 = CellsOnEdge(IEdge, 0);
      const int JCell1 = CellsOnEdge(IEdge, 1);

      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K = KStart + KVec;
         MeanLayerThickEdge(IEdge, K) =
             0.5_Real * (LayerThickCell(JCell0, K) + LayerThickCell(JCell1, K));
      }

      switch (FluxThickEdgeChoice) {
      case FluxThickEdgeOption::Center:
         for (int KVec = 0; KVec < KLen; ++KVec) {
            const int K = KStart + KVec;
            FluxLayerThickEdge(IEdge, K) =
                0.5_Real *
                (LayerThickCell(JCell0, K) + LayerThickCell(JCell1, K));
         }
         break;
      case FluxThickEdgeOption::Upwind:
         for (int KVec = 0; KVec < KLen; ++KVec) {
            const int K = KStart + KVec;
            if (NormalVelEdge(IEdge, K) > 0) {
               FluxLayerThickEdge(IEdge, K) = LayerThickCell(JCell0, K);
            } else if (NormalVelEdge(IEdge, K) < 0) {
               FluxLayerThickEdge(IEdge, K) = LayerThickCell(JCell1, K);
            } else {
               FluxLayerThickEdge(IEdge, K) = Kokkos::max(
                   LayerThickCell(JCell0, K), LayerThickCell(JCell1, K));
            }
         }
         break;
      }
   }

   template <class FC>
   KOKKOS_FUNCTION void computeVarsOnCells(int ICell, int KStart,
                                           const Array2DReal &LayerThickCell,
                                           FC FullChunk) const {

      const int KLen = FullChunk ? VecLength : MaxLayerCell(ICell) - KStart + 1;

      // Temporary for stacked shallow water
      for (int KVec = 0; KVec < VecLength; ++KVec) {
         const int K       = KStart + KVec;
         SshCell(ICell, K) = LayerThickCell(ICell, K) - BottomDepth(ICell);
      }

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
   Array1DI4 MaxLayerCell;
   Array1DI4 MaxLayerEdgeTop;
};

} // namespace OMEGA
#endif
