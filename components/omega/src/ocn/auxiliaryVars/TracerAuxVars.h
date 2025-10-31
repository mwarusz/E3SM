#ifndef OMEGA_AUX_TRACER_H
#define OMEGA_AUX_TRACER_H

#include "DataTypes.h"
#include "Field.h"
#include "HorzMesh.h"
#include "OmegaKokkos.h"
#include "VertCoord.h"

#include <string>

namespace OMEGA {

enum class FluxTracerEdgeOption { Center, Upwind };

class TracerAuxVars {
 public:
   Array3DReal HTracersEdge;
   Array3DReal Del2TracersCell;

   FluxTracerEdgeOption TracersOnEdgeChoice;

   TracerAuxVars(const std::string &AuxStateSuffix, const HorzMesh *Mesh,
                 const VertCoord *VCoord, const I4 NTracers);

   KOKKOS_FUNCTION void computeVarsOnEdge(const TeamMember &Team, int L,
                                          int IEdge,
                                          const Array2DReal &NormalVelEdge,
                                          const Array2DReal &HCell,
                                          const Array3DReal &TrCell) const {
      const int KMin   = MinLayerEdgeBot(IEdge);
      const int KMax   = MaxLayerEdgeTop(IEdge);
      const int KRange = vertRange(KMin, KMax);

      const int JCell0 = CellsOnEdge(IEdge, 0);
      const int JCell1 = CellsOnEdge(IEdge, 1);

      switch (TracersOnEdgeChoice) {
      case FluxTracerEdgeOption::Center:
         parallelForInner(
             Team, KRange, INNER_LAMBDA(int KOff) {
                const I4 K = KMin + KOff;
                HTracersEdge(L, IEdge, K) =
                    0.5_Real * (HCell(JCell0, K) * TrCell(L, JCell0, K) +
                                HCell(JCell1, K) * TrCell(L, JCell1, K));
             });
         break;
      case FluxTracerEdgeOption::Upwind:
         parallelForInner(
             Team, KRange, INNER_LAMBDA(int KOff) {
                const I4 K = KMin + KOff;
                if (NormalVelEdge(IEdge, K) > 0) {
                   HTracersEdge(L, IEdge, K) =
                       HCell(JCell0, K) * TrCell(L, JCell0, K);
                } else if (NormalVelEdge(IEdge, K) < 0) {
                   HTracersEdge(L, IEdge, K) =
                       HCell(JCell1, K) * TrCell(L, JCell1, K);
                } else {
                   HTracersEdge(L, IEdge, K) =
                       Kokkos::max(HCell(JCell0, K) * TrCell(L, JCell0, K),
                                   HCell(JCell1, K) * TrCell(L, JCell1, K));
                }
             });
         break;
      }
   }

   KOKKOS_FUNCTION void
   computeVarsOnCells(const TeamMember &Team, int L, int ICell,
                      const Array2DReal &LayerThickEdgeMean,
                      const Array3DReal &TrCell) const {

      auto Del2TrCellTmp = getScratch<Scratch1DReal>(Team, NVertLayers);

      const int KMinCell   = MinLayerCell(ICell);
      const int KMaxCell   = MaxLayerCell(ICell);
      const int KRangeCell = vertRange(KMinCell, KMaxCell);

      const Real InvAreaCell = 1._Real / AreaCell(ICell);

      parallelForInner(
          Team, KRangeCell, INNER_LAMBDA(int KOff) {
             const I4 K       = KMinCell + KOff;
             Del2TrCellTmp(K) = 0;
          });

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const int JEdge = EdgesOnCell(ICell, J);

         const int JCell0 = CellsOnEdge(JEdge, 0);
         const int JCell1 = CellsOnEdge(JEdge, 1);

         const Real DvDcEdge = DvEdge(JEdge) / DcEdge(JEdge);

         const int KMinEdge   = MinLayerEdgeBot(JEdge);
         const int KMaxEdge   = MaxLayerEdgeTop(JEdge);
         const int KRangeEdge = vertRange(KMinEdge, KMaxEdge);

         parallelForInner(
             Team, KRangeEdge, INNER_LAMBDA(int KOff) {
                const I4 K = KMinEdge + KOff;
                const Real TracerGrad =
                    TrCell(L, JCell1, K) - TrCell(L, JCell0, K);
                Del2TrCellTmp(K) -= EdgeMask(JEdge, K) *
                                    EdgeSignOnCell(ICell, J) * DvDcEdge *
                                    LayerThickEdgeMean(JEdge, K) * TracerGrad;
             });
      }
      parallelForInner(
          Team, KRangeCell, INNER_LAMBDA(int KOff) {
             const I4 K                   = KMinCell + KOff;
             Del2TracersCell(L, ICell, K) = Del2TrCellTmp(K) * InvAreaCell;
          });
   }

   void registerFields(const std::string &AuxGroupName,
                       const std::string &MeshName) const;
   void unregisterFields() const;

 private:
   I4 NVertLayers;
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DI4 CellsOnEdge;
   Array2DReal EdgeSignOnCell;
   Array1DReal DcEdge;
   Array1DReal DvEdge;
   Array1DReal AreaCell;
   Array2DReal EdgeMask;
   Array1DI4 MinLayerEdgeBot;
   Array1DI4 MaxLayerEdgeTop;
   Array1DI4 MinLayerCell;
   Array1DI4 MaxLayerCell;
};

} // namespace OMEGA
#endif
