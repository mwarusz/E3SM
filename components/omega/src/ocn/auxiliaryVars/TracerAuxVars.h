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

   template <class FC>
   KOKKOS_FUNCTION void
   computeVarsOnEdge(int L, int IEdge, int KStart,
                     const Array2DReal &NormalVelEdge, const Array2DReal &HCell,
                     const Array3DReal &TrCell, FC FullChunk) const {

      // const int KLen =
      //     FullChunk ? VecLength : MaxLayerEdgeTop(IEdge) - KStart + 1;
      const int KLen =
          Kokkos::min(VecLength, MaxLayerEdgeTop(IEdge) - KStart + 1);

      const int JCell0 = CellsOnEdge(IEdge, 0);
      const int JCell1 = CellsOnEdge(IEdge, 1);

      const auto LTrCell = subview(TrCell, L, Kokkos::ALL, Kokkos::ALL);
      const auto LHTracersEdge =
          subview(HTracersEdge, L, Kokkos::ALL, Kokkos::ALL);

      Real HTracerTmp[VecLength];

      switch (TracersOnEdgeChoice) {
      case FluxTracerEdgeOption::Center:
         for (int KVec = 0; KVec < KLen; ++KVec) {
            const int K = KStart + KVec;
            HTracerTmp[KVec] =
                0.5_Real * (HCell(JCell0, K) * LTrCell(JCell0, K) +
                            HCell(JCell1, K) * LTrCell(JCell1, K));
         }
         break;
      case FluxTracerEdgeOption::Upwind:
         for (int KVec = 0; KVec < KLen; ++KVec) {
            const int K = KStart + KVec;
            if (NormalVelEdge(IEdge, K) > 0) {
               HTracerTmp[KVec] = HCell(JCell0, K) * LTrCell(JCell0, K);
            } else if (NormalVelEdge(IEdge, K) < 0) {
               HTracerTmp[KVec] = HCell(JCell1, K) * LTrCell(JCell1, K);
            } else {
               HTracerTmp[KVec] =
                   Kokkos::max(HCell(JCell0, K) * LTrCell(JCell0, K),
                               HCell(JCell1, K) * LTrCell(JCell1, K));
            }
         }
         break;
      }

      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K             = KStart + KVec;
         LHTracersEdge(IEdge, K) = HTracerTmp[KVec];
      }
   }

   template <class FC>
   KOKKOS_FUNCTION void
   computeVarsOnCells(int L, int ICell, int KStart,
                      const Array2DReal &LayerThickEdgeMean,
                      const Array3DReal &TrCell, FC FullChunk) const {

      // const int KLen = FullChunk ? VecLength : MaxLayerCell(ICell) - KStart +
      // 1;
      const int KLen = Kokkos::min(VecLength, MaxLayerCell(ICell) - KStart + 1);

      const Real InvAreaCell = 1._Real / AreaCell(ICell);

      const auto LDel2TracersCell =
          subview(Del2TracersCell, L, Kokkos::ALL, Kokkos::ALL);
      const auto LTrCell = subview(TrCell, L, Kokkos::ALL, Kokkos::ALL);

      Real Del2TrCellTmp[VecLength] = {0};

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const int JEdge = EdgesOnCell(ICell, J);

         const int JCell0 = CellsOnEdge(JEdge, 0);
         const int JCell1 = CellsOnEdge(JEdge, 1);

         const Real DvDcEdge = DvEdge(JEdge) / DcEdge(JEdge);

         for (int KVec = 0; KVec < KLen; ++KVec) {
            const int K           = KStart + KVec;
            const Real TracerGrad = LTrCell(JCell1, K) - LTrCell(JCell0, K);
            Del2TrCellTmp[KVec] -= EdgeMask(JEdge, K) *
                                   EdgeSignOnCell(ICell, J) * DvDcEdge *
                                   LayerThickEdgeMean(JEdge, K) * TracerGrad;
         }
      }
      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K                  = KStart + KVec;
         Del2TracersCell(L, ICell, K) = Del2TrCellTmp[KVec] * InvAreaCell;
      }
   }

   void registerFields(const std::string &AuxGroupName,
                       const std::string &MeshName) const;
   void unregisterFields() const;

 private:
   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DI4 CellsOnEdge;
   Array2DReal EdgeSignOnCell;
   Array1DReal DcEdge;
   Array1DReal DvEdge;
   Array1DReal AreaCell;
   Array2DReal EdgeMask;
   Array1DI4 MaxLayerEdgeTop;
   Array1DI4 MaxLayerCell;
};

} // namespace OMEGA
#endif
