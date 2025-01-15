#ifndef OMEGA_VERTICALADV_H
#define OMEGA_VERTICALADV_H
//===-- ocn/VerticalAdv.h - vertical advection ------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "AuxiliaryState.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "MachEnv.h"
#include "HorzMesh.h"
#include "OceanState.h"
#include "OmegaKokkos.h"
#include "TimeMgr.h"

namespace OMEGA {

class VerticalAdv {
 public:
   Array2DReal VertTransportTop;
   Array1DReal VertCoordWeights;

   VerticalAdv(const HorzMesh *Mesh, int NVertLevels);

   int computeVerticalTransport(const OceanState *State,
                                const AuxiliaryState *AuxState,
                                int TimeLevel, TimeInterval Coeff);

   int computeVerticalTransport1(const OceanState *State,
                                const AuxiliaryState *AuxState,
                                int TimeLevel, TimeInterval Coeff);

   int computeVerticalTransport2(const OceanState *State,
                                const AuxiliaryState *AuxState,
                                int TimeLevel, TimeInterval Coeff);

   KOKKOS_FUNCTION void
   computeThicknessWtdDiv(int ICell, const Array2DReal &NormalVelEdge,
                          const Array2DReal &FluxLayerThickEdge,
                          const Array2DReal &DivHU, const Array2DReal &OldSSH,
                          const Array1DReal &ProjSSH, const Real CoeffSec) {

      const Real InvAreaCell = 1._Real / AreaCell(ICell);
      Real BtrDivHU = 0.;

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);
         for (int K = 0; K < NVertLevels; ++K) {
            const Real Flux = DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                              FluxLayerThickEdge(JEdge, K) *
                              NormalVelEdge(JEdge, K);
            DivHU(ICell, K) -= Flux;
            BtrDivHU -= Flux;
         }
      }
      ProjSSH(ICell) = OldSSH(ICell, 0) - CoeffSec * BtrDivHU;
   }

   KOKKOS_FUNCTION void
   computeAleThickness(int ICell, const Array2DReal &RestingThickness,
                       const Array1DReal &ProjSSH,
                       const Array2DReal &NewThickness) {

      R8 ThicknessSum = 1.e-14;
      for (int K = 0; K < NVertLevels; ++K) {
         ThicknessSum += VertCoordWeights(K) * RestingThickness(ICell, K);
      }

      for (int K = 0; K < NVertLevels; ++K) {
         NewThickness(ICell, K) = RestingThickness(ICell, K) + (ProjSSH(ICell) *
                                  VertCoordWeights(K) * RestingThickness(ICell, K)) /
                                  ThicknessSum;
      }
   }

   KOKKOS_FUNCTION void
   computeVerticalTransportTop(int ICell, const Array2DReal &OldThickness,
                               const Array2DReal &NewThickness,
                               const Array2DReal &DivHU, const Real CoeffSec) {

      VertTransportTop(ICell, NVertLevels) = 0.;
      for (int K = NVertLevels-1; K > 0; --K) {
         VertTransportTop(ICell, K) = VertTransportTop(ICell, K+1) -
         DivHU(ICell, K) - (NewThickness(ICell, K) - OldThickness(ICell, K)) /
         CoeffSec;
      }
   }

   KOKKOS_FUNCTION void
   computeThicknessWtdDiv1(int ICell, int KChunk,
                           const Array2DReal &NormalVelEdge,
                           const Array2DReal &FluxLayerThickEdge,
                           const Array2DReal &DivHU) {

      const I4 KStart = KChunk * VecLength;
      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);
         for (int KVec = 0; KVec < VecLength; ++KVec) {
            const I4 K = KStart + KVec;
            const Real Flux = DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                              FluxLayerThickEdge(JEdge, K) *
                              NormalVelEdge(JEdge, K);
            DivHU(ICell, K) -= Flux;
         }
      }
   }

   KOKKOS_FUNCTION void
   computeVertTranspTop1(int ICell, const Array2DReal &DivHU) {

      VertTransportTop(ICell, NVertLevels) = 0.;
      for (int K = NVertLevels-1; K > 0; --K) {
         VertTransportTop(ICell, K) = VertTransportTop(ICell, K + 1) -
         DivHU(ICell, K);
      }
   }

   KOKKOS_FUNCTION void
   computeVertTransportTop2(int ICell, const Array2DReal &NormalVelEdge,
                           const Array2DReal &FluxLayerThickEdge) {
      
      Real TmpDivHU[NVertLevels + 1] = {0};

      for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
         const I4 JEdge = EdgesOnCell(ICell, J);
         for (int K = 0; K < NVertLevels; ++K) {
            TmpDivHU[K] -= DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
                           FluxLayerThickEdge(JEdge, K) *
                           NormalVelEdge(JEdge, K);
         }
      }

      VertTransportTop(ICell, NVertLevels) = 0.;
      for (int K = NVertLevels - 1; K > 0; --K) {
         VertTransportTop(ICell, K) = VertTransportTop(ICell, K + 1) -
         TmpDivHU[ICell, K];
      }
   }



 private:
   const HorzMesh *Mesh;

   Array1DI4 NEdgesOnCell;
   Array2DI4 EdgesOnCell;
   Array2DReal EdgeSignOnCell;
   Array1DReal AreaCell;
   Array1DReal DvEdge;

   I4 NVertLevels;
   I4 NCellsAll;

};

}

//===----------------------------------------------------------------------===//
#endif // defined OMEGA_VERTICALADV_H
