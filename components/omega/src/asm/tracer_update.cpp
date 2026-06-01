#include "DataTypes.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void tracer_update(int L, int ICell, const TeamMember &Team,
                   const Array3DReal &NextTracers,
                   const Array3DReal &CurTracers, const Array3DReal &TracerTend,
                   const Array2DReal &PseudoThick1,
                   const Array2DReal &PseudoThick2,
                   const Array1DI4 &MinLayerCell, const Array1DI4 &MaxLayerCell,
                   Real CoeffSeconds) {

   const int KMin = MinLayerCell(ICell);
   const int KMax = MaxLayerCell(ICell);

   // const auto LNext = subviewUnmanaged(NextTracers, L, Kokkos::ALL,
   // Kokkos::ALL); const auto LCur = subviewUnmanaged(CurTracers, L,
   // Kokkos::ALL, Kokkos::ALL); const auto LTend = subviewUnmanaged(TracerTend,
   // L, Kokkos::ALL, Kokkos::ALL);

   parallelForInner(
       Team, Range{KMin, KMax}, INNER_LAMBDA(int K) {
          NextTracers(L, ICell, K) =
              (CurTracers(L, ICell, K) * PseudoThick2(ICell, K) +
               CoeffSeconds * TracerTend(L, ICell, K)) /
              PseudoThick1(ICell, K);

          // LNext(ICell, K) =
          //     (LCur(ICell, K) * PseudoThick2(ICell, K) +
          //      CoeffSeconds * LTend(ICell, K)) /
          //      PseudoThick1(ICell, K);
       });
}
