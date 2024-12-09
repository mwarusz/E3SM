#ifndef OMEGA_TRIDIAGONAL_H
#define OMEGA_TRIDIAGONAL_H

#include "DataTypes.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"

namespace OMEGA {

using ScratchArray2DReal = Kokkos::View<Real *[VecLength], MemLayout,
                                        ScratchMemSpace, MemoryUnmanaged>;

struct DiffusionScratch {
   ScratchArray2DReal G;
   ScratchArray2DReal H;
   ScratchArray2DReal X;
   ScratchArray2DReal Alpha;

   KOKKOS_FUNCTION DiffusionScratch(const TeamMember &Member, int NRow)
       : G(Member.team_scratch(0), NRow), H(Member.team_scratch(0), NRow),
         X(Member.team_scratch(0), NRow), Alpha(Member.team_scratch(0), NRow) {}
};

struct ThomasSolver {

   static void setScratchSize(TeamPolicy &Policy, int NRow) {
      Policy.set_scratch_size(
          0, Kokkos::PerTeam(4 * NRow * VecLength * sizeof(Real)));
   }

   static void KOKKOS_FUNCTION solve(const TeamMember &Member,
                                     const Array2DReal &DL,
                                     const Array2DReal &D,
                                     const Array2DReal &DU,
                                     const Array2DReal &X) {
      const int NBatch = X.extent_int(0);
      const int NRow   = X.extent_int(1);

      const int IStart = Member.league_rank() * VecLength;

      ScratchArray2DReal ScratchDL(Member.team_scratch(0), NRow);
      ScratchArray2DReal ScratchD(Member.team_scratch(0), NRow);
      ScratchArray2DReal ScratchDU(Member.team_scratch(0), NRow);
      ScratchArray2DReal ScratchX(Member.team_scratch(0), NRow);

      for (int K = 0; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               ScratchDL(K, IVec) = DL(I, K);
               ScratchD(K, IVec)  = D(I, K);
               ScratchDU(K, IVec) = DU(I, K);
               ScratchX(K, IVec)  = X(I, K);
            } else {
               ScratchDL(K, IVec) = 0;
               ScratchD(K, IVec)  = 1;
               ScratchDU(K, IVec) = 0;
               ScratchX(K, IVec)  = 0;
            }
         }
      }

      for (int K = 1; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const Real W = ScratchDL(K, IVec) / ScratchD(K - 1, IVec);
            ScratchD(K, IVec) -= W * ScratchDU(K - 1, IVec);
            ScratchX(K, IVec) -= W * ScratchX(K - 1, IVec);
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         ScratchX(NRow - 1, IVec) /= ScratchD(NRow - 1, IVec);
      }

      for (int K = NRow - 2; K >= 0; --K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            ScratchX(K, IVec) = (ScratchX(K, IVec) -
                                 ScratchDU(K, IVec) * ScratchX(K + 1, IVec)) /
                                ScratchD(K, IVec);
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         for (int K = 0; K < NRow; ++K) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               X(I, K) = ScratchX(K, IVec);
            }
         }
      }
   }

   static void KOKKOS_FUNCTION solveDiffusionSystem(
       const TeamMember &Member, const DiffusionScratch &Scratch) {
      const int NRow = Scratch.X.extent_int(0);

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.Alpha(0, IVec) =
             Scratch.G(0, IVec) * (1 - Scratch.G(0, IVec) / Scratch.H(0, IVec));
      }

      for (int K = 1; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            Scratch.Alpha(K, IVec) =
                Scratch.G(K - 1, IVec) *
                (Scratch.H(K - 1, IVec) + Scratch.Alpha(K - 1, IVec)) /
                (Scratch.H(K - 1, IVec) + Scratch.Alpha(K - 1, IVec) +
                 Scratch.G(K - 1, IVec));
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.H(0, IVec) += Scratch.G(0, IVec);
      }

      for (int K = 1; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const Real AddH = Scratch.Alpha(K, IVec) + Scratch.G(K, IVec);

            Scratch.H(K, IVec) += AddH;
            Scratch.X(K, IVec) += Scratch.G(K - 1, IVec) /
                                  Scratch.H(K - 1, IVec) *
                                  Scratch.X(K - 1, IVec);
         }
      }

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         Scratch.X(NRow - 1, IVec) /= Scratch.H(NRow - 1, IVec);
      }

      for (int K = NRow - 2; K >= 0; --K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            Scratch.X(K, IVec) = (Scratch.X(K, IVec) +
                                  Scratch.G(K, IVec) * Scratch.X(K + 1, IVec)) /
                                 Scratch.H(K, IVec);
         }
      }
   }

   static void KOKKOS_FUNCTION solveDiffusionSystem(const TeamMember &Member,
                                                    const Array2DReal &G,
                                                    const Array2DReal &H,
                                                    const Array2DReal &X) {
      const int NBatch = X.extent_int(0);
      const int NRow   = X.extent_int(1);

      const int IStart = Member.league_rank() * VecLength;

      DiffusionScratch Scratch(Member, NRow);

      for (int K = 0; K < NRow; ++K) {
         for (int IVec = 0; IVec < VecLength; ++IVec) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               Scratch.G(K, IVec) = G(I, K);
               Scratch.H(K, IVec) = H(I, K);
               Scratch.X(K, IVec) = X(I, K);
            } else {
               Scratch.G(K, IVec) = 0;
               Scratch.H(K, IVec) = 1;
               Scratch.X(K, IVec) = 0;
            }
         }
      }

      solveDiffusionSystem(Member, Scratch);

      for (int IVec = 0; IVec < VecLength; ++IVec) {
         for (int K = 0; K < NRow; ++K) {
            const int I = IStart + IVec;
            if (I < NBatch) {
               X(I, K) = Scratch.X(K, IVec);
            }
         }
      }
   }
};

} // namespace OMEGA
#endif
