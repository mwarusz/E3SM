#include "Tridiagonal.h"
#include <iostream>

using namespace OMEGA;

int testCorrectness() {
   int NBatch = 11;
   int NRow   = 64;

   Array2DReal DL("DL", NBatch, NRow);
   Array2DReal D("D", NBatch, NRow);
   Array2DReal DU("DU", NBatch, NRow);
   Array2DReal X("X", NBatch, NRow);

   parallelFor(
       {NBatch, NRow}, KOKKOS_LAMBDA(int I, int K) {
          X(I, K) = 0.1 * K;

          DL(I, K) = K == 0 ? 0 : 1 + 0.1 * (I % 3) + 0.2 * (K % 7);
          D(I, K)  = 4 + 0.2 * (I % 11) + 0.1 * (K % 5);
          DU(I, K) = K == NRow - 1 ? 0 : 1 + 0.05 * (I % 5) - 0.1 * (K % 11);
       });

   Array2DReal AX("AX", NBatch, NRow);

   parallelFor(
       {NBatch, NRow}, KOKKOS_LAMBDA(int I, int K) {
          AX(I, K) = D(I, K) * X(I, K);
          if (K > 0) {
             AX(I, K) += DL(I, K) * X(I, K - 1);
          }
          if (K < NRow - 1) {
             AX(I, K) += DU(I, K) * X(I, K + 1);
          }
       });

   TeamPolicy Policy((NBatch + VecLength - 1) / VecLength, 1, 1);
   ThomasSolver::setScratchSize(Policy, NRow);

   Kokkos::parallel_for(
       Policy, KOKKOS_LAMBDA(const TeamMember &Member) {
          ThomasSolver::solve(Member, DL, D, DU, AX);
       });

   Real Error;
   parallelReduce(
       {NBatch, NRow},
       KOKKOS_LAMBDA(int I, int K, Real &Accum) {
          Accum = Kokkos::max(Accum, Kokkos::abs(X(I, K) - AX(I, K)));
       },
       Kokkos::Max<Real>(Error));

   std::cout << Error << std::endl;

   return 0;
}

int testDiffusionCorrectness() {
   int NBatch = 11;
   int NRow   = 64;

   Array2DReal G("G", NBatch, NRow);
   Array2DReal H("H", NBatch, NRow);
   Array2DReal X("X", NBatch, NRow);

   parallelFor(
       {NBatch, NRow}, KOKKOS_LAMBDA(int I, int K) {
          X(I, K) = 0.1 * K;

          G(I, K) = K == 0 ? 0 : 1 + 0.1 * (I % 3) + 0.2 * (K % 7);
          H(I, K) = 4 + 0.2 * (I % 11) + 0.1 * (K % 5);
       });

   Array2DReal AX("AX", NBatch, NRow);

   parallelFor(
       {NBatch, NRow}, KOKKOS_LAMBDA(int I, int K) {
          const Real DL = K == 0 ? 0 : -G(I, K - 1);
          const Real DU = -G(I, K);
          const Real D  = H(I, K) - DL - DU;

          AX(I, K) = D * X(I, K);
          if (K > 0) {
             AX(I, K) += DL * X(I, K - 1);
          }
          if (K < NRow - 1) {
             AX(I, K) += DU * X(I, K + 1);
          }
       });

   TeamPolicy Policy((NBatch + VecLength - 1) / VecLength, 1, 1);
   ThomasSolver::setScratchSize(Policy, NRow);

   Kokkos::parallel_for(
       Policy, KOKKOS_LAMBDA(const TeamMember &Member) {
          ThomasSolver::solveDiffusionSystem(Member, G, H, AX);
       });

   Real Error;
   parallelReduce(
       {NBatch, NRow},
       KOKKOS_LAMBDA(int I, int K, Real &Accum) {
          Accum = Kokkos::max(Accum, Kokkos::abs(X(I, K) - AX(I, K)));
       },
       Kokkos::Max<Real>(Error));

   std::cout << Error << std::endl;

   return 0;
}

int testDiffusionStability() {
   const int NCells    = 100;
   const int NVertices = NCells + 1;

   const Real DX = 1._Real / NCells;
   const Real DT = 1;

   const Real LargeVal = 1e50;

   Array1DReal DiffCoeff("DiffCoeff", NVertices);
   parallelFor(
       {NVertices}, KOKKOS_LAMBDA(int IVertex) {
          const Real XVertex = IVertex * DX;
          DiffCoeff(IVertex) = Kokkos::abs(XVertex - 0.5) < 0.2 ? LargeVal : 0;
       });

   Array1DReal Tracer("Tracer", NCells);
   parallelFor(
       {NCells}, KOKKOS_LAMBDA(int ICell) {
          const Real XCell = (ICell + 0.5_Real) * DX;
          const Real XTmp  = XCell - 0.5_Real;
          Tracer(ICell)    = exp(-XTmp * XTmp);
       });

   TeamPolicy Policy((NVertices + VecLength - 1) / VecLength, 1, 1);
   ThomasSolver::setScratchSize(Policy, NVertices);

   for (int Iter = 0; Iter < 100; ++Iter) {
      Kokkos::parallel_for(
          Policy, KOKKOS_LAMBDA(const TeamMember &Member) {
             DiffusionScratch Scratch(Member, NCells);

             Kokkos::parallel_for(
                 TeamThreadRange(Member, NCells), KOKKOS_LAMBDA(int ICell) {
                    for (int IVec = 0; IVec < VecLength; ++IVec) {
                       Scratch.H(ICell, IVec) = DX;
                       Scratch.G(ICell, IVec) = DiffCoeff(ICell + 1) * DT / DX;
                       Scratch.X(ICell, IVec) =
                           Tracer(ICell) * Scratch.H(ICell, IVec);
                    }
                 });

             Member.team_barrier();
             ThomasSolver::solveDiffusionSystem(Member, Scratch);
             Member.team_barrier();

             Kokkos::parallel_for(
                 TeamThreadRange(Member, NCells), KOKKOS_LAMBDA(int ICell) {
                    Tracer(ICell) = Scratch.X(ICell, 0);
                 });
          });
   }

   Real Error;
   parallelReduce(
       {NCells},
       KOKKOS_LAMBDA(int ICell, Real &Accum) {
          Accum += Kokkos::abs(Tracer(ICell));
       },
       Error);

   std::cout << Error << std::endl;

   return 0;
}

int tridiagonalTest() {
   int Err = 0;

   Err += testCorrectness();
   Err += testDiffusionCorrectness();
   Err += testDiffusionStability();

   return Err;
}

int main(int argc, char *argv[]) {

   int RetVal = 0;

   Kokkos::initialize(argc, argv);

   RetVal += tridiagonalTest();

   Kokkos::finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
