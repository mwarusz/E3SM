//===-- Test driver for OMEGA Vertical Advection  ----------------*- C++ -*-===/
//
//===-----------------------------------------------------------------------===/

#include "VerticalAdv.h"
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "TimeStepper.h"

#include "Pacer.h"
#include <iostream>
#include <mpi.h>

using namespace OMEGA;

using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using TeamMember = TeamPolicy::member_type;
using Kokkos::PerTeam;
using Kokkos::TeamThreadRange;

constexpr int NVertLevels = 128;

constexpr int VLength = 64;
// constexpr int VLength = 32;
// constexpr int VLength = 16;
// constexpr int VLength = 8;
// constexpr int VLength = 4;
// constexpr int VLength = 2;
// constexpr int VLength = 1;

int initVadvTest() {

   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   if (DefEnv->isMasterTask()) {
      DefEnv->print();
   }

   initLogging(DefEnv);

   // Open config file
   Config("Omega");
   Err = Config::readAll("omega.yml");
   if (Err != 0) {
      LOG_CRITICAL("VertAdvTest: Error reading config file");
      return Err;
   }

   int TimeStepperErr = TimeStepper::init1();
   if (TimeStepperErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing default time stepper");
   }

   int IOErr = IO::init(DefComm);
   if (IOErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing parallel IO");
   }

   int DecompErr = Decomp::init();
   if (DecompErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing default decomposition");
   }

   int HaloErr = Halo::init();
   if (HaloErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing default halo");
   }

   int MeshErr = HorzMesh::init();
   if (MeshErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing default mesh");
   }

   int TracerErr = Tracers::init();
   if (TracerErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing tracer infrastructure");
   }

   const auto &Mesh = HorzMesh::getDefault();
   // Horz dimensions created in HorzMesh
   auto VertDim = Dimension::create("NVertLevels", NVertLevels);

   int StateErr = OceanState::init();
   if (StateErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing default state");
   }

   int AuxStateErr = AuxiliaryState::init();
   if (AuxStateErr != 0) {
      Err++;
      LOG_ERROR("VertAdvTest: error initializing default aux state");
   }

   return Err;
}

void finalizeVadvTest() {
   AuxiliaryState::clear();
   Tracers::clear();
   OceanState::clear();
   Field::clear();
   Dimension::clear();
   TimeStepper::clear();
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();
}

void initArrays() {

   const auto *Mesh      = HorzMesh::getDefault();
   const auto *DefDecomp = Decomp::getDefault();
   auto *State           = OceanState::getDefault();

   int TimeLevel = 0;
   State->copyToHost(TimeLevel);
   HostArray2DReal LayerThicknessH;
   State->getLayerThicknessH(LayerThicknessH, TimeLevel);
   HostArray2DReal NormalVelocityH;
   State->getNormalVelocityH(NormalVelocityH, TimeLevel);
   Array2DReal LayerThickness;
   State->getLayerThickness(LayerThickness, TimeLevel);
   Array2DReal NormalVelocity;
   State->getNormalVelocity(NormalVelocity, TimeLevel);

   AuxiliaryState *AuxState = AuxiliaryState::getDefault();

   OMEGA_SCOPE(LocEdgeSignOnCell, Mesh->EdgeSignOnCellH);

   //   int NCells = LayerThicknessH.extent(0);
   //   int NLevels = LayerThicknessH.extent(1);
   int NCells  = LayerThickness.extent(0);
   int NLevels = LayerThickness.extent(1);
   //   for (int ICell = 0; ICell < NCells; ++ICell) {
   //      for (int K = 0; K < NLevels; ++K) {
   ////         Real NewVal = DefDecomp->CellIDH(ICell) * NLevels + K;
   //         Real NewVal = 100._Real * (K + 1);
   //         LayerThicknessH(ICell, K) = NewVal;
   ////         std::cout << ICell << " " << K << " " << LayerThicknessH(ICell,
   /// K) << std::endl;
   //      }
   //      for (int J = 0; J < LocEdgeSignOnCell.extent(1); ++J) {
   ////         std::cout << ICell << " " << J << " " <<
   /// LocEdgeSignOnCell(ICell, J) << std::endl;
   //      }
   //   }
   parallelFor(
       "initLayThick", {NCells, NLevels}, KOKKOS_LAMBDA(int ICell, int K) {
          Real NewVal              = 100._Real * (K + 1);
          LayerThickness(ICell, K) = NewVal;
       });

   //   for (int IEdge = 0; IEdge < DefDecomp->NEdgesAll; ++IEdge) {
   ////      std::cout << IEdge << " " << Mesh->DvEdgeH(IEdge) << std::endl;
   //      Real Sign;
   //      if (IEdge % 2 == 0) {
   //        Sign = 1.;
   //      } else {
   //        Sign = -1.;
   //      }
   //      Real Mod = 0.1_Real * double(IEdge % 10);
   //      for (int K = 0; K < NLevels; ++K) {
   //         NormalVelocityH(IEdge, K) = Sign * (5._Real + Mod);
   ////         std::cout << IEdge << " " << K << " " << NormalVelocityH(IEdge,
   /// K)  << std::endl;
   //      }
   //   }

   parallelFor(
       "intNormVel", {DefDecomp->NEdgesAll, NLevels},
       KOKKOS_LAMBDA(int IEdge, int K) {
          Real Sign;
          if (IEdge % 2 == 0) {
             Sign = 1.;
          } else {
             Sign = -1.;
          }
          Real Mod                 = 0.1_Real * double(IEdge % 10);
          NormalVelocity(IEdge, K) = Sign * (5._Real + Mod);
       });

   //   State->copyToDevice(TimeLevel);
   State->copyToHost(TimeLevel);
   Kokkos::fence();

   AuxState->computeMomAux(State, TimeLevel, TimeLevel);
   Kokkos::fence();

   //   auto FluxLayerThickEdgeH =
   //   createHostMirrorCopy(AuxState->LayerThicknessAux.FluxLayerThickEdge);
   //   for (int I1 = 0; I1 < FluxLayerThickEdgeH.extent(0); ++I1) {
   //      for (int I2 = 0; I2 < FluxLayerThickEdgeH.extent(1); ++I2) {
   ////         std::cout << I1 << " " << I2 << " " << FluxLayerThickEdgeH(I1,
   /// I2) << std::endl;
   //      }
   //   }

   //   std::cout << FluxLayerThickEdgeH.extent(0) << " " <<
   //   FluxLayerThickEdgeH.extent(1) << std::endl;
}

int main(int argc, char *argv[]) {

   int RetErr = 0;

   // Initialize the global MPI environment
   MPI_Init(&argc, &argv);
   Kokkos::initialize();

   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");
   {

      Pacer::start("Init");
      initVadvTest();

      //      std::cout << " main " << std::endl;
      Pacer::start("SetArrays");
      initArrays();
      Pacer::stop("SetArrays");

      const auto *Mesh = HorzMesh::getDefault();

      int VectorLength = 1;
      //      int VectorLength = 2;
      //      int VectorLength = 4;
      //      int VectorLength = 8;
      //      int VectorLength = 16;
      //      int VectorLength = 32;
      //      int VectorLength = 64;
      //      int VadvErr = VerticalAdv::init(NVertLevels, VectorLength);
      //      VerticalAdv VadvObj(Mesh, NVertLevels, VectorLength);
      //      VerticalAdv VadvObj;
      //      VadvObj.init(NVertLevels, VectorLength);

      int TimeLev             = 0;
      std::string TimeStepStr = "0000_00:10:00";
      TimeInterval TimeStep(TimeStepStr);

      OceanState *State        = OceanState::getDefault();
      AuxiliaryState *AuxState = AuxiliaryState::getDefault();

      Array2DReal NormalVelEdge;
      State->getNormalVelocity(NormalVelEdge, TimeLev);
      const Array2DReal &FluxLayerThickEdge =
          AuxState->LayerThicknessAux.FluxLayerThickEdge;

      Array2DReal TmpArray("Tmp", Mesh->NCellsSize, NVertLevels);
      Array2DReal DivHU("DivHU", Mesh->NCellsSize, NVertLevels);
      Array2DReal VertTransportTop("VertTranspTop", Mesh->NCellsSize,
                                   NVertLevels + 1);
      std::cout << 0 << std::endl;

      OMEGA_SCOPE(LocAreaCell, Mesh->AreaCell);
      OMEGA_SCOPE(LocNEonC, Mesh->NEdgesOnCell);
      OMEGA_SCOPE(LocEonC, Mesh->EdgesOnCell);
      OMEGA_SCOPE(LocESonC, Mesh->EdgeSignOnCell);
      OMEGA_SCOPE(LocDvE, Mesh->DvEdge);

      Kokkos::fence();
      Pacer::stop("Init");

      Pacer::start("Run");
      Pacer::start("1stVadv1");
      {
         const int NVertLs = Mesh->NVertLevels;
         const int NChunks = NVertLevels / VLength;
         parallelFor(
             "Run1-0", {Mesh->NCellsAll, NChunks},
             KOKKOS_LAMBDA(int ICell, int KChunk) {
                const int KStart = KChunk * VLength;
                for (int KVec = 0; KVec < VLength; ++KVec) {
                   const I4 K      = KStart + KVec;
                   DivHU(ICell, K) = 0;
                }
             });
         Kokkos::fence();

         parallelFor(
             "Run1-1", {Mesh->NCellsAll, NChunks},
             KOKKOS_LAMBDA(int ICell, int KChunk) {
                const Real InvAreaCell = 1._Real / LocAreaCell(ICell);
                const I4 KStart        = KChunk * VLength;

                for (int J = 0; J < LocNEonC(ICell); ++J) {
                   const I4 JEdge = LocEonC(ICell, J);
                   for (int KVec = 0; KVec < VLength; ++KVec) {
                      const I4 K      = KStart + KVec;
                      const Real Flux = LocDvE(JEdge) * LocESonC(ICell, J) *
                                        FluxLayerThickEdge(JEdge, K) *
                                        NormalVelEdge(JEdge, K) * InvAreaCell;
                      DivHU(ICell, K) -= Flux;
                   }
                }
             });
         Kokkos::fence();

         parallelFor(
             "Run1-2", {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
                const int NVertLs                    = DivHU.extent(1);
                VertTransportTop(ICell, NVertLevels) = 0.;
                for (int K = NVertLevels - 1; K > 0; --K) {
                   VertTransportTop(ICell, K) =
                       VertTransportTop(ICell, K + 1) - DivHU(ICell, K);
                }
                VertTransportTop(ICell, 0) = 0.;
             });
         Kokkos::fence();
      }
      Pacer::stop("1stVadv1");

      Pacer::start("1stVadv2");
      {
         const int NVertLs = Mesh->NVertLevels;
         Array1DReal TmpDivHU("TmpDivHU", NVertLs);

         parallelFor(
             "Run2", {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
                const int NVertLs      = NormalVelEdge.extent(1);
                const Real InvAreaCell = 1._Real / LocAreaCell(ICell);
                for (int J = 0; J < LocNEonC(ICell); ++J) {
                   const I4 JEdge = LocEonC(ICell, J);
                   for (int K = 0; K < NVertLs; ++K) {
                      Real TmpVal = LocDvE(JEdge) * LocESonC(ICell, J) *
                                    FluxLayerThickEdge(JEdge, K) *
                                    NormalVelEdge(JEdge, K) * InvAreaCell;
                      TmpDivHU(K) -= TmpVal;
                   }
                }
                VertTransportTop(ICell, NVertLevels) = 0.;
                for (int K = NVertLevels - 1; K > 0; --K) {
                   VertTransportTop(ICell, K) =
                       VertTransportTop(ICell, K + 1) - TmpDivHU(K);
                }
                VertTransportTop(ICell, 0) = 0.;
             });
         Kokkos::fence();
      }
      Pacer::stop("1stVadv2");

      Pacer::start("1stVadv3");
      {
         const int NVertLs = Mesh->NVertLevels;
         const int NChunks = NVertLevels / VLength;

#ifdef OMEGA_TARGET_DEVICE
         const int TeamSize = 64;
#else
         const int TeamSize = 1;
#endif
         const auto Policy = TeamPolicy(Mesh->NCellsAll, TeamSize, 1);
         Kokkos::parallel_for(
             "Run3", Policy, KOKKOS_LAMBDA(const TeamMember &Member) {
                const int ICell = Member.league_rank();

                Kokkos::parallel_for(
                    TeamThreadRange(Member, NChunks), [&](int KChunk) {
                       const Real InvAreaCell = 1._Real / LocAreaCell(ICell);
                       const I4 KStart        = KChunk * VLength;

                       Real DivHUTmp[VLength] = {0};

                       for (int J = 0; J < LocNEonC(ICell); ++J) {
                          const I4 JEdge = LocEonC(ICell, J);
                          for (int KVec = 0; KVec < VLength; ++KVec) {
                             const I4 K = KStart + KVec;
                             DivHUTmp[KVec] -=
                                 LocDvE(JEdge) * LocESonC(ICell, J) *
                                 FluxLayerThickEdge(JEdge, K) *
                                 NormalVelEdge(JEdge, K) * InvAreaCell;
                          }
                       }
                       for (int KVec = 0; KVec < VLength; ++KVec) {
                          const int K     = KStart + KVec;
                          DivHU(ICell, K) = DivHUTmp[KVec];
                       }
                    });

                Member.team_barrier();

                Kokkos::parallel_scan(TeamThreadRange(Member, NVertLs),
                                      [&](int K, Real &Accum, bool IsFinal) {
                                         const int KRev = NVertLs - 1 - K;
                                         if (IsFinal) {
                                            VertTransportTop(ICell, KRev + 1) =
                                                Accum;
                                         }
                                         Accum -= DivHU(ICell, KRev);
                                      });

                Kokkos::single(PerTeam(Member),
                               [&]() { VertTransportTop(ICell, 0) = 0; });
             });
         Kokkos::fence();
      }
      Pacer::stop("1stVadv3");

      for (int Istep = 0; Istep < 100; ++Istep) {
         std::cout << Istep + 1 << std::endl;

         Pacer::start("vadv1");
         {
            const int NVertLs = Mesh->NVertLevels;
            const int NChunks = NVertLevels / VLength;

            parallelFor(
                "Run1-1", {Mesh->NCellsAll, NChunks},
                KOKKOS_LAMBDA(int ICell, int KChunk) {
                   const Real InvAreaCell = 1._Real / LocAreaCell(ICell);
                   const I4 KStart        = KChunk * VLength;

                   Real DivHUTmp[VLength] = {0};

                   for (int J = 0; J < LocNEonC(ICell); ++J) {
                      const I4 JEdge = LocEonC(ICell, J);
                      for (int KVec = 0; KVec < VLength; ++KVec) {
                         const I4 K = KStart + KVec;
                         DivHUTmp[KVec] -= LocDvE(JEdge) * LocESonC(ICell, J) *
                                           FluxLayerThickEdge(JEdge, K) *
                                           NormalVelEdge(JEdge, K) *
                                           InvAreaCell;
                      }
                   }
                   for (int KVec = 0; KVec < VLength; ++KVec) {
                      const int K     = KStart + KVec;
                      DivHU(ICell, K) = DivHUTmp[KVec];
                   }
                });
            Kokkos::fence();

            parallelFor(
                "Run1-2", {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
                   const int NVertLs                    = DivHU.extent(1);
                   VertTransportTop(ICell, NVertLevels) = 0.;
                   for (int K = NVertLevels - 1; K > 0; --K) {
                      VertTransportTop(ICell, K) =
                          VertTransportTop(ICell, K + 1) - DivHU(ICell, K);
                   }
                   VertTransportTop(ICell, 0) = 0.;
                });
         }

         Kokkos::fence();
         Pacer::stop("vadv1");

         Pacer::start("vadv2");
         {
            const int NVertLs = Mesh->NVertLevels;

            parallelFor(
                "Run2", {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
                   const Real InvAreaCell     = 1._Real / LocAreaCell(ICell);
                   Real DivHUTmp[NVertLevels] = {0};
                   for (int J = 0; J < LocNEonC(ICell); ++J) {
                      const I4 JEdge = LocEonC(ICell, J);
                      for (int K = 0; K < NVertLs; ++K) {
                         DivHUTmp[K] -= LocDvE(JEdge) * LocESonC(ICell, J) *
                                        FluxLayerThickEdge(JEdge, K) *
                                        NormalVelEdge(JEdge, K) * InvAreaCell;
                      }
                   }
                   for (int K = 0; K < NVertLevels; ++K) {
                      DivHU(ICell, K) = DivHUTmp[K];
                   }
                   VertTransportTop(ICell, NVertLevels) = 0.;
                   for (int K = NVertLevels - 1; K > 0; --K) {
                      VertTransportTop(ICell, K) =
                          VertTransportTop(ICell, K + 1) - DivHU(ICell, K);
                   }
                   VertTransportTop(ICell, 0) = 0.;
                });
            Kokkos::fence();
         }
         Pacer::stop("vadv2");

         Pacer::start("vadv3");
         {
            const int NVertLs = Mesh->NVertLevels;
            const int NChunks = NVertLevels / VLength;
#ifdef OMEGA_TARGET_DEVICE
            const int TeamSize = 64;
#else
            const int TeamSize = 1;
#endif
            const auto Policy = TeamPolicy(Mesh->NCellsAll, TeamSize, 1);
            Kokkos::parallel_for(
                "Run3", Policy, KOKKOS_LAMBDA(const TeamMember &Member) {
                   const int ICell = Member.league_rank();

                   Kokkos::parallel_for(
                       TeamThreadRange(Member, NChunks), [&](int KChunk) {
                          const Real InvAreaCell = 1._Real / LocAreaCell(ICell);
                          const I4 KStart        = KChunk * VLength;

                          Real DivHUTmp[VLength] = {0};

                          for (int J = 0; J < LocNEonC(ICell); ++J) {
                             const I4 JEdge = LocEonC(ICell, J);
                             for (int KVec = 0; KVec < VLength; ++KVec) {
                                const I4 K = KStart + KVec;
                                DivHUTmp[KVec] -=
                                    LocDvE(JEdge) * LocESonC(ICell, J) *
                                    FluxLayerThickEdge(JEdge, K) *
                                    NormalVelEdge(JEdge, K) * InvAreaCell;
                             }
                          }
                          for (int KVec = 0; KVec < VLength; ++KVec) {
                             const int K     = KStart + KVec;
                             DivHU(ICell, K) = DivHUTmp[KVec];
                          }
                       });

                   Member.team_barrier();

                   Kokkos::parallel_scan(TeamThreadRange(Member, NVertLs),
                                         [&](int K, Real &Accum, bool IsFinal) {
                                            const int KRev = NVertLs - 1 - K;
                                            if (IsFinal) {
                                               VertTransportTop(
                                                   ICell, KRev + 1) = Accum;
                                            }
                                            Accum -= DivHU(ICell, KRev);
                                         });

                   Kokkos::single(PerTeam(Member),
                                  [&]() { VertTransportTop(ICell, 0) = 0; });
                });
            Kokkos::fence();
         }
         Pacer::stop("vadv3");
      }

      Pacer::stop("Run");

      Pacer::start("Finalize");
      //      VerticalAdv::clear();
      finalizeVadvTest();
      Pacer::stop("Finalize");
   }

   Pacer::print("omega");
   Pacer::finalize();
   Kokkos::finalize();
   MPI_Finalize();

   return RetErr;
}
