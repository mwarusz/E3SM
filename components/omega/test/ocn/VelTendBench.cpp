//===-- Test driver for OMEGA tendency terms ---------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA tendency term functors
///
/// This driver tests the functors used to calculate the tendencies used to
/// update OMEGA state variables. The tests are designed to be run with the
/// planar and spherical meshes described in the OMEGA Quick Start. For each
/// functor, input arrays are initialized based on arbitrary periodic functions
/// defined in the structs for the planar and spherical configurations. The
/// difference between analytical solutions and the output of each function
/// are used to calculate L2 and L-Infinity error norms, which are compared to
/// expected values for the given mesh.
///
//
//===-----------------------------------------------------------------------===/
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Error.h"
#include "Field.h"
#include "GlobalConstants.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "OceanTestCommon.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "TendencyTerms.h"
#include "TimeStepper.h"
#include "Tracers.h"
#include "VertCoord.h"
#include "mpi.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace OMEGA;

constexpr Geometry Geom          = Geometry::Planar;
constexpr char DefaultMeshFile[] = "OmegaBenchMesh.nc";

void velTend(const Array2DReal &LocNormalVelocityTend,
             const Array2DReal &NormRVortEdge, const Array2DReal &NormFEdge,
             const Array2DReal &FluxLayerThickEdge,
             const Array2DReal &NormVelEdge, const Array2DReal &KECell,
             const Array2DReal &SSHCell, const Array2DReal &DivCell,
             const Array2DReal &RVortVertex, const Array2DReal &Del2DivCell,
             const Array2DReal &Del2RVortVertex, const HorzMesh *Mesh,
             const VertCoord *VCoord);

int benchVelTend(int NVertLayers) {

   I4 Err = 0;

   const auto Mesh             = HorzMesh::getDefault();
   const auto VCoord           = VertCoord::getDefault();
   const auto &MinLayerEdgeBot = VCoord->MinLayerEdgeBot;
   const auto &MaxLayerEdgeTop = VCoord->MinLayerEdgeTop;

   // PotentialVortHAdvOnEdge LocPotentialVortHAdv(Mesh, VCoord);
   // KEGradOnEdge LocKEGrad(Mesh, VCoord);
   // SSHGradOnEdge LocSSHGrad(Mesh, VCoord);
   // VelocityDiffusionOnEdge LocVelocityDiffusion(Mesh, VCoord);
   // VelocityHyperDiffOnEdge LocVelocityHyperDiff(Mesh, VCoord);

   // LocPotentialVortHAdv.Enabled = true;
   // LocKEGrad.Enabled            = true;
   // LocSSHGrad.Enabled           = true;
   // LocVelocityDiffusion.Enabled = true;
   // LocVelocityHyperDiff.Enabled = true;

   Array2DReal NormRVortEdge("", Mesh->NEdgesSize, NVertLayers);
   Array2DReal NormFEdge("", Mesh->NEdgesSize, NVertLayers);
   Array2DReal FluxLayerThickEdge("", Mesh->NEdgesSize, NVertLayers);
   Array2DReal NormVelEdge("", Mesh->NEdgesSize, NVertLayers);
   Array2DReal KECell("", Mesh->NCellsSize, NVertLayers);
   Array2DReal SSHCell("", Mesh->NCellsSize, NVertLayers);
   Array2DReal DivCell("", Mesh->NCellsSize, NVertLayers);
   Array2DReal RVortVertex("", Mesh->NVerticesSize, NVertLayers);
   Array2DReal Del2DivCell("", Mesh->NCellsSize, NVertLayers);
   Array2DReal Del2RVortVertex("", Mesh->NVerticesSize, NVertLayers);

   deepCopy(NormRVortEdge, 0.9);
   deepCopy(NormFEdge, 0.15);
   deepCopy(FluxLayerThickEdge, 0.23);
   deepCopy(NormVelEdge, 0.12);
   deepCopy(KECell, 0.11);
   deepCopy(SSHCell, 0.5);
   deepCopy(DivCell, 0.4);
   deepCopy(RVortVertex, 0.3);
   deepCopy(Del2DivCell, 0.2);
   deepCopy(Del2RVortVertex, 0.1);

   // output array
   Array2DReal LocNormalVelocityTend("", Mesh->NEdgesSize, NVertLayers);

   // const auto LConfig = LaunchConfig({Mesh->NEdgesOwned}, NVertLayers,
   // parallelForOuter(
   //     LConfig, KOKKOS_LAMBDA(int IEdge, const TeamMember &Team) {
   //        ArrayScratch1DReal VelTendScratch(teamScratch(Team), NVertLayers);

   //       const int KMin = MinLayerEdgeBot(IEdge);
   //       const int KMax = MaxLayerEdgeTop(IEdge);

   //       parallelForInnerOpt(
   //           Team, Range{KMin, KMax},
   //           INNER_LAMBDA(int K) { VelTendScratch(K) = 0; });

   //       if (LocPotentialVortHAdv.Enabled) {
   //          LocPotentialVortHAdv(Team, VelTendScratch, IEdge, NormRVortEdge,
   //                               NormFEdge, FluxLayerThickEdge, NormVelEdge);
   //       }

   //       if (LocKEGrad.Enabled) {
   //          LocKEGrad(Team, VelTendScratch, IEdge, KECell);
   //       }

   //       if (LocSSHGrad.Enabled) {
   //          LocSSHGrad(Team, VelTendScratch, IEdge, SSHCell);
   //       }

   //       if (LocVelocityDiffusion.Enabled) {
   //          LocVelocityDiffusion(Team, VelTendScratch, IEdge, DivCell,
   //                               RVortVertex);
   //       }

   //       if (LocVelocityHyperDiff.Enabled) {
   //          LocVelocityHyperDiff(Team, VelTendScratch, IEdge, Del2DivCell,
   //                               Del2RVortVertex);
   //       }

   //       parallelForInnerOpt(
   //           Team, Range{KMin, KMax}, INNER_LAMBDA(int K) {
   //              LocNormalVelocityTend(IEdge, K) = VelTendScratch(K);
   //           });
   //    });

   velTend(LocNormalVelocityTend, NormRVortEdge, NormFEdge, FluxLayerThickEdge,
           NormVelEdge, KECell, SSHCell, DivCell, RVortVertex, Del2DivCell,
           Del2RVortVertex, Mesh, VCoord);

   int NRep = 20;
   Kokkos::fence();
   std::vector<double> times(NRep);
   Kokkos::Timer timer;

   for (int Rep = 0; Rep < NRep; ++Rep) {
      timer.reset();
      // parallelForOuter(
      //     LConfig, KOKKOS_LAMBDA(int IEdge, const TeamMember &Team) {
      //        ArrayScratch1DReal VelTendScratch(teamScratch(Team),
      //        NVertLayers);

      //       const int KMin = MinLayerEdgeBot(IEdge);
      //       const int KMax = MaxLayerEdgeTop(IEdge);

      //       parallelForInnerOpt(
      //           Team, Range{KMin, KMax},
      //           INNER_LAMBDA(int K) { VelTendScratch(K) = 0; });

      //       if (LocPotentialVortHAdv.Enabled) {
      //          LocPotentialVortHAdv(Team, VelTendScratch, IEdge,
      //          NormRVortEdge,
      //                               NormFEdge, FluxLayerThickEdge,
      //                               NormVelEdge);
      //       }

      //       if (LocKEGrad.Enabled) {
      //          LocKEGrad(Team, VelTendScratch, IEdge, KECell);
      //       }

      //       if (LocSSHGrad.Enabled) {
      //          LocSSHGrad(Team, VelTendScratch, IEdge, SSHCell);
      //       }

      //       if (LocVelocityDiffusion.Enabled) {
      //          LocVelocityDiffusion(Team, VelTendScratch, IEdge, DivCell,
      //                               RVortVertex);
      //       }

      //       if (LocVelocityHyperDiff.Enabled) {
      //          LocVelocityHyperDiff(Team, VelTendScratch, IEdge, Del2DivCell,
      //                               Del2RVortVertex);
      //       }

      //       parallelForInnerOpt(
      //           Team, Range{KMin, KMax}, INNER_LAMBDA(int K) {
      //              LocNormalVelocityTend(IEdge, K) = VelTendScratch(K);
      //           });
      //    });
      velTend(LocNormalVelocityTend, NormRVortEdge, NormFEdge,
              FluxLayerThickEdge, NormVelEdge, KECell, SSHCell, DivCell,
              RVortVertex, Del2DivCell, Del2RVortVertex, Mesh, VCoord);
      Kokkos::fence();
      times[Rep] = timer.seconds();
   }

   auto [min_itr, max_itr] = std::minmax_element(times.begin(), times.end());
   auto min_time           = *min_itr;
   auto max_time           = *max_itr;
   auto avg_time = std::accumulate(times.begin(), times.end(), 0.) / NRep;

   Kokkos::printf("%24s %e %e %e\n", "VelTend", avg_time, min_time, max_time);

   return Err;
} // end testTracerDiffOnCell

void initTendTest(const std::string &MeshFile, int NVertLayers) {

   Error Err;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   // Initialize logging
   initLogging(DefEnv);

   // Open config file
   Config::Initialize();
   Config::readAll("omega.yml");

   TimeStepper::init1();

   IO::init(DefComm);

   Decomp::init(MeshFile);

   int HaloErr = Halo::init();
   if (HaloErr != 0) {
      ABORT_ERROR("TendencyTermsTest: error initializing default halo");
   }

   HorzMesh::init();

   // initialize vertical coordinate, do not read stream and use local
   // NVertLayers value
   VertCoord::init(false, NVertLayers);
   Tracers::init();

} // end initTendTest

void finalizeTendTest() {
   Tracers::clear();
   HorzMesh::clear();
   VertCoord::clear();
   Field::clear();
   Dimension::clear();
   TimeStepper::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();

} // end finalizeTendTest

int tendencyTermsTest(const std::string &MeshFile = DefaultMeshFile) {
   int Err         = 0;
   int NVertLayers = 96;

   initTendTest(MeshFile, NVertLayers);

   Err += benchVelTend(NVertLayers);

   if (Err == 0) {
      LOG_INFO("TendencyTermsTest: Successful completion");
   }

   finalizeTendTest();

   return Err;

} // end tendencyTermsTest

int main(int argc, char *argv[]) {

   int RetErr = 0;

   MPI_Init(&argc, &argv);
   Kokkos::initialize(argc, argv);
   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");

   RetErr = tendencyTermsTest();

   Pacer::finalize();
   Kokkos::finalize();
   MPI_Finalize();

   return RetErr;

} // end of main
//===-----------------------------------------------------------------------===/
