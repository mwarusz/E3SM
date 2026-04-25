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

struct TestSetupPlane {

   Real Lx = 1;
   Real Ly = SqrtThree / 2;

   ErrorMeasures ExpectedDivErrors         = {0.00124886886594453264,
                                              0.00124886886590977139};
   ErrorMeasures ExpectedPVErrors          = {0.00807347170900282914,
                                              0.00794755105765788429};
   ErrorMeasures ExpectedGradErrors        = {0.00125026071878537952,
                                              0.00134354611117262161};
   ErrorMeasures ExpectedLaplaceErrors     = {0.00113090174765822192,
                                              0.00134324628763667899};
   ErrorMeasures ExpectedTrHAdvErrors      = {0.0029211089892916243,
                                              0.0024583038518548855};
   ErrorMeasures ExpectedTrDel2Errors      = {0.00334357193650093847,
                                              0.00290978146207349032};
   ErrorMeasures ExpectedTrDel4Errors      = {0.00508833446725232875,
                                              0.00523080740758275625};
   ErrorMeasures ExpectedSurfTrRestErrors  = {0, 0};
   ErrorMeasures ExpectedWindForcingErrors = {0, 0};
   ErrorMeasures ExpectedBottomDragErrors  = {0.033848740052302935,
                                              0.01000133508329411};

   KOKKOS_FUNCTION Real vectorX(Real X, Real Y) const {
      return std::sin(TwoPi * X / Lx) * std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real vectorY(Real X, Real Y) const {
      return std::cos(TwoPi * X / Lx) * std::sin(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real divergence(Real X, Real Y) const {
      return TwoPi * (1. / Lx + 1. / Ly) * std::cos(TwoPi * X / Lx) *
             std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real scalar(Real X, Real Y) const {
      return std::sin(TwoPi * X / Lx) * std::sin(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real gradX(Real X, Real Y) const {
      return TwoPi / Lx * std::cos(TwoPi * X / Lx) * std::sin(TwoPi * Y / Ly);
   }
   KOKKOS_FUNCTION Real gradY(Real X, Real Y) const {
      return TwoPi / Ly * std::sin(TwoPi * X / Lx) * std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real curl(Real X, Real Y) const {
      return TwoPi * (-1. / Lx + 1. / Ly) * std::sin(TwoPi * X / Lx) *
             std::sin(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real laplaceVecX(Real X, Real Y) const {
      return -TwoPi * TwoPi * (1. / Lx / Lx + 1. / Ly / Ly) *
             std::sin(TwoPi * X / Lx) * std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real laplaceVecY(Real X, Real Y) const {
      return -TwoPi * TwoPi * (1. / Lx / Lx + 1. / Ly / Ly) *
             std::cos(TwoPi * X / Lx) * std::sin(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real layerThick(Real X, Real Y) const {
      return 2. + std::sin(TwoPi * X / Lx) * std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real planetaryVort(Real X, Real Y) const {
      return std::cos(TwoPi * X / Lx) * std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real normRelVort(Real X, Real Y) const {
      return curl(X, Y) / layerThick(X, Y);
   }

   KOKKOS_FUNCTION Real normPlanetVort(Real X, Real Y) const {
      return planetaryVort(X, Y) / layerThick(X, Y);
   }

   KOKKOS_FUNCTION Real tracerFluxDiv(Real X, Real Y) const {
      return (TwoPi / (Lx * Ly)) *
             (std::cos(TwoPi * X / Lx) *
              (2 * (Lx + Ly) * std::cos(TwoPi * Y / Ly) +
               (Lx + 2 * Ly) * std::sin(TwoPi * X / Lx) *
                   std::pow(std::cos(TwoPi * Y / Ly), 2) -
               Lx * std::sin(TwoPi * X / Lx) *
                   std::pow(std::sin(TwoPi * Y / Ly), 2)));
   }

   KOKKOS_FUNCTION Real scalarA(Real X, Real Y) const {
      return std::cos(TwoPi * X / Lx) * std::sin(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real scalarB(Real X, Real Y) const {
      return 2. + std::cos(TwoPi * X / Lx) * std::cos(TwoPi * Y / Ly);
   }

   KOKKOS_FUNCTION Real tracerDiff(Real X, Real Y) const {
      return -TwoPi * TwoPi * std::sin(TwoPi * Y / Ly) *
             (2 * (1 / Lx / Lx + 1 / Ly / Ly) * std::cos(TwoPi * X / Lx) +
              (1 / Ly / Ly +
               (1 / Lx / Lx + 1 / Ly / Ly) * std::cos(2 * TwoPi * X / Lx)) *
                  std::cos(TwoPi * Y / Ly));
   }

   KOKKOS_FUNCTION Real scalarC(Real X, Real Y) const {
      return std::pow(std::cos(TwoPi * X / Lx), 2) -
             std::pow(std::sin(TwoPi * Y / Ly), 2);
   }

   KOKKOS_FUNCTION Real tracerHyperDiff(Real X, Real Y) const {
      return -2 * TwoPi * TwoPi *
             (std::cos(2 * TwoPi * X / Lx) / Lx / Lx +
              std::cos(2 * TwoPi * Y / Ly) / Ly / Ly);
   }

   KOKKOS_FUNCTION Real windForcingX(Real X, Real Y) const {
      const Real StressU = vectorX(X, Y);
      const Real Thick   = scalarB(X, Y);
      return StressU / (Thick * RhoSw);
   }

   KOKKOS_FUNCTION Real windForcingY(Real X, Real Y) const {
      const Real StressV = vectorY(X, Y);
      const Real Thick   = scalarB(X, Y);
      return StressV / (Thick * RhoSw);
   }

   KOKKOS_FUNCTION Real bottomDragX(Real X, Real Y, Real Coeff) const {
      const Real UVel = vectorX(X, Y);
      return -Coeff * std::abs(scalarA(X, Y)) / scalarB(X, Y) * UVel;
   }

   KOKKOS_FUNCTION Real bottomDragY(Real X, Real Y, Real Coeff) const {
      const Real VVel = vectorY(X, Y);
      return -Coeff * std::abs(scalarA(X, Y)) / scalarB(X, Y) * VVel;
   }

}; // end TestSetupPlane

constexpr Geometry Geom          = Geometry::Planar;
constexpr char DefaultMeshFile[] = "OmegaBenchMesh.nc";
using TestSetup                  = TestSetupPlane;

int benchVelTend(int NVertLayers) {

   I4 Err = 0;
   TestSetup Setup;

   const auto Mesh             = HorzMesh::getDefault();
   const auto VCoord           = VertCoord::getDefault();
   const auto &MinLayerEdgeBot = VCoord->MinLayerEdgeBot;
   const auto &MaxLayerEdgeTop = VCoord->MinLayerEdgeTop;

   PotentialVortHAdvOnEdge LocPotentialVortHAdv(Mesh, VCoord);
   KEGradOnEdge LocKEGrad(Mesh, VCoord);
   SSHGradOnEdge LocSSHGrad(Mesh, VCoord);
   VelocityDiffusionOnEdge LocVelocityDiffusion(Mesh, VCoord);
   VelocityHyperDiffOnEdge LocVelocityHyperDiff(Mesh, VCoord);

   LocPotentialVortHAdv.Enabled = true;
   LocKEGrad.Enabled            = true;
   LocSSHGrad.Enabled           = true;
   LocVelocityDiffusion.Enabled = true;
   LocVelocityHyperDiff.Enabled = true;

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

   const auto LConfig = LaunchConfig({Mesh->NEdgesOwned}, NVertLayers,
                                     TeamScratch<Real>(NVertLayers));
   parallelForOuter(
       LConfig, KOKKOS_LAMBDA(int IEdge, const TeamMember &Team) {
          ArrayScratch1DReal VelTendScratch(teamScratch(Team), NVertLayers);

          const int KMin = MinLayerEdgeBot(IEdge);
          const int KMax = MaxLayerEdgeTop(IEdge);

          parallelForInnerOpt(
              Team, Range{KMin, KMax},
              INNER_LAMBDA(int K) { VelTendScratch(K) = 0; });

          if (LocPotentialVortHAdv.Enabled) {
             LocPotentialVortHAdv(Team, VelTendScratch, IEdge, NormRVortEdge,
                                  NormFEdge, FluxLayerThickEdge, NormVelEdge);
          }

          if (LocKEGrad.Enabled) {
             LocKEGrad(Team, VelTendScratch, IEdge, KECell);
          }

          if (LocSSHGrad.Enabled) {
             LocSSHGrad(Team, VelTendScratch, IEdge, SSHCell);
          }

          if (LocVelocityDiffusion.Enabled) {
             LocVelocityDiffusion(Team, VelTendScratch, IEdge, DivCell,
                                  RVortVertex);
          }

          if (LocVelocityHyperDiff.Enabled) {
             LocVelocityHyperDiff(Team, VelTendScratch, IEdge, Del2DivCell,
                                  Del2RVortVertex);
          }

          parallelForInnerOpt(
              Team, Range{KMin, KMax}, INNER_LAMBDA(int K) {
                 LocNormalVelocityTend(IEdge, K) = VelTendScratch(K);
              });
       });

   int NRep = 20;
   Kokkos::fence();
   std::vector<double> times(NRep);
   Kokkos::Timer timer;

   for (int Rep = 0; Rep < NRep; ++Rep) {
      timer.reset();
      parallelForOuter(
          LConfig, KOKKOS_LAMBDA(int IEdge, const TeamMember &Team) {
             ArrayScratch1DReal VelTendScratch(teamScratch(Team), NVertLayers);

             const int KMin = MinLayerEdgeBot(IEdge);
             const int KMax = MaxLayerEdgeTop(IEdge);

             parallelForInnerOpt(
                 Team, Range{KMin, KMax},
                 INNER_LAMBDA(int K) { VelTendScratch(K) = 0; });

             if (LocPotentialVortHAdv.Enabled) {
                LocPotentialVortHAdv(Team, VelTendScratch, IEdge, NormRVortEdge,
                                     NormFEdge, FluxLayerThickEdge,
                                     NormVelEdge);
             }

             if (LocKEGrad.Enabled) {
                LocKEGrad(Team, VelTendScratch, IEdge, KECell);
             }

             if (LocSSHGrad.Enabled) {
                LocSSHGrad(Team, VelTendScratch, IEdge, SSHCell);
             }

             if (LocVelocityDiffusion.Enabled) {
                LocVelocityDiffusion(Team, VelTendScratch, IEdge, DivCell,
                                     RVortVertex);
             }

             if (LocVelocityHyperDiff.Enabled) {
                LocVelocityHyperDiff(Team, VelTendScratch, IEdge, Del2DivCell,
                                     Del2RVortVertex);
             }

             parallelForInnerOpt(
                 Team, Range{KMin, KMax}, INNER_LAMBDA(int K) {
                    LocNormalVelocityTend(IEdge, K) = VelTendScratch(K);
                 });
          });
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
