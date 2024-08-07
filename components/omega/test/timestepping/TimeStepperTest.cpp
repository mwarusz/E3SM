#include "TimeStepper.h"
#include "../ocn/OceanTestCommon.h"
#include "AuxiliaryState.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "IOField.h"
#include "Logging.h"
#include "MachEnv.h"
#include "MetaData.h"
#include "OceanState.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"
#include "mpi.h"

#include <cmath>
#include <iomanip>

using namespace OMEGA;

constexpr Geometry Geom   = Geometry::Planar;
constexpr int NVertLevels = 1;
constexpr Real Pi         = M_PI;

struct ManufacturedSolution {
   Real m_grav  = 9.80616;
   Real m_f0    = 1e-4;
   Real m_lx    = 10000e3;
   Real m_ly    = std::sqrt(3) / 2 * m_lx;
   Real m_eta0  = 1;
   Real m_h0    = 1000;
   int m_mx     = 2;
   int m_my     = 2;
   Real m_kx    = m_mx * (2 * Pi / m_lx);
   Real m_ky    = m_my * (2 * Pi / m_ly);
   Real m_omega = std::sqrt(m_grav * m_h0 * (m_kx * m_kx + m_ky * m_ky));

   KOKKOS_FUNCTION Real layerThickness(Real x, Real y, Real t) const {
      return m_h0 + m_eta0 * std::sin(m_kx * x + m_ky * y - m_omega * t);
   }

   KOKKOS_FUNCTION Real velocityX(Real x, Real y, Real t) const {
      return m_eta0 * std::cos(m_kx * x + m_ky * y - m_omega * t);
   }

   KOKKOS_FUNCTION Real velocityY(Real x, Real y, Real t) const {
      return m_eta0 * std::cos(m_kx * x + m_ky * y - m_omega * t);
   }

   KOKKOS_FUNCTION Real h_tend(Real x, Real y, Real t) const {
      using std::cos;
      using std::sin;

      Real phi = m_kx * x + m_ky * y - m_omega * t;
      return m_eta0 * (-m_h0 * (m_kx + m_ky) * sin(phi) - m_omega * cos(phi) +
                       m_eta0 * (m_kx + m_ky) * cos(2 * phi));
   }

   KOKKOS_FUNCTION Real vx_tend(Real x, Real y, Real t) const {
      using std::cos;
      using std::sin;

      Real phi = m_kx * x + m_ky * y - m_omega * t;
      return m_eta0 * ((-m_f0 + m_grav * m_kx) * cos(phi) + m_omega * sin(phi) -
                       m_eta0 * (m_kx + m_ky) * sin(2 * phi) / 2);
   }

   KOKKOS_FUNCTION Real vy_tend(Real x, Real y, Real t) const {
      using std::cos;
      using std::sin;

      Real phi = m_kx * x + m_ky * y - m_omega * t;
      return m_eta0 * ((m_f0 + m_grav * m_ky) * cos(phi) + m_omega * sin(phi) -
                       m_eta0 * (m_kx + m_ky) * sin(2 * phi) / 2);
   }
};

int initState() {
   int Err = 0;

   ManufacturedSolution Setup;
   auto *Mesh  = HorzMesh::getDefault();
   auto *State = OceanState::getDefault();

   const auto &LayerThickCell = State->LayerThickness[0];
   const auto &NormalVelEdge  = State->NormalVelocity[0];

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) { return Setup.layerThickness(X, Y, 0); },
       LayerThickCell, Geom, Mesh, OnCell, NVertLevels);

   Real ThickSum;
   parallelReduce(
       {Mesh->NCellsOwned, NVertLevels},
       KOKKOS_LAMBDA(int ICell, int K, Real &Accum) {
          Accum += LayerThickCell(ICell, K) * LayerThickCell(ICell, K);
       },
       ThickSum);
   std::cout << "thick sum: " << ThickSum << std::endl;

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.velocityX(X, Y, 0);
          VecField[1] = Setup.velocityY(X, Y, 0);
       },
       NormalVelEdge, EdgeComponent::Normal, Geom, Mesh, NVertLevels,
       ExchangeHalos::Yes, CartProjection::No);

   Real NormalVelSum;
   parallelReduce(
       {Mesh->NEdgesOwned, NVertLevels},
       KOKKOS_LAMBDA(int IEdge, int K, Real &Accum) {
          Accum += NormalVelEdge(IEdge, K) * NormalVelEdge(IEdge, K);
       },
       NormalVelSum);
   std::cout << "vel sum: " << NormalVelSum << std::endl;
   // need to override FVertex with prescribed values
   // cannot use setScalar because it doesn't support setting 1D arrays
   const auto &FVertex = Mesh->FVertex;

   auto XVertex = createDeviceMirrorCopy(Mesh->XVertexH);
   auto YVertex = createDeviceMirrorCopy(Mesh->YVertexH);

   auto LonVertex = createDeviceMirrorCopy(Mesh->LonVertexH);
   auto LatVertex = createDeviceMirrorCopy(Mesh->LatVertexH);

   parallelFor(
       {Mesh->NVerticesOwned},
       KOKKOS_LAMBDA(int IVertex) { FVertex(IVertex) = Setup.m_f0; });

   auto MyHalo    = Halo::getDefault();
   auto &FVertexH = Mesh->FVertexH;
   deepCopy(FVertexH, FVertex);
   Err += MyHalo->exchangeFullArrayHalo(FVertexH, OnVertex);
   deepCopy(FVertex, FVertexH);

   return Err;
}

int createExactSolution(Real TimeEnd) {
   int Err = 0;

   auto *DefDecomp = Decomp::getDefault();
   auto *DefHalo   = Halo::getDefault();
   auto *DefMesh   = HorzMesh::getDefault();

   auto *ExactState =
       OceanState::create("Exact", DefMesh, DefDecomp, DefHalo, NVertLevels, 1);

   ManufacturedSolution Setup;

   const auto &LayerThickCell = ExactState->LayerThickness[0];
   const auto &NormalVelEdge  = ExactState->NormalVelocity[0];

   Err += setScalar(
       KOKKOS_LAMBDA(Real X, Real Y) {
          return Setup.layerThickness(X, Y, TimeEnd);
       },
       LayerThickCell, Geom, DefMesh, OnCell, NVertLevels);

   Err += setVectorEdge(
       KOKKOS_LAMBDA(Real(&VecField)[2], Real X, Real Y) {
          VecField[0] = Setup.velocityX(X, Y, TimeEnd);
          VecField[1] = Setup.velocityY(X, Y, TimeEnd);
       },
       NormalVelEdge, EdgeComponent::Normal, Geom, DefMesh, NVertLevels,
       ExchangeHalos::Yes, CartProjection::No);

   return Err;
}

void computeErrors() {
   const auto *DefMesh = HorzMesh::getDefault();

   const auto *State      = OceanState::getDefault();
   const auto *ExactState = OceanState::get("Exact");

   const auto &ThickCell     = State->LayerThickness[0];
   const auto &NormalVelEdge = State->NormalVelocity[0];

   const auto &ExactThickCell     = ExactState->LayerThickness[0];
   const auto &ExactNormalVelEdge = ExactState->NormalVelocity[0];

   ErrorMeasures ThickErrors;
   computeErrors(ThickErrors, ThickCell, ExactThickCell, DefMesh, OnCell,
                 NVertLevels);

   ErrorMeasures VelErrors;
   computeErrors(VelErrors, NormalVelEdge, ExactNormalVelEdge, DefMesh, OnEdge,
                 NVertLevels);

   if (MachEnv::getDefault()->isMasterTask()) {
      std::cout.precision(18);
      std::cout << "MW: " << ThickErrors.LInf << " " << ThickErrors.L2
                << std::endl;
      std::cout << "MW: " << VelErrors.LInf << " " << VelErrors.L2 << std::endl;
   }
}

//------------------------------------------------------------------------------
// The initialization routine for aux vars testing
int initTimeStepperTest(const std::string &mesh) {
   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   int IOErr = IO::init(DefComm);
   if (IOErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing parallel IO");
   }

   int DecompErr = Decomp::init(mesh);
   if (DecompErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default decomposition");
   }

   int HaloErr = Halo::init();
   if (HaloErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default halo");
   }

   int MeshErr = HorzMesh::init();
   if (MeshErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default mesh");
   }

   const auto &Mesh = HorzMesh::getDefault();
   MetaDim::create("NCells", Mesh->NCellsSize);
   MetaDim::create("NVertices", Mesh->NVerticesSize);
   MetaDim::create("NEdges", Mesh->NEdgesSize);
   MetaDim::create("NVertLevels", NVertLevels);

   int StateErr = OceanState::init();
   if (StateErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default state");
   }

   int AuxStateErr = AuxiliaryState::init();
   if (AuxStateErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default aux state");
   }

   int TendenciesErr = Tendencies::init();
   if (TendenciesErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default tendencies");
   }

   return Err;
}

int testTimeStepping(int Scale) {
   int Err = 0;

   TimeStepper::init();

   const auto *Stepper = TimeStepper::getDefault();
   auto *State         = OceanState::getDefault();

   const Real TimeEnd = 10 * 60 * 60;
   // const Real TimeEnd = 10 * 60;
   Real TimeStep    = 3 * Scale / 1e3;
   const int NSteps = std::ceil(TimeEnd / TimeStep);
   TimeStep         = TimeEnd / NSteps;

   std::cout << "TimeStep: " << TimeStep << std::endl;
   std::cout << "NSteps: " << NSteps << std::endl;

   for (int Step = 0; Step < NSteps; ++Step) {
      // std::cout << "Step: " << Step << std::endl;
      const Real Time = Step * TimeStep;
      Stepper->doStep(State, Time, TimeStep);
   }

   createExactSolution(TimeEnd);
   computeErrors();

   TimeStepper::clear();
   return Err;
}

void finalizeTimeStepperTest() {

   MetaDim::destroy("NCells");
   MetaDim::destroy("NVertices");
   MetaDim::destroy("NEdges");
   MetaDim::destroy("NVertLevels");

   Tendencies::clear();
   AuxiliaryState::clear();
   OceanState::clear();
   IOField::clear();
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();
}

int timeStepperTest(int Scale, const std::string &MeshFile = "OmegaMesh.nc") {
   int Err = initTimeStepperTest(MeshFile);
   if (Err != 0) {
      LOG_CRITICAL("TimeStepperTest: Error initializing");
   }

   Err += initState();

   Err += testTimeStepping(Scale);

   if (Err == 0) {
      LOG_INFO("TimeStepperTest: Successful completion");
   }
   finalizeTimeStepperTest();

   return Err;
}

int main(int argc, char *argv[]) {

   int RetVal = 0;

   MPI_Init(&argc, &argv);
   Kokkos::initialize(argc, argv);

   RetVal += timeStepperTest(
       200000, "/Users/mwarusz/mpas_meshes/planar_periodic_10000km/"
               "planar_periodic_50x50.nc");
   RetVal += timeStepperTest(
       100000, "/Users/mwarusz/mpas_meshes/planar_periodic_10000km/"
               "planar_periodic_100x100.nc");

   // RetVal += timeStepperTest();

   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
