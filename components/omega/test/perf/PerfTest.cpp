#include "../ocn/OceanTestCommon.h"
#include "AuxiliaryState.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Field.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OceanState.h"
#include "OmegaKokkos.h"
#include "TendencyTerms.h"
#include "TimeMgr.h"
#include "TimeStepper.h"
#include "Tracers.h"
#include "mpi.h"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace OMEGA;

constexpr Real shrink_factor = 1;
constexpr Real grav          = 9.80665;
constexpr Real earth_radius  = 6.37122e6 / shrink_factor;
constexpr Real day           = 24 * 60 * 60 / shrink_factor;
constexpr Real omg           = 7.292e-5 * shrink_factor;
constexpr Real Pi            = M_PI;

struct SteadyZonal {
   Real m_u0    = 2 * Pi * earth_radius / (12 * day);
   Real m_alpha = Pi / 4;
   Real m_h0    = 2.94e4 / grav;

   KOKKOS_FUNCTION Real coriolis(Real lon, Real lat) const {
      using std::cos;
      using std::sin;
      return 2 * omg *
             (-cos(lon) * cos(lat) * sin(m_alpha) + sin(lat) * cos(m_alpha));
   }

   KOKKOS_FUNCTION Real layerThickness(Real lon, Real lat) const {
      using std::cos;
      using std::sin;
      Real tmp = -cos(lat) * cos(lon) * sin(m_alpha) + sin(lat) * cos(m_alpha);
      return m_h0 -
             (earth_radius * omg * m_u0 + m_u0 * m_u0 / 2) * tmp * tmp / grav;
   }

   KOKKOS_FUNCTION Real psi(Real lon, Real lat) const {
      using std::cos;
      using std::sin;
      return -earth_radius * m_u0 *
             (sin(lat) * cos(m_alpha) - cos(lon) * cos(lat) * sin(m_alpha));
   }
};

constexpr Geometry Geom = Geometry::Spherical;

int initState() {
   int Err = 0;

   auto *Mesh  = HorzMesh::getDefault();
   auto *State = OceanState::getDefault();

   const auto &LayerThickCell = State->LayerThickness[0];
   const auto &NormalVelEdge  = State->NormalVelocity[0];

   const int NVertLevels = Mesh->NVertLevels;

   SteadyZonal Setup;

   Err += setScalar(
       KOKKOS_LAMBDA(Real Lon, Real Lat) {
          return Setup.layerThickness(Lon, Lat);
       },
       LayerThickCell, Geom, Mesh, OnCell, NVertLevels);

   Array2DReal PsiVertex("PsiVertex", Mesh->NVerticesSize, NVertLevels);
   Err += setScalar(
       KOKKOS_LAMBDA(Real Lon, Real Lat) { return Setup.psi(Lon, Lat); },
       PsiVertex, Geom, Mesh, OnVertex, NVertLevels);

   const auto &VerticesOnEdge = Mesh->VerticesOnEdge;
   const auto &DvEdge         = Mesh->DvEdge;
   parallelFor(
       {Mesh->NEdgesOwned, Mesh->NVertLevels}, KOKKOS_LAMBDA(int IEdge, int K) {
          const int JVertex0 = VerticesOnEdge(IEdge, 0);
          const int JVertex1 = VerticesOnEdge(IEdge, 1);
          NormalVelEdge(IEdge, K) =
              -(PsiVertex(JVertex1, K) - PsiVertex(JVertex0, K)) /
              DvEdge(IEdge);
       });

   // need to override FVertex with prescribed values
   // cannot use setScalar because it doesn't support setting 1D arrays
   const auto &FVertex = Mesh->FVertex;
   auto LonVertex      = createDeviceMirrorCopy(Mesh->LonVertexH);
   auto LatVertex      = createDeviceMirrorCopy(Mesh->LatVertexH);

   parallelFor(
       {Mesh->NVerticesOwned}, KOKKOS_LAMBDA(int IVertex) {
          const Real XV    = LonVertex(IVertex);
          const Real YV    = LatVertex(IVertex);
          FVertex(IVertex) = Setup.coriolis(XV, YV);
       });

   State->exchangeHalo(0);

   auto MyHalo    = Halo::getDefault();
   auto &FVertexH = Mesh->FVertexH;
   deepCopy(FVertexH, FVertex);
   Err += MyHalo->exchangeFullArrayHalo(FVertexH, OnVertex);
   deepCopy(FVertex, FVertexH);

   deepCopy(Mesh->FEdge, 0);
   deepCopy(Mesh->FEdgeH, 0);
   deepCopy(Mesh->FCell, 0);
   deepCopy(Mesh->FCellH, 0);
   deepCopy(Mesh->FCellH, 0);
   deepCopy(Mesh->BottomDepth, 0);
   deepCopy(Mesh->BottomDepthH, 0);

   return Err;
}

int createExactSolution(Real TimeEnd) {
   int Err = 0;

   auto *DefHalo  = Halo::getDefault();
   auto *TestMesh = HorzMesh::getDefault();

   auto *TestState = OceanState::getDefault();

   auto *ExactState =
       OceanState::create("Exact", TestMesh, DefHalo, TestMesh->NVertLevels, 1);

   const auto &ExactLayerThickCell = ExactState->LayerThickness[0];
   const auto &ExactNormalVelEdge  = ExactState->NormalVelocity[0];

   const auto &LayerThickCell = TestState->LayerThickness[0];
   const auto &NormalVelEdge  = TestState->NormalVelocity[0];

   deepCopy(ExactLayerThickCell, LayerThickCell);
   deepCopy(ExactNormalVelEdge, NormalVelEdge);

   return Err;
}

ErrorMeasures computeErrors() {
   const auto *TestMesh = HorzMesh::getDefault();

   const auto *State      = OceanState::getDefault();
   const auto *ExactState = OceanState::get("Exact");

   const auto &NormalVelEdge      = State->NormalVelocity[0];
   const auto &ExactNormalVelEdge = ExactState->NormalVelocity[0];

   const auto &LayerThickCell      = State->LayerThickness[0];
   const auto &ExactLayerThickCell = ExactState->LayerThickness[0];

   ErrorMeasures VelErrors;
   computeErrors(VelErrors, NormalVelEdge, ExactNormalVelEdge, TestMesh, OnEdge,
                 TestMesh->NVertLevels);

   ErrorMeasures ThickErrors;
   computeErrors(ThickErrors, LayerThickCell, ExactLayerThickCell, TestMesh,
                 OnCell, TestMesh->NVertLevels);

   return ThickErrors;
}

Real minDcEdge() {
   const auto *TestMesh = HorzMesh::getDefault();

   const auto &DcEdge = TestMesh->DcEdge;

   Real MinDcEdgeLoc;
   parallelReduce(
       {TestMesh->NEdgesOwned},
       KOKKOS_LAMBDA(int IEdge, Real &Accum) {
          Accum = Kokkos::min(DcEdge(IEdge), Accum);
       },
       Kokkos::Min<Real>(MinDcEdgeLoc));

   MPI_Comm Comm = MachEnv::getDefault()->getComm();
   Real MinDcEdge;
   MPI_Allreduce(&MinDcEdgeLoc, &MinDcEdge, 1, MPI_RealKind, MPI_MIN, Comm);

   return MinDcEdge;
}

int initTimeStepperTest(const std::string &mesh) {
   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   // Default init

   initLogging(DefEnv);

   // Open config file
   OMEGA::Config("Omega");
   Err = OMEGA::Config::readAll("omega.yml");

   if (Err != 0) {
      LOG_CRITICAL("TimeStepperTest: Error reading config file");
      return Err;
   }

   // Note that the default time stepper is not used in subsequent tests
   // but is initialized here because the number of time levels is needed
   // to initialize the Tracers. If a later timestepper test uses more time
   // levels than the default, this unit test may fail.
   int TSErr = TimeStepper::init1();
   if (TSErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing default time stepper");
   }

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

   int TracerErr = Tracers::init();
   if (TracerErr != 0) {
      Err++;
      LOG_ERROR("TimeStepperTest: error initializing tracers infrastructure");
   }

   Err = AuxiliaryState::init();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default aux state");
      return Err;
   }

   Err = Tendencies::init();
   if (Err != 0) {
      LOG_CRITICAL("Error initializing default tendencies");
      return Err;
   }

   // finish initializing default time stepper
   TSErr = TimeStepper::init2();
   if (TSErr != 0) {
      Err++;
      LOG_ERROR("error initializing default time stepper");
   }

   Err = OceanState::init();
   if (Err != 0) {
      LOG_CRITICAL("ocnInit: Error initializing default state");
      return Err;
   }

   return Err;
}

// Slightly adjust time step so that it evenly divides TimeEnd and return number
// of steps
int adjustTimeStep(TimeStepper *Stepper, Real TimeEnd) {
   TimeInterval TimeStep = Stepper->getTimeStep();
   R8 TimeStepSeconds;
   TimeStep.get(TimeStepSeconds, TimeUnits::Seconds);

   const int NSteps = std::ceil(TimeEnd / TimeStepSeconds);

   TimeStepSeconds = TimeEnd / NSteps;
   TimeStep.set(TimeStepSeconds, TimeUnits::Seconds);

   Stepper->changeTimeStep(TimeInterval(TimeStepSeconds, TimeUnits::Seconds));

   return NSteps;
}

void timeLoop(TimeInstant TimeStart, Real TimeEnd) {
   auto *Stepper = TimeStepper::getDefault();
   auto *State   = OceanState::getDefault();

   const int NSteps            = adjustTimeStep(Stepper, TimeEnd);
   const TimeInterval TimeStep = Stepper->getTimeStep();

   // Time loop
   Stepper->doStep(State, TimeStart);

   timer_start("time_loop");
   // for (int Step = 0; Step < NSteps - 1; ++Step) {
   for (int Step = 0; Step < 1; ++Step) {
      TimeInstant Time = TimeStart + (Step + 1) * TimeStep;
      Stepper->doStep(State, Time);
   }
   Kokkos::fence();
   timer_stop("time_loop");
}

void finalizeTimeStepperTest() {

   Tracers::clear();
   TimeStepper::clear();
   Tendencies::clear();
   AuxiliaryState::clear();
   OceanState::clear();
   Dimension::clear();
   Field::clear();
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();
}

int testSteadyZonal() {
   int Err = 0;

   auto *TestMesh       = HorzMesh::getDefault();
   auto *DefHalo        = Halo::getDefault();
   auto *TestAuxState   = AuxiliaryState::getDefault();
   auto *TestTendencies = Tendencies::getDefault();
   auto *Stepper        = TimeStepper::getDefault();

   SteadyZonal Setup;

   // Start time = 0
   const TimeInstant TimeStart(0, 0, 0, 0, 0, 0);
   const Real TimeEndSeconds = day / 8;
   const TimeInstant TimeEnd(0, 0, 0, 0, 0, day / 8);

   Real CFL = 0.6;
   Real TimeStepSeconds =
       CFL * minDcEdge() / (Setup.m_u0 + std::sqrt(grav * Setup.m_h0));
   TimeInterval TimeStep(TimeStepSeconds, TimeUnits::Seconds);
   Stepper->changeTimeStep(TimeStep);

   Err += initState();
   createExactSolution(TimeEndSeconds);

   timeLoop(TimeStart, TimeEndSeconds);

   ErrorMeasures Errf = computeErrors();

   if (MachEnv::getDefault()->isMasterTask()) {
      std::cout << Errf.L2 << " " << Errf.LInf << std::endl;
   }

   return Err;
}

int timeStepperTest(const std::string &MeshFile = "OmegaMesh.nc") {

   int Err = initTimeStepperTest(MeshFile);

   if (Err != 0) {
      LOG_CRITICAL("TimeStepperTest: Error initializing");
   }

   Err += testSteadyZonal();

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

   RetVal += timeStepperTest("PerfMesh.nc");

   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
