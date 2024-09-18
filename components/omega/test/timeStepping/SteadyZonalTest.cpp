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

constexpr Geometry Geom   = Geometry::Spherical;
constexpr int NVertLevels = 64;

int initState() {
   int Err = 0;

   auto *Mesh  = HorzMesh::get("TestMesh");
   auto *State = OceanState::get("TestState");

   const auto &LayerThickCell = State->LayerThickness[0];
   const auto &NormalVelEdge  = State->NormalVelocity[0];

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

   return Err;
}

int createExactSolution(Real TimeEnd) {
   int Err = 0;

   auto *DefHalo  = Halo::getDefault();
   auto *TestMesh = HorzMesh::get("TestMesh");

   auto *TestState = OceanState::get("TestState");

   auto *ExactState =
       OceanState::create("Exact", TestMesh, DefHalo, NVertLevels, 1);

   const auto &ExactLayerThickCell = ExactState->LayerThickness[0];
   const auto &ExactNormalVelEdge  = ExactState->NormalVelocity[0];

   const auto &LayerThickCell = TestState->LayerThickness[0];
   const auto &NormalVelEdge  = TestState->NormalVelocity[0];

   deepCopy(ExactLayerThickCell, LayerThickCell);
   deepCopy(ExactNormalVelEdge, NormalVelEdge);

   return Err;
}

ErrorMeasures computeErrors() {
   const auto *TestMesh = HorzMesh::get("TestMesh");

   const auto *State      = OceanState::get("TestState");
   const auto *ExactState = OceanState::get("Exact");

   const auto &NormalVelEdge      = State->NormalVelocity[0];
   const auto &ExactNormalVelEdge = ExactState->NormalVelocity[0];

   const auto &LayerThickCell      = State->LayerThickness[0];
   const auto &ExactLayerThickCell = ExactState->LayerThickness[0];

   ErrorMeasures VelErrors;
   computeErrors(VelErrors, NormalVelEdge, ExactNormalVelEdge, TestMesh, OnEdge,
                 NVertLevels);

   ErrorMeasures ThickErrors;
   computeErrors(ThickErrors, LayerThickCell, ExactLayerThickCell, TestMesh,
                 OnCell, NVertLevels);

   return ThickErrors;
}

Real minDcEdge() {
   const auto *TestMesh = HorzMesh::get("TestMesh");

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

   // Non-default init
   // Creating non-default state and auxiliary state to use only one vertical
   // level

   auto *DefDecomp = Decomp::getDefault();
   auto *DefHalo   = Halo::getDefault();

   auto *TestMesh = HorzMesh::create("TestMesh", DefDecomp, NVertLevels);

   // Horz dimensions created in HorzMesh
   auto VertDim = Dimension::create("NVertLevels", NVertLevels);

   const int NTimeLevels = 2;
   auto *TestOceanState  = OceanState::create("TestState", TestMesh, DefHalo,
                                              NVertLevels, NTimeLevels);
   if (!TestOceanState) {
      Err++;
      LOG_ERROR("TimeStepperTest: error creating test state");
   }

   auto *TestAuxState =
       AuxiliaryState::create("TestAuxState", TestMesh, NVertLevels);
   if (!TestAuxState) {
      Err++;
      LOG_ERROR("TimeStepperTest: error creating test auxiliary state");
   }

   Config Options;

   // Creating non-default tendencies with custom velocity tendencies
   auto *TestTendencies =
       Tendencies::create("TestTendencies", TestMesh, NVertLevels, &Options);
   if (!TestTendencies) {
      Err++;
      LOG_ERROR("TimeStepperTest: error creating test tendencies");
   }

   // Disable all other tendencies
   TestTendencies->ThicknessFluxDiv.Enabled   = true;
   TestTendencies->PotientialVortHAdv.Enabled = true;
   TestTendencies->KEGrad.Enabled             = true;
   TestTendencies->SSHGrad.Enabled            = true;
   TestTendencies->VelocityDiffusion.Enabled  = false;
   TestTendencies->VelocityHyperDiff.Enabled  = false;

   return Err;
}

// Slightly adjust time step so that it evenly divides TimeEnd and return number
// of steps
int adjustTimeStep(TimeStepper *Stepper, Real TimeEnd) {
   TimeInterval TimeStep = Stepper->getTimeStep();
   Real TimeStepSeconds;
   TimeStep.get(TimeStepSeconds, TimeUnits::Seconds);

   const int NSteps = std::ceil(TimeEnd / TimeStepSeconds);

   TimeStepSeconds = TimeEnd / NSteps;
   TimeStep.set(TimeStepSeconds, TimeUnits::Seconds);
   Stepper->setTimeStep(TimeInterval(TimeStepSeconds, TimeUnits::Seconds));

   std::cout << "TimeStep: " << TimeStepSeconds << std::endl;

   return NSteps;
}

void timeLoop(TimeInstant TimeStart, Real TimeEnd) {
   auto *Stepper = TimeStepper::get("TestTimeStepper");
   auto *State   = OceanState::get("TestState");

   const int NSteps            = adjustTimeStep(Stepper, TimeEnd);
   const TimeInterval TimeStep = Stepper->getTimeStep();

   // Time loop
   //
   Kokkos::Timer Timer;
   Timer.reset();
   for (int Step = 0; Step < NSteps; ++Step) {
      const TimeInstant Time = TimeStart + Step * TimeStep;
      Stepper->doStep(State, Time);
   }
   std::cout << "RunTime: " << Timer.seconds() << std::endl;
}

void finalizeTimeStepperTest() {

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

int testSteadyZonal(const std::string &Name, TimeStepperType Type) {
   int Err = 0;

   auto *TestMesh       = HorzMesh::get("TestMesh");
   auto *DefHalo        = Halo::getDefault();
   auto *TestAuxState   = AuxiliaryState::get("TestAuxState");
   auto *TestTendencies = Tendencies::get("TestTendencies");

   Calendar TestCalendar("TestCalendar", CalendarNoCalendar);

   auto *TestTimeStepper =
       TimeStepper::create("TestTimeStepper", Type, TestTendencies,
                           TestAuxState, TestMesh, DefHalo);

   if (!TestTimeStepper) {
      Err++;
      LOG_ERROR("TimeStepperTest: error creating test time stepper {}", Name);
   }

   // Start time = 0
   const TimeInstant TimeStart(&TestCalendar, 0, 0, 0, 0, 0, 0);

   const Real TimeEnd = day;

   Err += initState();
   createExactSolution(TimeEnd);

   Real CFL = 0.6;

   SteadyZonal Setup;
   Real TimeStepSeconds =
       CFL * minDcEdge() / (Setup.m_u0 + std::sqrt(grav * Setup.m_h0));

   TestTimeStepper->setTimeStep(
       TimeInterval(TimeStepSeconds, TimeUnits::Seconds));

   timeLoop(TimeStart, TimeEnd);

   ErrorMeasures Errf = computeErrors();

   std::cout << Errf.L2 << " " << Errf.LInf << std::endl;

   TimeStepper::erase("TestTimeStepper");

   return Err;
}

int timeStepperTest(const std::string &MeshFile = "OmegaMesh.nc") {

   int Err = initTimeStepperTest(MeshFile);

   if (Err != 0) {
      LOG_CRITICAL("TimeStepperTest: Error initializing");
   }

   Err += testSteadyZonal("RungeKutta4", TimeStepperType::RungeKutta4);

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
