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
constexpr int NVertLevels = 60;

struct DecayThicknessTendency {
   void operator()(Array2DReal ThicknessTend, OceanState *State,
                   AuxiliaryState *AuxState, int TimeLevel, Real Time) const {
      // auto *Mesh       = HorzMesh::getDefault();
      // parallelFor(
      //     {Mesh->NCellsAll, NVertLevels}, KOKKOS_LAMBDA(int ICell, int K) {
      //        ThicknessTend(ICell, K) += h_tend(X, Y, Time);
      //     });
   }
};

struct DecayVelocityTendency {
   Real Coeff = 1;

   Real exactSolution(Real Time) { return std::exp(-Coeff * Time); }

   void operator()(Array2DReal NormalVelTend, OceanState *State,
                   AuxiliaryState *AuxState, int TimeLevel, Real Time) const {

      auto *Mesh                = HorzMesh::getDefault();
      auto NVertLevels          = NormalVelTend.extent_int(1);
      const auto &NormalVelEdge = State->NormalVelocity[TimeLevel];

      parallelFor(
          {Mesh->NEdgesAll, NVertLevels}, KOKKOS_LAMBDA(int IEdge, int K) {
             NormalVelTend(IEdge, K) -= Coeff * NormalVelEdge(IEdge, K);
          });
   }
};

int initState() {
   int Err = 0;

   auto *Mesh  = HorzMesh::getDefault();
   auto *State = OceanState::getDefault();

   const auto &LayerThickCell = State->LayerThickness[0];
   const auto &NormalVelEdge  = State->NormalVelocity[0];

   deepCopy(LayerThickCell, 1);
   deepCopy(NormalVelEdge, 1);

   return Err;
}

int createExactSolution(Real TimeEnd) {
   int Err = 0;

   auto *DefHalo = Halo::getDefault();
   auto *DefMesh = HorzMesh::getDefault();

   auto *ExactState =
       OceanState::create("Exact", DefMesh, DefHalo, NVertLevels, 1);

   const auto &LayerThickCell = ExactState->LayerThickness[0];
   const auto &NormalVelEdge  = ExactState->NormalVelocity[0];

   deepCopy(LayerThickCell, 1);
   deepCopy(NormalVelEdge, DecayVelocityTendency{}.exactSolution(TimeEnd));

   return Err;
}

ErrorMeasures computeErrors() {
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

   OceanState::erase("Exact");

   return VelErrors;
}

//------------------------------------------------------------------------------
// The initialization routine for time stepper testing
int initTimeStepperTest(TimeStepperType Type, const std::string &mesh) {
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

   Config Options;
   Tendencies::create("TestTendencies", HorzMesh::getDefault(), NVertLevels,
                      &Options, DecayThicknessTendency{},
                      DecayVelocityTendency{});

   TimeStepper::create("TestTimeStepper", Type,
                       Tendencies::get("TestTendencies"),
                       AuxiliaryState::getDefault(), HorzMesh::getDefault(),
                       Halo::getDefault());

   return Err;
}

ErrorMeasures runWithTimeStep(Real TimeStep) {
   int Err = 0;

   Err += initState();

   const auto *Stepper = TimeStepper::get("TestTimeStepper");
   auto *State         = OceanState::getDefault();

   const Real TimeEnd = 1;
   const int NSteps   = std::ceil(TimeEnd / TimeStep);
   TimeStep           = TimeEnd / NSteps;

   // std::cout << "TimeStep: " << TimeStep << std::endl;
   // std::cout << "NSteps: " << NSteps << std::endl;

   for (int Step = 0; Step < NSteps; ++Step) {
      const Real Time = Step * TimeStep;
      Stepper->doStep(State, Time, TimeStep);
   }

   createExactSolution(TimeEnd);

   return computeErrors();
}

void finalizeTimeStepperTest() {

   MetaDim::destroy("NCells");
   MetaDim::destroy("NVertices");
   MetaDim::destroy("NEdges");
   MetaDim::destroy("NVertLevels");

   TimeStepper::clear();
   Tendencies::clear();
   AuxiliaryState::clear();
   OceanState::clear();
   IOField::clear();
   HorzMesh::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();
}

int timeStepperTest(TimeStepperType Type,
                    const std::string &MeshFile = "OmegaMesh.nc") {
   int Err = initTimeStepperTest(Type, MeshFile);
   if (Err != 0) {
      LOG_CRITICAL("TimeStepperTest: Error initializing");
   }

   int NRefinements = 2;
   std::vector<Real> Errors(NRefinements);

   const Real BaseTimeStep = 0.2;

   Real TimeStep = BaseTimeStep;
   for (int RefLevel = 0; RefLevel < NRefinements; ++RefLevel) {
      Errors[RefLevel] = runWithTimeStep(TimeStep).L2;
      TimeStep /= 2;
   }

   std::cout << "Rate: " << std::log2(Errors[0] / Errors[1]) << std::endl;

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

   RetVal += timeStepperTest(TimeStepperType::RungeKutta4);

   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
