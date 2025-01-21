//===-- Test driver for OMEGA Vertical Advection  ----------------*- C++ -*-===/
//
//===-----------------------------------------------------------------------===/

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
#include "VerticalAdv.h"

#include "Pacer.h"
#include <mpi.h>
#include <iostream>

using namespace OMEGA;

constexpr int NVertLevels = 128;

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
      LOG_CRITICAL("AuxStateTest: Error reading config file");
      return Err;
   }

   int TimeStepperErr = TimeStepper::init1();
   if (TimeStepperErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing default time stepper");
   }

   int IOErr = IO::init(DefComm);
   if (IOErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing parallel IO");
   }

   int DecompErr = Decomp::init();
   if (DecompErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing default decomposition");
   }

   int HaloErr = Halo::init();
   if (HaloErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing default halo");
   }

   int MeshErr = HorzMesh::init();
   if (MeshErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing default mesh");
   }

   int TracerErr = Tracers::init();
   if (TracerErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing tracer infrastructure");
   }

   const auto &Mesh = HorzMesh::getDefault();
   // Horz dimensions created in HorzMesh
   auto VertDim = Dimension::create("NVertLevels", NVertLevels);

   int StateErr = OceanState::init();
   if (StateErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing default state");
   }

   int AuxStateErr = AuxiliaryState::init();
   if (AuxStateErr != 0) {
      Err++;
      LOG_ERROR("AuxStateTest: error initializing default aux state");
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

   const auto *Mesh = HorzMesh::getDefault();
   const auto *DefDecomp = Decomp::getDefault();
   auto *State = OceanState::getDefault();

   int TimeLevel = 0;
   State->copyToHost(TimeLevel);
   HostArray2DReal LayerThicknessH;
   State->getLayerThicknessH(LayerThicknessH, TimeLevel);
   HostArray2DReal NormalVelocityH;
   State->getNormalVelocityH(NormalVelocityH, TimeLevel);

   OMEGA_SCOPE(LocEdgeSignOnCell, Mesh->EdgeSignOnCellH);

   int NCells = LayerThicknessH.extent(0);
   int NLevels = LayerThicknessH.extent(1);
   for (int ICell = 0; ICell < NCells; ++ICell) {
      for (int K = 0; K < NLevels; ++K) {
//         Real NewVal = DefDecomp->CellIDH(ICell) * NLevels + K;
         Real NewVal = 100._Real * (K + 1);
         LayerThicknessH(ICell, K) = NewVal;
//         std::cout << ICell << " " << K << " " << LayerThicknessH(ICell, K) << std::endl;
      }
      for (int J = 0; J < LocEdgeSignOnCell.extent(1); ++J) {
//         std::cout << ICell << " " << J << " " << LocEdgeSignOnCell(ICell, J) << std::endl;
      }
   }

   for (int IEdge = 0; IEdge < DefDecomp->NEdgesAll; ++IEdge) {
//      std::cout << IEdge << " " << Mesh->DvEdgeH(IEdge) << std::endl;
      Real Sign;
      if (IEdge % 2 == 0) {
        Sign = 1.; 
      } else {
        Sign = -1.;
      }
      Real Mod = 0.1_Real * double(IEdge % 10);
      for (int K = 0; K < NLevels; ++K) {
         NormalVelocityH(IEdge, K) = Sign * (5._Real + Mod);
//         std::cout << IEdge << " " << K << " " << NormalVelocityH(IEdge, K)  << std::endl;
      }
   }

   State->copyToDevice(TimeLevel);

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

      int VectorLength = 32;
      VerticalAdv VadvObj(Mesh, NVertLevels, VectorLength);

      int TimeLev = 0;
      std::string TimeStepStr = "0000_00:10:00";
      TimeInterval TimeStep(TimeStepStr);

      Kokkos::fence();
      Pacer::stop("Init");

      Pacer::start("Run");

      OceanState *State = OceanState::getDefault();
      AuxiliaryState *AuxState = AuxiliaryState::getDefault();

      Pacer::start("1stVadv1");
      VadvObj.computeVerticalTransport1(State, AuxState, TimeLev, TimeStep);
      Kokkos::fence();
      Pacer::stop("1stVadv1");



      Pacer::start("1stVadv2");
      VadvObj.computeVerticalTransport2(State, AuxState, TimeLev, TimeStep);
      Kokkos::fence();
      Pacer::stop("1stVadv2");

      for (int Istep = 0; Istep < 100; ++Istep) {

         Pacer::start("vadv1");
         VadvObj.computeVerticalTransport1(State, AuxState, TimeLev, TimeStep);
         Kokkos::fence();
         Pacer::stop("vadv1");

         Pacer::start("vadv2");
         VadvObj.computeVerticalTransport2(State, AuxState, TimeLev, TimeStep);
         Kokkos::fence();
         Pacer::stop("vadv2");

      }

      Pacer::stop("Run");

      Pacer::start("Finalize");
      finalizeVadvTest();
      Pacer::stop("Finalize");
   }

   Pacer::print("omega");
   Pacer::finalize();
   Kokkos::finalize();
   MPI_Finalize();

   return RetErr;
}
