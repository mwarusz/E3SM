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

#include <iostream>

using namespace OMEGA;

constexpr int NVertLevels = 128;

int initVadvTest() {

   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

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

void finalizeAuxStateTest() {
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
   const auto *Decomp = Decomp::getDefault();
   auto *State = OceanState::getDefault();

   int TimeLevel = 0;
   State->copyToHost(TimeLevel);
   Array2DReal LayerThickness;
   State->getLayerThickness(LayerThickness, TimeLevel);   
   Array2DReal LayerThicknessH;
   State->getLayerThicknessH(LayerThicknessH, TimeLevel);

   int NCell = LayerThicknessH.extent(0);
   int NLevels = LayerThicknessH.extent(1);
   for (int ICell = 0; ICell < NCell; ++ICell) {
      for (int K = 0; K < NLevels; ++K) {
         std::cout << ICell << " " << K << " " << LayerThicknessH(ICell, K) << std::endl;
      }
   }

}

int main(int argc, char *argv[]) {

   int RetErr = 0;

   // Initialize the global MPI environment
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   {

      initVadvTest();

//      std::cout << " main " << std::endl;
      initArrays();

      finalizeAuxStateTest();
   }
   Kokkos::finalize();
   MPI_Finalize();

   return RetErr;
}
