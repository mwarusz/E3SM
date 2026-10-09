#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "IOStream.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "TimeStepper.h"
#include "VertCoord.h"
#include "mpi.h"
#include <chrono>

using namespace OMEGA;

void initHaloBench() {
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   initLogging(DefEnv);

   Config("Omega");
   Config::readAll("omega.yml");

   TimeStepper::init1();
   TimeStepper *DefStepper = TimeStepper::getDefault();

   Clock *ModelClock = DefStepper->getClock();
   IO::init(DefComm);
   Decomp::init();

   Field::init(ModelClock);
   IOStream::init(ModelClock);

   Halo::init();

   HorzMesh::init(ModelClock);
   VertCoord::init();
}

void finishHaloBench() {
   VertCoord::clear();
   HorzMesh::clear();
   Field::clear();
   Dimension::clear();
   TimeStepper::clear();
   Halo::clear();
   Decomp::clear();
   MachEnv::removeAll();
}

int main(int argc, char *argv[]) {
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   const auto Comm = MPI_COMM_WORLD;
   Pacer::initialize(Comm);
   Pacer::setPrefix("Omega:");

   initHaloBench();
   {
      auto *DefMachEnv = MachEnv::getDefault();
      auto *DefHalo    = Halo::getDefault();
      auto *DefDecomp  = Decomp::getDefault();
      auto *VCoord     = VertCoord::getDefault();

      int NCellsOwned = DefDecomp->NCellsOwned;
      int NCellsSize  = DefDecomp->NCellsSize;
      int NCellsAll   = DefDecomp->NCellsAll;
      int NVertLayers = VCoord->NVertLayers;

      Array2DReal Arr("Arr", NCellsSize, NVertLayers);
      deepCopy(Arr, 1);

      int NRecv = 0;
      int NSend = 0;
      for (const auto &Nb : DefHalo->Neighbors) {
         NRecv += Nb.RecvLists[0].NTot;
         NSend += Nb.SendLists[0].NTot;
      }

      int NRep = 100;

      auto TStart = std::chrono::steady_clock::now();
      for (int Rep = 0; Rep < NRep; ++Rep) {
         DefHalo->exchangeFullArrayHalo(Arr, OnCell);
      }
      auto TEnd = std::chrono::steady_clock::now();

      auto Sec = std::chrono::duration<double>(TEnd - TStart).count();

      double NMoved = (NSend + NRecv) * NVertLayers;

      auto BW = (NMoved * sizeof(Real)) / (Sec / NRep) / 1e9;

      Kokkos::printf("%d %f\n", DefMachEnv->getMyTask(), BW);
   }
   finishHaloBench();

   Pacer::finalize();
   Kokkos::finalize();
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

   return 0;
}
