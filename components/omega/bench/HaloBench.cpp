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

void benchHalo1D(int NRecv, int NSend, int NRep, int NWarm) {
   auto *DefMachEnv = MachEnv::getDefault();
   auto *DefHalo    = Halo::getDefault();
   auto *DefDecomp  = Decomp::getDefault();
   auto *VCoord     = VertCoord::getDefault();
   MPI_Comm Comm    = DefMachEnv->getComm();

   const int NTasks = DefMachEnv->getNumTasks();

   int NCellsOwned = DefDecomp->NCellsOwned;
   int NCellsSize  = DefDecomp->NCellsSize;
   int NCellsAll   = DefDecomp->NCellsAll;
   int NVertLayers = VCoord->NVertLayers;

   Array1DReal Arr("Arr", NCellsSize);
   deepCopy(Arr, 1);

   for (int Rep = 0; Rep < NWarm; ++Rep) {
      DefHalo->exchangeFullArrayHalo(Arr, OnCell);
   }
   MPI_Barrier(Comm);

   auto TStart = std::chrono::steady_clock::now();
   for (int Rep = 0; Rep < NRep; ++Rep) {
      DefHalo->exchangeFullArrayHalo(Arr, OnCell);
   }
   auto TEnd = std::chrono::steady_clock::now();

   auto TotalTime = std::chrono::duration<double>(TEnd - TStart).count();

   double NMoved = NSend + NRecv;

   double TimePerExch = TotalTime / NRep;

   auto BW = (NMoved * sizeof(Real)) / TimePerExch / 1e9;

   double BW_Min, BW_Max, BW_Avg;
   MPI_Allreduce(&BW, &BW_Min, 1, MPI_DOUBLE, MPI_MIN, Comm);
   MPI_Allreduce(&BW, &BW_Max, 1, MPI_DOUBLE, MPI_MAX, Comm);

   MPI_Allreduce(&BW, &BW_Avg, 1, MPI_DOUBLE, MPI_SUM, Comm);
   BW_Avg /= DefMachEnv->getNumTasks();

   double Time_Min, Time_Max, Time_Avg;
   MPI_Allreduce(&TimePerExch, &Time_Min, 1, MPI_DOUBLE, MPI_MIN, Comm);
   MPI_Allreduce(&TimePerExch, &Time_Max, 1, MPI_DOUBLE, MPI_MAX, Comm);

   MPI_Allreduce(&TimePerExch, &Time_Avg, 1, MPI_DOUBLE, MPI_SUM, Comm);
   Time_Avg /= DefMachEnv->getNumTasks();

   LOG_INFO("1D Time: {} {} {}", Time_Avg, Time_Min, Time_Max);
   LOG_INFO("1D BW  : {} {} {} {}", BW_Avg, BW_Min, BW_Max,
            (BW_Avg / 2) * NTasks);
}

void benchHalo2D(int NRecv, int NSend, int NRep, int NWarm) {
   auto *DefMachEnv = MachEnv::getDefault();
   auto *DefHalo    = Halo::getDefault();
   auto *DefDecomp  = Decomp::getDefault();
   auto *VCoord     = VertCoord::getDefault();
   MPI_Comm Comm    = DefMachEnv->getComm();

   const int NTasks = DefMachEnv->getNumTasks();

   int NCellsOwned = DefDecomp->NCellsOwned;
   int NCellsSize  = DefDecomp->NCellsSize;
   int NCellsAll   = DefDecomp->NCellsAll;
   int NVertLayers = VCoord->NVertLayers;

   Array2DReal Arr("Arr", NCellsSize, NVertLayers);
   deepCopy(Arr, 1);

   for (int Rep = 0; Rep < NWarm; ++Rep) {
      DefHalo->exchangeFullArrayHalo(Arr, OnCell);
   }
   MPI_Barrier(Comm);

   auto TStart = std::chrono::steady_clock::now();
   for (int Rep = 0; Rep < NRep; ++Rep) {
      DefHalo->exchangeFullArrayHalo(Arr, OnCell);
   }
   auto TEnd = std::chrono::steady_clock::now();

   auto TotalTime = std::chrono::duration<double>(TEnd - TStart).count();

   double NMoved = (NSend + NRecv) * NVertLayers;

   double TimePerExch = TotalTime / NRep;

   auto BW = (NMoved * sizeof(Real)) / TimePerExch / 1e9;

   double BW_Min, BW_Max, BW_Avg;
   MPI_Allreduce(&BW, &BW_Min, 1, MPI_DOUBLE, MPI_MIN, Comm);
   MPI_Allreduce(&BW, &BW_Max, 1, MPI_DOUBLE, MPI_MAX, Comm);

   MPI_Allreduce(&BW, &BW_Avg, 1, MPI_DOUBLE, MPI_SUM, Comm);
   BW_Avg /= DefMachEnv->getNumTasks();

   double Time_Min, Time_Max, Time_Avg;
   MPI_Allreduce(&TimePerExch, &Time_Min, 1, MPI_DOUBLE, MPI_MIN, Comm);
   MPI_Allreduce(&TimePerExch, &Time_Max, 1, MPI_DOUBLE, MPI_MAX, Comm);

   MPI_Allreduce(&TimePerExch, &Time_Avg, 1, MPI_DOUBLE, MPI_SUM, Comm);
   Time_Avg /= DefMachEnv->getNumTasks();

   LOG_INFO("2D Time: {} {} {}", Time_Avg, Time_Min, Time_Max);
   LOG_INFO("2D BW  : {} {} {} {}", BW_Avg, BW_Min, BW_Max,
            (BW_Avg / 2) * NTasks);
}

void benchHalo3D(int NRecv, int NSend, int NRep, int NWarm) {
   auto *DefMachEnv = MachEnv::getDefault();
   auto *DefHalo    = Halo::getDefault();
   auto *DefDecomp  = Decomp::getDefault();
   auto *VCoord     = VertCoord::getDefault();
   MPI_Comm Comm    = DefMachEnv->getComm();

   const int NTasks = DefMachEnv->getNumTasks();

   int NCellsOwned = DefDecomp->NCellsOwned;
   int NCellsSize  = DefDecomp->NCellsSize;
   int NCellsAll   = DefDecomp->NCellsAll;
   int NVertLayers = VCoord->NVertLayers;
   int NTracers    = 2;

   Array3DReal Arr("Arr", NTracers, NCellsSize, NVertLayers);
   deepCopy(Arr, 1);

   for (int Rep = 0; Rep < NWarm; ++Rep) {
      DefHalo->exchangeFullArrayHalo(Arr, OnCell);
   }
   MPI_Barrier(Comm);

   auto TStart = std::chrono::steady_clock::now();
   for (int Rep = 0; Rep < NRep; ++Rep) {
      DefHalo->exchangeFullArrayHalo(Arr, OnCell);
   }
   auto TEnd = std::chrono::steady_clock::now();

   auto TotalTime = std::chrono::duration<double>(TEnd - TStart).count();

   double NMoved = (NSend + NRecv) * NVertLayers * NTracers;

   double TimePerExch = TotalTime / NRep;

   auto BW = (NMoved * sizeof(Real)) / TimePerExch / 1e9;

   double BW_Min, BW_Max, BW_Avg;
   MPI_Allreduce(&BW, &BW_Min, 1, MPI_DOUBLE, MPI_MIN, Comm);
   MPI_Allreduce(&BW, &BW_Max, 1, MPI_DOUBLE, MPI_MAX, Comm);

   MPI_Allreduce(&BW, &BW_Avg, 1, MPI_DOUBLE, MPI_SUM, Comm);
   BW_Avg /= DefMachEnv->getNumTasks();

   double Time_Min, Time_Max, Time_Avg;
   MPI_Allreduce(&TimePerExch, &Time_Min, 1, MPI_DOUBLE, MPI_MIN, Comm);
   MPI_Allreduce(&TimePerExch, &Time_Max, 1, MPI_DOUBLE, MPI_MAX, Comm);

   MPI_Allreduce(&TimePerExch, &Time_Avg, 1, MPI_DOUBLE, MPI_SUM, Comm);
   Time_Avg /= DefMachEnv->getNumTasks();

   LOG_INFO("3D Time: {} {} {}", Time_Avg, Time_Min, Time_Max);
   LOG_INFO("3D BW  : {} {} {} {}", BW_Avg, BW_Min, BW_Max,
            (BW_Avg / 2) * NTasks);
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

   int NRep  = 100;
   int NWarm = 20;

   initHaloBench();
   {
      auto *DefMachEnv = MachEnv::getDefault();
      auto *DefHalo    = Halo::getDefault();

      const int NTasks = DefMachEnv->getNumTasks();

      int Nb_Min, Nb_Max, Nb_Sum;
      MPI_Allreduce(&DefHalo->NNghbr, &Nb_Min, 1, MPI_INT, MPI_MIN, Comm);
      MPI_Allreduce(&DefHalo->NNghbr, &Nb_Max, 1, MPI_INT, MPI_MAX, Comm);
      MPI_Allreduce(&DefHalo->NNghbr, &Nb_Sum, 1, MPI_INT, MPI_SUM, Comm);
      auto Nb_Avg = double(Nb_Sum) / NTasks;

      int NRecv = 0;
      int NSend = 0;
      for (const auto &Nb : DefHalo->Neighbors) {
         NRecv += Nb.RecvLists[0].NTot;
         NSend += Nb.SendLists[0].NTot;
      }

      LOG_INFO("Nb  : {} {} {}", Nb_Avg, Nb_Min, Nb_Max);

      MPI_Barrier(Comm);
      benchHalo1D(NRecv, NSend, NRep, NWarm);

      MPI_Barrier(Comm);
      benchHalo2D(NRecv, NSend, NRep, NWarm);

      MPI_Barrier(Comm);
      benchHalo3D(NRecv, NSend, NRep, NWarm);
   }
   finishHaloBench();

   Pacer::finalize();
   Kokkos::finalize();
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();

   return 0;
}
