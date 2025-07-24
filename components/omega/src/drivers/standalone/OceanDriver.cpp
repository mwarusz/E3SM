//===-- drivers/standalone/OceanDriver.cpp - Standalone driver --*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#include "OceanDriver.h"
#include "DataTypes.h"
#include "OceanState.h"
#include "OmegaKokkos.h"
#include "TimeMgr.h"
#include "TimeStepper.h"
#include "Timing.h"
#include <mpi.h>

#include <iostream>

int main(int argc, char **argv) {

   int ErrAll;
   int ErrCurr;
   int ErrFinalize;

   MPI_Init(&argc, &argv); // initialize MPI
   Kokkos::initialize();   // initialize Kokkos
   OMEGA::initTiming();

   OMEGA::timerStart("Init", 0, OMEGA::AddMpiBarrier);
   ErrCurr = OMEGA::ocnInit(MPI_COMM_WORLD);
   if (ErrCurr != 0)
      LOG_ERROR("Error initializing OMEGA");
   OMEGA::timerStop("Init", 0);

   // Get time information
   OMEGA::TimeStepper *DefStepper = OMEGA::TimeStepper::getDefault();
   OMEGA::Alarm *EndAlarm         = DefStepper->getEndAlarm();
   OMEGA::Clock *ModelClock       = DefStepper->getClock();
   OMEGA::TimeInstant CurrTime    = ModelClock->getCurrentTime();

   OMEGA::timerStart("RunLoop", 0, OMEGA::AddMpiBarrier);
   while (ErrCurr == 0 && !(EndAlarm->isRinging())) {

      ErrCurr = OMEGA::ocnRun(CurrTime);

      if (ErrCurr != 0)
         LOG_ERROR("Error advancing Omega run interval");
   }
   OMEGA::timerStop("RunLoop", 0);

   OMEGA::timerStart("Finalize", 0, OMEGA::AddMpiBarrier);
   ErrFinalize = OMEGA::ocnFinalize(CurrTime);
   if (ErrFinalize != 0)
      LOG_ERROR("Error finalizing OMEGA");
   OMEGA::timerStop("Finalize", 0);

   ErrAll = abs(ErrCurr) + abs(ErrFinalize);
   if (ErrAll == 0) {
      LOG_INFO("OMEGA successfully completed");
   } else {
      LOG_ERROR("OMEGA terminating due to error");
   }

   OMEGA::finalizeTiming("omega");

   Kokkos::finalize();
   MPI_Finalize();

   if (ErrAll >= 256)
      ErrAll = 255;
   return ErrAll;
}

//===----------------------------------------------------------------------===//
