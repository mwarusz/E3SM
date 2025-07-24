//===-- Timing.cpp - Omega timing implementation ---------*- C++ -*-===//
//
/// \file
/// \brief Utilities for timing
///
/// This file contains the implementation of the timing utility functions for
/// Omega. The implementation is based on the E3SM Pacer library and
/// vendor-specific extensions on GPU platforms.
//
//===----------------------------------------------------------------------===//

#include "Timing.h"

#include "Config.h"
#include "Error.h"
#include "OmegaKokkos.h"
#include "Pacer.h"

#include <mpi.h>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

namespace OMEGA {

// Global timing level
static int TimingLevel = 0;

// Global option for automatic Kokkos fences
static bool AutoFence = true;

// Counter used to implement the DisableChildTimers flag
static int DisableChildTimersCounter = 0;

// Vector-based stack of active timers
static std::vector<std::string> ActiveTimers;

// Initialize Pacer and set Omega prefix
void initTiming() {
   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");
}

// Read Timing config
void readTimingConfig() {
   Error Err;

   Config *OmegaConfig = Config::getOmegaConfig();
   Config TimingConfig("Timing");
   Err += OmegaConfig->get(TimingConfig);
   CHECK_ERROR_ABORT(Err, "Timing: Timing group not found in Config");

   Err += TimingConfig.get("Level", TimingLevel);
   CHECK_ERROR_ABORT(Err, "Timing: Level not found in TimingConfig");
   OMEGA_REQUIRE(TimingLevel >= 0, "Invalid timing level {} < 0", TimingLevel);

   Err += TimingConfig.get("AutoFence", AutoFence);
   CHECK_ERROR_ABORT(Err, "Timing: AutoFence not found in TimingConfig");
}

// Write timing output and finalize Pacer
void finalizeTiming(const std::string &FileName) {
   Pacer::print(FileName);
   Pacer::finalize();
}

// Start a named timer that is active at TimingLevel >= Level with optional
// flags
void timerStart(const std::string &TimerName, int Level, int Flags) {
   OMEGA_REQUIRE(Level >= 0, "Invalid timer level {} < 0 for timer {}", Level,
                 TimerName);

   // Return immediately if this timer level is above the global timing level
   if (Level > TimingLevel) {
      return;
   }

   // Need to save the value of this counter before incrementing it
   const int DisableCounterInit = DisableChildTimersCounter;

   // Increment DisableChildCounter if the corresponding flag is set
   if (Flags & DisableChildTimers) {
      DisableChildTimersCounter++;
   }

   // Return if we were in a DisableChildTimers region from the start
   if (DisableCounterInit > 0) {
      return;
   }

   // Add a Kokkos fence if the auto-fence option is on
   if (AutoFence) {
      Kokkos::fence();
   }

   // Add an MPI barrier if the corresponding flag is set
   if (Flags & AddMpiBarrier) {
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // If CUDA is enabled start an NVTX range
#ifdef KOKKOS_ENABLE_CUDA
   nvtxRangePush(TimerName.c_str());
#endif

   // Add a prefix based on the enclosing timer if the corresponding flag is set
   if (Flags & PrefixParent) {
      Pacer::start(ActiveTimers.back() + ":" + TimerName);
   } else {
      Pacer::start(TimerName);
   }

   // Push this timer onto the stack
   ActiveTimers.push_back(TimerName);
}

// Stop a named timer that is active at TimingLevel >= Level with optional flags
void timerStop(const std::string &TimerName, int Level, int Flags) {
   OMEGA_REQUIRE(Level >= 0, "Invalid timer level {} < 0 for timer {}", Level,
                 TimerName);

   // Return immediately if this timer level is above the global timing level
   if (Level > TimingLevel) {
      return;
   }

   // Dectement DisableChildCounter if the corresponding flag is set
   if (Flags & DisableChildTimers) {
      DisableChildTimersCounter--;
   }

   // Return if this timer is still in a DisableChildTimers region
   if (DisableChildTimersCounter > 0) {
      return;
   }

   // If CUDA is enabled end the NVTX range
#ifdef KOKKOS_ENABLE_CUDA
   nvtxRangePop();
#endif

   // Add a Kokkos fence if the auto-fence option is on
   if (AutoFence) {
      Kokkos::fence();
   }

   // Add an MPI barrier if the corresponding flag is set
   if (Flags & AddMpiBarrier) {
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Pop this timer from the stack
   ActiveTimers.pop_back();

   // Add a prefix based on the enclosing timer if the corresponding flag is set
   if (Flags & PrefixParent) {
      Pacer::stop(ActiveTimers.back() + ":" + TimerName);
   } else {
      Pacer::stop(TimerName);
   }
}

} // namespace OMEGA
