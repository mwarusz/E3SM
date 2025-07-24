#ifndef OMEGA_TIMING_H
#define OMEGA_TIMING_H
//===-- infra/Timing.h - Timing --------------------*- C++ -*-===//
//
/// \file
/// \brief Timing utilities
///
/// This header defines utility functions for timing and optional flags that
/// they can accept.
//===----------------------------------------------------------------------===//

#include <string>

namespace OMEGA {

// Optional flags for timerStart and timerStop
enum TimerFlag { AddMpiBarrier = 1, PrefixParent = 2, DisableChildTimers = 4 };

// Initalize timing infrastructure
void initTiming();

// Read TimingLevel and other timing options from the config file
void readTimingConfig();

// Write timing output and finalize timing infrastructure
void finalizeTiming(const std::string &FileName);

// Start a named timer that is active at TimingLevel >= Level with optional
// flags
void timerStart(const std::string &TimerName, int Level, int Flags = 0);

// Stop a named timer that is active at TimingLevel >= Level with optional flags
void timerStop(const std::string &TimerName, int Level, int Flags = 0);

} // namespace OMEGA

#endif // OMEGA_TIMING_H
