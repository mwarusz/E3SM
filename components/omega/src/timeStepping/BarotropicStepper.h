#ifndef OMEGA_BAROS_H
#define OMEGA_BAROS_H
//===-- BartropicTimeStepper.h - barotropic time stepper --*- C++ -*-===//
//
/// \file
/// \brief Contains the base class for all barotropic time steppers
//
//===----------------------------------------------------------------------===//

#include "OceanState.h"
#include "TimeMgr.h"

namespace OMEGA {

/// An enum for every barotropic stepper type
/// needs to extended every time a new time stepper is added
enum class BarotropicStepperType { Unsplit, Invalid };

//------------------------------------------------------------------------------
// Utility routine
/// Translate string for barotropic stepper type into enum
BarotropicStepperType getBarotropicStepperFromStr(
    const std::string &InString ///< [in] choice of time stepping method
);

class BarotropicStepper {
 public:
   /// Advance the barotropic sub-system
   virtual void
   doBarotropicStep(OceanState *State,   ///< [inout] model state
                    TimeInstant &SimTime ///< [inout] current simulation time
   ) const = 0;

   virtual ~BarotropicStepper() = default;
};

} // namespace OMEGA
#endif
