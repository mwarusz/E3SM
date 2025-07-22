//===-- SplitForwardBackwardStepper.cpp - split forward-backward methods --*-
// C++ -*--===//
//
// Contains methods for the split Forward-Backward time stepper
//
//===----------------------------------------------------------------------===//

#include "SplitForwardBackwardStepper.h"

namespace OMEGA {

//------------------------------------------------------------------------------
// Constructor creates an instance of a split forward-backward stepper and
// fills with some time information. Data pointers are added later.
// Mostly passes relevant info to the base constructor.
SplitForwardBackwardStepper::SplitForwardBackwardStepper(
    const std::string &InName,      ///< [in] name of time stepper
    const TimeInstant &InStartTime, ///< [in] start time for time stepping
    const TimeInstant &InStopTime,  ///< [in] stop  time for time stepping
    const TimeInterval &InTimeStep  ///< [in] time step
    )
    : SplitTimeStepper(InName, TimeStepperType::SplitForwardBackward, 2,
                       InStartTime, InStopTime, InTimeStep) {}

// First part of the baroclinic step of the forward-backward scheme (before
// barotropic step)
void SplitForwardBackwardStepper::doBaroclinicStep1(
    OceanState *State,   // input model state
    TimeInstant &SimTime // current simulation time
) const {}

// Final part of the baroclinic step of the forward-backward scheme (after
// barotropic step)
void SplitForwardBackwardStepper::doBaroclinicStep2(
    OceanState *State,   // input model state
    TimeInstant &SimTime // current simulation time
) const {}

} // namespace OMEGA
