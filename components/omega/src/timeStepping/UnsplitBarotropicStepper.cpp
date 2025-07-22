//===-- UnsplitBarotropicStepper.cpp - unsplit bartoropic --*- C++ -*-===//
//
// Contains methods for the unsplit barotropic time stepper
//
//===----------------------------------------------------------------------===//

#include "UnsplitBarotropicStepper.h"

namespace OMEGA {

//------------------------------------------------------------------------------
// Perform barotropic step of the unsplit scheme
void UnsplitBarotropicStepper::doBarotropicStep(
    OceanState *State,   // model state
    TimeInstant &SimTime // current simulation time
) const {}

} // namespace OMEGA
