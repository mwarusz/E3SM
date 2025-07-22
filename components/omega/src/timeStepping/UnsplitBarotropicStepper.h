#ifndef OMEGA_BTRUNS_H
#define OMEGA_BTRUNS_H
//===-- UnsplitBartoropicStepper.h - unsplit barotropic time step --*- C++
//-*-===//
//
/// \file
/// \brief Contains the class for the unsplit barotropic scheme
//
//===----------------------------------------------------------------------===//

#include "BarotropicStepper.h"

namespace OMEGA {

class UnsplitBarotropicStepper : public BarotropicStepper {
 public:
   UnsplitBarotropicStepper() = default;

   /// Perform barotropic step of the unsplit scheme
   void
   doBarotropicStep(OceanState *State,   ///< [inout] model state
                    TimeInstant &SimTime ///< [inout] current simulation time
   ) const override;
};

} // namespace OMEGA
#endif
