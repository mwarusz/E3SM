//===-- BarotropicStepper.cpp - barotropic stepper methods --*- C++ -*-===//
//
// Contains methods for the barotropic stepper base class
//
//===----------------------------------------------------------------------===//

#include "BarotropicStepper.h"
#include "Error.h"

namespace OMEGA {

//------------------------------------------------------------------------------
// utility functions
// convert string into BarotropicStepperType enum
BarotropicStepperType getBarotropicStepperFromStr(const std::string &InString) {

   // Initialize BarotropicStepperChoice with Invalid
   BarotropicStepperType BarotropicStepperChoice =
       BarotropicStepperType::Invalid;

   if (InString == "Unsplit") {
      BarotropicStepperChoice = BarotropicStepperType::Unsplit;
   } else {
      ABORT_ERROR("BarotropicStepper should be one of 'Unsplit', but got {}:",
                  InString);
   }

   return BarotropicStepperChoice;
}

//------------------------------------------------------------------------------

} // namespace OMEGA
