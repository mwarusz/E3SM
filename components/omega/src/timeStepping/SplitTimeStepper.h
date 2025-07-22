#ifndef OMEGA_TSSPLIT_H
#define OMEGA_TSSPLIT_H
//===-- SplitTimeStepper.h - split time stepper --*- C++ -*-===//
//
/// \file
/// \brief Contains the base class for all baroclinic-barotropic split time
/// steppers
//
//===----------------------------------------------------------------------===//

#include "BarotropicStepper.h"
#include "TimeStepper.h"
#include <memory>

namespace OMEGA {

class SplitTimeStepper : public TimeStepper {
 public:
   /// Advance the state by one step of the split method
   void doStep(OceanState *State,   ///< [inout] model state
               TimeInstant &SimTime ///< [inout] current simulation time
   ) const override;

 protected:
   /// Performs initialization of the barotropic time stepper
   void finalizeInit() override;

   /// This constructor simply calls the base class constuctor with its
   /// arguments
   SplitTimeStepper(
       const std::string &InName,      ///< [in] name of time stepper
       TimeStepperType InType,         ///< [in] type (time stepping method)
       I4 InNTimeLevels,               ///< [in] num time levels for method
       const TimeInstant &InStartTime, ///< [in] start time for time stepping
       const TimeInstant &InStopTime,  ///< [in] stop  time for time stepping
       const TimeInterval &InTimeStep  ///< [in] time step
   );

   virtual void
   doBaroclinicStep1(OceanState *State,   ///< [inout] model state
                     TimeInstant &SimTime ///< [inout] current simulation time
   ) const = 0;

   virtual void
   doBaroclinicStep2(OceanState *State,   ///< [inout] model state
                     TimeInstant &SimTime ///< [inout] current simulation time
   ) const = 0;

   std::unique_ptr<BarotropicStepper> BaroStepper;
};

} // namespace OMEGA
#endif
