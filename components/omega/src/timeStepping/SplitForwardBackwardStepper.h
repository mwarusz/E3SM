#ifndef OMEGA_TSSFB_H
#define OMEGA_TSSFB_H
//===-- SplitForwardBackwardStepper.h - split forward-backward time step --*-
// C++ -*--===//
//
/// \file
/// \brief Contains the class for split forward-backward time stepping scheme
//===----------------------------------------------------------------------===//

#include "SplitTimeStepper.h"

namespace OMEGA {

class SplitForwardBackwardStepper : public SplitTimeStepper {
 public:
   /// Constructor creates an instance of a split forward-backward stepper and
   /// fills with some time information. Data pointers are added later.
   SplitForwardBackwardStepper(
       const std::string &InName,      ///< [in] name of time stepper
       const TimeInstant &InStartTime, ///< [in] start time for time stepping
       const TimeInstant &InStopTime,  ///< [in] stop  time for time stepping
       const TimeInterval &InTimeStep  ///< [in] time step
   );

 private:
   /// First part of the baroclinic step of the forward-backward scheme (before
   /// barotropic step)
   void
   doBaroclinicStep1(OceanState *State,   ///< [inout] model state
                     TimeInstant &SimTime ///< [inout] current simulation time
   ) const override;

   /// Final part of the baroclinic step of the forward-backward scheme (after
   /// barotropic step)
   void
   doBaroclinicStep2(OceanState *State,   ///< [inout] model state
                     TimeInstant &SimTime ///< [inout] current simulation time
   ) const override;
};

} // namespace OMEGA
#endif
