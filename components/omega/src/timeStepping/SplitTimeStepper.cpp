#include "SplitTimeStepper.h"
#include "UnsplitBarotropicStepper.h"

namespace OMEGA {

// This constructor simply calls the base class constuctor with its arguments
SplitTimeStepper::SplitTimeStepper(
    const std::string &InName,      // [in] name of time stepper
    TimeStepperType InType,         // [in] type (time stepping method)
    I4 InNTimeLevels,               // [in] num time levels needed by method
    const TimeInstant &InStartTime, // [in] start time for time stepping
    const TimeInstant &InStopTime,  // [in] stop  time for time stepping
    const TimeInterval &InTimeStep  // [in] time step
    )
    : TimeStepper(InName, InType, InNTimeLevels, InStartTime, InStopTime,
                  InTimeStep) {}

//------------------------------------------------------------------------------
// Advance the state by one step of the split scheme
void SplitTimeStepper::doStep(OceanState *State,   // model state
                              TimeInstant &SimTime // current simulation time
) const {

   int Err;

   doBaroclinicStep1(State, SimTime);

   BaroStepper->doBarotropicStep(State, SimTime);

   doBaroclinicStep2(State, SimTime);

   // Advance the clock and update the simulation time
   Err     = StepClock->advance();
   SimTime = StepClock->getCurrentTime();
}

void SplitTimeStepper::finalizeInit() {

   if (!Tend)
      LOG_CRITICAL("Tendency not initialized");
   if (!Mesh)
      LOG_CRITICAL("Invalid mesh");
   if (!MeshHalo)
      LOG_CRITICAL("Invalid MeshHalo");

   Error Err;

   // Retrieve TimeStepper options from Config if available
   Config *OmegaConfig = Config::getOmegaConfig();
   Config TimeIntConfig("TimeIntegration");
   Err = OmegaConfig->get(TimeIntConfig);
   CHECK_ERROR_ABORT(Err, "TimeIntegration group not found in Config");

   // Initialize choice of barotropic time stepper
   std::string BarotropicStepperStr;
   Err += TimeIntConfig.get("BarotropicStepper", BarotropicStepperStr);
   CHECK_ERROR_ABORT(Err,
                     "BarotropicStepper not found in TimeIntegration Config");
   BarotropicStepperType BarotropicStepperChoice =
       getBarotropicStepperFromStr(BarotropicStepperStr);

   switch (BarotropicStepperChoice) {
   case BarotropicStepperType::Unsplit:
      BaroStepper = std::make_unique<UnsplitBarotropicStepper>();
      break;
   case BarotropicStepperType::Invalid:
      ABORT_ERROR("Invalid barotropic stepper");
   default:
      ABORT_ERROR("Unknown barotropic stepper");
   }
}

} // namespace OMEGA
