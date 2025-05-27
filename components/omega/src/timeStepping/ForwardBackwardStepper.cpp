//===-- ForwardBackwardStepper.cpp - forward-backward methods --*- C++ -*--===//
//
// Contains methods for the Forward-Backward time stepper
//
//===----------------------------------------------------------------------===//

#include "ForwardBackwardStepper.h"
#include "Pacer.h"

#define TIME_HALO
// #define TIME_TEND

namespace OMEGA {

//------------------------------------------------------------------------------
// Constructor creates an instance of a forward-backward stepper and
// fills with some time information. Data pointers are added later.
// Mostly passes relevant info to the base constructor.
ForwardBackwardStepper::ForwardBackwardStepper(
    const std::string &InName,      ///< [in] name of time stepper
    const TimeInstant &InStartTime, ///< [in] start time for time stepping
    const TimeInstant &InStopTime,  ///< [in] stop  time for time stepping
    const TimeInterval &InTimeStep  ///< [in] time step
    )
    : TimeStepper(InName, TimeStepperType::ForwardBackward, 2, InStartTime,
                  InStopTime, InTimeStep) {}

//------------------------------------------------------------------------------
// Advance the state by one step of the forward-backward scheme
void ForwardBackwardStepper::doStep(
    OceanState *State,   // input model state
    TimeInstant &SimTime // current simulation time
) const {

   int Err = 0;

   const int CurLevel  = 0;
   const int NextLevel = 1;

   static int step = 0;

   Array3DReal CurTracerArray, NextTracerArray;
   Err = Tracers::getAll(CurTracerArray, CurLevel);
   Err = Tracers::getAll(NextTracerArray, NextLevel);

   if (State == nullptr)
      LOG_CRITICAL("Invalid State");
   if (AuxState == nullptr)
      LOG_CRITICAL("Invalid AuxState");

      // R_h^{n} = RHS_h(u^{n}, h^{n}, t^{n})
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::start("computeThicknessTendencies");
   }
#endif
   Tend->computeThicknessTendencies(State, AuxState, CurLevel, CurLevel,
                                    SimTime);
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("computeThicknessTendencies");
   }
#endif

   // h^{n+1} = h^{n} + R_h^{n}
#ifdef TIME_TEND
   if (step > 0) {
      Pacer::start("updateThicknessByTend");
   }
#endif
   updateThicknessByTend(State, NextLevel, State, CurLevel, TimeStep);
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("updateThicknessByTend");
   }
#endif

   // R_phi^{n} = RHS_phi(u^{n}, h^{n}, phi^{n}, t^{n})
#ifdef TIME_TEND
   if (step > 0) {
      Pacer::start("computeTracerTendencies");
   }
#endif
   Tend->computeTracerTendencies(State, AuxState, CurTracerArray, CurLevel,
                                 CurLevel, SimTime);
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("computeTracerTendencies");
   }
#endif

   // phi^{n+1} = (phi^{n} * h^{n} + R_phi^{n}) / h^{n+1}
#ifdef TIME_TEND
   if (step > 0) {
      Pacer::start("updateTracersByTend");
   }
#endif
   updateTracersByTend(NextTracerArray, CurTracerArray, State, NextLevel, State,
                       CurLevel, TimeStep);
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("updateTracersByTend");
   }
#endif

   // R_u^{n+1} = RHS_u(u^{n}, h^{n+1}, t^{n+1})
#ifdef TIME_TEND
   if (step > 0) {
      Pacer::start("computeVelocityTendencies");
   }
#endif
   Tend->computeVelocityTendencies(State, AuxState, NextLevel, CurLevel,
                                   SimTime + TimeStep);
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("computeVelocityTendencies");
   }
#endif

   // u^{n+1} = u^{n} + R_u^{n+1}
#ifdef TIME_TEND
   if (step > 0) {
      Pacer::start("updateVelocityByTend");
   }
#endif
   updateVelocityByTend(State, NextLevel, State, CurLevel, TimeStep);
#ifdef TIME_TEND
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("updateVelocityByTend");
   }
#endif

   // Update time levels (New -> Old) of prognostic variables with halo
   // exchanges

   // if (step == 0) {
#ifdef TIME_HALO
   if (step > 0) {
      const auto Comm = MachEnv::getDefault()->getComm();
      Kokkos::fence();
      MPI_Barrier(Comm);
      Pacer::start("updateTimeLevels");
   }
#endif
   State->updateTimeLevels();
   Tracers::updateTimeLevels();
#ifdef TIME_HALO
   if (step > 0) {
      Kokkos::fence();
      Pacer::stop("updateTimeLevels");
   }
#endif

   // Advance the clock and update the simulation time
   Err     = StepClock->advance();
   SimTime = StepClock->getCurrentTime();

   if (step == 0) {
      step++;
   }
}

} // namespace OMEGA
