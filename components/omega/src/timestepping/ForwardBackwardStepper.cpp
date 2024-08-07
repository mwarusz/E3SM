#include "ForwardBackwardStepper.h"

namespace OMEGA {

ForwardBackwardStepper::ForwardBackwardStepper(const std::string &Name,
                                               Tendencies *Tend,
                                               AuxiliaryState *AuxState,
                                               HorzMesh *Mesh, Halo *MeshHalo)
    : TimeStepper(Name, TimeStepperType::ForwardBackward, Tend, AuxState, Mesh,
                  MeshHalo) {}

void ForwardBackwardStepper::doStep(OceanState *State, Real Time,
                                    Real TimeStep) const {
   computeThickTend(State, 0, Time);
   updateThicknessByTend(State, 1, State, 0, TimeStep);
   // TODO(mwarusz): this copy could be avoided with a more flexible interface
   deepCopy(State->NormalVelocity[1], State->NormalVelocity[0]);

   computeVelTend(State, 1, Time + TimeStep);
   updateVelocityByTend(State, 1, TimeStep);

   State->updateTimeLevels();
}

} // namespace OMEGA
