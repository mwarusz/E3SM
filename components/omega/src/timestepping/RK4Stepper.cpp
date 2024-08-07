#include "RK4Stepper.h"

namespace OMEGA {

RK4Stepper::RK4Stepper(const std::string &Name, Tendencies *Tend,
                       AuxiliaryState *AuxState, HorzMesh *Mesh, Halo *MeshHalo)
    : TimeStepper(Name, TimeStepperType::RungeKutta4, Tend, AuxState, Mesh,
                  MeshHalo) {

   auto *DefDecomp  = Decomp::getDefault();
   auto NVertLevels = OceanState::getDefault()->LayerThickness[0].extent_int(1);

   ProvisState =
       OceanState::create("Provis", Mesh, DefDecomp, MeshHalo, NVertLevels, 1);

   RKA[0] = 0;
   RKA[1] = 1. / 2;
   RKA[2] = 1. / 2;
   RKA[3] = 1;

   RKB[0] = 1. / 6;
   RKB[1] = 1. / 3;
   RKB[2] = 1. / 3;
   RKB[3] = 1. / 6;

   RKC[0] = 0;
   RKC[1] = 1. / 2;
   RKC[2] = 1. / 2;
   RKC[3] = 1;
}

void RK4Stepper::doStep(OceanState *State, Real Time, Real TimeStep) const {

   for (int Stage = 0; Stage < NStages; ++Stage) {
      const Real StageTime = Time + RKC[Stage] * TimeStep;
      if (Stage == 0) {
         computeTendencies(State, 0, StageTime);
         updateStateByTend(State, 1, State, 0, RKB[Stage] * TimeStep);
      } else {
         updateStateByTend(ProvisState, 0, State, 0, RKA[Stage] * TimeStep);

         if (Stage == 2) {
            ProvisState->copyToHost(0);
            MeshHalo->exchangeFullArrayHalo(ProvisState->NormalVelocityH[0],
                                            OnEdge);
            MeshHalo->exchangeFullArrayHalo(ProvisState->LayerThicknessH[0],
                                            OnCell);
            ProvisState->copyToDevice(0);
         }

         computeTendencies(ProvisState, 0, StageTime);
         updateStateByTend(State, 1, RKB[Stage] * TimeStep);
      }
   }

   State->updateTimeLevels();
}

} // namespace OMEGA
