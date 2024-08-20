(omega-dev-time-stepping)=

# Time stepping

The Omega time stepping module is responsible for advancing the model state in
time. The module provides an abstract base class for all Omega time steppers, and
concrete time stepper implementations. Each concrete
time stepper is a separate class that inherits from the base class and, at minimum,
implements the main `doStep` method.

## Configuration enums
Each concrete time stepper
```c++
enum FluxThickEdgeOption { Center, Upwind };
```

## Implemented time steppers
The following auxiliary variable groups are currently implemented:
| Group | Auxiliary Variable | Available options |
| ----- | ------------------- | ------- |
| KineticAuxVars | KineticEnergyCell ||
|| VelocityDiv ||
| LayerThicknessAuxVars | FluxLayerThickEdge | Center or Upwind|
|| MeanLayerThickEdge ||
| VorticityAuxVars |  RelVortVertex ||
||  NormRelVortVertex ||
||  NormPlanetVortVertex ||
||  NormRelVortEdge ||
||  NormPlanetVortEdge ||
| VelocityDel2AuxVars |  Del2Edge ||
||  Del2DivCell ||
||  Del2RelVortVertex ||
