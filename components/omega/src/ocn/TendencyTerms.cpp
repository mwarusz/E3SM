//===-- ocn/TendencyTerms.cpp - Tendency Terms ------------------*- C++ -*-===//
//
// The tendency terms that update state variables are implemented as functors,
// i.e. as classes that act like functions. This source defines the class
// constructors for these functors, which initialize the functor objects using
// the Mesh objects and info from the Config. The function call operators () are
// defined in the corresponding header file.
//
//===----------------------------------------------------------------------===//

#include "TendencyTerms.h"
#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"
#include "OceanState.h"

namespace OMEGA {

Tendencies *Tendencies::DefaultTendencies = nullptr;
std::map<std::string, Tendencies> Tendencies::AllTendencies;

//------------------------------------------------------------------------------
// Initialize the tendencies. Assumes that HorzMesh as alread been initialized.
int Tendencies::init() {
   int Err = 0;

   HorzMesh *DefHorzMesh = HorzMesh::getDefault();
   Config *TendConfig;

   int NVertLevels = 60;

   Tendencies DefTendencies("Default", DefHorzMesh, NVertLevels, TendConfig);

   Tendencies::DefaultTendencies = get("Default");

   return Err;

} // end init

//------------------------------------------------------------------------------
// Destroys the tendencies
Tendencies::~Tendencies() {

   // No operations needed, Kokkos arrays removed when no longer in scope

} // end destructor

//------------------------------------------------------------------------------
// Removes all tendencies instances before exit
void Tendencies::clear() { AllTendencies.clear(); } // end clear

//------------------------------------------------------------------------------
// Removes tendencies from list by name
void Tendencies::erase(const std::string Name) {

   AllTendencies.erase(Name);

} // end erase

//------------------------------------------------------------------------------
// Get default tendencies
Tendencies *Tendencies::getDefault() {

   return Tendencies::DefaultTendencies;

} // end get default

//------------------------------------------------------------------------------
// Get tendencies by name
Tendencies *Tendencies::get(const std::string Name ///< [in] Name of tendencies
) {

   auto it = AllTendencies.find(Name);

   if (it != AllTendencies.end()) {
      return &(it->second);
   } else {
      LOG_ERROR(
          "Tendencies::get: Attempt to retrieve non-existent tendencies:");
      LOG_ERROR("{} has not been defined or has been removed", Name);
      return nullptr;
   }

} // end get tendencies

//------------------------------------------------------------------------------
// Construct a new group of tendencies
Tendencies::Tendencies(const std::string &Name, ///< [in] Name for tendencies
                       const HorzMesh *Mesh,    ///< [in] Horizontal mesh
                       int NVertLevels, ///< [in] Number of vertical levels
                       Config *Options  ///< [in] Configuration options
                       )
    : ThicknessFluxDiv(Mesh, Options), PotientialVortHAdv(Mesh, Options),
      KEGrad(Mesh, Options), SHHGrad(Mesh, Options),
      VelocityDiffusion(Mesh, Options), VelocityHyperDiff(Mesh, Options) {

   // Tendency arrays
   LayerThicknessTend =
       Array2DReal("LayerThicknessTend", Mesh->NCellsSize, NVertLevels);
   NormalVelocityTend =
       Array2DReal("NormalVelocityTend", Mesh->NEdgesSize, NVertLevels);

   //
   NCellsOwned = Mesh->NCellsOwned;
   NEdgesOwned = Mesh->NEdgesOwned;
   NChunks     = NVertLevels / VecLength;

   // Associate this instance with a name
   AllTendencies.emplace(Name, *this);

} // end constructor

//------------------------------------------------------------------------------
// Compute tendencies for layer thickness equation
// TODO Add AuxilaryState as argument
void Tendencies::computeThicknessTendencies(
    OceanState *State,        ///< [in] State variables
    AuxiliaryState *AuxState, ///< [in] Auxilary state variables
    int TimeLevel,            ///< [in] Time level
    Real Time                 ///< [in] Time
) {

   OMEGA_SCOPE(LocLayerThicknessTend, LayerThicknessTend);
   OMEGA_SCOPE(LocNCellsOwned, NCellsOwned);
   OMEGA_SCOPE(LocNChunks, NChunks);
   OMEGA_SCOPE(LocThicknessFluxDiv, ThicknessFluxDiv);

   // Compute thickness flux divergence
   const Array2DReal &ThickFluxEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;
   parallelFor(
       {LocNCellsOwned, LocNChunks}, KOKKOS_LAMBDA(int ICell, int KChunk) {
          LocThicknessFluxDiv(LocLayerThicknessTend, ICell, KChunk,
                              ThickFluxEdge);
       });

} // end thickness tendency compute

//------------------------------------------------------------------------------
// Compute tendencies for normal velocity equation
void Tendencies::computeVelocityTendencies(
    OceanState *State,        ///< [in] State variables
    AuxiliaryState *AuxState, ///< [in] Auxilary state variables
    int TimeLevel,            ///< [in] Time level
    Real Time                 ///< [in] Time
) {

   OMEGA_SCOPE(LocNormalVelocityTend, NormalVelocityTend);
   OMEGA_SCOPE(LocNEdgesOwned, NEdgesOwned);
   OMEGA_SCOPE(LocNChunks, NChunks);
   OMEGA_SCOPE(LocPotientialVortHAdv, PotientialVortHAdv);
   OMEGA_SCOPE(LocKEGrad, KEGrad);
   OMEGA_SCOPE(LocSHHGrad, SHHGrad);
   OMEGA_SCOPE(LocVelocityDiffusion, VelocityDiffusion);
   OMEGA_SCOPE(LocVelocityHyperDiff, VelocityHyperDiff);

   // Compute potential vorticity horizontal advection
   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;
   const Array2DReal &NormRVortEdge = AuxState->VorticityAux.NormRelVortEdge;
   const Array2DReal &NormFEdge     = AuxState->VorticityAux.NormPlanetVortEdge;
   const Array2DReal &NormVelEdge   = State->NormalVelocity[TimeLevel];
   parallelFor(
       {LocNEdgesOwned, LocNChunks}, KOKKOS_LAMBDA(int IEdge, int KChunk) {
          LocPotientialVortHAdv(LocNormalVelocityTend, IEdge, KChunk,
                                NormRVortEdge, NormFEdge, FluxLayerThickEdge,
                                NormVelEdge);
       });

   // Compute kinetic energy gradient
   const Array2DReal &KECell = AuxState->KineticAux.KineticEnergyCell;
   parallelFor(
       {LocNEdgesOwned, LocNChunks}, KOKKOS_LAMBDA(int IEdge, int KChunk) {
          LocKEGrad(LocNormalVelocityTend, IEdge, KChunk, KECell);
       });

   // Compute sea surface height gradient
   const Array2DReal &SSHCell = AuxState->LayerThicknessAux.SshCell;
   parallelFor(
       {LocNEdgesOwned, LocNChunks}, KOKKOS_LAMBDA(int IEdge, int KChunk) {
          LocSHHGrad(LocNormalVelocityTend, IEdge, KChunk, SSHCell);
       });

   // Compute del2 horizontal diffusion
   const Array2DReal &DivCell     = AuxState->KineticAux.VelocityDivCell;
   const Array2DReal &RVortVertex = AuxState->VorticityAux.RelVortVertex;
   parallelFor(
       {LocNEdgesOwned, LocNChunks}, KOKKOS_LAMBDA(int IEdge, int KChunk) {
          LocVelocityDiffusion(LocNormalVelocityTend, IEdge, KChunk, DivCell,
                               RVortVertex);
       });

   // Compute del4 horizontal diffusion
   const Array2DReal &Del2DivCell = AuxState->VelocityDel2Aux.Del2DivCell;
   const Array2DReal &Del2RVortVertex =
       AuxState->VelocityDel2Aux.Del2RelVortVertex;
   parallelFor(
       {LocNEdgesOwned, LocNChunks}, KOKKOS_LAMBDA(int IEdge, int KChunk) {
          LocVelocityHyperDiff(LocNormalVelocityTend, IEdge, KChunk,
                               Del2DivCell, Del2RVortVertex);
       });

} // end velocity tendency compute

//------------------------------------------------------------------------------
// Compute both layer thickness and normal velocity tendencies
void Tendencies::computeAllTendencies(
    OceanState *State,        ///< [in] State variables
    AuxiliaryState *AuxState, ///< [in] Auxilary state variables
    int TimeLevel,            ///< [in] Time level
    Real Time                 ///< [in] Time
) {

   AuxState->computeAll(State, TimeLevel);
   computeThicknessTendencies(State, AuxState, TimeLevel, Time);
   computeVelocityTendencies(State, AuxState, TimeLevel, Time);

} // end all tendency compute

// TODO: Implement Config options for all constructors
ThicknessFluxDivOnCell::ThicknessFluxDivOnCell(const HorzMesh *Mesh,
                                               Config *Options)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      DvEdge(Mesh->DvEdge), AreaCell(Mesh->AreaCell),
      EdgeSignOnCell(Mesh->EdgeSignOnCell) {

   // Options->get("ThicknessFluxTendencyEnable", Enabled);
}

PotentialVortHAdvOnEdge::PotentialVortHAdvOnEdge(const HorzMesh *Mesh,
                                                 Config *Options)
    : NEdgesOnEdge(Mesh->NEdgesOnEdge), EdgesOnEdge(Mesh->EdgesOnEdge),
      WeightsOnEdge(Mesh->WeightsOnEdge) {

   // Options->get("PVTendencyEnable", Enabled);
}

KEGradOnEdge::KEGradOnEdge(const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {

   // Options->get("KETendencyEnable", Enabled);
}

SSHGradOnEdge::SSHGradOnEdge(const HorzMesh *Mesh, Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), DcEdge(Mesh->DcEdge) {

   // Options->get("SSHTendencyEnable", Enabled);
}

VelocityDiffusionOnEdge::VelocityDiffusionOnEdge(const HorzMesh *Mesh,
                                                 Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge),
      MeshScalingDel2(Mesh->MeshScalingDel2), EdgeMask(Mesh->EdgeMask) {

   // Options->get("VelDiffTendencyEnable", Enabled);
   // Options->get("ViscDel2", ViscDel2);
}

VelocityHyperDiffOnEdge::VelocityHyperDiffOnEdge(const HorzMesh *Mesh,
                                                 Config *Options)
    : CellsOnEdge(Mesh->CellsOnEdge), VerticesOnEdge(Mesh->VerticesOnEdge),
      DcEdge(Mesh->DcEdge), DvEdge(Mesh->DvEdge),
      MeshScalingDel4(Mesh->MeshScalingDel4), EdgeMask(Mesh->EdgeMask) {

   // Options->get("VelHyperDiffTendencyEnable", Enabled);
   // Options->get("ViscDel4", ViscDel4);
}

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
