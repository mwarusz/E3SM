//===-- ocn/VerticalAdv.cpp - vertical advection ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "VerticalAdv.h"

namespace OMEGA {

VerticalAdv::VerticalAdv(const HorzMesh *Mesh, int InNVertLevels, int InVectorLength)
    : Mesh(Mesh), VertTransportTop("VertTransportTop", Mesh->NCellsSize,
                                     InNVertLevels+1),
      VertCoordWeights("VertCoordWeights", InNVertLevels),
      NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      EdgeSignOnCell(Mesh->EdgeSignOnCell), AreaCell(Mesh->AreaCell),
      DvEdge(Mesh->DvEdge), NVertLevels(InNVertLevels),
      NCellsAll(Mesh->NCellsAll), VectorLength(InVectorLength) {}

int VerticalAdv::computeVerticalTransport(const OceanState *State,
                                          const AuxiliaryState *AuxState,
                                          int TimeLevel, TimeInterval Coeff) {
   int Err = 0;

   R8 CoeffSeconds;
   Coeff.get(CoeffSeconds, TimeUnits::Seconds);

   Array2DReal RestingThickness, OldThickness;
   Array2DReal NormalVelEdge;
   State->getLayerThickness(OldThickness, TimeLevel);
   RestingThickness = OldThickness;
   State->getNormalVelocity(NormalVelEdge, TimeLevel);

   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;
   const Array2DReal &OldSSH = AuxState->LayerThicknessAux.SshCell;

   Array2DReal DivHU("DivHU", Mesh->NCellsSize, NVertLevels);
   Array1DReal ProjSSH("ProjSSH", Mesh->NCellsSize);

   parallelFor(
       {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
          computeThicknessWtdDiv(ICell, NormalVelEdge, FluxLayerThickEdge,
                                 DivHU, OldSSH, ProjSSH, CoeffSeconds);
       });

   Array2DReal NewThickness("NewThickness", Mesh->NCellsSize, NVertLevels);

   parallelFor(
       {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
          computeAleThickness(ICell, RestingThickness, ProjSSH, NewThickness);
   });

   parallelFor(
       {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
          computeVerticalTransportTop(ICell, OldThickness, NewThickness, DivHU,
                                      CoeffSeconds);
   });

   return Err;
}

int VerticalAdv::computeVerticalTransport1(const OceanState *State,
                                           const AuxiliaryState *AuxState,
                                           int TimeLevel, TimeInterval Coeff) {

   int Err = 0;

   int NChunks = NVertLevels / VectorLength;

   Array2DReal NormalVelEdge;
   State->getNormalVelocity(NormalVelEdge, TimeLevel);
   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;
   Array2DReal DivHU("DivHU", Mesh->NCellsSize, NVertLevels);

   parallelFor(
      {NCellsAll, NChunks}, KOKKOS_LAMBDA(int ICell, int KChunk) {
         computeThicknessWtdDiv1(ICell, KChunk, NormalVelEdge,
                                 FluxLayerThickEdge, DivHU);
   });

   Kokkos::fence();

   parallelFor(
      {NCellsAll} , KOKKOS_LAMBDA(int ICell) {
         computeVertTranspTop1(ICell, DivHU);
   });

   return Err;
}

int VerticalAdv::computeVerticalTransport2(const OceanState *State,
                                           const AuxiliaryState *AuxState,
                                           int TimeLevel, TimeInterval Coeff) {

   int Err = 0;

   Array2DReal NormalVelEdge;
   State->getNormalVelocity(NormalVelEdge, TimeLevel);
   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;

   parallelFor(
      {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
         computeVertTransportTop2(ICell, NormalVelEdge, FluxLayerThickEdge);
   });

   return Err;
}

}

