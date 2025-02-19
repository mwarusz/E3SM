//===-- ocn/VerticalAdv.cpp - vertical advection ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "VerticalAdv.h"

namespace OMEGA {

VerticalAdv *VerticalAdv::DefVAdv = nullptr;
std::map<std::string, std::unique_ptr<VerticalAdv>> VerticalAdv::AllVAdv;

VerticalAdv::VerticalAdv(const HorzMesh *Mesh, int NVertLevels, int InVectorLength)
    : NEdgesOnCell(Mesh->NEdgesOnCell), EdgesOnCell(Mesh->EdgesOnCell),
      EdgeSignOnCell(Mesh->EdgeSignOnCell), AreaCell(Mesh->AreaCell),
      DvEdge(Mesh->DvEdge),
      VectorLength(InVectorLength)
   {
      Mesh = HorzMesh::getDefault();
      this->VertTransportTop = Array2DReal("VertTransportTop", Mesh->NCellsSize,
                                     NVertLevels + 1);
      this->DivHU = Array2DReal("DivHU", Mesh->NCellsSize, NVertLevels);
   }
//VerticalAdv::VerticalAdv() = default;
//
//VerticalAdv::~VerticalAdv()  {}

int VerticalAdv::init(int NVertLevels, int InVectorLength) {

   I4 Err = 0;

   auto *Mesh = HorzMesh::getDefault();
   VerticalAdv::DefVAdv = create("Default", Mesh, NVertLevels, InVectorLength);

   return Err;
//   HorzMesh *DefHorzMesh = HorzMesh::getDefault();
//   Mesh = DefHorzMesh;
//
//   VectorLength = InVectorLength;
//
//   NEdgesOnCell = Mesh->NEdgesOnCell;
//   EdgesOnCell = Mesh->EdgesOnCell;
//   EdgeSignOnCell = Mesh->EdgeSignOnCell;
//   AreaCell = Mesh->AreaCell;
//   DvEdge = Mesh->DvEdge;
//
//   this->VertTransportTop = Array2DReal("VertTransportTop", Mesh->NCellsSize,
//                                  NVertLevels + 1);
//   this->DivHU = Array2DReal("DivHU", Mesh->NCellsSize, NVertLevels);
}

VerticalAdv *VerticalAdv::create(const std::string &Name, const HorzMesh *Mesh,
                                 const int NVertLevels, const int InVectorLength) {
   if (AllVAdv.find(Name) != AllVAdv.end()) {
      LOG_ERROR("Vadv create error");
      return nullptr;
   }

   auto *NewVadv = new VerticalAdv(Mesh, NVertLevels, InVectorLength);
   AllVAdv.emplace(Name, NewVadv);

}

void VerticalAdv::clear() {

   AllVAdv.clear();

}

int VerticalAdv::computeVerticalTransport(const OceanState *State,
                                          const AuxiliaryState *AuxState,
                                          int TimeLevel, TimeInterval Coeff) {
   int Err = 0;

   const I4 NVertLevels = Mesh->NVertLevels;

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

//   Array2DReal DivHU("DivHU", Mesh->NCellsSize, NVertLevels);
   Array1DReal ProjSSH("ProjSSH", Mesh->NCellsSize);

   parallelFor("compVert0a",
       {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
          computeThicknessWtdDiv(ICell, NormalVelEdge, FluxLayerThickEdge,
                                 DivHU, OldSSH, ProjSSH, CoeffSeconds);
       });

   Array2DReal NewThickness("NewThickness", Mesh->NCellsSize, NVertLevels);

   parallelFor("compVert0b",
       {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
          computeAleThickness(ICell, RestingThickness, ProjSSH, NewThickness);
   });

   parallelFor("compVert0c",
       {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
          computeVerticalTransportTop(ICell, OldThickness, NewThickness, DivHU,
                                      CoeffSeconds);
   });

   return Err;
}

int VerticalAdv::computeVerticalTransport1(const OceanState *State,
                                           const AuxiliaryState *AuxState,
                                           int TimeLevel, TimeInterval Coeff) {

   int Err = 0;

   const I4 NVertLevels = Mesh->NVertLevels;
   const int NChunks = NVertLevels / VectorLength;

   Array2DReal NormalVelEdge;
   State->getNormalVelocity(NormalVelEdge, TimeLevel);
   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;
//   Array2DReal DivHU("DivHU", Mesh->NCellsSize, NVertLevels);

//   parallelFor(
//      {Mesh->NCellsAll, NChunks}, KOKKOS_LAMBDA(int ICell, int KChunk) {
//         computeThicknessWtdDiv1(ICell, KChunk, NormalVelEdge,
//                                 FluxLayerThickEdge, DivHU);
//   });

//   Kokkos::fence();

//   parallelFor(
//      {Mesh->NCellsAll} , KOKKOS_LAMBDA(int ICell) {
//         computeVertTranspTop1(ICell, DivHU);
//   });

   return Err;
}

int VerticalAdv::computeVerticalTransport2(const OceanState *State,
                                           const AuxiliaryState *AuxState,
                                           int TimeLevel, TimeInterval Coeff) {

   int Err = 0;

   OMEGA_SCOPE(LocDivHU, DivHU);

   Array2DReal NormalVelEdge;
   State->getNormalVelocity(NormalVelEdge, TimeLevel);
   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;
//   Array2DReal DivHU("DivHU", Mesh->NCellsSize, NVertLevels);

   parallelFor("compVertAdv2",
      {Mesh->NCellsAll}, KOKKOS_LAMBDA(int ICell) {
         computeVertTransportTop2(ICell, NormalVelEdge, FluxLayerThickEdge,
                                  LocDivHU);
   });

   return Err;
}

int VerticalAdv::computeVerticalTransport3(const OceanState *State,
                                           const AuxiliaryState *AuxState,
                                           const Array2DReal &TmpArray,
                                           int TimeLevel, TimeInterval Coeff) {
   int Err = 0;

//   OMEGA_SCOPE(LocDivHU, DivHU);

   const I4 NVertLevels = Mesh->NVertLevels;
   const int NChunks = NVertLevels / VectorLength;

   Array2DReal LocDivHU("DivHU", Mesh->NCellsSize, NVertLevels);

   std::cout << NVertLevels << " " << NChunks << " " << VectorLength << std::endl;

   Array2DReal NormalVelEdge;
   State->getNormalVelocity(NormalVelEdge, TimeLevel);
   const Array2DReal &FluxLayerThickEdge =
       AuxState->LayerThicknessAux.FluxLayerThickEdge;

   parallelFor("compVertTransp3-1",
      {Mesh->NCellsOwned, NChunks}, KOKKOS_LAMBDA(int ICell, int KChunk) {
//         const Real InvAreaCell = 1._Real / AreaCell(ICell);
         const I4 KStart = KChunk * VectorLength;

//         for (int J = 0; J < NEdgesOnCell(ICell); ++J) {
//            const I4 JEdge = EdgesOnCell(ICell, J);
            for (int KVec = 0; KVec < VectorLength; ++KVec) {
//               const I4 K = KStart + KVec;
               TmpArray(ICell, KVec) = 0;
//               const I4 K = KStart + KVec;
//               const Real Flux = DvEdge(JEdge) * EdgeSignOnCell(ICell, J) *
//                                 FluxLayerThickEdge(JEdge, K) *
//                                 NormalVelEdge(JEdge, K) * InvAreaCell;
//               LocDivHU(ICell, K) -= Flux;
            }
//         }
   });

   return Err;
}

}

