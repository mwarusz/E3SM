#ifndef OMEGA_AUX_VORTICITY_H
#define OMEGA_AUX_VORTICITY_H

#include "DataTypes.h"
#include "HorzMesh.h"
#include "OmegaKokkos.h"
#include "VertCoord.h"

#include <string>

namespace OMEGA {

class VorticityAuxVars {
 public:
   Array2DReal RelVortVertex;
   Array2DReal NormRelVortVertex;
   Array2DReal NormPlanetVortVertex;

   Array2DReal NormRelVortEdge;
   Array2DReal NormPlanetVortEdge;

   VorticityAuxVars(const std::string &AuxStateSuffix, const HorzMesh *Mesh,
                    const VertCoord *VCoord);

   KOKKOS_FUNCTION void
   computeVarsOnVertex(const TeamMember &Team, int IVertex,
                       const Array2DReal &LayerThickCell,
                       const Array2DReal &NormalVelEdge) const {

      Scratch1DReal LayerThickVertex(Team.team_scratch(0), NVertLayers);
      Scratch1DReal RelVortVertexTmp(Team.team_scratch(0), NVertLayers);

      const int KMinVertex   = MinLayerVertexTop(IVertex);
      const int KMaxVertex   = MaxLayerVertexBot(IVertex);
      const int KRangeVertex = vertRange(KMinVertex, KMaxVertex);

      const Real InvAreaTriangle = 1._Real / AreaTriangle(IVertex);

      parallelForInner(
          Team, KRangeVertex, INNER_LAMBDA(int KOff) {
             const I4 K          = KMinVertex + KOff;
             LayerThickVertex(K) = 0;
             RelVortVertexTmp(K) = 0;
          });

      for (int J = 0; J < VertexDegree; ++J) {
         const int JCell = CellsOnVertex(IVertex, J);

         const int KMinCell   = MinLayerCell(JCell);
         const int KMaxCell   = MaxLayerCell(JCell);
         const int KRangeCell = vertRange(KMinCell, KMaxCell);

         parallelForInner(
             Team, KRangeCell, INNER_LAMBDA(int KOff) {
                const I4 K = KMinCell + KOff;
                LayerThickVertex(K) += InvAreaTriangle *
                                       KiteAreasOnVertex(IVertex, J) *
                                       LayerThickCell(JCell, K);
             });

         const int JEdge = EdgesOnVertex(IVertex, J);

         const int KMinEdge   = MinLayerEdgeTop(JEdge);
         const int KMaxEdge   = MaxLayerEdgeBot(JEdge);
         const int KRangeEdge = vertRange(KMinEdge, KMaxEdge);

         parallelForInner(
             Team, KRangeEdge, INNER_LAMBDA(int KOff) {
                const I4 K = KMinEdge + KOff;
                RelVortVertexTmp(K) += InvAreaTriangle * DcEdge(JEdge) *
                                       EdgeSignOnVertex(IVertex, J) *
                                       NormalVelEdge(JEdge, K);
             });
      }

      parallelForInner(
          Team, KRangeVertex, INNER_LAMBDA(int KOff) {
             const I4 K                     = KMinVertex + KOff;
             const Real InvLayerThickVertex = 1._Real / LayerThickVertex(K);

             RelVortVertex(IVertex, K) = RelVortVertexTmp(K);
             NormRelVortVertex(IVertex, K) =
                 RelVortVertexTmp(K) * InvLayerThickVertex;
             NormPlanetVortVertex(IVertex, K) =
                 FVertex(IVertex) * InvLayerThickVertex;
          });
   }

   KOKKOS_FUNCTION void computeVarsOnEdge(const TeamMember &Team,
                                          int IEdge) const {
      const int KMin   = MinLayerEdgeTop(IEdge);
      const int KMax   = MaxLayerEdgeBot(IEdge);
      const int KRange = vertRange(KMin, KMax);

      const int JVertex0 = VerticesOnEdge(IEdge, 0);
      const int JVertex1 = VerticesOnEdge(IEdge, 1);

      parallelForInner(
          Team, KRange, INNER_LAMBDA(int KOff) {
             const I4 K = KMin + KOff;
             NormRelVortEdge(IEdge, K) =
                 0.5_Real * (NormRelVortVertex(JVertex0, K) +
                             NormRelVortVertex(JVertex1, K));

             NormPlanetVortEdge(IEdge, K) =
                 0.5_Real * (NormPlanetVortVertex(JVertex0, K) +
                             NormPlanetVortVertex(JVertex1, K));
          });
   }

   void registerFields(const std::string &AuxGroupName,
                       const std::string &MeshName) const;
   void unregisterFields() const;

 private:
   I4 NVertLayers;
   I4 VertexDegree;
   Array2DI4 CellsOnVertex;
   Array2DI4 EdgesOnVertex;
   Array2DReal EdgeSignOnVertex;
   Array1DReal DcEdge;
   Array2DReal KiteAreasOnVertex;
   Array1DReal AreaTriangle;
   Array2DI4 VerticesOnEdge;
   Array1DReal FVertex;

   Array1DI4 MinLayerVertexTop;
   Array1DI4 MaxLayerVertexBot;
   Array1DI4 MinLayerCell;
   Array1DI4 MaxLayerCell;
   Array1DI4 MinLayerEdgeTop;
   Array1DI4 MaxLayerEdgeBot;
};

} // namespace OMEGA
#endif
