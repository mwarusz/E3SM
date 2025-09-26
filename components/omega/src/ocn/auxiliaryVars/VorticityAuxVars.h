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

   template <class FC>
   KOKKOS_FUNCTION void computeVarsOnVertex(int IVertex, int KStart,
                                            const Array2DReal &LayerThickCell,
                                            const Array2DReal &NormalVelEdge,
                                            FC FullChunk) const {

      // const int KLen =
      //     FullChunk ? VecLength : MaxLayerVertexBot(IVertex) - KStart + 1;
      const int KLen =
          Kokkos::min(VecLength, MaxLayerVertexBot(IVertex) - KStart + 1);

      const Real InvAreaTriangle = 1._Real / AreaTriangle(IVertex);

      Real LayerThickVertex[VecLength] = {0};
      Real RelVortVertexTmp[VecLength] = {0};

      for (int J = 0; J < VertexDegree; ++J) {
         const int JCell = CellsOnVertex(IVertex, J);
         const int JEdge = EdgesOnVertex(IVertex, J);

         for (int KVec = 0; KVec < KLen; ++KVec) {
            const int K = KStart + KVec;
            LayerThickVertex[KVec] += InvAreaTriangle *
                                      KiteAreasOnVertex(IVertex, J) *
                                      LayerThickCell(JCell, K);
            RelVortVertexTmp[KVec] += InvAreaTriangle * DcEdge(JEdge) *
                                      EdgeSignOnVertex(IVertex, J) *
                                      NormalVelEdge(JEdge, K);
         }
      }

      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K                    = KStart + KVec;
         const Real InvLayerThickVertex = 1._Real / LayerThickVertex[KVec];

         RelVortVertex(IVertex, K) = RelVortVertexTmp[KVec];
         NormRelVortVertex(IVertex, K) =
             RelVortVertexTmp[KVec] * InvLayerThickVertex;
         NormPlanetVortVertex(IVertex, K) =
             FVertex(IVertex) * InvLayerThickVertex;
      }
   }

   template <class FC>
   KOKKOS_FUNCTION void computeVarsOnEdge(int IEdge, int KStart,
                                          FC FullChunk) const {

      // const int KLen =
      //     FullChunk ? VecLength : MaxLayerEdgeBot(IEdge) - KStart + 1;
      const int KLen =
          Kokkos::min(VecLength, MaxLayerEdgeTop(IEdge) - KStart + 1);

      const int JVertex0 = VerticesOnEdge(IEdge, 0);
      const int JVertex1 = VerticesOnEdge(IEdge, 1);

      Real NRelVortTmp[VecLength];
      Real NPlanetVortTmp[VecLength];

      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K       = KStart + KVec;
         NRelVortTmp[KVec] = 0.5_Real * (NormRelVortVertex(JVertex0, K) +
                                         NormRelVortVertex(JVertex1, K));
      }
      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K          = KStart + KVec;
         NPlanetVortTmp[KVec] = 0.5_Real * (NormPlanetVortVertex(JVertex0, K) +
                                            NormPlanetVortVertex(JVertex1, K));
      }

      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K               = KStart + KVec;
         NormRelVortEdge(IEdge, K) = NRelVortTmp[KVec];
      }

      for (int KVec = 0; KVec < KLen; ++KVec) {
         const int K                  = KStart + KVec;
         NormPlanetVortEdge(IEdge, K) = NPlanetVortTmp[KVec];
      }
   }

   void registerFields(const std::string &AuxGroupName,
                       const std::string &MeshName) const;
   void unregisterFields() const;

 private:
   I4 VertexDegree;
   Array2DI4 CellsOnVertex;
   Array2DI4 EdgesOnVertex;
   Array2DReal EdgeSignOnVertex;
   Array1DReal DcEdge;
   Array2DReal KiteAreasOnVertex;
   Array1DReal AreaTriangle;
   Array2DI4 VerticesOnEdge;
   Array1DReal FVertex;
   Array1DI4 MaxLayerVertexBot;
   Array1DI4 MaxLayerEdgeTop;
};

} // namespace OMEGA
#endif
