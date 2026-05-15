//===-- Test driver for OMEGA tendency terms ---------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA tendency term functors
///
/// This driver tests the functors used to calculate the tendencies used to
/// update OMEGA state variables. The tests are designed to be run with the
/// planar and spherical meshes described in the OMEGA Quick Start. For each
/// functor, input arrays are initialized based on arbitrary periodic functions
/// defined in the structs for the planar and spherical configurations. The
/// difference between analytical solutions and the output of each function
/// are used to calculate L2 and L-Infinity error norms, which are compared to
/// expected values for the given mesh.
///
//
//===-----------------------------------------------------------------------===/
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Error.h"
#include "Field.h"
#include "GlobalConstants.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "OceanTestCommon.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "TendencyTerms.h"
#include "TimeStepper.h"
#include "Tracers.h"
#include "VertCoord.h"
#include "mpi.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace OMEGA;

void velTend(const Array2DReal &LocNormalVelocityTend,
             const Array2DReal &NormRVortEdge, const Array2DReal &NormFEdge,
             const Array2DReal &FluxLayerThickEdge,
             const Array2DReal &NormVelEdge, const Array2DReal &KECell,
             const Array2DReal &SSHCell, const Array2DReal &DivCell,
             const Array2DReal &RVortVertex, const Array2DReal &Del2DivCell,
             const Array2DReal &Del2RVortVertex, const HorzMesh *Mesh,
             const VertCoord *VCoord) {

   const int NVertLayers       = VCoord->NVertLayers;
   const auto &MinLayerEdgeBot = VCoord->MinLayerEdgeBot;
   const auto &MaxLayerEdgeTop = VCoord->MinLayerEdgeTop;

   PotentialVortHAdvOnEdge LocPotentialVortHAdv(Mesh, VCoord);
   KEGradOnEdge LocKEGrad(Mesh, VCoord);
   SSHGradOnEdge LocSSHGrad(Mesh, VCoord);
   VelocityDiffusionOnEdge LocVelocityDiffusion(Mesh, VCoord);
   VelocityHyperDiffOnEdge LocVelocityHyperDiff(Mesh, VCoord);

   LocPotentialVortHAdv.Enabled = true;
   LocKEGrad.Enabled            = true;
   LocSSHGrad.Enabled           = true;
   LocVelocityDiffusion.Enabled = true;
   LocVelocityHyperDiff.Enabled = true;

   const auto LConfig = LaunchConfig({Mesh->NEdgesOwned}, NVertLayers,
                                     TeamScratch<Real>(NVertLayers));
   parallelForOuter(
       LConfig, KOKKOS_LAMBDA(int IEdge, const TeamMember &Team) {
          ArrayScratch1DReal VelTendScratch(teamScratch(Team), NVertLayers);

          const int KMin = MinLayerEdgeBot(IEdge);
          const int KMax = MaxLayerEdgeTop(IEdge);

          parallelForInnerOpt(
              Team, Range{KMin, KMax},
              INNER_LAMBDA(int K) { VelTendScratch(K) = 0; });

          if (LocPotentialVortHAdv.Enabled) {
             LocPotentialVortHAdv(Team, VelTendScratch, IEdge, NormRVortEdge,
                                  NormFEdge, FluxLayerThickEdge, NormVelEdge);
          }

          if (LocKEGrad.Enabled) {
             LocKEGrad(Team, VelTendScratch, IEdge, KECell);
          }

          if (LocSSHGrad.Enabled) {
             LocSSHGrad(Team, VelTendScratch, IEdge, SSHCell);
          }

          if (LocVelocityDiffusion.Enabled) {
             LocVelocityDiffusion(Team, VelTendScratch, IEdge, DivCell,
                                  RVortVertex);
          }

          if (LocVelocityHyperDiff.Enabled) {
             LocVelocityHyperDiff(Team, VelTendScratch, IEdge, Del2DivCell,
                                  Del2RVortVertex);
          }

          parallelForInnerOpt(
              Team, Range{KMin, KMax}, INNER_LAMBDA(int K) {
                 LocNormalVelocityTend(IEdge, K) = VelTendScratch(K);
              });
       });
}
