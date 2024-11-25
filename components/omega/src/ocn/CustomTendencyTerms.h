#ifndef OMEGA_CUSTOMTENDENCYTERMS_H
#define OMEGA_CUSTOMTENDENCYTERMS_H
//===-- ocn/CustomTendencyTerms.h - Custom tendency terms -------*- C++ -*-===//
//
/// \file
/// \brief Contains customized tendency terms for the thickness and momentum
///        equations
///
/// For details on the manufactured solution class, see Bishnu et al. (2024)
/// (https://doi.org/10.1029/2022MS003545) and the manufactured solution test
/// case in Polaris. The Polaris package leverages this feature to validate
/// an expected order of convergence of Omega.
//
//===----------------------------------------------------------------------===//

#include "HorzMesh.h"
#include "TendencyTerms.h"
#include "TimeMgr.h"

namespace OMEGA {

//===-----------------------------------------------------------------------===/
// A class for the manufactured solution tendency terms
//===-----------------------------------------------------------------------===/
class ManufacturedSolution {

   //===--------------------------------------------------------------------===/
   // Manufactured tendency term for the thickness equation
   //===--------------------------------------------------------------------===/
   struct ManufacturedThicknessTendency {

      // Constants defined in 'init'
      TimeInstant ReferenceTime;
      R8 H0;
      R8 Eta0;
      R8 Kx;
      R8 Ky;
      R8 AngFreq;

      void operator()(Array2DReal ThicknessTend, const OceanState *State,
                      const AuxiliaryState *AuxState, int ThickTimeLevel,
                      int VelTimeLevel, TimeInstant Time) {

         // Get elapsed time since reference time
         R8 ElapsedTimeSec;
         TimeInterval ElapsedTimeInterval = Time - ReferenceTime;
         ElapsedTimeInterval.get(ElapsedTimeSec, TimeUnits::Seconds);

         auto *Mesh       = HorzMesh::getDefault();
         auto NVertLevels = ThicknessTend.extent_int(1);

         Array1DReal XCell = Mesh->XCell;
         Array1DReal YCell = Mesh->YCell;

         OMEGA_SCOPE(LocXCell, XCell);
         OMEGA_SCOPE(LocYCell, YCell);
         OMEGA_SCOPE(LocH0, H0);
         OMEGA_SCOPE(LocEta0, Eta0);
         OMEGA_SCOPE(LocKx, Kx);
         OMEGA_SCOPE(LocKy, Ky);
         OMEGA_SCOPE(LocAngFreq, AngFreq);

         parallelFor(
             {Mesh->NCellsAll, NVertLevels},
             KOKKOS_LAMBDA(int ICell, int KLevel) {
                R8 X     = LocXCell(ICell);
                R8 Y     = LocYCell(ICell);
                R8 Phase = LocKx * X + LocKy * Y - LocAngFreq * ElapsedTimeSec;
                ThicknessTend(ICell, KLevel) +=
                    LocEta0 *
                    (-LocH0 * (LocKx + LocKy) * sin(Phase) -
                     LocAngFreq * cos(Phase) +
                     LocEta0 * (LocKx + LocKy) * cos(2.0_Real * Phase));
             });
      }
   }; // end struct ManufacturedThicknessTendency

   //===--------------------------------------------------------------------===/
   // Manufactured tendency term for the momentum equation
   //===--------------------------------------------------------------------===/
   struct ManufacturedVelocityTendency {

      // Constants defined in 'init'
      TimeInstant ReferenceTime;
      R8 Grav;
      R8 Eta0;
      R8 Kx;
      R8 Ky;
      R8 AngFreq;
      R8 ViscDel2;
      R8 ViscDel4;
      bool VelDiffTendencyEnable;
      bool VelHyperDiffTendencyEnable;

      void operator()(Array2DReal NormalVelTend, const OceanState *State,
                      const AuxiliaryState *AuxState, int ThickTimeLevel,
                      int VelTimeLevel, TimeInstant Time) {

         // Get elapsed time since reference time
         R8 ElapsedTimeSec;
         TimeInterval ElapsedTimeInterval = Time - ReferenceTime;
         ElapsedTimeInterval.get(ElapsedTimeSec, TimeUnits::Seconds);

         auto *Mesh       = HorzMesh::getDefault();
         auto NVertLevels = NormalVelTend.extent_int(1);

         Array1DReal FEdge     = Mesh->FEdge;
         Array1DReal XEdge     = Mesh->XEdge;
         Array1DReal YEdge     = Mesh->YEdge;
         Array1DReal AngleEdge = Mesh->AngleEdge;

         OMEGA_SCOPE(LocFEdge, FEdge);
         OMEGA_SCOPE(LocXEdge, XEdge);
         OMEGA_SCOPE(LocYEdge, YEdge);
         OMEGA_SCOPE(LocAngleEdge, AngleEdge);
         OMEGA_SCOPE(LocGrav, Grav);
         OMEGA_SCOPE(LocEta0, Eta0);
         OMEGA_SCOPE(LocKx, Kx);
         OMEGA_SCOPE(LocKy, Ky);
         OMEGA_SCOPE(LocAngFreq, AngFreq);
         OMEGA_SCOPE(LocViscDel2, ViscDel2);
         OMEGA_SCOPE(LocViscDel4, ViscDel4);
         OMEGA_SCOPE(LocVelDiffTendencyEnable, VelDiffTendencyEnable);
         OMEGA_SCOPE(LocVelHyperDiffTendencyEnable, VelHyperDiffTendencyEnable);

         R8 LocKx2 = LocKx * LocKx;
         R8 LocKy2 = LocKy * LocKy;
         R8 LocKx4 = LocKx2 * LocKx2;
         R8 LocKy4 = LocKy2 * LocKy2;

         parallelFor(
             {Mesh->NEdgesAll, NVertLevels},
             KOKKOS_LAMBDA(int IEdge, int KLevel) {
                R8 X = LocXEdge(IEdge);
                R8 Y = LocYEdge(IEdge);

                R8 Phase = LocKx * X + LocKy * Y - LocAngFreq * ElapsedTimeSec;
                R8 SourceTerm0 = LocAngFreq * sin(Phase) -
                                 0.5_Real * LocEta0 * (LocKx + LocKy) *
                                     sin(2.0_Real * Phase);

                R8 U = LocEta0 *
                       ((-LocFEdge(IEdge) + LocGrav * LocKx) * cos(Phase) +
                        SourceTerm0);
                R8 V = LocEta0 *
                       ((LocFEdge(IEdge) + LocGrav * LocKy) * cos(Phase) +
                        SourceTerm0);

                // Del2 and del4 source terms
                if (LocVelDiffTendencyEnable) {
                   U += LocViscDel2 * LocEta0 * LocKx2 * cos(Phase);
                   V += LocViscDel2 * LocEta0 * LocKy2 * cos(Phase);
                }
                if (LocVelHyperDiffTendencyEnable) {
                   U -= LocViscDel4 * LocEta0 *
                        (LocKx4 * cos(Phase) + LocKx2 * LocKy2 * cos(Phase));
                   V -= LocViscDel4 * LocEta0 *
                        (LocKy4 * cos(Phase) + LocKx2 * LocKy2 * cos(Phase));
                }

                R8 NormalCompSourceTerm =
                    cos(LocAngleEdge(IEdge)) * U + sin(LocAngleEdge(IEdge)) * V;
                NormalVelTend(IEdge, KLevel) += NormalCompSourceTerm;
             });
      }
   }; // end struct ManufacturedVelocityTendency

 public:
   // Instances of manufactured tendencies
   ManufacturedThicknessTendency ManufacturedThickTend;
   ManufacturedVelocityTendency ManufacturedVelTend;

   int init();

}; // end class ManufacturedSolution

} // end namespace OMEGA

//===-----------------------------------------------------------------------===/

#endif
