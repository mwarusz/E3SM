//===-- ocn/CustomTendencyTerms.cpp - Custom tendency terms -----*- C++ -*-===//
//
// The customized tendency terms can be added to the tendency terms based
// based on an option 'UseCustomTendency' in Tendencies Config group.
// This file contains functions for initializing customized tendency terms.
//
//===----------------------------------------------------------------------===//

#include "CustomTendencyTerms.h"
#include "Config.h"
#include "TimeStepper.h"

namespace OMEGA {

//===-----------------------------------------------------------------------===/
// Initialize the manufactured solution tendency terms.
//===-----------------------------------------------------------------------===/
int ManufacturedSolution::init() {
   int Err;

   // Get ManufacturedSolConfig group
   Config *OmegaConfig = Config::getOmegaConfig();
   Config ManufacturedSolConfig("ManufacturedSolution");
   Err = OmegaConfig->get(ManufacturedSolConfig);
   if (Err != 0) {
      LOG_CRITICAL("ManufacturedSolution:: ManufacturedSolution group "
                   "not found in Config");
      return Err;
   }

   // Get TendConfig group
   Config TendConfig("Tendencies");
   Err = OmegaConfig->get(TendConfig);
   if (Err != 0) {
      LOG_CRITICAL("ManufacturedSolution:: Tendencies group "
                   "not found in Config");
      return Err;
   }

   // Get manufactured solution parameters from Config
   R8 WavelengthX;
   R8 WavelengthY;
   R8 Amplitude;

   Err = ManufacturedSolConfig.get("WavelengthX", WavelengthX);
   if (Err != 0) {
      LOG_ERROR("ManufacturedSolution:: WavelengthX not found in "
                "ManufacturedSolConfig");
      return Err;
   }

   Err = ManufacturedSolConfig.get("WavelengthY", WavelengthY);
   if (Err != 0) {
      LOG_ERROR("ManufacturedSolution:: WavelengthY not found in "
                "ManufacturedSolConfig");
      return Err;
   }

   Err = ManufacturedSolConfig.get("Amplitude", Amplitude);
   if (Err != 0) {
      LOG_ERROR("ManufacturedSolution:: Amplitude not found in "
                "ManufacturedSolConfig");
      return Err;
   }

   // Get Tendendices parameters for del2 and del4 source terms
   Err = TendConfig.get("VelDiffTendencyEnable",
                        ManufacturedVelTend.VelDiffTendencyEnable);
   Err += TendConfig.get("VelHyperDiffTendencyEnable",
                         ManufacturedVelTend.VelHyperDiffTendencyEnable);
   Err += TendConfig.get("ViscDel2", ManufacturedVelTend.ViscDel2);
   Err += TendConfig.get("ViscDel4", ManufacturedVelTend.ViscDel4);

   if (Err != 0) {
      LOG_ERROR("ManufacturedSolution::Error reading Tendencies config");
      return Err;
   }

   // Get the reference time to compute the model elapsed time
   /// Get model clock from time stepper
   TimeStepper *DefStepper             = TimeStepper::getDefault();
   Clock *ModelClock                   = DefStepper->getClock();
   ManufacturedThickTend.ReferenceTime = ModelClock->getCurrentTime();
   ManufacturedVelTend.ReferenceTime   = ManufacturedThickTend.ReferenceTime;

   // Get BottomDepth for the resting thickness
   /// This test case assumes that the restingThickness is horizontally uniform
   /// and that only one vertical level is used so only one set of indices is
   /// used here.
   HorzMesh *DefHorzMesh = HorzMesh::getDefault();
   R8 H0                 = DefHorzMesh->BottomDepthH(0);

   // Define and compute common constants
   R8 Grav    = 9.80665_Real;                          // Gravity acceleration
   R8 Pii     = 3.141592653589793_Real;                // Pi
   R8 Kx      = 2.0_Real * Pii / WavelengthX;          // Wave in X-dir
   R8 Ky      = 2.0_Real * Pii / WavelengthY;          // Wave in Y-dir
   R8 AngFreq = sqrt(H0 * Grav * (Kx * Kx + Ky * Ky)); // Angular frequency

   // Assign constants for thickness tendency function
   ManufacturedThickTend.H0      = H0;
   ManufacturedThickTend.Eta0    = Amplitude;
   ManufacturedThickTend.Kx      = Kx;
   ManufacturedThickTend.Ky      = Ky;
   ManufacturedThickTend.AngFreq = AngFreq;

   // Assign constants for velocity tendency function
   ManufacturedVelTend.Grav    = Grav;
   ManufacturedVelTend.Eta0    = Amplitude;
   ManufacturedVelTend.Kx      = Kx;
   ManufacturedVelTend.Ky      = Ky;
   ManufacturedVelTend.AngFreq = AngFreq;

   return Err;

} // end ManufacturedSolution init

} // end namespace OMEGA

//=-------------------------------------------------------------------------===/
