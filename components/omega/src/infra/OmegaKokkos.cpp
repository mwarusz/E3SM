#include "OmegaKokkos.h"
#include "Config.h"
#include "Error.h"

#include <cstdlib>

namespace OMEGA {

static TeamConfig DefaultTeamCfg{};

TeamConfig defaultTeamConfig() { return DefaultTeamCfg; }

void readKokkosConfig() {
   Error Err;

   Config *OmegaConfig = Config::getOmegaConfig();
   Config KokkosConfig("Kokkos");
   Err += OmegaConfig->get(KokkosConfig);
   CHECK_ERROR_ABORT(Err, "OmegaKokkos: Kokkos group not found in Config");

   std::string TeamSizeStr;
   Err += KokkosConfig.get("TeamSize", TeamSizeStr);
   CHECK_ERROR_ABORT(Err, "OmegaKokkos: TeamSize not found in KokkosConfig");

   std::string VectorSizeStr;
   Err += KokkosConfig.get("VectorSize", VectorSizeStr);
   CHECK_ERROR_ABORT(Err, "OmegaKokkos: VectorSize not found in KokkosConfig");

   if (TeamSizeStr != "Auto") {
      DefaultTeamCfg.TeamSize = std::atoi(TeamSizeStr.c_str());
   }

   if (VectorSizeStr != "Auto") {
      DefaultTeamCfg.VectorSize = std::atoi(VectorSizeStr.c_str());
   }
}

} // namespace OMEGA
