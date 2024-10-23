#include "TendencyTerms.h"
#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"
#include "OceanState.h"

using namespace OMEGA;

void pv_flux_func(const Array2DReal &vn_tend_edge,
	          const Array2DReal &rvort_edge,
	          const Array2DReal &pvort_edge,
	          const Array2DReal &thick_edge,
	          const Array2DReal &vn_edge,
		  int nedges,
		  int nlayers_vec,
		  const PotentialVortHAdvOnEdge &pv_flux_edge) {
    pv_flux_edge(vn_tend_edge, 0, 0, rvort_edge, pvort_edge, thick_edge, vn_edge);
    //parallelFor(
    //    "compute_vtend3", {nedges, nlayers_vec},
    //    KOKKOS_LAMBDA(int iedge, int kchunk) {
    //      pv_flux_edge(vn_tend_edge, iedge, kchunk, rvort_edge, pvort_edge, thick_edge, vn_edge);
    //    });
}
