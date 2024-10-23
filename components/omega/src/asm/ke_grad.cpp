#include "TendencyTerms.h"
#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"
#include "OceanState.h"

using namespace OMEGA;

void ke_grad_func(const Array2DReal &vn_tend_edge,
	          const Array2DReal &ke_cell,
		  int nedges,
		  int nlayers_vec,
                  const KEGradOnEdge &ke_grad_edge) {
    parallelFor(
        "compute_vtend3", {nedges, nlayers_vec},
        KOKKOS_LAMBDA(int iedge, int kchunk) {
          ke_grad_edge(vn_tend_edge, iedge, kchunk, ke_cell);
        });
}
