#include "AuxiliaryState.h"
#include "Config.h"
#include "DataTypes.h"
#include "HorzMesh.h"
#include "OceanState.h"
#include "TendencyTerms.h"

using namespace OMEGA;

void thick_div_func(const Array2DReal &thick_tend_cell,
	         	const Array2DReal &thick_flux_edge,
	         	const Array2DReal &normal_vel_edge,
                  int ncells, int nlayers_vec,
                   const ThicknessFluxDivOnCell &thick_flux_div) {
   parallelFor(
       "compute_vtend3", {ncells, nlayers_vec},
       KOKKOS_LAMBDA(int icell, int kchunk) {

             thick_flux_div(thick_tend_cell, icell, kchunk,
                                 thick_flux_edge, normal_vel_edge);
       });
}
