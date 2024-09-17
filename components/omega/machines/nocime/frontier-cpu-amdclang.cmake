set(OMEGA_SKIP_CIME TRUE CACHE BOOL "")
set(OMEGA_BUILD_TYPE Release CACHE STRING "")

set(OMEGA_C_COMPILER "mpicc" CACHE STRING "")
set(OMEGA_CXX_COMPILER "mpicxx" CACHE STRING "")
set(OMEGA_Fortran_COMPILER "mpifort" CACHE STRING "")

set(OMEGA_CXX_FLAGS "-g" CACHE STRING "")
set(OMEGA_ARCH SERIAL CACHE STRING "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")

set(OMEGA_PARMETIS_ROOT /ccs/proj/cli115/software/polaris/frontier/spack/dev_polaris_0_5_0_gnu_mpich/var/spack/environments/dev_polaris_0_5_0_gnu_mpich/.spack-env/view CACHE STRING "")

set(OMEGA_VECTOR_LENGTH 4 CACHE STRING "")

set(OMEGA_MPI_EXEC srun CACHE STRING "")
set(OMEGA_BUILD_TEST ON CACHE BOOL "")

set(NetCDF_PATH $ENV{NETCDF_DIR} CACHE STRING "")
set(PnetCDF_PATH $ENV{NETCDF_DIR} CACHE STRING "")
