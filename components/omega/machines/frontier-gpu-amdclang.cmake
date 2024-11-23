set(OMEGA_SKIP_CIME TRUE CACHE BOOL "")
set(OMEGA_C_COMPILER "mpicc" CACHE STRING "")
set(OMEGA_CXX_COMPILER "mpicxx" CACHE STRING "")
set(OMEGA_Fortran_COMPILER "mpifort" CACHE STRING "")

set(OMEGA_CXX_FLAGS "-g" CACHE STRING "")
set(MPILIB_NAME mpich CACHE STRING "")
set(OMEGA_ARCH HIP CACHE STRING "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")
set(Kokkos_ARCH_GFX90A ON CACHE BOOL "")

set(OMEGA_PARMETIS_ROOT /ccs/proj/cli115/software/polaris/frontier/spack/dev_polaris_0_4_0_gnu_mpich/var/spack/environments/dev_polaris_0_4_0_gnu_mpich/.spack-env/view CACHE STRING "")

set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "")
set(OMEGA_MPI_ON_DEVICE ON CACHE BOOL "")
set(OMEGA_LINK_OPTIONS -L$ENV{MPICH_DIR}/lib -lmpi $ENV{CRAY_XPMEM_POST_LINK_OPTS} -lxpmem $ENV{PE_MPICH_GTL_DIR_amd_gfx90a} $ENV{PE_MPICH_GTL_LIBS_amd_gfx90a} CACHE STRING "")

set(OMEGA_MPI_EXEC srun CACHE STRING "")
set(OMEGA_BUILD_TEST ON CACHE BOOL "")

set(NetCDF_PATH $ENV{NETCDF_DIR} CACHE STRING "")
set(PnetCDF_PATH $ENV{NETCDF_DIR} CACHE STRING "")

set(OMEGA_USE_CALIPER ON CACHE BOOL "")
set(caliper_DIR $ENV{HOME}/installs/caliper-amdclang/share/cmake/caliper CACHE STRING "")
