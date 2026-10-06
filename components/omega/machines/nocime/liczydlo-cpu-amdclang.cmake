set(OMEGA_SKIP_CIME TRUE CACHE BOOL "")

set(OMEGA_C_COMPILER "$ENV{HOME}/installs/openmpi-amdclang/bin/mpicc" CACHE STRING "")
set(OMEGA_CXX_COMPILER "$ENV{HOME}/installs/openmpi-amdclang/bin/mpicxx" CACHE STRING "")
set(OMEGA_Fortran_COMPILER "$ENV{HOME}/installs/openmpi-amdclang/bin/mpifort" CACHE STRING "")

set(OMEGA_ARCH SERIAL CACHE STRING "")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")

set(OMEGA_MPI_EXEC "$ENV{HOME}/installs/openmpi-amdclang/bin/mpirun" CACHE STRING "")

set(OMEGA_GKLIB_ROOT "/usr/" CACHE STRING "")
set(OMEGA_METIS_ROOT "/usr/" CACHE STRING "")
set(OMEGA_PARMETIS_ROOT "/usr/" CACHE STRING "")

set(OMEGA_BUILD_TEST ON CACHE BOOL "")
