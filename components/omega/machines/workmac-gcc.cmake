set(OMEGA_SKIP_CIME TRUE CACHE BOOL "")
set(OMEGA_C_COMPILER "mpicc" CACHE STRING "")
set(OMEGA_CXX_COMPILER "mpicxx" CACHE STRING "")
set(OMEGA_Fortran_COMPILER "mpifort" CACHE STRING "")

set(OMEGA_ARCH SERIAL CACHE STRING "")

set(OMEGA_GKLIB_ROOT "/Users/mwarusz/omega/gklib" CACHE STRING "")
set(OMEGA_METIS_ROOT "/Users/mwarusz/omega/metis" CACHE STRING "")
set(OMEGA_PARMETIS_ROOT "/Users/mwarusz/omega/parmetis" CACHE STRING "")

set(OMEGA_CXX_FLAGS -march=native CACHE STRING "")

set(OMEGA_MPI_EXEC mpirun CACHE STRING "")
set(OMEGA_BUILD_TEST ON CACHE BOOL "")
