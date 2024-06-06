string(APPEND CMAKE_C_FLAGS " -mcmodel=small")
string(APPEND CMAKE_Fortran_FLAGS " -mcmodel=small")

set(MPICC "mpicc")
set(MPICXX "mpicxx")
set(MPIFC "mpifort")
set(SCC "gcc-14")
set(SCXX "g++-14")
set(SFC "gfortran-14")
