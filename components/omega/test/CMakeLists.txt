# Omega Unit Tests

##################
# Test meshes
##################

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(
  QU240Mesh
  URL      https://web.lcrc.anl.gov/public/e3sm/inputdata/ocn/mpas-o/oQU240/ocean.QU.240km.151209.nc
  URL_HASH SHA256=a3758f88ceff3d91e86dba7922f6dd7d5672157b4793ef78214624ab8b2724ae
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/meshes
  DOWNLOAD_NO_EXTRACT TRUE
)

FetchContent_Declare(
  PlanarMesh
  URL      https://gist.github.com/mwarusz/f8caf260398dbe140d2102ec46a41268/raw/e3c29afbadc835797604369114321d93fd69886d/PlanarPeriodic48x48.nc
  URL_HASH SHA256=fbe73f796fcad0fcd0ea2df6f9deb40108755464c5fa3f802719372548833b44
  DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/meshes
  DOWNLOAD_NO_EXTRACT TRUE
)

FetchContent_MakeAvailable(QU240Mesh PlanarMesh)

add_test(NAME SetupQU240Mesh
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/meshes/ocean.QU.240km.151209.nc
    ${CMAKE_CURRENT_BINARY_DIR}/OmegaMesh.nc)
set_tests_properties(SetupQU240Mesh PROPERTIES FIXTURES_SETUP QU240Fixture)

add_test(NAME CleanupQU240Mesh
  COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_BINARY_DIR}/OmegaMesh.nc)
set_tests_properties(CleanupQU240Mesh PROPERTIES FIXTURES_CLEANUP QU240Fixture)

add_test(NAME SetupPlanarMesh
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_BINARY_DIR}/meshes/PlanarPeriodic48x48.nc
    ${CMAKE_CURRENT_BINARY_DIR}/OmegaMesh.nc)
set_tests_properties(SetupPlanarMesh PROPERTIES FIXTURES_SETUP PlanarFixture)
set_tests_properties(SetupPlanarMesh PROPERTIES DEPENDS CleanupQU240Mesh)

add_test(NAME CleanupPlanarMesh
  COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_BINARY_DIR}/OmegaMesh.nc)
set_tests_properties(CleanupPlanarMesh PROPERTIES FIXTURES_CLEANUP PlanarFixture)

##################
# Data type test
##################

set(_TestDataTypesName testDataTypes.exe)

add_executable(${_TestDataTypesName} base/DataTypesTest.cpp)

target_compile_options(
  ${_TestDataTypesName}
  PRIVATE
  "-I${OMEGA_SOURCE_DIR}/src/base"
  "-I${OMEGA_SOURCE_DIR}/src/infra"
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestDataTypesName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestDataTypesName} ${OMEGA_LIB_NAME} yakl parmetis metis)

if(GKlib_FOUND)
  target_link_libraries(${_TestDataTypesName} gklib)
endif()

add_test(
  NAME DATA_TYPES_TEST
  COMMAND ${MPI_EXEC} -n 1 -- ./${_TestDataTypesName}
)


##################
# Machine env test
##################

set(_TestMachEnvName testMachEnv.exe)

add_executable(${_TestMachEnvName} base/MachEnvTest.cpp)

target_compile_options(
  ${_TestMachEnvName}
  PRIVATE
  "-I${OMEGA_SOURCE_DIR}/src/base"
  "-I${OMEGA_SOURCE_DIR}/src/infra"
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestMachEnvName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestMachEnvName} ${OMEGA_LIB_NAME} yakl)

add_test(
  NAME MACHINE_ENV_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestMachEnvName}
)


##################
# Broadcast test
##################

set(_TestBroadcastName testBroadcast.exe)

# Add broadcast test
add_executable(${_TestBroadcastName} base/BroadcastTest.cpp)

target_compile_options(
  ${_TestBroadcastName}
  PRIVATE
  "-I${OMEGA_SOURCE_DIR}/src/base"
  "-I${OMEGA_SOURCE_DIR}/src/infra"
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestBroadcastName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestBroadcastName} ${OMEGA_LIB_NAME} spdlog yakl)

add_test(
  NAME BROADCAST_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestBroadcastName}
)


##################
# Logging test
##################

set(_TestLoggingName testLogging.exe)

add_executable(${_TestLoggingName} infra/LoggingTest.cpp)

target_compile_options(
  ${_TestLoggingName}
  PRIVATE
  "-I${OMEGA_SOURCE_DIR}/src/base"
  "-I${OMEGA_SOURCE_DIR}/src/infra"
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestLoggingName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestLoggingName} ${OMEGA_LIB_NAME} spdlog yakl)

add_test(NAME LOGGING_TEST COMMAND ./${_TestLoggingName})

#############
# Decomp test
#############

set(_TestDecompName testDecomp.exe)

add_executable(${_TestDecompName} base/DecompTest.cpp)

target_include_directories(
  ${_TestDecompName}
  PRIVATE
  ${OMEGA_SOURCE_DIR}/src/base
  ${OMEGA_SOURCE_DIR}/src/infra
  ${Parmetis_INCLUDE_DIRS}
)

target_compile_options(
  ${_TestDecompName}
  PRIVATE
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestDecompName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestDecompName} ${OMEGA_LIB_NAME} spdlog yakl parmetis metis pioc)

if(GKlib_FOUND)
  target_link_libraries(${_TestDecompName} gklib)
endif()

add_test(
  NAME DECOMP_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestDecompName}
)
set_tests_properties(DECOMP_TEST PROPERTIES FIXTURES_REQUIRED QU240Fixture)

##################
# Halo test
##################

set(_TestHaloName testHalo.exe)

# Add halo test
add_executable(${_TestHaloName} base/HaloTest.cpp)

target_include_directories(
  ${_TestHaloName}
  PRIVATE
  ${OMEGA_SOURCE_DIR}/src/base
  ${OMEGA_SOURCE_DIR}/src/infra
  ${Parmetis_INCLUDE_DIRS}
)

target_compile_options(
  ${_TestHaloName}
  PRIVATE
  ${OMEGA_CXX_FLAGS}
)

target_link_libraries(${_TestHaloName} ${OMEGA_LIB_NAME} spdlog yakl parmetis metis pioc)

add_test(
  NAME HALO_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestHaloName}
)
set_tests_properties(HALO_TEST PROPERTIES FIXTURES_REQUIRED QU240Fixture)

################
# HorzMesh test
################

set(_TestHorzMeshName testHorzMesh.exe)

# Add broadcast test
add_executable(${_TestHorzMeshName} ocn/HorzMeshTest.cpp)

target_include_directories(
  ${_TestHorzMeshName}
  PRIVATE
  ${OMEGA_SOURCE_DIR}/src/base
  ${OMEGA_SOURCE_DIR}/src/infra
  ${OMEGA_SOURCE_DIR}/src/ocn
  ${Parmetis_INCLUDE_DIRS}
)

target_compile_options(
  ${_TestHorzMeshName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestHorzMeshName} ${OMEGA_LIB_NAME} spdlog yakl parmetis metis pioc)

add_test(
  NAME HORZMESH_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestHorzMeshName}
)
set_tests_properties(HORZMESH_TEST PROPERTIES FIXTURES_REQUIRED QU240Fixture)

################
# Operators test
################

set(_TestOperatorsName testOperators.exe)

# Add broadcast test
add_executable(${_TestOperatorsName} ocn/OperatorsTest.cpp)

target_include_directories(
  ${_TestOperatorsName}
  PRIVATE
  ${OMEGA_SOURCE_DIR}/src/base
  ${OMEGA_SOURCE_DIR}/src/infra
  ${OMEGA_SOURCE_DIR}/src/ocn
  ${Parmetis_INCLUDE_DIRS}
)

target_compile_options(
  ${_TestOperatorsName}
  PRIVATE
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestOperatorsName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestOperatorsName} ${OMEGA_LIB_NAME} spdlog yakl parmetis metis pioc)

add_test(
  NAME OPERATORS_TEST
  COMMAND ${MPI_EXEC} -n 2 -- ./${_TestOperatorsName}
)
set_tests_properties(OPERATORS_TEST PROPERTIES FIXTURES_REQUIRED PlanarFixture)
yakl_process_target(${_TestOperatorsName})

#############
# IO test
#############

set(_TestIOName testIO.exe)

add_executable(${_TestIOName} base/IOTest.cpp)

target_include_directories(
   ${_TestIOName}
  PRIVATE
  ${OMEGA_SOURCE_DIR}/src/base
  ${OMEGA_SOURCE_DIR}/src/infra
  ${Parmetis_INCLUDE_DIRS}
)

target_compile_options(
   ${_TestIOName}
  PRIVATE
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
   ${_TestIOName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestIOName}     ${OMEGA_LIB_NAME} spdlog yakl parmetis metis pioc)

if(GKlib_FOUND)
   target_link_libraries(${_TestIOName} gklib)
endif()

add_test(
  NAME IO_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestIOName}
)
set_tests_properties(IO_TEST PROPERTIES FIXTURES_REQUIRED QU240Fixture)

##################
# Config test
##################

set(_TestConfigName testConfig.exe)

# Add broadcast test
add_executable(${_TestConfigName} infra/ConfigTest.cpp)

target_include_directories(
  ${_TestConfigName}
  PRIVATE
  ${OMEGA_SOURCE_DIR}/src/base
  ${OMEGA_SOURCE_DIR}/src/infra
)

target_compile_options(
  ${_TestConfigName}
  PRIVATE
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestConfigName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestConfigName} ${OMEGA_LIB_NAME} yaml-cpp)

add_test(
  NAME CONFIG_TEST
  COMMAND ${MPI_EXEC} -n 8 -- ./${_TestConfigName}
)

##################
# Metadata test
##################

set(_TestMetadataName testMetadata.exe)

add_executable(${_TestMetadataName} infra/MetadataTest.cpp)

target_compile_options(
  ${_TestMetadataName}
  PRIVATE
  "-I${OMEGA_SOURCE_DIR}/src/base"
  "-I${OMEGA_SOURCE_DIR}/src/infra"
  ${OMEGA_CXX_FLAGS}
)

target_link_options(
  ${_TestMetadataName}
  PRIVATE
  ${OMEGA_LINK_OPTIONS}
)

target_link_libraries(${_TestMetadataName} ${OMEGA_LIB_NAME} yakl)

add_test(NAME METADATA_TEST COMMAND ./${_TestMetadataName})

##################
# YAKL test
##################

set(_TestYaklName testYakl.exe)

add_executable(
  ${_TestYaklName} ${E3SM_EXTERNALS_ROOT}/YAKL/unit/ParForC/ParForC.cpp
)

include(${E3SM_EXTERNALS_ROOT}/YAKL/yakl_utils.cmake)
yakl_process_target(${_TestYaklName})

# handles cuda cases
if (YAKL_ARCH STREQUAL "CUDA")
  set_target_properties(${_TestYaklName} PROPERTIES LINKER_LANGUAGE CXX)
  if (CMAKE_VERSION VERSION_GREATER "3.18.0")
    set_target_properties(${_TestYaklName} PROPERTIES CUDA_ARCHITECTURES OFF)
  endif()
endif()

add_test(NAME YAKL_TEST COMMAND ./${_TestYaklName})

##################
# test properties
##################

set_tests_properties(
  DATA_TYPES_TEST
  MACHINE_ENV_TEST
  BROADCAST_TEST
  LOGGING_TEST
  CONFIG_TEST
  DECOMP_TEST
  HALO_TEST
  HORZMESH_TEST
  OPERATORS_TEST
  IO_TEST
  METADATA_TEST
  PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL" PASS_REGULAR_EXPRESSION "PASS"
)

set_tests_properties(
  YAKL_TEST
  PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR"
)
