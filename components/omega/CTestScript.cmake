cmake_minimum_required(VERSION 3.20)

if (DEFINED SRCDIR)
  set(CTEST_SOURCE_DIRECTORY "${SRCDIR}")
else ()
  set(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
endif ()

if (DEFINED BINDIR)
  set(CTEST_BINARY_DIRECTORY "${BINDIR}")
else ()
  set(CTEST_BINARY_DIRECTORY "$ENV{HOME}/omega-dashboard")
endif ()

if (NOT MACHINE)
  message(FATAL_ERROR "Error: MACHINE is not defined.")
endif ()

if (NOT COMPILER)
  message(FATAL_ERROR "Error: COMPILER is not defined.")
endif ()

if (NOT PARMETIS)
  message(FATAL_ERROR "Error: PARMETIS is not defined.")
endif ()

if (NOT ARCH)
  message(FATAL_ERROR "Error: ARCH is not defined.")
endif ()

if (NOT OCEANMESH)
  message(FATAL_ERROR "Error: OCEANMESH is not defined.")
endif ()

if (NOT SPHEREMESH)
  message(FATAL_ERROR "Error: SPHEREMESH is not defined.")
endif ()

if (NOT PLANARMESH)
  message(FATAL_ERROR "Error: PLANARMESH is not defined.")
endif ()

execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set (CTEST_SITE "${MACHINE}")
set (CTEST_BUILD_GROUP "Omega Unit-test")
set (CTEST_BUILD_NAME "unitest-${GIT_BRANCH}-${COMPILER}")

set (CTEST_UPDATE_COMMAND "git")
set (CTEST_DROP_SITE_CDASH TRUE)

file(REMOVE_RECURSE ${CTEST_BINARY_DIRECTORY})

ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}/test")
execute_process(COMMAND ln -sf ${OCEANMESH}
  ${CTEST_BINARY_DIRECTORY}/test/OmegaMesh.nc)
execute_process(COMMAND ln -sf ${SPHEREMESH}
  ${CTEST_BINARY_DIRECTORY}/test/OmegaSphereMesh.nc)
execute_process(COMMAND ln -sf ${PLANARMESH}
  ${CTEST_BINARY_DIRECTORY}/test/OmegaPlanarMesh.nc)

set(CTEST_NIGHTLY_START_TIME "06:00:00 UTC")

ctest_start(Nightly GROUP Unit-test)

ctest_update(
  RETURN_VALUE UpdateRetval
  CAPTURE_CMAKE_ERROR UpdateResult
)

set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CONFIG_OPTIONS
  "-DOMEGA_CIME_MACHINE=${MACHINE};"
  "-DOMEGA_CIME_COMPILER=${COMPILER};"
  "-DOMEGA_ARCH=${ARCH};"
  "-DOMEGA_BUILD_TEST=ON;"
  "-DOMEGA_PARMETIS_ROOT=${PARMETIS}"
)
ctest_configure(
  RETURN_VALUE ConfigRetval
  CAPTURE_CMAKE_ERROR ConfigResult
  OPTIONS "${CONFIG_OPTIONS}"
)

set(CTEST_BUILD_COMMAND "./omega_build.sh")
set(CTEST_BUILD_CONFIGURATION "Release")
ctest_build(
  RETURN_VALUE BuildRetval
  CAPTURE_CMAKE_ERROR BuildResult
)

ctest_test(
  BUILD "${CTEST_BINARY_DIRECTORY}"
  RETURN_VALUE TestRetval
  CAPTURE_CMAKE_ERROR TestResult
)

#set(CTEST_SUBMIT_URL "https://my.cdash.org/submit.php?project=omega")
set(CTEST_SUBMIT_URL "https://my.cdash.org/submit.php?project=e3sm")
ctest_submit(
  RETURN_VALUE SubmitRetval
  CAPTURE_CMAKE_ERROR SubmitResult
)
