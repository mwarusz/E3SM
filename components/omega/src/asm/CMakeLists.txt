add_library(asm
  ke_grad.cpp
  thick_adv.cpp
  tracer_adv_cell.cpp
  tracer_adv_edge.cpp
  tracer_diff.cpp
  teos10.cpp
  tracer_update.cpp
)

target_link_libraries(
  asm
  PRIVATE
  ${OMEGA_LIB_NAME}
  OmegaLibFlags
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(asm PRIVATE
    -fopt-info-vec-all=vec-opt.txt
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  target_compile_options(asm PRIVATE
    -qopt-report=3
    -qopt-report-phase=vec
    -qopt-report-file=vec-opt.txt
  )
endif()
