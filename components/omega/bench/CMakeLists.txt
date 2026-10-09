add_executable(benchHalo HaloBench.cpp)
target_link_libraries(
  benchHalo
  PRIVATE
  OmegaLibFlags
  ${OMEGA_LIB_NAME}
)
