add_library(ke_grad ke_grad.cpp)
target_link_libraries(ke_grad OmegaLibFlags)
target_compile_options(ke_grad PRIVATE -save-temps)

add_library(pv_flux pv_flux.cpp)
target_link_libraries(pv_flux OmegaLibFlags)
target_compile_options(pv_flux PRIVATE -save-temps)

add_library(thick_div thick_div.cpp)
target_link_libraries(thick_div OmegaLibFlags)
target_compile_options(thick_div PRIVATE -save-temps)
