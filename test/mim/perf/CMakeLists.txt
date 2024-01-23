include_directories(../../src/include/)

add_executable(perf_warp_affine perf_warp_affine.cpp)
target_link_libraries(perf_warp_affine mim)