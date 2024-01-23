cmake_minimum_required(VERSION 3.16)
project(MIM LANGUAGES CXX)

if(NOT DO_NOT_USE_CUDA)
    enable_language(CUDA)
endif()

add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_ZERO)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-ftrapv)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

### Eigen with source
include_directories(external/eigen-3.4.0)

include_directories(src/mim)

add_subdirectory(src)

if(NOT DO_NOT_TEST)
    add_subdirectory(test)

    include(CTest)
    if(WIN32)
        add_test(imgcmp-test test/mim/${CMAKE_BUILD_TYPE}/test_imgcmp)
        add_test(warp-affine-test test/mim/${CMAKE_BUILD_TYPE}/test_warp_affine)
        add_test(lucas-kanade-test test/mim/${CMAKE_BUILD_TYPE}/test_lucas_kanade)
    else()
        add_test(imgcmp-test test/mim/test_imgcmp)
        add_test(warp-affine-test test/mim/test_warp_affine)
        add_test(lucas-kanade-test test/mim/test_lucas_kanade)
    endif()
endif()

