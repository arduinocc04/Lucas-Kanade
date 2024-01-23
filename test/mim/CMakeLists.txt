if(NOT DO_NOT_USE_OPENCV)
    # find openCV installed directory
    if(WIN32)
        find_path(OpenCV_DIR OpenCVConfig.cmake
        PATHS "C:/Users/jhkong/Documents/opencv/build"
              "D:/opencv/build")
        # clang에서는 실패
        # 실행시 opencv_world480.dll 경로 path 설정해주어야!
        # 4.8 에서는 build/x64/vc16/bin/ 에 위치함.
    endif()

    find_package(OpenCV REQUIRED)
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    file(COPY opencv_logo.jpg DESTINATION ${CMAKE_BINARY_DIR})

    file(GLOB png_imgs "*.png")
    foreach(png ${png_imgs})
        file(COPY ${png} DESTINATION ${CMAKE_BINARY_DIR})
    endforeach()

    file(GLOB png_imgs "*.PNG")
    foreach(png ${png_imgs})
        file(COPY ${png} DESTINATION ${CMAKE_BINARY_DIR})
    endforeach()
endif()

add_subdirectory(tools)

include_directories(tools)
include_directories(../../src/include/)

add_executable(test_imgcmp test_imgcmp.cpp)
target_link_libraries(test_imgcmp mim_test)

add_executable(test_warp_affine test_warp_affine.cpp)
target_link_libraries(test_warp_affine mim_test)

add_executable(test_lucas_kanade test_lucas_kanade.cpp)
target_link_libraries(test_lucas_kanade mim_test)

add_executable(test_pyr_down test_pyr_down.cpp)
target_link_libraries(test_pyr_down mim_test)

add_subdirectory(perf)