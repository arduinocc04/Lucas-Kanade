if(DO_NOT_USE_OPENCV)
    add_compile_definitions( DO_NOT_USE_OPENCV )
endif()
add_library(mim_test
    test_tools.cpp
)

set_target_properties(mim_test PROPERTIES VERSION 0.0.1)
target_link_libraries(mim_test mim)

if(NOT DO_NOT_USE_OPENCV)
    target_link_libraries(mim_test ${OpenCV_LIBS})
endif()
