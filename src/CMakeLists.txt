add_library(mim
    imgalign.cpp
    imgcmp.cpp
    imgproc.cpp
)

set_target_properties(mim PROPERTIES VERSION 0.0.1)
target_include_directories(mim PRIVATE include/)