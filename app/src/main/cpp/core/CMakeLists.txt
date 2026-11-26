cmake_minimum_required(VERSION 3.22.1)
project(momotranscript_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mmt_core STATIC
    windowing.cpp
    postprocess.cpp
    avifs_pack.cpp
    cv_lines.cpp
    tensor_prep.cpp
    momoscript.cpp
    visual_match.cpp
)
target_include_directories(mmt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT ANDROID)
    enable_testing()
    add_executable(test_mmt_core tests/test_core.cpp)
    target_link_libraries(test_mmt_core PRIVATE mmt_core)
    add_test(NAME test_mmt_core COMMAND test_mmt_core)
endif()
