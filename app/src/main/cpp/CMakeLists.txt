cmake_minimum_required(VERSION 3.22.1)
project(momotranscript_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(core)

set(MNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mnn/include")
set(MNN_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

add_library(mnn_prebuilt SHARED IMPORTED)
set_target_properties(mnn_prebuilt PROPERTIES
    IMPORTED_LOCATION "${MNN_LIB_DIR}/libMNN.so"
)

add_library(mnn_express_prebuilt SHARED IMPORTED)
set_target_properties(mnn_express_prebuilt PROPERTIES
    IMPORTED_LOCATION "${MNN_LIB_DIR}/libMNN_Express.so"
)

add_library(momotranscript_core SHARED
    jni_bridge.cpp
    mnn_engine.cpp
)

target_include_directories(momotranscript_core
    PRIVATE
        "${MNN_INCLUDE_DIR}"
)

target_link_libraries(momotranscript_core
    PRIVATE
        mmt_core
        mnn_prebuilt
        mnn_express_prebuilt
        log
)

# 16 KB page alignment for Android 15/16+ devices
target_link_options(momotranscript_core PRIVATE "-Wl,-z,max-page-size=16384")
