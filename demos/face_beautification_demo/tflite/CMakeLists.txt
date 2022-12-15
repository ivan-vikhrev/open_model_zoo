# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (TFLITE_BUILD_DIR)
    find_library(TFLITE_LIB libtensorflow-lite.so ${TFLITE_BUILD_DIR})
    set(FLATBUFFERS_INCLUDE_DIR ${TFLITE_BUILD_DIR}/flatbuffers/include)
else()
    message(FATAL_ERROR "Could not find tf-lite build directory")
endif()

if (TFLITE_SOURCE_DIR)
    set(TFLITE_INCLUDE_DIR "${TFLITE_SOURCE_DIR};${FLATBUFFERS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Could not find tf-lite source directory")
endif()

set(TFLITE_LIBRARY "tensorflow-lite")
add_library(${TFLITE_LIBRARY} SHARED IMPORTED)
set_target_properties(${TFLITE_LIBRARY} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TFLITE_INCLUDE_DIR}"
    IMPORTED_LOCATION ${TFLITE_LIB}
    IMPORTED_IMPLIB ${TFLITE_LIB})

add_demo(NAME face_beautification_tflite_demo
    SOURCES ${SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES monitors ${TFLITE_LIBRARY})

target_compile_features(face_beautification_tflite_demo PRIVATE cxx_std_17)
