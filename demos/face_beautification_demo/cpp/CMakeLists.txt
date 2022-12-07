# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)

add_demo(NAME face_beautification_demo
    SOURCES ${MAIN_SRC}
    HEADERS ${MAIN_HEADERS}
    DEPENDENCIES monitors)
