#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(audiogen_runner)

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")

# Use release build type by default if none is specified
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified. Defaulting to Release.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set options for executorch build.
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
option(EXECUTORCH_ENABLE_LOGGING "" ON)

# Disable tests for ExecuTorch
set(BUILD_TESTING OFF)

if(CMAKE_TOOLCHAIN_FILE)
  list(APPEND TOOLCHAIN_CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif()

if(ANDROID_ABI)
  list(APPEND TOOLCHAIN_CMAKE_ARGS -DANDROID_ABI=${ANDROID_ABI})
endif()

message(STATUS "Fetching/configuring ExecuTorch...")
if(NOT ET_SRC_PATH)
  include(FetchContent)

  # ExecuTorch requires the directory name to be 'executorch'

  set(EXECUTORCH_SOURCE_DIR ${CMAKE_SOURCE_DIR}/executorch)
  FetchContent_Declare(executorch
    SOURCE_DIR ${EXECUTORCH_SOURCE_DIR}
    # For Sine support in autoencoder
    GIT_REPOSITORY https://github.com/GregoryComer/executorch.git
    GIT_TAG 68f6f0e74ff82173889446ae894c4685dad77da2
  )
  FetchContent_Populate(executorch)

else()
  set(EXECUTORCH_SOURCE_DIR ${ET_SRC_PATH})
endif()

message(STATUS "---- Running ./install_executorch.sh")
execute_process(
  COMMAND ./install_executorch.sh
  WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}
  RESULT_VARIABLE result
)

if(NOT result EQUAL 0)
  message(FATAL_ERROR "Failed to run ./install_executorch.sh")
endif()

# Add Executorch source directory
add_subdirectory(${EXECUTORCH_SOURCE_DIR} ${CMAKE_BINARY_DIR}/executorch)

# Add tokenizers from ExecuTorch extensions
add_subdirectory(${EXECUTORCH_SOURCE_DIR}/extension/llm/tokenizers ${CMAKE_BINARY_DIR}/tokenizers)

add_executable(sao_runner main.cpp)

target_link_libraries(
  sao_runner PUBLIC executorch optimized_native_cpu_ops_lib
                           xnnpack_backend extension_module_static extension_tensor tokenizers
)
