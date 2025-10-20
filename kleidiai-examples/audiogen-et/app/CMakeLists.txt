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
option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
option(EXECUTORCH_ENABLE_LOGGING "" ON)

option(ENABLE_F16_IGEMM_SME2_EXPERIMENTAL "" OFF)
set(SME2_GIT_BRANCH "pull/8687/head:f16_igemm")
set(SME2_GIT_HASH "4ee40b90b9")

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
    GIT_REPOSITORY https://github.com/pytorch/executorch.git
    GIT_TAG v1.0.0
  )
  FetchContent_Populate(executorch)

  if(ENABLE_F16_IGEMM_SME2_EXPERIMENTAL)
    message(STATUS "Enabling SME2 IGEMM experimental optimizations in ExecuTorch")
    set(XNNPACK_ENABLE_ARM_SME2 ON CACHE BOOL "" FORCE)

    # Find Git executable
    find_program(GIT_EXE git REQUIRED)

    # Fetch the specific branch with SME2 IGEMM optimizations
    execute_process(
      COMMAND ${GIT_EXE} fetch origin ${SME2_GIT_BRANCH}
      WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}/backends/xnnpack/third-party/XNNPACK/
      RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Failed to fetch SME2 experimental optimizations in XNNPACK")
    endif()

    # Checkout SME2 IGEMM commit
    execute_process(
      COMMAND ${GIT_EXE} checkout ${SME2_GIT_HASH}
      WORKING_DIRECTORY ${EXECUTORCH_SOURCE_DIR}/backends/xnnpack/third-party/XNNPACK/
      RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Failed to checkout SME2 experimental optimizations in XNNPACK")
    endif()

  endif()

else()
  set(EXECUTORCH_SOURCE_DIR ${ET_SRC_PATH})
endif()

# Add Executorch source directory
add_subdirectory(${EXECUTORCH_SOURCE_DIR} ${CMAKE_BINARY_DIR}/executorch)

# Add tokenizers from ExecuTorch extensions
add_subdirectory(${EXECUTORCH_SOURCE_DIR}/extension/llm/tokenizers ${CMAKE_BINARY_DIR}/tokenizers)

add_executable(audiogen main.cpp)

target_link_libraries(
  audiogen PUBLIC executorch optimized_native_cpu_ops_lib
                           xnnpack_backend extension_module_static extension_tensor tokenizers
)
