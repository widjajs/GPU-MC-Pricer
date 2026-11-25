cmake_minimum_required(VERSION 3.24)
project(gpu_mc_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA is optional so the CPU side builds and tests on a Mac.
option(PRICER_ENABLE_CUDA "Build the CUDA backend if a CUDA compiler is found" ON)
include(CheckLanguage)
check_language(CUDA)

set(PRICER_SOURCES src/analytic.cpp src/cpu_pricer.cpp)

if(PRICER_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  find_package(CUDAToolkit REQUIRED)
  list(APPEND PRICER_SOURCES src/gpu_pricer.cu)
  message(STATUS "CUDA backend: ON (${CMAKE_CUDA_COMPILER})")
else()
  list(APPEND PRICER_SOURCES src/gpu_stub.cpp)
  message(STATUS "CUDA backend: OFF (no CUDA compiler; GPU calls will throw)")
endif()

add_library(pricer_core ${PRICER_SOURCES})
target_include_directories(pricer_core PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(pricer_core PUBLIC Threads::Threads)
if(CMAKE_CUDA_COMPILER AND PRICER_ENABLE_CUDA)
  target_link_libraries(pricer_core PUBLIC CUDA::cudart CUDA::curand)
  # -lineinfo lets Nsight Compute map hotspots back to source lines.
  target_compile_options(pricer_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

add_executable(mcprice src/main.cpp)
target_link_libraries(mcprice PRIVATE pricer_core)

enable_testing()
add_executable(pricer_tests tests/test_pricer.cpp)
target_link_libraries(pricer_tests PRIVATE pricer_core)
add_test(NAME pricer_tests COMMAND pricer_tests)
