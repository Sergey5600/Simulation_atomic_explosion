cmake_minimum_required(VERSION 3.24)
project(NukeSim VERSION 0.1.0 LANGUAGES CXX)

option(SIM_ENABLE_CUDA "Build the CUDA backend (falls back to CPU-only if no CUDA compiler is found)" ON)
option(SIM_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
set(SIM_HAS_CUDA OFF)
if(SIM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 30xx (Ampere)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(SIM_HAS_CUDA ON)
  else()
    message(WARNING "No CUDA compiler found: building the CPU-only backend")
  endif()
endif()
message(STATUS "NukeSim: CUDA backend = ${SIM_HAS_CUDA}")

# ---------------------------------------------------------------------------
# Dependencies (vcpkg on Windows, system packages on Linux)
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(OpenVDB CONFIG QUIET)
if(NOT TARGET OpenVDB::openvdb)
  # OpenVDB usually ships a Find module instead of a config package.
  list(APPEND CMAKE_MODULE_PATH
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/openvdb"
    "/usr/lib/x86_64-linux-gnu/cmake/OpenVDB"
    "/usr/local/lib/cmake/OpenVDB"
    "/usr/lib/cmake/OpenVDB")
  find_package(OpenVDB MODULE REQUIRED)
endif()

find_path(TOMLPP_INCLUDE_DIR toml++/toml.hpp REQUIRED)
find_path(STB_INCLUDE_DIR stb_image_write.h PATH_SUFFIXES stb REQUIRED)
if(SIM_BUILD_TESTS)
  find_path(DOCTEST_INCLUDE_DIR doctest/doctest.h REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
# Kernel sources call sim::parallel_for. They are plain C++ in CPU-only builds
# and are compiled by nvcc in CUDA builds, so every kernel exists exactly once.
set(SIM_KERNEL_SOURCES
  src/core/field_ops.cpp
  src/lowmach/advection.cpp
  src/lowmach/pressure.cpp
  src/lowmach/solver.cpp
  src/scenes/test_fireball.cpp
  src/scenes/fireball_rise.cpp
)

set(SIM_HOST_SOURCES
  src/core/backend.cpp
  src/core/memory.cpp
  src/core/log.cpp
  src/core/schedule.cpp
  src/io/config.cpp
  src/io/vdb_writer.cpp
  src/io/manifest.cpp
  src/preview/preview.cpp
  src/physics/atmosphere.cpp
  src/scenes/scene.cpp
  src/app/run.cpp
)

add_library(simcore STATIC ${SIM_KERNEL_SOURCES} ${SIM_HOST_SOURCES})
target_include_directories(simcore PUBLIC src)
target_include_directories(simcore SYSTEM PRIVATE ${TOMLPP_INCLUDE_DIR} ${STB_INCLUDE_DIR})
target_link_libraries(simcore PUBLIC OpenMP::OpenMP_CXX OpenVDB::openvdb)
target_compile_features(simcore PUBLIC cxx_std_20)

if(SIM_HAS_CUDA)
  set_source_files_properties(${SIM_KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(simcore PUBLIC SIM_HAS_CUDA=1)
  target_link_libraries(simcore PUBLIC CUDA::cudart)
  # Host compiler needs OpenMP for the CPU path of kernel sources compiled by nvcc.
  string(REPLACE " " "," _sim_omp_flags "${OpenMP_CXX_FLAGS}")
  target_compile_options(simcore PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr -Xcompiler=${_sim_omp_flags}>)
  if(MSVC)
    target_compile_options(simcore PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus,/utf-8>)
  endif()
else()
  target_compile_definitions(simcore PUBLIC SIM_HAS_CUDA=0)
endif()

if(MSVC)
  target_compile_definitions(simcore PUBLIC NOMINMAX _USE_MATH_DEFINES)
  target_compile_options(simcore PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:__cplusplus /utf-8 /bigobj>)
else()
  target_compile_options(simcore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

add_executable(nukesim src/app/main.cpp)
target_link_libraries(nukesim PRIVATE simcore)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(SIM_BUILD_TESTS)
  enable_testing()
  add_executable(simtests
    tests/test_main.cpp
    tests/test_backend.cpp
    tests/test_schedule.cpp
    tests/test_vdb.cpp
    tests/test_config.cpp
    tests/test_lowmach.cpp
  )
  target_include_directories(simtests SYSTEM PRIVATE ${DOCTEST_INCLUDE_DIR})
  target_link_libraries(simtests PRIVATE simcore)
  target_compile_definitions(simtests PRIVATE SIM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME simtests COMMAND simtests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
