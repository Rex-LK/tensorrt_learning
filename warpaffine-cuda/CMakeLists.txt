cmake_minimum_required(VERSION 2.6)

project(warpaffine-cuda-test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/rex/TensorRT-8.2.0.6/include/)
link_directories(/home/rex/TensorRT-8.2.0.6/targets/x86_64-linux-gnu/lib)

cuda_add_library(warpaffine SHARED src/warpaffine.cu)
target_link_libraries(warpaffine)

cuda_add_executable(test src/test.cpp)
target_link_libraries(test  warpaffine ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


