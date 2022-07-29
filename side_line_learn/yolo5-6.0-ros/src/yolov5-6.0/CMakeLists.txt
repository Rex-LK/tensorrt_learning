cmake_minimum_required(VERSION 3.20)

project(yolov5_trt)
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
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/rex/TensorRT-8.2.0.6/include/)
link_directories(/home/rex/TensorRT-8.2.0.6/targets/x86_64-linux-gnu/lib)


cuda_add_library(myplugins SHARED src/yololayer.cu src/basic_transform.cu)
target_link_libraries(myplugins nvinfer cudart)



cuda_add_executable(yolov5 src/calibrator.cpp src/yolov5.cpp src/preprocess.cu)
target_link_libraries(yolov5 nvinfer cudart  myplugins  ${OpenCV_LIBS})

cuda_add_executable(detect  src/detect.cpp src/preprocess.cu)
target_link_libraries(detect nvinfer cudart myplugins ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


