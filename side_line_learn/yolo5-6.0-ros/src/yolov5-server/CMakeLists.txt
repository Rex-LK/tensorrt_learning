cmake_minimum_required(VERSION 3.0.2)
project(yolov5_server_nodelet)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  base
  nodelet
  roscpp
  rospy
  std_msgs
)


find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})


include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/home/rex/TensorRT-8.2.0.6/include)
link_directories(/home/rex/TensorRT-8.2.0.6/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


include_directories(${CUDA_INCLUDE_DIRS} ${TRT_INCLUDE_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
include_directories(${OpenCV_INCLUDE_DIRS})



#设置yolov5路径
set(yolov5_dir /home/rex/Desktop/tensorrt_learning/side_line_learn/yolo5-6.0-ros/src/yolov5-6.0)

include_directories(${yolov5_dir}/include)

cuda_add_library(myplugins SHARED ${yolov5_dir}/src/yololayer.cu ${yolov5_dir}/src/basic_transform.cu)
target_link_libraries(myplugins nvinfer cudart ${CUDA_npp_LIBRARY})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES yolov5_server_nodelet
  CATKIN_DEPENDS  base nodelet roscpp rospy std_msgs message_runtime
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

cuda_add_library(yolov5_server_nodelet SHARED src/yolov5_server_nodelet.cpp)
target_link_libraries(yolov5_server_nodelet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        -lpthread cuda
        nvinfer
        cudart
        myplugins 
        ${OpenCV_LIBS}
        )
