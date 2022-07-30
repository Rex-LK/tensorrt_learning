cmake_minimum_required(VERSION 3.0.2)
project(client_nodelet)

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

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES client_nodelet
  CATKIN_DEPENDS base nodelet roscpp rospy std_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


add_library(client_nodelet SHARED src/client_nodelet.cpp)
target_link_libraries(client_nodelet
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        -lpthread
        ${OpenCV_LIBS}
)
