cmake_minimum_required(VERSION 3.20)
project(base)

add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  nodelet
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
  message_generation

)

find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

add_service_files(
  FILES
  RosImage.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES base
 CATKIN_DEPENDS nodelet roscpp rospy sensor_msgs cv_bridge std_msgs message_runtime

)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

cuda_add_library(base SHARED src/base.cpp)

add_dependencies(base ${PROJECT_NAME}_gencpp)

target_link_libraries(base
        ${catkin_LIBRARIES}
        -lpthread
        )

