cmake_minimum_required(VERSION 3.0)
project(FastestDet)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})




include_directories(${PROJECT_SOURCE_DIR}/ncnn/include/ncnn)


include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(fastdet_lib SHARED ${PROJECT_SOURCE_DIR}/src/fastdet.cpp)
target_link_libraries(fastdet_lib ${OpenCV_LIBS} ncnn)

add_library(infer_lib SHARED ${PROJECT_SOURCE_DIR}/src/infer.cpp)
target_link_libraries(infer_lib pthread)


set(ncnn_DIR ${PROJECT_SOURCE_DIR}/ncnn/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

add_executable(fastdet_test fastdet_test.cpp)
target_link_libraries(fastdet_test fastdet_lib)

# add_executable(FastestDet FastestDet.cpp)
# target_link_libraries(FastestDet fastdet_lib)


add_executable(multi_thead_infer multi_thead_infer.cpp)
target_link_libraries(multi_thead_infer infer_lib fastdet_lib)