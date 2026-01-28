cmake_minimum_required(VERSION 3.20)
project(trt_inference LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(OpenCV_DIR /home/tfy/document/HybrIK/cpp/opencv-4.12.0/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(TENSORRT_INCLUDE_DIRS /home/tfy/docker_share/TensorRT-10.6.0.26/include)
set(TENSORRT_LIBRARY_DIRS /home/tfy/docker_share/TensorRT-10.6.0.26/targets/x86_64-linux-gnu/lib)
include_directories(${TENSORRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${TENSORRT_LIBRARY_DIRS})

add_executable(demo test_yolov5.cpp preprocess.cpp postprocess.cpp yolov5.cpp)
add_executable(thread_pool test_yolov5_thread_pool.cpp preprocess.cpp postprocess.cpp yolov5.cpp yolov5_thread_pool.cpp)
target_link_libraries(demo PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so)
target_link_libraries(thread_pool PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so)