cmake_minimum_required(VERSION 3.20)
project(yolov5_tensorrt_threadpool LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

option(WITH_NVCODEC "build USE_NVCODEC" ON)
if(WITH_NVCODEC)
    add_compile_options(-DUSE_NVCODEC)
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(OpenCV_DIR /home/tfy/document/HybrIK/cpp/opencv-4.12.0/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(TENSORRT_INCLUDE_DIRS /home/tfy/docker_share/TensorRT-10.6.0.26/include)
set(TENSORRT_LIBRARY_DIRS /home/tfy/docker_share/TensorRT-10.6.0.26/targets/x86_64-linux-gnu/lib)
include_directories(${TENSORRT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${TENSORRT_LIBRARY_DIRS})

set(FFMPEG_PREFIX "/home/tfy/document/5.yolov5_silu_relu/FFmpeg-n6.1.4/install" CACHE PATH "FFmpeg install prefix")
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h HINTS ${FFMPEG_PREFIX}/include REQUIRED)
find_library(AVFORMAT_LIBRARY avformat HINTS ${FFMPEG_PREFIX}/lib REQUIRED)

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h HINTS ${FFMPEG_PREFIX}/include REQUIRED)
find_library(AVCODEC_LIBRARY avcodec HINTS ${FFMPEG_PREFIX}/lib REQUIRED)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h HINTS ${FFMPEG_PREFIX}/include REQUIRED)
find_library(AVUTIL_LIBRARY avutil HINTS ${FFMPEG_PREFIX}/lib REQUIRED)

find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h HINTS ${FFMPEG_PREFIX}/include REQUIRED)
find_library(SWSCALE_LIBRARY swscale HINTS ${FFMPEG_PREFIX}/lib REQUIRED)

find_library(NPP_LIB nppig HINTS /usr/local/cuda/lib64 REQUIRED)

include_directories(${AVFORMAT_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR})

add_executable(demo test_yolov5.cpp preprocess.cpp preprocess.cu postprocess.cpp yolov5.cpp)
add_executable(thread_pool test_yolov5_thread_pool.cpp preprocess.cpp preprocess.cu postprocess.cpp yolov5.cpp yolov5_thread_pool.cpp)

target_link_libraries(demo PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so)
target_link_libraries(thread_pool PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY_DIRS}/libnvinfer.so)

if(WITH_NVCODEC)
    target_link_libraries(thread_pool PRIVATE
        ${AVFORMAT_LIBRARY}
        ${AVCODEC_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${SWSCALE_LIBRARY}
        /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnppicc.so
        /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnppc.so
        /usr/local/cuda-12.4/targets/x86_64-linux/lib/libnppig.so
    )
endif()