#include <cstdint>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


struct AffineMatrix
{
    float value[6];
};


void preprocess_cuda(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, float* affine_matrix, float* affine_matrix_inverse);
