#include <cuda_runtime.h>


#define BLOCK_SIZE  1024

/**
 * @description: 							cuda post-processing decoding 
 * @param {float* } 						input predict array
 * @param {int} num_bboxes					number of bboxes
 * @param {int} num_classes					number of classes
 * @param {float} score_threshold			score threshold
 * @param {float* } inverse_affine_matrix	inverse of affine_matrix
 * @param {float* } parray					output array
 * @param {int} max_objects					max number of objects
 * @param {int} num_box_element				number of box element
 * @param {cudaStream_t} stream				cuda stream
 * @return {*}
 */
void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float score_threshold, float* inverse_affine_matrix, float* parray, int max_objects, int num_box_element, cudaStream_t stream);

/**
 * @description: 					cuda NMS kernel
 * @param {float* } 				input predict array
 * @param {float} nms_threshold		NMS threshold
 * @param {int} max_objects			max number of objects
 * @param {int} num_box_element		number of box element
 * @param {cudaStream_t} stream		cuda stream
 * @return {*}
 */
void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, int num_box_element, cudaStream_t stream);
