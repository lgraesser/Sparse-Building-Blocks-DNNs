#ifndef _CONV_H__
#define _CONV_H__

/* Header file defining functions for carrying out
 * dense and sparse 2D convolutions
 *
 * Matrices are assumed to be stored in xxx major order
 */

#include <cudnn.h>
#include "sparse_conversion.h"
#include "matrix_io.h"

struct Kernel {
  float *vals;
  int dims[4];
  char is_column_first;
  float *vals_device;
  cudnnFilterDescriptor_t kernel_descriptor;
  bool is_on_device;
};

void convolve2DDense(struct Matrix * mat,
                struct Kernel * kernel,
                struct Matrix * result,
                cudnnHandle_t cudnn);

void convolve2DDenseProjectImp(struct Matrix * mat,
                struct Kernel * kernel,
                struct Matrix * result);

void destroyKernel(struct Kernel * kernel, struct Matrix * k_mat);

__global__ void convolve2DKernel(float * matrix,
                        float * kernel,
                        float * result,
                        int mat_h,
                        int mat_w,
                        int k_h,
                        int k_w);

#endif
