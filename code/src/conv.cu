/* Functions for carrying out
 * dense and sparse 2D convolutions
 *
 * Matrices are assumed to be stored in row major order
 */

#include <cudnn.h>
#include "conv.h"
#include "indexing_defs.h"
#include "safe_call_defs.h"
#include "sparse_conversion.h"
#include "matrix_io.h"

void convolve2DDenseProjectImp(struct Matrix * mat,
                struct Kernel * kernel,
                struct Matrix * result)
{
  // Initialize result matrix
  result->dims[0] = mat->dims[0];
  result->dims[1] = mat->dims[1];
  result->dims[2] = mat->dims[2];
  result->dims[3] = mat->dims[3];
  result->is_column_first = mat->is_column_first;

  // Initialize cuda memory and copy in matrix and result to device
  float * d_input;
  float * d_output;
  size_t image_bytes = mat->dims[2] * mat->dims[3] * sizeof(float);
  size_t out_im_bytes = result->dims[2] * result->dims[3] * sizeof(float);
  size_t kernel_bytes = kernel->dims[2] * kernel->dims[3] * sizeof(float);
  CudaSafeCall(cudaMalloc(&d_input, image_bytes));
  CudaSafeCall(cudaMemcpy(d_input, mat->vals, image_bytes, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMalloc(&d_output, out_im_bytes));
  CudaSafeCall(cudaMemset(d_output, 0, out_im_bytes));

  if (!kernel->is_on_device)
  {
    printf("Kernel is not on the device. Allocating to constant memory\n");

    CudaSafeCall(cudaMalloc(&kernel->vals_device, kernel_bytes));
    CudaSafeCall(cudaMemcpy(kernel->vals_device, kernel->vals, kernel_bytes, cudaMemcpyHostToDevice));
    kernel->is_on_device = 1;
  }

  // Create kernel dims
  int t_col = MIN(mat->dims[2], 16);
  int t_row = MIN(mat->dims[3], 16);
  int b_col = mat->dims[2] / t_col;
  int b_row = mat->dims[3] / t_row;
  printf("Grid dim: (%d, %d), block dim: (%d, %d)\n", b_row, b_col, t_row, t_col);
  dim3 dimGrid(b_col, b_row);
  dim3 dimBlock(t_col, t_row);

  // Call convolve kernel
  convolve2DKernel<<<dimGrid, dimBlock, kernel_bytes>>>(
                  d_input,
                  kernel->vals_device,
                  d_output,
                  mat->dims[2],
                  mat->dims[3],
                  kernel->dims[2],
                  kernel->dims[3]);
  CudaCheckError();

  // Copy result back to host
  result->vals = (float *)calloc(out_im_bytes, sizeof(float));
  CudaSafeCall(cudaMemcpy(result->vals, d_output, out_im_bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
}


// This implementation follows the approach in chapter 7 of
// Programming Massively Parallel Processors
__global__ void convolve2DKernel(float * matrix,
                        float * kernel,
                        float * result,
                        int mat_h,
                        int mat_w,
                        int k_h,
                        int k_w)
{
    int row_o = blockIdx.y * blockDim.y + threadIdx.x;
    int col_o = blockIdx.x * blockDim.x + threadIdx.y;
    int tidx = row_o * (gridDim.y * blockDim.y) + col_o;
    int row_i = row_o - k_h / 2;
    int col_i = col_o - k_w / 2;
    int i, j;

    // Load kernel into shared memory
    extern __shared__ float k_shared[];
    if (tidx < (k_h * k_w))
    {
      k_shared[tidx] = kernel[tidx];
    }
    __syncthreads();

    // Compute convolution
    float out = 0.0f;
    if (row_o < mat_h && col_o < mat_w)
    {
      for (i = 0; i < k_h; i++)
      {
        for (j = 0; j < k_w; j++)
        {
          if (row_i + i >= 0 && col_i + j >= 0 &&
              row_i + i < mat_h && col_i + j < mat_w)
          {
            out += k_shared[index2D(i, j, k_w)] * matrix[index2D(row_i + i, col_i + j, mat_w)];
          }
        }
      }
      result[index2D(row_o, col_o, mat_w)] = out;
    }
}


// This implementation closely follows this excellent tutorial
// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/

void convolve2DDense(struct Matrix * mat,
                struct Kernel * kernel,
                struct Matrix * result, // Not initialized
                cudnnHandle_t cudnn)
{
  //Initialize input, kernel and output descriptors
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_FLOAT,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/mat->dims[2],
                                      /*image_width=*/mat->dims[3]));

  int out_height = mat->dims[2];
  int out_width = mat->dims[3];
  int pad_height = kernel->dims[2] / 2;
  int pad_width = kernel->dims[3] / 2;
  result->dims[0] = mat->dims[0];
  result->dims[1] = mat->dims[1];
  result->dims[2] = out_height;
  result->dims[3] = out_width;
  result->is_column_first = mat->is_column_first;

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/1,
                                        /*image_height=*/out_height,
                                        /*image_width=*/out_width));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/pad_height,
                                             /*pad_width=*/pad_width,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION));

  // Initialize kernel descriptor if it is not on the device already
  if (!kernel->is_on_device)
  {
    printf("Kernel is not on the device. Creating kernel descriptor\n");
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel->kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel->kernel_descriptor,
                                         /*dataType=*/CUDNN_DATA_FLOAT,
                                         /*format=*/CUDNN_TENSOR_NCHW,
                                         /*out_channels=*/1,
                                         /*in_channels=*/1,
                                         /*kernel_height=*/kernel->dims[2],
                                         /*kernel_width=*/kernel->dims[3]));
  }

  // Describe algorithm
  cudnnConvolutionFwdAlgo_t convolution_algorithm;
  checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel->kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

  // Allocate memory on device
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     kernel->kernel_descriptor,
                                                     convolution_descriptor,
                                                     output_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));

  void * d_workspace;
  CudaSafeCall(cudaMalloc(&d_workspace, workspace_bytes));
  size_t image_bytes = mat->dims[2] * mat->dims[3] * sizeof(float);
  size_t out_im_bytes = result->dims[2] * result->dims[3] * sizeof(float);
  size_t kernel_bytes = kernel->dims[2] * kernel->dims[3] * sizeof(float);
  float * d_input;
  float * d_output;

  CudaSafeCall(cudaMalloc(&d_input, image_bytes));
  CudaSafeCall(cudaMemcpy(d_input, mat->vals, image_bytes, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMalloc(&d_output, out_im_bytes));
  CudaSafeCall(cudaMemset(d_output, 0, out_im_bytes));

  if (!kernel->is_on_device)
  {
    printf("Kernel is not on the device. Allocating memory\n");
    CudaSafeCall(cudaMalloc(&kernel->vals_device, kernel_bytes));
    CudaSafeCall(cudaMemcpy(kernel->vals_device, kernel->vals, kernel_bytes, cudaMemcpyHostToDevice));
    kernel->is_on_device = 1;
  }

  // Convolve
  const float alpha = 1, beta = 0;
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel->kernel_descriptor,
                                     kernel->vals_device,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));

  // Copy result back to host
  result->vals = (float *)calloc(out_im_bytes, sizeof(float));
  CudaSafeCall(cudaMemcpy(result->vals, d_output, out_im_bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
}


void destroyKernel(struct Kernel * kernel, struct Matrix * kernel_mat)
{
  cudnnDestroyFilterDescriptor(kernel->kernel_descriptor);
  cudaFree(kernel->vals_device);
  destroyMatrix(kernel_mat);
}


void convolve2DSparse(struct SparseMat * mat,
                struct Kernel * kernel,
                struct SparseMat * result)
{
  // TODO
  // Copy kernel to device if not there - constant memory
  // Put matrix, and sparse mat result in constant memory
  // Convolve (look up csr indexing)
  // Copy back
}
