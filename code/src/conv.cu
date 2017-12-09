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
                struct Matrix * result,
                bool pitch)
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
  size_t pit = 0;
  size_t * d_pitch = &pit;
  size_t image_bytes = mat->dims[2] * mat->dims[3] * sizeof(float);
  size_t image_width_bytes = mat->dims[3] * sizeof(float);
  size_t out_im_bytes = result->dims[2] * result->dims[3] * sizeof(float);
  size_t kernel_bytes = kernel->dims[2] * kernel->dims[3] * sizeof(float);

  if (pitch)
  {
    printf("Adding pitch to input matrix\n");
    CudaSafeCall(cudaMallocPitch(&d_input, d_pitch, image_width_bytes, (size_t) mat->dims[2]));
    CudaSafeCall(cudaMemcpy2D(d_input,
                        *d_pitch,
                        mat->vals,
                        image_width_bytes,
                        image_width_bytes,
                        mat->dims[2],
                        cudaMemcpyHostToDevice));
    printf("Pitch is: %d\n", *d_pitch);
  }
  else
  {
    CudaSafeCall(cudaMalloc(&d_input, image_bytes));
    CudaSafeCall(cudaMemcpy(d_input, mat->vals, image_bytes, cudaMemcpyHostToDevice));
    printf("No pitch, pitch is: %d\n", *d_pitch);
  }


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
  int t_col = MIN(mat->dims[2], 3);
  int t_row = MIN(mat->dims[3], 3);
  int t_col_halo = t_col + kernel->dims[2] - 1;
  int t_row_halo = t_row + kernel->dims[3] - 1;
  int b_col = ceil(mat->dims[2] / (t_col * 1.));
  int b_row = ceil(mat->dims[3] / (t_row * 1.));
  printf("Grid dim: (%d, %d), block dim: (%d, %d)\n", b_row, b_col, t_row_halo, t_col_halo);
  printf("Total threads: %d\n", t_col_halo * t_row_halo * b_col * b_row);
  dim3 dimGrid(b_col, b_row);
  dim3 dimBlock(t_col_halo, t_row_halo);

  // Actual matrix width
  int actual_mat_width = image_width_bytes / 4;
  if (pitch)
  {
    actual_mat_width = *d_pitch / 4;
  }
  printf("Actual matrix width: %d\n", actual_mat_width);

  // Compute size of shared memory block (kernel + tile)
  size_t tile_size = t_col_halo * t_row_halo * sizeof(float);
  size_t shared_mem_size = tile_size + kernel_bytes;
  printf("Shared mem is: %d elems, tile: %d, kernel: %d\n",
                                    shared_mem_size / 4,
                                    tile_size / 4,
                                    kernel_bytes / 4);

  // Call convolve kernel
  convolve2DKernel<<<dimGrid, dimBlock, shared_mem_size>>>(
                  d_input,
                  kernel->vals_device,
                  d_output,
                  mat->dims[2],
                  mat->dims[3],
                  actual_mat_width,
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
                        int actual_mat_width,
                        int k_h,
                        int k_w)
{
    int out_tile_width = blockDim.x - k_w + 1;
    int out_tile_height = blockDim.y - k_h + 1;
    int t_row = threadIdx.y;
    int t_col = threadIdx.x;
    int blk_idx = t_row * blockDim.x + t_col;
    int row_o = blockIdx.y * out_tile_height + t_row;
    int col_o = blockIdx.x * out_tile_width + t_col;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int tidx = blockId * (blockDim.x * blockDim.y) +
              threadIdx.y * blockDim.x + threadIdx.x;
    int row_i = row_o - k_h / 2;
    int col_i = col_o - k_w / 2;
    int i, j;
    // printf("tidx: %d, tile_w: %d, tile_h: %d, row_o: %d, col_o: %d, row_i: %d, col_i: %d, t_row: %d, t_col: %d\n",
                  // tidx, out_tile_width, out_tile_height, row_o, col_o, row_i, col_i, t_row, t_col);

    // Load kernel into shared memory
    // Tile first, then kernel
    extern __shared__ float shared_data[];
    float * mat_tile = &shared_data[0];
    float * kernel_shared = &shared_data[(blockDim.y * blockDim.x)];
    if ((row_i >= 0) && (row_i < mat_h) &&
        (col_i >= 0) && (col_i < mat_w))
    {
      mat_tile[t_row * blockDim.y + t_col] = matrix[row_i * actual_mat_width + col_i];
    }
    else
    {
      mat_tile[t_row * blockDim.y + t_col] = 0.0f;
    }
    if (blk_idx < (k_h * k_w))
    {
      kernel_shared[blk_idx] = kernel[blk_idx];
    }
    __syncthreads();

    // // Print shared mem
    // if (tidx == 25)
    // {
    //   printf("Kernel shared start: %d\n", (blockDim.y + k_h - 1 ) * (blockDim.x + k_w - 1));
    //   for (i = 0; i < blockDim.y; i ++)
    //   {
    //     for (j = 0; j < blockDim.x; j++)
    //     {
    //       printf("%.3f ", mat_tile[i * blockDim.y + j]);
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    //   for (i = 0; i < k_h; i ++)
    //   {
    //     for (j = 0; j < k_w; j++)
    //     {
    //       printf("%.3f ", kernel_shared[i * k_w + j]);
    //     }
    //     printf("\n");
    //   }
    // }

    // Compute convolution
    float out = 0.0f;
    if (t_row < out_tile_height && t_col < out_tile_width)
    {
      if (row_o < mat_h && col_o < mat_w)
      {
        for (i = 0; i < k_h; i++)
        {
          for (j = 0; j < k_w; j++)
          {
            out += kernel_shared[i * k_w + j] *
                      mat_tile[(t_row + i) * blockDim.y + t_col + j];
          }
        }
        result[index2D(row_o, col_o, mat_w)] = out;
      }
    }
}


// void convolve2DSparseProjectImp(struct Matrix * mat,
//                 struct Kernel * kernel,
//                 struct Matrix * result,
//                 bool pitch)
// {
//   // Initialize result matrix
//   result->dims[0] = mat->dims[0];
//   result->dims[1] = mat->dims[1];
//   result->dims[2] = mat->dims[2];
//   result->dims[3] = mat->dims[3];
//   result->is_column_first = mat->is_column_first;
//
//   // Initialize cuda memory and copy in matrix and result to device
//   float * d_input;
//   float * d_output;
//   size_t pit = 0;
//   size_t * d_pitch = &pit;
//   size_t image_bytes = mat->dims[2] * mat->dims[3] * sizeof(float);
//   size_t image_width_bytes = mat->dims[3] * sizeof(float);
//   size_t out_im_bytes = result->dims[2] * result->dims[3] * sizeof(float);
//   size_t kernel_bytes = kernel->dims[2] * kernel->dims[3] * sizeof(float);
//
//   if (pitch)
//   {
//     printf("Adding pitch to input matrix\n");
//     CudaSafeCall(cudaMallocPitch(&d_input, d_pitch, image_width_bytes, (size_t) mat->dims[2]));
//     CudaSafeCall(cudaMemcpy2D(d_input,
//                         *d_pitch,
//                         mat->vals,
//                         image_width_bytes,
//                         image_width_bytes,
//                         mat->dims[2],
//                         cudaMemcpyHostToDevice));
//     printf("Pitch is: %d\n", *d_pitch);
//   }
//   else
//   {
//     CudaSafeCall(cudaMalloc(&d_input, image_bytes));
//     CudaSafeCall(cudaMemcpy(d_input, mat->vals, image_bytes, cudaMemcpyHostToDevice));
//     printf("No pitch, pitch is: %d\n", *d_pitch);
//   }
//
//
//   CudaSafeCall(cudaMalloc(&d_output, out_im_bytes));
//   CudaSafeCall(cudaMemset(d_output, 0, out_im_bytes));
//
//   if (!kernel->is_on_device)
//   {
//     printf("Kernel is not on the device. Allocating to constant memory\n");
//
//     CudaSafeCall(cudaMalloc(&kernel->vals_device, kernel_bytes));
//     CudaSafeCall(cudaMemcpy(kernel->vals_device, kernel->vals, kernel_bytes, cudaMemcpyHostToDevice));
//     kernel->is_on_device = 1;
//   }
//
//   // Create kernel dims
//   int t_col = MIN(mat->dims[2], 16);
//   int t_row = MIN(mat->dims[3], 16);
//   int b_col = mat->dims[2] / t_col;
//   int b_row = mat->dims[3] / t_row;
//   printf("Grid dim: (%d, %d), block dim: (%d, %d)\n", b_row, b_col, t_row, t_col);
//   dim3 dimGrid(b_col, b_row);
//   dim3 dimBlock(t_col, t_row);
//
//   // Actual matrix width
//   int actual_mat_width = image_width_bytes / 4;
//   if (pitch)
//   {
//     actual_mat_width = *d_pitch / 4;
//   }
//   printf("Actual matrix width: %d\n", actual_mat_width);
//
//   // Call convolve kernel
//   convolve2DKernel<<<dimGrid, dimBlock, kernel_bytes>>>(
//                   d_input,
//                   kernel->vals_device,
//                   d_output,
//                   mat->dims[2],
//                   mat->dims[3],
//                   actual_mat_width,
//                   kernel->dims[2],
//                   kernel->dims[3]);
//   CudaCheckError();
//
//   // Copy result back to host
//   result->vals = (float *)calloc(out_im_bytes, sizeof(float));
//   CudaSafeCall(cudaMemcpy(result->vals, d_output, out_im_bytes, cudaMemcpyDeviceToHost));
//   cudaFree(d_input);
//   cudaFree(d_output);
// }
//
//
// // This implementation follows the approach in chapter 7 of
// // Programming Massively Parallel Processors
// __global__ void convolve2DKernelSparse(float * matrix,
//                         float * kernel_vals,
//                         int * kernel_rows,
//                         int * kernel_cols,
//                         int num_k_elems,
//                         float * result,
//                         int mat_h,
//                         int mat_w,
//                         int actual_mat_width,
//                         int k_h,
//                         int k_w)
// {
//     int row_o = blockIdx.y * blockDim.y + threadIdx.x;
//     int col_o = blockIdx.x * blockDim.x + threadIdx.y;
//     int tidx = row_o * (gridDim.y * blockDim.y) + col_o;
//     int row_i = row_o - k_h / 2;
//     int col_i = col_o - k_w / 2;
//     int i, j;
//
//     // Load kernel into shared memory
//     // Three kernel matrices included in one shared array
//     // First is kernel_rows: size k_h + 1
//     // Then kernel_cols: size num_k_elems
//     // Then kernel_vals: size num_k_elems
//     extern __shared__ float k_shared[];
//     int b1 = k_h + 1;
//     int b2 = b1 + num_k_elems;
//     int b3 = b2 + num_k_elems;
//     if (tidx < (b1))
//     {
//       k_shared[tidx] = kernel_rows[tidx];
//     }
//     else if (tidx < b2)
//     {
//       k_shared[tidx] = kernel_cols[tidx - b1];
//     }
//     else if (tidx < b3)
//     {
//       k_shared[tidx] = kernel_cols[tidx - b2];
//     }
//     __syncthreads();
//     int row_start;
//     int row_end;
//     int row;
//     int col;
//     // Compute convolution
//     float out = 0.0f;
//     if (row_o < mat_h && col_o < mat_w)
//     {
//       // Sparse kernel - only access matrix where kernel is non zero
//       for (i = 1; i < k_h + 1; i++)
//       {
//         row_start = k_shared[i -1];
//         row_end = k_shared[i];
//         for (j = row_start; j < row_end; j++)
//         {
//           row = i - 1;
//           col = k_shared[b1 + j];
//           if (row_i + row >= 0 && col_i + col >= 0 &&
//               row_i + row < mat_h && col_i + col < mat_w)
//           {
//             out += k_shared[b2 + j] * matrix[(row_i + row) * actual_mat_width + col_i + col];
//           }
//         }
//       }
//       result[index2D(row_o, col_o, mat_w)] = out;
//     }
// }


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
