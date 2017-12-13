/* Functions for carrying out
 * dense and sparse 2D convolutions
 *
 * Matrices are assumed to be stored in row major order
 */
#include <cuda.h>
#include <cudnn.h>
#include "conv.h"
#include "indexing_defs.h"
#include "safe_call_defs.h"
#include "sparse_conversion.h"
#include "matrix_io.h"
double time_taken;
clock_t start, end;

void convolve2DDenseProjectImp(struct Matrix * mat,
                struct Kernel * kernel,
                struct Matrix * result,
                bool pitch,
                int num_its)
{
  start = clock();
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
    printf("Kernel is not on the device. Allocating to device memory\n");

    CudaSafeCall(cudaMalloc(&kernel->vals_device, kernel_bytes));
    CudaSafeCall(cudaMemcpy(kernel->vals_device, kernel->vals, kernel_bytes, cudaMemcpyHostToDevice));
    kernel->is_on_device = 1;
  }
  else
  {
    printf("Kernel is already on the device.\n");
  }
  // Create kernel dims
  int t_col = MIN(mat->dims[2], 16);
  int t_row = MIN(mat->dims[3], 16);
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
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to allocate mem and copy to device: %lf \n",time_taken);
  // Call convolve kernel
  start = clock();
  int i;
  for (i = 0; i < num_its; i++)
  {
    convolve2DKernel<<<dimGrid, dimBlock, shared_mem_size>>>(
                    d_input,
                    kernel->vals_device,
                    d_output,
                    mat->dims[2],
                    mat->dims[3],
                    actual_mat_width,
                    kernel->dims[2],
                    kernel->dims[3]);
  //CudaCheckError();
  }
  cudaDeviceSynchronize();
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to execute kernel: %lf \n",time_taken);
  start = clock();
  // Copy result back to host
  result->vals = (float *)calloc(out_im_bytes, sizeof(float));
  CudaSafeCall(cudaMemcpy(result->vals, d_output, out_im_bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to copy memory back to host and free device mem: %lf \n",time_taken);
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


// ASSUMES SQUARE KERNEL
void convolve2DSparseProjectImp(struct Matrix * mat,
                struct SparseMat * kernel,
                struct Matrix * result,
                bool pitch,
                int num_its)
{
  start = clock();
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
  size_t kernel_bytes = kernel->total_non_zero * sizeof(float);
  size_t kernel_col_bytes = kernel->total_non_zero * sizeof(int);
  size_t kernel_row_bytes = (kernel->num_rows + 1) * sizeof(int);

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

  if (kernel->is_on_device != 1)
  {
    printf("Kernel is not on the device. Allocating to device memory\n");

    CudaSafeCall(cudaMalloc(&kernel->csrValA_device,
                            kernel_bytes));
    CudaSafeCall(cudaMemcpy(kernel->csrValA_device,
                            kernel->csrValA,
                            kernel_bytes,
                            cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&kernel->csrColIndA_device,
                            kernel_col_bytes));
    CudaSafeCall(cudaMemcpy(kernel->csrColIndA_device,
                            kernel->csrColIndA,
                            kernel_col_bytes,
                            cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&kernel->csrRowPtrA_device,
                            kernel_row_bytes));
    CudaSafeCall(cudaMemcpy(kernel->csrRowPtrA_device,
                            kernel->csrRowPtrA,
                            kernel_row_bytes,
                            cudaMemcpyHostToDevice));
    kernel->is_on_device = 1;
  }
  else
  {
    printf("Kernel is ALREADY on the device.\n");
  }

  // Create kernel dims
  int t_col = MIN(mat->dims[2], 16);
  int t_row = MIN(mat->dims[3], 16);
  int t_col_halo = t_col + kernel->num_rows - 1;
  int t_row_halo = t_row + kernel->num_rows - 1;
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

  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to allocate mem and copy to device: %lf \n",time_taken);
  // Call convolve kernel
  start = clock();
  int i;
  for (i = 0; i < num_its; i++)
  {
    convolve2DKernelSparse<<<dimGrid, dimBlock, shared_mem_size>>>(
                    d_input,
                    kernel->csrValA_device,
                    kernel->csrRowPtrA_device,
                    kernel->csrColIndA_device,
                    kernel->total_non_zero,
                    d_output,
                    mat->dims[2],
                    mat->dims[3],
                    actual_mat_width,
                    kernel->num_rows,
                    kernel->num_rows);
    //CudaCheckError();
  }
  cudaDeviceSynchronize();
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to execute kernel: %lf \n",time_taken);
  start = clock();
  // Copy result back to host
  result->vals = (float *)calloc(out_im_bytes, sizeof(float));
  CudaSafeCall(cudaMemcpy(result->vals, d_output, out_im_bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to copy memory back to host and free device mem: %lf \n",time_taken);
}


__global__ void convolve2DKernelSparse(float * matrix,
                        float * kernel_vals,
                        int * kernel_rows,
                        int * kernel_cols,
                        int num_k_elems,
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
    int row_i = row_o - k_h / 2;
    int col_i = col_o - k_w / 2;
    int i, j;
    // printf("tidx: %d, tile_w: %d, tile_h: %d, row_o: %d, col_o: %d, row_i: %d, col_i: %d, t_row: %d, t_col: %d\n",
                  // tidx, out_tile_width, out_tile_height, row_o, col_o, row_i, col_i, t_row, t_col);

    // Load kernel into shared memory
    // Tile first, then kernel
    // Kernel is sparse
    extern __shared__ float shared_data[];
    float * mat_tile = &shared_data[0];
    float * kernel_shared = &shared_data[(blockDim.y * blockDim.x)];
    float * kernel_rows_shared = &kernel_shared[0];
    float * kernel_cols_shared = &kernel_shared[k_h + 1];
    float * kernel_vals_shared = &kernel_shared[k_h + 1 + num_k_elems];

    if ((row_i >= 0) && (row_i < mat_h) &&
        (col_i >= 0) && (col_i < mat_w))
    {
      mat_tile[t_row * blockDim.y + t_col] = matrix[row_i * actual_mat_width + col_i];
    }
    else
    {
      mat_tile[t_row * blockDim.y + t_col] = 0.0f;
    }
    if (blk_idx < (k_h + 1))
    {
      kernel_rows_shared[blk_idx] = kernel_rows[blk_idx];
    }
    if (blk_idx < (num_k_elems))
    {
      kernel_cols_shared[blk_idx] = kernel_cols[blk_idx];
      kernel_vals_shared[blk_idx] = kernel_vals[blk_idx];
    }
    __syncthreads();

    int row_start;
    int row_end;
    int col;
    // Compute convolution
    float out = 0.0f;
    if (t_row < out_tile_height && t_col < out_tile_width)
    {
      if (row_o < mat_h && col_o < mat_w)
      {
        for (i = 0; i < k_h; i++)
        {
          row_start = kernel_rows_shared[i];
          row_end = kernel_rows_shared[i + 1];
          for (j = row_start; j < row_end; j++)
          {
            col = kernel_cols_shared[j];
            out += kernel_vals_shared[j] *
                      mat_tile[(t_row + i) * blockDim.y + t_col + col];
          }
        }
        result[index2D(row_o, col_o, mat_w)] = out;
      }
    }
}


// This implementation closely follows this excellent tutorial
// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/

void convolve2DDense(struct Matrix * mat,
                struct Kernel * kernel,
                struct Matrix * result, // Not initialized
                cudnnHandle_t cudnn,
                int num_its)
{
  start = clock();
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
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));

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

  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to allocate mem and copy to device: %lf \n",time_taken);
  // Convolve
  start = clock();
  const float alpha = 1, beta = 0;
  int i;
  for (i = 0; i < num_its; i++)
  {
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
  }
  cudaDeviceSynchronize();
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to execute kernel: %lf \n",time_taken);
  start = clock();
  // Copy result back to host
  result->vals = (float *)calloc(out_im_bytes, sizeof(float));
  CudaSafeCall(cudaMemcpy(result->vals, d_output, out_im_bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken to copy memory back to host and free device mem: %lf \n",time_taken);
}


void destroyKernel(struct Kernel * kernel, struct Matrix * kernel_mat)
{
  // cudnnDestroyFilterDescriptor(kernel->kernel_descriptor);
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
