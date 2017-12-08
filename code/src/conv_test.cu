/*
 * Test file for convolution
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cudnn.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include "conv.h"
#include "safe_call_defs.h"
#include "sparse_conversion.h"
#include "matrix_io.h"

int main(int argc, char * argv[])
{
  struct Matrix row_mat;
  struct Matrix k_mat;
  struct Kernel kernel;
  struct Matrix col_kernel;
  struct Matrix dense_out;
  if (argc != 3){
    printf("usage ./conv_test filename kernel\n");
    exit(1);
  }
  char * filename = argv[1];
  char * kernel_name = argv[2];
  int k_elems;
  int num_elems;

  // Read in matrix and kernel
  read_matrix_dims(filename, &row_mat, &num_elems);
  row_mat.vals = (float *)calloc(num_elems, sizeof(float));
  read_matrix_vals(filename, &row_mat, 0);
  print_matrix(&row_mat);
  read_matrix_dims(kernel_name, &k_mat, &k_elems);
  k_mat.vals = (float *)calloc(k_elems, sizeof(float));
  read_matrix_vals(kernel_name, &k_mat, 0);
  print_matrix(&k_mat);
  kernel.vals = (float *)calloc(k_elems, sizeof(float));
  kernel.vals = k_mat.vals;
  for (int i = 0; i < 4; i++)
  {
    kernel.dims[i] = k_mat.dims[i];
  }
  kernel.is_column_first = k_mat.is_column_first;
  kernel.is_on_device = 0;

  // Initialize cuda librarries
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  // Convolve dense
  convolve2DDense(&row_mat,
                  &kernel,
                  &dense_out, // Not initialized
                  cudnn);
  print_matrix(&dense_out);
  destroyMatrix(&dense_out);

  // Convolve dense alternate
  convolve2DDenseProjectImp(&row_mat,
                            &kernel,
                            &dense_out,
                            0);
  print_matrix(&dense_out);
  destroyMatrix(&dense_out);

  // Convolve dense alternate
  convolve2DDenseProjectImp(&row_mat,
                            &kernel,
                            &dense_out,
                            1);
  print_matrix(&dense_out);
  destroyMatrix(&dense_out);

  // Convert kernel to sparse matrix
  for (int i = 0; i < 4; i++)
  {
    col_kernel.dims[i] = kernel.dims[i];
  }
  num_elems = kernel.dims[2] * kernel.dims[3];
  col_kernel.vals = (float *)calloc(num_elems, sizeof(float));
  convert_to_column_major(&k_mat, &col_kernel);
  print_matrix(&col_kernel);
  struct SparseMat spm;
  convert_to_sparse(&spm, &col_kernel, handle);
  copyDeviceCSR2Host(&spm);
  printf("Num rows: %d\n", col_kernel.dims[2]);
  print_sparse_matrix(&spm);

  // Convolve sparse
  // TODO

  // Free memory
  cusparseDestroy(handle);
  cudnnDestroy(cudnn);
  destroySparseMatrix(&spm);
  destroyMatrix(&col_kernel);
  destroyMatrix(&row_mat);
  destroyKernel(&kernel, &k_mat);
}
