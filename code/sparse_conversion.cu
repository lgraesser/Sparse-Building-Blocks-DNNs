/*
 * File for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include "sparse_conversion.h"
#include "matrix_io.h"
#include "indexing_defs.h"
#include "safe_call_defs.h"

struct SparseMat * convert_to_sparse(struct Matrix * mat,
                              cusparseHandle_t handle,
                              cusparseMatDescr_t descrA)
{
  // Example for 2D matrix
  struct SparseMat spm;
  float * matrix = mat->vals;
  printf("[%d, %d, %d, %d]\n", mat->dims[0], mat->dims[1], mat->dims[2], mat->dims[3]);
  float * matrix_device;
  const int lda = mat->dims[2];
  int num_non_zero = 0;

  // Allocate device dense array and copy over
  CudaSafeCall(cudaMalloc(&matrix_device,
                        mat->dims[2] * mat->dims[3] * sizeof(float)));
  CudaSafeCall(cudaMemcpy(matrix_device,
                          matrix,
                          mat->dims[2] * mat->dims[3] * sizeof(float),
                          cudaMemcpyHostToDevice));

  // Device side number of nonzero element per row of matrix
  CudaSafeCall(cudaMalloc(&spm.nz_per_row_device,
                          mat->dims[2] * sizeof(int)));
  cusparseSafeCall(cusparseSnnz(handle,
                                CUSPARSE_DIRECTION_ROW,
                                mat->dims[2],
                                mat->dims[3],
                                descrA,
                                matrix_device,
                                lda,
                                spm.nz_per_row_device,
                                &num_non_zero));

  // Host side number of nonzero elements per row of matrix
  spm.nz_per_row = (int *)calloc(mat->dims[2], sizeof(int));
  CudaSafeCall(cudaMemcpy(spm.nz_per_row,
                          spm.nz_per_row_device,
                          mat->dims[2] * sizeof(int),
                          cudaMemcpyDeviceToHost));
  // // Error checking
  // int i;
  // printf("Num non zero elements: %d\n", num_non_zero);
  // for (i = 0; i < mat->dims[2]; i++)
  // {
  //   printf("row %d: %d\n", i, spm.nz_per_row[i]);
  // }

  // Allocate device sparse matrices
  CudaSafeCall(cudaMalloc(&(spm.csrRowPtrA_device),
                        (mat->dims[2] + 1) * sizeof(int)));
  CudaSafeCall(cudaMalloc(&(spm.csrColIndA_device),
                        num_non_zero * sizeof(int)));
  CudaSafeCall(cudaMalloc(&(spm.csrValA_device),
                        num_non_zero * sizeof(float)));

  // Call cusparse
  cusparseSafeCall(cusparseSdense2csr(
                handle, // cusparse handle
                mat->dims[2], // Number of rows
                mat->dims[3], // Number of cols
                descrA, // cusparse matrix descriptor
                matrix_device, // Matrix
                lda, // Leading dimension of the array
                spm.nz_per_row_device, // Non zero elements per row
                spm.csrValA_device, // Holds the matrix values
                spm.csrRowPtrA_device,
                spm.csrColIndA_device));

  // Allocate host memory and copy device vals back to host
  spm.csrRowPtrA = (int *)calloc((mat->dims[2] + 1), sizeof(int));
  spm.csrColIndA = (int *)calloc(num_non_zero, sizeof(int));
  spm.csrValA = (float *)calloc(num_non_zero, sizeof(float));
  CudaSafeCall(cudaMemcpy(spm.csrRowPtrA,
                          spm.csrRowPtrA_device,
                          (mat->dims[2] + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(spm.csrColIndA,
                          spm.csrColIndA_device,
                          num_non_zero * sizeof(int),
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(spm.csrValA,
                          spm.csrValA_device,
                          num_non_zero * sizeof(int),
                          cudaMemcpyDeviceToHost));
  cudaFree(matrix_device);
  // Error checking
  printf("Converted matrix from dense to sparse\n");
  // int i;
  // for (i = 0; i < mat->dims[2] + 1; i++)
  // {
  //   printf("Row ptr: %d\n", spm.csrRowPtrA[i]);
  // }
  // for (i = 0; i < num_non_zero; i++)
  // {
  //   printf("Vals: %f, \t", spm.csrValA[i]);
  //   printf("Col idx: %d\n", spm.csrColIndA[i]);
  // }
  struct SparseMat *spm_ptr = &spm;
  return spm_ptr;
}


void print_sparse_matrix(struct SparseMat spm, int num_rows)
{
  printf("Sparse representation of the matrix\n");
  int i, j, row_st, row_end, col;
  float num;
  for (i = 0; i < num_rows; i++)
  {
    row_st = spm.csrRowPtrA[i];
    row_end = spm.csrRowPtrA[i + 1] - 1;
    for (j = row_st; j <= row_end; j++)
    {
      col = spm.csrColIndA[j];
      num = spm.csrValA[j];
      printf("(%d, %d): %05.2f\n", i, col, num);
    }
  }
}
