/*
 * File for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 * and that they are 2D
 */

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include "sparse_conversion.h"
#include "matrix_io.h"
#include "safe_call_defs.h"


void convert_to_sparse(struct SparseMat * spm,
                      struct Matrix * mat,
                      cusparseHandle_t handle)
{
  int * nz_per_row_device;
  float * matrix = mat->vals;
  printf("[%d, %d, %d, %d]\n", mat->dims[0], mat->dims[1], mat->dims[2], mat->dims[3]);
  float * matrix_device;
  const int lda = mat->dims[2];
  spm->num_rows = mat->dims[2];
  spm->total_non_zero = 0;

  // Create cusparse matrix descriptors
  cusparseCreateMatDescr(&(spm->descrA));
  cusparseSetMatType(spm->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(spm->descrA, CUSPARSE_INDEX_BASE_ZERO);

  // Allocate device dense array and copy over
  CudaSafeCall(cudaMalloc(&matrix_device,
                        mat->dims[2] * mat->dims[3] * sizeof(float)));
  CudaSafeCall(cudaMemcpy(matrix_device,
                          matrix,
                          mat->dims[2] * mat->dims[3] * sizeof(float),
                          cudaMemcpyHostToDevice));

  // Device side number of nonzero element per row of matrix
  CudaSafeCall(cudaMalloc(&(nz_per_row_device),
                          spm->num_rows * sizeof(int)));
  cusparseSafeCall(cusparseSnnz(handle,
                                CUSPARSE_DIRECTION_ROW,
                                mat->dims[2],
                                mat->dims[3],
                                spm->descrA,
                                matrix_device,
                                lda,
                                nz_per_row_device,
                                &(spm->total_non_zero)));

  // Host side number of nonzero elements per row of matrix

  printf("Num non zero elements: %d\n", spm->total_non_zero);

  // Allocate device sparse matrices
  CudaSafeCall(cudaMalloc(&(spm->csrRowPtrA_device),
                        (spm->num_rows + 1) * sizeof(int)));
  CudaSafeCall(cudaMalloc(&(spm->csrColIndA_device),
                        spm->total_non_zero * sizeof(int)));
  CudaSafeCall(cudaMalloc(&(spm->csrValA_device),
                        spm->total_non_zero * sizeof(float)));

  // Call cusparse
  cusparseSafeCall(cusparseSdense2csr(
                handle, // cusparse handle
                mat->dims[2], // Number of rows
                mat->dims[3], // Number of cols
                spm->descrA, // cusparse matrix descriptor
                matrix_device, // Matrix
                lda, // Leading dimension of the array
                nz_per_row_device, // Non zero elements per row
                spm->csrValA_device, // Holds the matrix values
                spm->csrRowPtrA_device,
                spm->csrColIndA_device));

  cudaFree(matrix_device);
  printf("Converted matrix from dense to sparse\n");
}


void convert_to_dense(struct SparseMat * spm,
                      struct Matrix * mat,
                      cusparseHandle_t handle)
{
  /* Assumes that csrValA_device, csrRowPtrA_device, csrColIndA_device
   * all exist on the device and have been correctly populated
   * Also assumes the cusparse matrix descriptor in the spm has between
   * properly initialized
   */
  int num_elems = mat->dims[2] * mat->dims[3];
  float * matrix_device;
  const int lda = mat->dims[2];
  // Allocate device dense array
  CudaSafeCall(cudaMalloc(&matrix_device,
                        num_elems * sizeof(float)));

  cusparseSafeCall(cusparseScsr2dense(handle,
                                      mat->dims[2],
                                      mat->dims[3],
                                      spm->descrA,
                                      spm->csrValA_device,
                                      spm->csrRowPtrA_device,
                                      spm->csrColIndA_device,
                                      matrix_device,
                                      lda));

  // Copy matrix back to cpu and free device storage
  CudaSafeCall(cudaMemcpy(mat->vals,
                          matrix_device,
                          num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  cudaFree(matrix_device);
}


void copyDeviceCSR2Host(struct SparseMat * spm_ptr)
{
  // Allocate host memory and copy device vals back to host
  // WARNING this may result in memory leak if you continously call this function
  spm_ptr->csrRowPtrA = (int *)calloc((spm_ptr->num_rows + 1), sizeof(int));
  spm_ptr->csrColIndA = (int *)calloc(spm_ptr->total_non_zero, sizeof(int));
  spm_ptr->csrValA = (float *)calloc(spm_ptr->total_non_zero, sizeof(float));
  CudaSafeCall(cudaMemcpy(spm_ptr->csrRowPtrA,
                          spm_ptr->csrRowPtrA_device,
                          (spm_ptr->num_rows + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(spm_ptr->csrColIndA,
                          spm_ptr->csrColIndA_device,
                          spm_ptr->total_non_zero * sizeof(int),
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(spm_ptr->csrValA,
                          spm_ptr->csrValA_device,
                          spm_ptr->total_non_zero * sizeof(int),
                          cudaMemcpyDeviceToHost));
  printf("Copied sparse matrix from device to host\n");
}


void destroySparseMatrix(struct SparseMat *spm){
  cudaFree(spm->csrRowPtrA_device);
  cudaFree(spm->csrColIndA_device);
  cudaFree(spm->csrValA_device);
  free(spm->csrRowPtrA);
  free(spm->csrColIndA);
  free(spm->csrValA);
}


void print_sparse_matrix(struct SparseMat *spm)
{
  printf("Sparse representation of the matrix\n");
  int i, j, row_st, row_end, col;
  float num;
  for (i = 0; i < spm->num_rows; i++)
  {
    row_st = spm->csrRowPtrA[i];
    row_end = spm->csrRowPtrA[i + 1] - 1;
    for (j = row_st; j <= row_end; j++)
    {
      col = spm->csrColIndA[j];
      num = spm->csrValA[j];
      printf("(%d, %d): %05.2f\n", i, col, num);
    }
  }
}
