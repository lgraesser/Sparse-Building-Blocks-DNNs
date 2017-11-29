/*
 * File for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>
#include <stdio.h>
#include "sparse_conversion.h"

struct SparseMat * convert_to_sparse(float * matrix,
                              int matrix_dims[4],
                              cusparseHandle_t handle,
                              cusparseMatDescr_t descrA)
{
  // Example for 2D matrix
  int i, j;
  struct SparseMat spm;
  int num_non_zero = 0;
  int * non_zero_per_row = (int *)calloc(matrix_dims[2], sizeof(int));
  for (i = 0; i < matrix_dims[2]; i++)
  {
    int nzpr = 0;
    for (j = 0; j < matrix_dims[3]; j++)
    {
      if (matrix[index2D(i, j, matrix_dims[3])] < SMALL_NUM &&
          matrix[index2D(i, j, matrix_dims[3])] > -SMALL_NUM)
      {
        num_non_zero++;
        nzpr++;
      }
      non_zero_per_row[i] = nzpr;
    }
  }
  spm.nz_per_row = (const int *) non_zero_per_row;
  spm.csrRowPtrA = (int *)calloc(matrix_dims[2] + 1, sizeof(int));
  spm.csrColIndA = (int *)calloc(num_non_zero, sizeof(int));
  spm.csrValA = (float *)calloc(num_non_zero, sizeof(float));
  spm.total_non_zero = num_non_zero;
  struct SparseMat * spm_ptr = &spm;

  cusparseStatus_t result = cusparseSdense2csr(
                handle, // cusparse handle
                matrix_dims[2], // Number of rows
                matrix_dims[3], // Number of cols
                descrA, // cusparse matrix descriptor
                (const float *) matrix, // Matrix
                matrix_dims[2], // Leading dimension of the array
                spm.nz_per_row, // Non zero elements per row
                spm.csrValA, // Holds the matrix values
                spm.csrRowPtrA,
                spm.csrColIndA);
  printf("%d\n",result);
  printf("Converted matrix from dense to sparse\n");
  for (i = 0; i < matrix_dims[2] + 1; i++)
  {
    printf("Row ptr: %d\n", spm.csrRowPtrA[i]);
  }
  return spm_ptr;
}


void print_sparse_matrix(struct SparseMat * spm, int num_rows)
{
  int i, j, row_st, row_end, col;
  float num;
  for (i = 0; i < num_rows; i++)
  {
    row_st = (*spm).csrRowPtrA[i];
    row_end = (*spm).csrRowPtrA[i + 1] - 1;
    printf("Row start: {}, row_end: {}\n", row_st, row_end);
    for (j = row_st; j <= row_end; j++)
    {
      col = (*spm).csrColIndA[j];
      num = (*spm).csrValA[j];
      printf("(%d, %d): %05.2f\n", i, col, num);
    }
  }
}
