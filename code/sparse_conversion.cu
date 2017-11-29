/*
 * File for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>
#include "sparse_conversion.h"
#include "matrix_io.h"

SparseMat * convert_to_sparse(float * matrix,
                              int matrix_dims,
                              cusparseHandle_t * handle,
                              cusparseMatDescr_t descrX)
{
  // Example for 2D matrix
  int i, j;
  struct SparseMat * spm;
  int num_non_zero = 0;
  spm.nz_per_row = (int *)calloc(matrix_dims[2], sizeof(int));
  for (i = 0; i < matrix_dims[2], i++)
  {
    int nzpr = 0;
    for (j = 0; j < matrix_dims[3], j++)
    {
      if (matrix[index2D(i, j, matrix_dims[3])] < SMALL_NUM &&
          matrix[index2D(i, j, matrix_dims[3])] > -SMALL_NUM &&)
      {
        num_non_zero++;
        nzpr++;
      }
      nz_per_row[i] = nzpr;
    }
  }
  spm.csrRowPtrA = (int *)calloc(matrix_dims[2] + 1, sizeof(int));
  spm.csrColIndA = (int *)calloc(num_non_zero, sizeof(int));
  spm.csrColValA = (float *)calloc(num_non_zero, sizeof(float));

  cusparseStatus_t result = cusparseSdense2csr(
                handle, // cusparse handle
                matrix_dims[2] // Number of rows
                matrix_dims[3] // Number of cols
                descrA, // cusparse matrix descriptor
                matrix, // Matrix
                matrix_dims[2], // Leading dimension of the array
                spm.nz_per_row, // Non zero elements per row
                spm.csrValA, // Holds the matrix values
                spm.csrRowPtrA,
                spm.csrColIndA);
  printf("Converted matrix from dense to sparse\n")
}


void print_sparse_matrix(SparseMat *)
{
  // TODO
}
