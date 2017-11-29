/*
 * Test file for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>
#include "sparse_conversion.h"
#include "matrix_io.h"

int main(int argc, char * argv[])
{
  float * matrix;
  int matrix_dims[4] = {0};
  char * filename = "test.mat";
  int num_elems = 1;
  read_matrix_dims(filename, matrix_dims, &num_elems);
  matrix = (float *)calloc(num_elems, sizeof(float));
  matrix_cols = (float *)calloc(num_elems, sizeof(float));

  // Read and convert matrices
  read_matrix_vals(filename, matrix, matrix_dims, 1);
  print_matrix(matrix, matrix_dims, 1);

  // Initialize cusparse library
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descrX;
  cusparseCreateMatDescr(&descrX);

  // Call conversion func
  struct SparseMat * spm = convert_to_sparse(matrix,
                                int matrix_dims,
                                cusparseHandle_t * handle,
                                cusparseMatDescr_t descrX)

  cusparseDestroy(handle);
  free(spm.csrRowPtrA);
  free(spm.csrColIndA);
  free(spm.csrValA)
  free(spm.nz_per_row);
  free(matrix);
  free(matrix_cols);
}
