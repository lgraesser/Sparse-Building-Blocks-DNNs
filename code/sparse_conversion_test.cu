/*
 * Test file for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>
#include <stdio.h>
#include "sparse_conversion.h"
#include "matrix_io.h"

int main(int argc, char * argv[])
{
  float * matrix;
  int matrix_dims[4] = {0};
  if (argc != 2){
    printf("usage ./sparse_conversion_test filename\n");
    exit(1);
  }
  char * filename = argv[1];
  int num_elems = 1;
  read_matrix_dims(filename, matrix_dims, &num_elems);
  matrix = (float *)calloc(num_elems, sizeof(float));

  // Read and convert matrices
  read_matrix_vals(filename, matrix, matrix_dims, 1);
  print_matrix(matrix, matrix_dims, 1);

  // Initialize cusparse library
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descrX;
  cusparseCreateMatDescr(&descrX);
  cusparseSetMatType(descrX, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrX, CUSPARSE_INDEX_BASE_ZERO);

  // Call conversion func
  struct SparseMat * spm = convert_to_sparse(
                                matrix,
                                matrix_dims,
                                handle,
                                descrX);

  printf("Num rows: %d\n",matrix_dims[2]);
  print_sparse_matrix(spm, matrix_dims[2]);

  cusparseDestroy(handle);
  free((*spm).csrRowPtrA);
  free((*spm).csrColIndA);
  free((*spm).csrValA);
  free((int *)(*spm).nz_per_row);
  free(matrix);
}
