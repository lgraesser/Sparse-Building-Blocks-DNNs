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
  // struct MMatrix mat;
  if (argc != 2){
    printf("usage ./sparse_conversion_test filename\n");
    exit(1);
  }
  char * filename = argv[1];
  int num_elems;
  read_matrix_dims(filename, &mat, &num_elems);
  mat.vals = (float *)calloc(num_elems, sizeof(float));
  read_matrix_vals(filename, &mat, 1);
  print_matrix(&mat);

  // Initialize cusparse library
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descrX;
  cusparseCreateMatDescr(&descrX);
  cusparseSetMatType(descrX, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrX, CUSPARSE_INDEX_BASE_ZERO);

  // Call conversion func
  struct SparseMat * spm_ptr = convert_to_sparse(
                                &mat,
                                handle,
                                descrX);
  struct SparseMat spm = *spm_ptr;

  printf("Num rows: %d\n", mat.dims[2]);
  print_sparse_matrix(spm, mat.dims[2]);

  // Free memory
  cusparseDestroy(handle);
  cudaFree(spm.csrRowPtrA_device);
  cudaFree(spm.csrColIndA_device);
  cudaFree(spm.csrValA_device);
  cudaFree(spm.nz_per_row_device);
  free(spm.csrRowPtrA);
  free(spm.csrColIndA);
  free(spm.csrValA);
  free(spm.nz_per_row);
  free(mat.vals);
}
