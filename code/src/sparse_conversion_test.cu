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
  struct Matrix mat;
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

  // Call conversion func
  struct SparseMat spm;
  convert_to_sparse(&spm, &mat, handle);
  copyDeviceCSR2Host(&spm);

  printf("Num rows: %d\n", mat.dims[2]);
  print_sparse_matrix(&spm);

  struct Matrix mat2;
  mat2.vals = (float *)calloc(num_elems, sizeof(float));
  for (int i = 0; i < 4; i++)
  {
      mat2.dims[i] = mat.dims[i];
  }
  mat2.is_column_first = mat.is_column_first;
  convert_to_dense(&spm, &mat2, handle);
  print_matrix(&mat2);

  // Free memory
  cusparseDestroy(handle);
  destroySparseMatrix(&spm);
  destroyMatrix(&mat);
  destroyMatrix(&mat2);
}
