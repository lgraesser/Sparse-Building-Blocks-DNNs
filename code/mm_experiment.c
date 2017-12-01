
/*
 * MMatrix multiplication experiments.
 */
#include "indexing_defs.h"
#include "matrix_io.h"
#include "mm.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char * argv[])
{
  struct MMatrix matrix1, matrix2, matrixRes;
  int n_elements;
  const char * filename1 = "a.mat";

  read_matrix_dims(filename1, &matrix1, &n_elements);
  matrix1.vals = (float *)calloc(n_elements, sizeof(float));
  read_matrix_vals(filename1, &matrix1,1);
  print_matrix(&matrix1);

  const char * filename2 = "b.mat";
  read_matrix_dims(filename2, &matrix2, &n_elements);
  matrix2.vals = (float *)calloc(n_elements, sizeof(float));
  read_matrix_vals(filename2, &matrix2,1);
  print_matrix(&matrix2);


  int m = matrix1.dims[2];
  int k = matrix1.dims[3];
  int n = matrix2.dims[3];

  matrixRes.dims[0] = matrixRes.dims[1] = 0;
  matrixRes.dims[2]=m;
  matrixRes.dims[3]=n;

  matrixRes.vals = (float *)calloc(m*n,sizeof(float));
  cpu_mm(&matrix1,&matrix2,&matrixRes,m,n,k);
  printf("Cpu result:\n");
  print_matrix(&matrixRes);

  memset(matrixRes.vals,0,m*n*sizeof(float));

  gpu_mm_dense(&matrix1,&matrix2,&matrixRes,m,n,k);
  printf("CuBLAS result:\n");
  print_matrix(&matrixRes);

  free(matrix1.vals);
  free(matrix2.vals);
  free(matrixRes.vals);
}
