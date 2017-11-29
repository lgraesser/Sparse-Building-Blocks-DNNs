
/*
 * Matrix multiplication experiments.
 */

#include "matrix_io.h"
#include "mm.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char * argv[])
{
  float * matrix1, * matrix2,*matrixRes;
  int n_elements1,n_elements2;
  int matrix_dims1[4] = {0};
  int matrix_dims2[4] = {0};
  int matrix_dimsRes[4] = {0};
  const char * filename1 = "a.mat";
  read_matrix_dims(filename1, matrix_dims1, &n_elements1);
  matrix1 = (float *)calloc(n_elements1, sizeof(float));
  read_matrix_vals(filename1, matrix1, matrix_dims1,1);
  print_matrix(matrix1, matrix_dims1,1);

  const char * filename2 = "b.mat";
  read_matrix_dims(filename2, matrix_dims2, &n_elements2);
  matrix2 = (float *)calloc(n_elements2, sizeof(float));
  read_matrix_vals(filename2, matrix2, matrix_dims2,1);
  print_matrix(matrix2, matrix_dims2,1);

  int m = matrix_dims1[2];
  int k = matrix_dims1[3];
  int n = matrix_dims2[3];
  matrix_dimsRes[2]=m;
  matrix_dimsRes[3]=n;

  matrixRes = (float *)calloc(m*n,sizeof(float));
  cpu_mm(matrix1,matrix2,matrixRes,m,n,k);
  printf("Cpu result:\n");
  print_matrix(matrixRes,matrix_dimsRes ,1);
  free(matrixRes);

  matrixRes = (float *)calloc(m*n,sizeof(float));
  gpu_mm_dense(matrix1,matrix2,matrixRes,m,n,k);
  printf("CuBLAS result:\n");
  print_matrix(matrixRes,matrix_dimsRes ,1);

  free(matrix1);
  free(matrix2);
  free(matrixRes);
}
