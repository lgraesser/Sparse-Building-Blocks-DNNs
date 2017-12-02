
/*
 * Matrix multiplication experiments.
 */
#include "matrix_io.h"
#include "mm.h"
#include <stdlib.h>
#include <stdio.h>
// #define DEBUG

int main(int argc, char * argv[])
{
  struct Matrix matrix1, matrix2, matrixResCPU, matrixResGPU;
  int n_elements;
  const char * filename1;
  const char * filename2;

  if (argc != 3){
    printf("usage ./mm matrixA matrixB\n");
    printf("Default values are going to be used ./mm data/a.mat data/b.mat\n");
    filename1 = "a.mat";
    filename2 = "b.mat";
  }
  else{
    filename1 = argv[1];
    filename2 = argv[2];
  }

  read_matrix_dims(filename1, &matrix1, &n_elements);
  matrix1.vals = (float *)calloc(n_elements, sizeof(float));
  read_matrix_vals(filename1, &matrix1,1);
  #ifdef DEBUG
    print_matrix(&matrix1);
  #endif


  read_matrix_dims(filename2, &matrix2, &n_elements);
  matrix2.vals = (float *)calloc(n_elements, sizeof(float));
  read_matrix_vals(filename2, &matrix2,1);
  #ifdef DEBUG
    print_matrix(&matrix2);
  #endif

  int m = matrix1.dims[2];
  int k = matrix1.dims[3];
  int n = matrix2.dims[3];

  initiliaze2dMatrix(&matrixResCPU,m,n);
  initiliaze2dMatrix(&matrixResGPU,m,n);


  cpu_mm(&matrix1,&matrix2,&matrixResCPU,m,n,k);
  #ifdef DEBUG
    printf("Cpu result:\n");
    print_matrix(&matrixResCPU);
  #endif

  gpu_mm_dense(&matrix1,&matrix2,&matrixResGPU,m,n,k);
  #ifdef DEBUG
    printf("CuBLAS result:\n");
    print_matrix(&matrixResGPU);
  #endif

  float diff = calculateDistanceMatrix(&matrixResCPU,&matrixResGPU);
  if (diff>1e-7){
      printf("Diff is %.8f\n",diff);
      printf("There might be a problem\n");
  }
  else{
      printf("Diff is less then 1e-7\n",diff);
  }

  destroyMatrix(&matrix1);
  destroyMatrix(&matrix2);
  destroyMatrix(&matrixResCPU);
  destroyMatrix(&matrixResGPU);
}
