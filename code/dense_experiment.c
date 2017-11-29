
/*
 * Matrix multiplication experiments.
 */

#include "matrix_io.h"
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

/* The input is 2d 2d
 */

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
   int lda=m,ldb=k,ldc=m;
   const float alf = 1;
   const float bet = 0;
   const float *alpha = &alf;
   const float *beta = &bet;

   // Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);

   // Do the actual multiplication
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

   // Destroy the handle
   cublasDestroy(handle);
}

void mm_cpu(float *a, float *b, float *c, int m, int n, int k){
    int i,j,r;
    float c_sum;
    for(i=0;i<m;i++){
      for(j=0;j<n;j++){
        c_sum = 0;
        for(r=0;r<k;r++){
            c_sum += a[index2DCol(i,r,m)]*b[index2DCol(r,j,k)];
        }
        c[index2DCol(i,j,m)] = c_sum;
      }
    }
}

int main(int argc, char * argv[])
{
  float * matrix1, * matrix2,*matrixRes;
  int n_elements1,n_elements2;
  int matrix_dims1[4] = {0};
  int matrix_dims2[4] = {0};
  int matrix_dimsRes[4] = {0};
  char * filename1 = "a.mat";
  read_matrix_dims(filename1, matrix_dims1, &n_elements1);
  matrix1 = (float *)calloc(n_elements1, sizeof(float));
  read_matrix_vals(filename1, matrix1, matrix_dims1,1);
  print_matrix(matrix1, matrix_dims1,1);

  char * filename2 = "b.mat";
  read_matrix_dims(filename2, matrix_dims2, &n_elements2);
  matrix2 = (float *)calloc(n_elements2, sizeof(float));
  read_matrix_vals(filename2, matrix2, matrix_dims2,1);
  print_matrix(matrix2, matrix_dims2,1);

  int m = matrix_dims1[2];
  int k = matrix_dims1[3];
  int n = matrix_dims2[3];
  printf("m:%d,k:%d,n:%d\n",m,k,n);
  matrixRes = (float *)calloc(m*n,sizeof(float));
  mm_cpu(matrix1,matrix2,matrixRes,m,n,k);

  matrix_dimsRes[2]=m;
  matrix_dimsRes[3]=n;
  print_matrix(matrixRes,matrix_dimsRes ,1);
  free(matrix1);
  free(matrix2);
  free(matrixRes);
}
