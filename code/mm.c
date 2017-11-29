#include "mm.h"
#include "matrix_io.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_mm_dense(const float *h_A, const float *h_B, float *h_C, const int m, const int n, const int k) {
   int lda=m,ldb=k,ldc=m;
   const float alf = 1;
   const float bet = 0;
   const float *alpha = &alf;
   const float *beta = &bet;
   float *d_A, *d_B, *d_C;
   cudaMalloc(&d_A,m*k*sizeof(float));
   cudaMalloc(&d_B,k*n*sizeof(float));
   cudaCalloc(&d_C,m*n*sizeof(float));

   cudaMemcpy(d_A,h_A,m*k * sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,h_B,k*n * sizeof(float),cudaMemcpyHostToDevice);

   // Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);

   // Do the actual multiplication
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

   cudaMemcpy(h_C,d_C,m*n * sizeof(float),cudaMemcpyDeviceToHost);
   // Destroy the handle
   cublasDestroy(handle);
}

void cpu_mm(float *a, float *b, float *c, int m, int n, int k){
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
