#include "mm.h"
#include "matrix_io.h"
#include "indexing_defs.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_mm_dense(struct MMatrix *h_A, struct MMatrix *h_B, struct MMatrix *h_C, const int m, const int n, const int k) {
   int lda=m,ldb=k,ldc=m;
   const float alf = 1;
   const float bet = 0;
   const float *alpha = &alf;
   const float *beta = &bet;
   float *d_A, *d_B, *d_C;
   cudaMalloc(&d_A,m*k*sizeof(float));
   cudaMalloc(&d_B,k*n*sizeof(float));
   cudaCalloc(&d_C,m*n*sizeof(float));

   cudaMemcpy(d_A,h_A->vals,m*k * sizeof(float),cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,h_B->vals,k*n * sizeof(float),cudaMemcpyHostToDevice);

   // Create a handle for CUBLAS
   cublasHandle_t handle;
   cublasCreate(&handle);

   // Do the actual multiplication
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

   cudaMemcpy(h_C->vals,d_C,m*n * sizeof(float),cudaMemcpyDeviceToHost);
   h_C->is_column_first=1;
   // Destroy the handle
   cublasDestroy(handle);
}

void cpu_mm(struct MMatrix *h_A, struct MMatrix *h_B, struct MMatrix *h_C, int m, int n, int k){
    int i,j,r;
    float c_sum;
    for(i=0;i<m;i++){
      for(j=0;j<n;j++){
        c_sum = 0;
        for(r=0;r<k;r++){
            c_sum += h_A->vals[index2DCol(i,r,m)]*h_B->vals[index2DCol(r,j,k)];
        }
        h_C->vals[index2DCol(i,j,m)] = c_sum;
      }
    }
    h_C->is_column_first=1;
}
