#include "mm.h"
#include "matrix_io.h"
#include "sparse_conversion.h"
#include "safe_call_defs.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_mm_dense(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k) {
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


void gpu_mm_sparse(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k) {
   cusparseOperation_t nop = CUSPARSE_OPERATION_NON_TRANSPOSE;
   // Initialize cusparse library
   cusparseHandle_t handle;
   cusparseCreate(&handle);
   struct SparseMat spmA,spmB,spmC;

   convert_to_sparse(&spmA, h_A, handle);
   convert_to_sparse(&spmB, h_B, handle);

   // init result sparse matrix
   cusparseCreateMatDescr(&(spmC.descr));
   cusparseSetMatType(spmC.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatIndexBase(spmC.descr, CUSPARSE_INDEX_BASE_ZERO);
   spmC.num_rows = m;
   CudaSafeCall(cudaMalloc(&(spmC.csrRowPtrA_device),
                         (spmC.num_rows + 1) * sizeof(int)));

   cusparseSafeCall( cusparseXcsrgemmNnz(handle, nop, nop, m, n, k,
                spmA.descr,spmA.total_non_zero,spmA.csrRowPtrA_device,spmA.csrColIndA_device,
                spmB.descr,spmB.total_non_zero,spmB.csrRowPtrA_device,spmB.csrColIndA_device,
                spmC.descr, spmC.csrRowPtrA_device, &spmC.total_non_zero ));

  CudaSafeCall(cudaMalloc(&(spmC.csrColIndA_device),
                        spmC.total_non_zero * sizeof(int)));
  CudaSafeCall(cudaMalloc(&(spmC.csrValA_device),
                        spmC.total_non_zero * sizeof(float)));
   // Do the actual multiplication
   cusparseSafeCall(cusparseScsrgemm(handle, nop, nop, m, n, k,
                    spmA.descr,spmA.total_non_zero,spmA.csrValA_device,spmA.csrRowPtrA_device,spmA.csrColIndA_device,
                    spmB.descr,spmB.total_non_zero,spmB.csrValA_device,spmB.csrRowPtrA_device,spmB.csrColIndA_device,
                    spmC.descr,spmC.csrValA_device,spmC.csrRowPtrA_device,spmC.csrColIndA_device));

  convert_to_dense(&spmC, h_C, handle);
  h_C->is_column_first=1;

   // Destroy the handle
  cusparseDestroy(handle);
  destroySparseMatrix(&spmA);
  destroySparseMatrix(&spmB);
  destroySparseMatrix(&spmC);
}

void cpu_mm(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, int m, int n, int k){
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
