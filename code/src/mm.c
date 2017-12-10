#include "mm.h"
#include "matrix_io.h"
#include "sparse_conversion.h"
#include "safe_call_defs.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void gpu_mm_dense(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k, cublasHandle_t handle) {
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

   // Do the actual multiplication
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);

   cudaMemcpy(h_C->vals,d_C,m*n * sizeof(float),cudaMemcpyDeviceToHost);
   h_C->is_column_first=1;
}


void gpu_mm_sparse(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k,cusparseHandle_t handle) {
   cusparseOperation_t nop = CUSPARSE_OPERATION_NON_TRANSPOSE;

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
  destroySparseMatrix(&spmA);
  destroySparseMatrix(&spmB);
  destroySparseMatrix(&spmC);
}

__global__ void sparseMM(float *a_csrVal, int *a_csrRowPtr, int *a_csrColInd,
   float *b_csrVal, int *b_csrRowPtr, int *b_csrColInd,
   float *c_matrix)
   /* blockDim.x-> n,  threadIdx.x-> j
      gridDim.x-> m, blockIdx.x-> i
      c_matrix -> m*n (columnbased)

      Each thread calculates i,j of matrix C.
      - To do that we look i th row of A
      - And j th row of B and add the matching values to the sum
   */
   {
     float sum = 0;
     int b_i,a_i,b_lim,a_lim,b_j,a_j;
     int id = index2DCol(blockIdx.x,threadIdx.x,gridDim.x);
     a_lim = a_csrRowPtr[blockIdx.x+1];
     b_lim = b_csrRowPtr[threadIdx.x+1];
     a_i = a_csrRowPtr[blockIdx.x];
     b_i = b_csrRowPtr[threadIdx.x];
     // if(blockIdx.x==0 && threadIdx.x==0){
     //   printf("Before:a_i=%d,b_i:%d,a_lim=%d,b_lim:%d\n",a_i,b_i,a_lim,b_lim);
     // }
     while((a_i<a_lim) && (b_i <b_lim))
        {
            b_j = b_csrColInd[b_i];
            a_j = a_csrColInd[a_i];
            // if(blockIdx.x==0 && threadIdx.x==0){
            //   printf("a_i=%d,b_i:%d,a_j=%d,b_j:%d\n",a_i,b_i,a_j,b_j);
            // }
            if ( a_j==b_j ){
              sum += a_csrVal[a_i]*b_csrVal[b_i];
            //   if(blockIdx.x==0 && threadIdx.x==0){
            //   printf("HIT:%f=%f*%f\n",a_csrVal[a_j]*b_csrVal[b_j],a_csrVal[a_j],b_csrVal[b_j]);
            // }
              a_i++;
              b_i++;
            }
            else if (a_j<b_j){
              a_i++;
            }
            else{
              b_i++;
            }
        }
        // if(blockIdx.x==0 && threadIdx.x==0){
        //   printf("sum:%d,id:%d\n",sum,id);
        // }
      c_matrix[id] = sum;
   }

void gpu_mm_sparse_ProjectImp(struct Matrix *h_A, struct Matrix *h_B, struct Matrix *h_C, const int m, const int n, const int k){
  struct Matrix h_B_transposed;
  transpose2dMatrix(h_B,&h_B_transposed);

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  struct SparseMat spmA,spmB;

  convert_to_sparse(&spmA, h_A, handle);
  // copyDeviceCSR2Host(&spmA);
  // print_sparse_matrix(&spmA);

  convert_to_sparse(&spmB, &h_B_transposed, handle);
  // copyDeviceCSR2Host(&spmB);
  // print_sparse_matrix(&spmB);

  int num_elems = h_C->dims[2] * h_C->dims[3];
  float * matrix_device;
  // Allocate device dense array
  CudaSafeCall(cudaMalloc(&matrix_device,
                        num_elems * sizeof(float)));

  sparseMM<<<m,n>>>(spmA.csrValA_device,spmA.csrRowPtrA_device,spmA.csrColIndA_device,
                      spmB.csrValA_device,spmB.csrRowPtrA_device,spmB.csrColIndA_device,matrix_device);

  CudaSafeCall(cudaMemcpy(h_C->vals,
                          matrix_device,
                          num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  h_C->is_column_first=1;
  // cudaFree(matrix_device);
  cusparseDestroy(handle);

 destroyMatrix(&h_B_transposed);
 destroySparseMatrix(&spmA);
 destroySparseMatrix(&spmB);
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
