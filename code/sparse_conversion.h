/*
 * Header file for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>

const float SMALL_NUM = 0.0000000001;
struct SparseMat {
  int * csrRowPtrA;
  int * csrColIndA;
  float * csrValA;
  const int * nz_per_row;
  int total_non_zero;
};

struct SparseMat * convert_to_sparse(float *, int [], cusparseHandle_t, const cusparseMatDescr_t);
void print_sparse_matrix(struct SparseMat *, int);
