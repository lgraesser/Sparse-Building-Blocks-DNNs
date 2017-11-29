/*
 * Header file for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>

/* To index a 2D, 3D or 4D array stored as 1D, in row major order */
/* Arrays always assumed to be S * K * M * N
 *  S: number of samples, s = index of a single sample
 *  K: number of channels, ch = index of a single channel
 *  M: number of rows, i = index of a single row
 *  N: number of columns, j = index of a single column
 */
#define index2D(i, j, N) ((i)*(N)) + (j)
#define index3D(ch, i, j, M, N) ((ch)*(M)*(N)) + ((i)*(N)) + (j)
#define index4D(s, ch, i, j, K, M, N) ((s)*(K)*(M)*(N)) + ((ch)*(M)*(N)) + ((i)*(N)) + (j)
#define index4DCol(s, ch, i, j, K, M, N) ((s)*(K)*(M)*(N)) + ((ch)*(M)*(N)) + ((j)*(M)) + (i)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

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
