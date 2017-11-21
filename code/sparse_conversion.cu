/*
 * File for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * Matrices are read in and stored in row major order
 * However, cuSPARSE assumes matrices are stored in column major order
 * Use convert_to_column_major to convert the matrix
 */

#include <cuda.h>
#include <cusparse.h>
#include "sparse_conversion.h"
#include "matrix_io.h"

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

int main(int argc, char * argv[])
{
  float * matrix;
  float * matrix_cols;
  int matrix_dims[4] = {0};
  char * filename = "test.mat";
  read_matrix_dims(filename, matrix_dims);
  // Allocate memory
  int num_elems = 1;
  int k;
  for (k = 0; k < 4; k++)
  {
    if (matrix_dims[k] != 0)
    {
      num_elems *= matrix_dims[k];
    }
  }
  matrix = (float *)calloc(num_elems, sizeof(float));
  matrix_cols = (float *)calloc(num_elems, sizeof(float));

  // Read and convert matrices
  read_matrix_vals(filename, matrix, matrix_dims);
  convert_to_column_major(matrix, matrix_cols, matrix_dims);

  // Initialize cusparse library
  cusparseHandle_t * handle;
  cusparseCreate(handle);

  // Example for 2D matrix
  int i, j;
  int num_non_zero = 0;
  int * nz_per_row;
  nz_per_row = (int *)calloc(matrix_dims[2], sizeof(int));
  for (i = 0; i < matrix_dims[2], i++)
  {
    int nzpr = 0;
    for (j = 0; j < matrix_dims[3], j++)
    {
      if (matrix[index2D(i, j, matrix_dims[3])] < SMALL_NUM &&
          matrix[index2D(i, j, matrix_dims[3])] > -SMALL_NUM &&)
      {
        num_non_zero++;
        nzpr++;
      }
      nz_per_row[i] = nzpr;
    }
  }
  int * csrRowPtrA;
  int * csrColIndA;
  float *csrValA;
  csrRowPtrA = (int *)calloc(matrix_dims[2] + 1, sizeof(int));
  csrColIndA = (int *)calloc(num_non_zero, sizeof(int));
  csrColValA = (float *)calloc(num_non_zero, sizeof(float));

  cusparseStatus_t result = cusparseSdense2csr(
                handle, // cusparse handle
                matrix_dims[2] // Number of rows
                matrix_dims[3] // Number of cols
                const cusparseMatDescr_t descrA,
                matrix, // Matrix
                matrix_dims[2], // Leading dimension of the array
                const int *nnzPerRow,
                float *csrValA,
                int *csrRowPtrA,
                int *csrColIndA);

  cusparseDestroy(handle);
  free(csrRowPtrA);
  free(csrColIndA);
  free(csrValA)
  free(nz_per_row);
  free(matrix);
  free(matrix_cols);
}
