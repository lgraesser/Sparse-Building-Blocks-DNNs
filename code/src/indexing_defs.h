#ifndef __INDEX_DEFS_H__
#define __INDEX_DEFS_H__

/*
 * Header file for matrix indexing
 */

 /* To index a 2D, 3D or 4D array stored as 1D, in row major order */
 /* Arrays always assumed to be S * K * M * N
  *  S: number of samples, s = index of a single sample
  *  K: number of channels, ch = index of a single channel
  *  M: number of rows, i = index of a single row
  *  N: number of columns, j = index of a single column
  */
 #define index2D(i, j, N) ((i)*(N)) + (j)
 #define index2DCol(i, j, M) ((j)*(M)) + (i)

 #define index3D(ch, i, j, M, N) ((ch)*(M)*(N)) + ((i)*(N)) + (j)
 #define index3DCol(ch, i, j, M, N) ((ch)*(M)*(N)) + ((j)*(M)) + (i)

 #define index4D(s, ch, i, j, K, M, N) ((s)*(K)*(M)*(N)) + ((ch)*(M)*(N)) + ((i)*(N)) + (j)
 #define index4DCol(s, ch, i, j, K, M, N) ((s)*(K)*(M)*(N)) + ((ch)*(M)*(N)) + ((j)*(M)) + (i)

 #define MAX(x, y) (((x) > (y)) ? (x) : (y))
 #define MIN(x, y) (((x) < (y)) ? (x) : (y))
 #define ABS(x) ((x)<0 ? (-x) : (x))

#endif
