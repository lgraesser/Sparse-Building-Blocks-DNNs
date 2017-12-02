#ifndef _MATRIX_IO_H__
#define _MATRIX_IO_H__
#include "indexing_defs.h"
/*
 * Header file for matrix reading, writing, and conversion
 * Read and write operations for matrices of 2, 3, and 4 dimensions
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * Matrices are read in and stored in row major order
 * However, cuSPARSE assumes matrices are stored in column major order
 * Use convert_to_column_major to convert the matrix
 */
 struct Matrix {
   float *vals;
   int dims[4];
   char is_column_first;
 };

/* ======================== Function declarations=========================== */
void read_matrix_vals(const char * filename, struct Matrix *mat,int is_column_first_flag);
void read_matrix_dims(const char * filename, struct Matrix *mat, int* product);
void print_matrix(struct Matrix *mat);
void convert_to_column_major(struct Matrix *matrix_row_major,
                             struct Matrix *matrix_col_major);
int isMatricesHaveSameDim(struct Matrix *matrix_row_major,
                            struct Matrix *matrix_col_major);
float calculateDistanceMatrix(struct Matrix *matrix1,struct Matrix *matrix2);
void initiliaze2dMatrix(struct Matrix *mat,int nRow,int nCol);
void destroyMatrix(struct Matrix *mat);

// void convert_dense_to_coo_4D(float *, float *, int *, int *, int *, int *, int [])
/* ========================================================================= */
#endif
