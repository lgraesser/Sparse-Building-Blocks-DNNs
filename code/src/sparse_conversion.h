#ifndef _SPARSE_CONVERSION_H__
#define _SPARSE_CONVERSION_H__
/*
 * Header file for conversion between sparse and dense matrices
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * cuSPARSE assumes matrices are stored in column major order
 */

#include <cuda.h>
#include <cusparse.h>
#include "matrix_io.h"

const float SMALL_NUM = 0.0000000001;
struct SparseMat {
  cusparseMatDescr_t descrA;
  int * csrRowPtrA;
  int * csrColIndA;
  float * csrValA;
  int * nz_per_row;
  int * csrRowPtrA_device;
  int * csrColIndA_device;
  float * csrValA_device;
  int * nz_per_row_device;
  int total_non_zero;
};

void convert_to_sparse(
        struct SparseMat *,
        struct Matrix *,
        cusparseHandle_t);
void copyDeviceCSR2Host(struct SparseMat *, struct Matrix *);
void destroySparseMatrix(struct SparseMat *);
void print_sparse_matrix(struct SparseMat, int);

#endif
