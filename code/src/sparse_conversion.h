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
  int * csrRowPtrA_device;
  int * csrColIndA_device;
  float * csrValA_device;
  int total_non_zero;
  int num_rows;
  int is_on_device;
};

void convert_to_sparse(
        struct SparseMat *,
        struct Matrix *,
        cusparseHandle_t);
void convert_to_dense(
        struct SparseMat *,
        struct Matrix *,
        cusparseHandle_t);
void copyDeviceCSR2Host(struct SparseMat *);
void destroySparseMatrix(struct SparseMat *);
void print_sparse_matrix(struct SparseMat *);

#endif
