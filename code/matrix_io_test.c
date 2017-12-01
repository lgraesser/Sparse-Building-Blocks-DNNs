#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix_io.h"
#include "indexing_defs.h"

int main(int argc, char * argv[])
{
  // TODO: add filename read in

  float * matrix;
  float * matrix_cols;
  int matrix_dims[4] = {0};
  const char * filename = "test.mat";
  int num_elems;
  read_matrix_dims(filename, matrix_dims,&num_elems);
  // Allocate memory
  matrix = (float *)calloc(num_elems, sizeof(float));
  matrix_cols = (float *)calloc(num_elems, sizeof(float));

  // Read and convert matrices
  read_matrix_vals(filename, matrix, matrix_dims,0);
  read_matrix_vals(filename, matrix_cols, matrix_dims,1);
  print_matrix(matrix, matrix_dims,0);
  print_matrix(matrix_cols, matrix_dims,1);

  // Check row major and col major layout
  int k;
  printf("Row major matrix\n");
  for (k = 0; k < num_elems; k++)
  {
    printf("%05.2f ", matrix[k]);
  }
  printf("\n");
  printf("Column major matrix\n");
  for (k = 0; k < num_elems; k++)
  {
    printf("%05.2f ", matrix_cols[k]);
  }
  printf("\n");

  // Free memory
  free(matrix);
  free(matrix_cols);
}
