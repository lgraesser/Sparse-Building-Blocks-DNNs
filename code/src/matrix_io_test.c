#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix_io.h"

int main(int argc, char * argv[])
{
  // TODO: add filename read in
  struct Matrix matrix_row;
  struct Matrix matrix_col;
  const char * filename = "test.mat";
  int num_elems;
  read_matrix_dims(filename, &matrix_row, &num_elems);
  read_matrix_dims(filename, &matrix_col, &num_elems);
  // Allocate memory
  matrix_row.vals = (float *)calloc(num_elems, sizeof(float));
  matrix_col.vals = (float *)calloc(num_elems, sizeof(float));

  // Read and convert matrices
  read_matrix_vals(filename, &matrix_row,0);
  read_matrix_vals(filename, &matrix_col,1);
  print_matrix(&matrix_row);
  print_matrix(&matrix_col);

  // Check row major and col major layout
  int k;
  printf("Row major matrix\n");
  for (k = 0; k < num_elems; k++)
  {
    printf("%05.2f ", matrix_row.vals[k]);
  }
  printf("\n");
  printf("Column major matrix\n");
  for (k = 0; k < num_elems; k++)
  {
    printf("%05.2f ", matrix_col.vals[k]);
  }
  printf("\n");

  // Free memory
  free(matrix_col.vals);
  free(matrix_row.vals);
}
