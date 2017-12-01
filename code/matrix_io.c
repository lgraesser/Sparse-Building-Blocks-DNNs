/*
 * Read and write operations for matrices of 2, 3, and 4 dimensions
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * Matrices are read in and stored in row major order
 * However, cuSPARSE assumes matrices are stored in column major order
 * Use convert_to_column_major to convert the matrix
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix_io.h"
#include "indexing_defs.h"

/* To index a 2D, 3D or 4D array stored as 1D, in row major order */
/* Arrays always assumed to be S * K * M * N
 *  S: number of samples, s = index of a single sample
 *  K: number of channels, ch = index of a single channel
 *  M: number of rows, i = index of a single row
 *  N: number of columns, j = index of a single column
 */
int isMatricesHaveSameDim(struct Matrix *matrix_row_major,
                             struct Matrix *matrix_col_major)
 {
   for (int i =0;i<4;i++){
     if (matrix_row_major->dims[i] != matrix_col_major->dims[i]){
       return 0;
     }
   }
   return 1;
 }

void convert_to_column_major(struct Matrix *matrix_row_major,
                             struct Matrix *matrix_col_major)
{
  // Write planes
  int s, ch, i, j;
  // TODO check whether they have same dimensions.
  if (! isMatricesHaveSameDim(matrix_row_major,matrix_col_major)){
    printf("Matrix dimensions doesn't match\n");
    exit(1);
  }
  else if (matrix_row_major->is_column_first!=0){
    printf("First matrix should be row major ordered\n");
    exit(1);
  }

  //Otherwise lets change
  for (s = 0; s < MAX(matrix_row_major->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(matrix_row_major->dims[1], 1); ch++)
    {
      for (i = 0; i < matrix_row_major->dims[2]; i++)
      {
        for (j = 0; j < matrix_row_major->dims[3]; j++)
        {
          matrix_col_major->vals[index4DCol(s, ch, i, j,
            matrix_row_major->dims[1], matrix_row_major->dims[2], matrix_row_major->dims[3])] =
            matrix_row_major->vals[index4D(s, ch, i, j,
            matrix_row_major->dims[1], matrix_row_major->dims[2], matrix_row_major->dims[3])];
        }
      }
    }
  }
  matrix_col_major->is_column_first=1;
}


void convert_to_row_major(float * matrix_row_major,
                             float * matrix_col_major,
                             int matrix_dims[])
{
  // Write planes
  int s, ch, i, j;
  for (s = 0; s < MAX(matrix_dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(matrix_dims[1], 1); ch++)
    {
      for (i = 0; i < matrix_dims[2]; i++)
      {
        for (j = 0; j < matrix_dims[3]; j++)
        {
          matrix_row_major[index4D(s, ch, i, j,
            matrix_dims[1], matrix_dims[2], matrix_dims[3])] =
            matrix_col_major[index4DCol(s, ch, i, j,
            matrix_dims[1], matrix_dims[2], matrix_dims[3])];
        }
      }
    }
  }
}


void read_matrix_dims(const char * filename, int matrix_dims[],int* product)
{
  // Return the multiplication of the dimensions, a.k. number of elements
  FILE *fp = fopen(filename, "r");
  size_t len = 0;
  char *line = NULL;
  getline(&line, &len, fp);
  printf("%s", line);
  int dim = atoi(&line[6]);
  printf("DIM: %d\n", dim);
  int i;
  //initiliaze dimensions to 0
  for (i = 0; i < 4; i ++)
  {
    mat->dims[i]=0;
  }

  // lets read the dims
  *product = 1;
  int offset = 4 - dim;
  for (i = 0; i < dim; i ++)
  {
    getline(&line, &len, fp);
    printf("%d: %s", i, line);
    mat->dims[i + offset] = atoi(&line[5]);
    *product *= mat->dims[i + offset];
  }
  printf("Matrix dimensions: [%d, %d, %d, %d]\n",
                              mat->dims[0],
                              mat->dims[1],
                              mat->dims[2],
                              mat->dims[3]);
  free(line);
  fclose(fp);
}


void read_matrix_vals(const char * filename, struct Matrix *mat,int is_column_first_flag)
{
  FILE *fp = fopen(filename, "r");
  size_t len = 0;
  char *line = NULL;
    // Discard matrix descriptor rows and gather matrix stats
  getline(&line, &len, fp);
  int k;
  for (k = 0; k < 4; k++)
  {
    if (mat->dims[k] != 0)
    {
      // printf("Dim[%d]: %d\n", k,mat->dims[k]);
      getline(&line, &len, fp);
    }
  }
  // Read planes
  int s, ch, i, j;
  for (s = 0; s < MAX(mat->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(mat->dims[1], 1); ch++)
    {
      // Read Matrix breaker line
      getline(&line, &len, fp);
       // printf("Line: %s\n", line);
      for (i = 0; i < mat->dims[2]; i++)
      {
        // Read row
        getline(&line, &len, fp);
        // printf("Row length: %d Line: %s\n", len, line);
        char * token;
        const char * delim = ",";
        token = strtok(line, delim);
        for (j = 0; j < mat->dims[3]; j++)
        {
          // printf("[%d, %d, %d, %d]\n", s, ch, i, j);
          float tok_f = atof(token);
          if (is_column_first_flag){
            mat->vals[index4DCol(s, ch, i, j, mat->dims[1], mat->dims[2], mat->dims[3])] = tok_f;
          }
          else{
            mat->vals[index4D(s, ch, i, j, mat->dims[1], mat->dims[2], mat->dims[3])] = tok_f;
          }
          token = strtok(NULL, delim);
        }
      }
    }
  }
  mat->is_column_first = is_column_first_flag;
  free(line);
  fclose(fp);
}


void print_matrix(struct Matrix *mat)
{
  printf("Matrix dimensions: [%d, %d, %d, %d]\n",
                              mat->dims[0],
                              mat->dims[1],
                              mat->dims[2],
                              mat->dims[3]);
  // Write planes
  if (is_col_order_flag)
  {
    printf("Column major ordering\n");
  }
  else
  {
    printf("Row major ordering\n");
  }
  int s, ch, i, j;
  for (s = 0; s < MAX(mat->dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(mat->dims[1], 1); ch++)
    {
      printf("(Sample, Channel) = (%d, %d)\n", s, ch);
      if (mat->is_column_first)
      {
        for (j = 0; j < mat->dims[2]; j++)
        {
          for (i = 0; i < mat->dims[3]; i++)
          {
            printf("%05.2f ", mat->vals[index4DCol(s, ch, i, j,
              mat->dims[1], mat->dims[2], mat->dims[3])]);
            printf("\n");
          }
          printf("\n");
        }
      }
      else
      {
        for (i = 0; i < mat->dims[2]; i++)
        {
          for (j = 0; j < mat->dims[3]; j++)
          {
            printf("%05.2f ", mat->vals[index4D(s, ch, i, j,
              mat->dims[1], mat->dims[2], mat->dims[3])]);
            printf("\n");
          }
          printf("\n");
        }
      }
    }
  }
}
