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

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* ======================== Function declarations=========================== */
void read_matrix_vals(char *, float *, int []);
void read_matrix_dims(char *, int []);
void print_matrix(float *, int []);
void write_matrix(char *, float *, int []);
/* ========================================================================= */


int main(int argc, char * argv[])
{
  // TODO: add filename read in

  float * matrix;
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
  read_matrix_vals(filename, matrix, matrix_dims);
  print_matrix(matrix, matrix_dims);
  free(matrix);
}


void read_matrix_dims(char * filename, int matrix_dims[])
{
  FILE *fp = fopen(filename, "r");
  size_t len = 0;
  ssize_t read;
  char *line = NULL;
  read = getline(&line, &len, fp);
  printf("%s", line);
  int dim = atoi(&line[6]);
  printf("DIM: %d\n", dim);
  int i;
  int offset = 4 - dim;
  for (i = 0; i < dim; i ++)
  {
    read = getline(&line, &len, fp);
    printf("%d: %s", i, line);
    matrix_dims[i + offset] = atoi(&line[5]);
  }
  printf("Matrix dimensions: [%d, %d, %d, %d]\n",
                              matrix_dims[0],
                              matrix_dims[1],
                              matrix_dims[2],
                              matrix_dims[3]);
  free(line);
}

void read_matrix_vals(char * filename, float * matrix, int matrix_dims[])
{
  FILE *fp = fopen(filename, "r");
  size_t len = 0;
  ssize_t read;
  char *line = NULL;
    // Discard matrix descriptor rows and gather matrix stats
  read = getline(&line, &len, fp);
  int k;
  for (k = 0; k < 4; k++)
  {
    if (matrix_dims[k] != 0)
    {
      read = getline(&line, &len, fp);
    }
  }
  // Read planes
  int s, ch, i, j;
  for (s = 0; s < MAX(matrix_dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(matrix_dims[1], 1); ch++)
    {
      // Read Matrix breaker line
      read = getline(&line, &len, fp);
      // printf("Line: %s\n", line);
      for (i = 0; i < matrix_dims[2]; i++)
      {
        // Read row
        read = getline(&line, &len, fp);
        // printf("Row length: %d Line: %s\n", len, line);
        char * token;
        char * delim = ",";
        token = strtok(line, delim);
        for (j = 0; j < matrix_dims[3]; j++)
        {
          // printf("[%d, %d, %d, %d]\n", s, ch, i, j);
          float tok_f = atof(token);
          matrix[index4D(s, ch, i, j, matrix_dims[1], matrix_dims[2], matrix_dims[3])] = tok_f;
          token = strtok(NULL, delim);
        }
      }
    }
  }
  free(line);
}

void print_matrix(float * matrix, int matrix_dims[])
{
  printf("Matrix dimensions: [%d, %d, %d, %d]\n",
                              matrix_dims[0],
                              matrix_dims[1],
                              matrix_dims[2],
                              matrix_dims[3]);
  // Write planes
  int s, ch, i, j;
  for (s = 0; s < MAX(matrix_dims[0], 1); s++)
  {
    for (ch = 0; ch < MAX(matrix_dims[1], 1); ch++)
    {
      printf("(Sample, Channel) = (%d, %d)\n", s, ch);
      for (i = 0; i < matrix_dims[2]; i++)
      {
        for (j = 0; j < matrix_dims[3]; j++)
        {
          printf("%05.2f ", matrix[index4D(s, ch, i, j,
            matrix_dims[1], matrix_dims[2], matrix_dims[3])]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}
