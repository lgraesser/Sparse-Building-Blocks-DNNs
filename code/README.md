To sample a sparse matrix use following

```bash
./generate_sparse_mat.py 3,20,200,300 0.9 > test.mat
less test.mat
```

To compile matrix_io and load a sample matrix

```bash
gcc -o io matrix_io.c
./generate_sparse_mat.py 3,2,5,10 0.4 > test.mat
./io
```

Otherwise to use the functions just include matrix_io.h and follow the following recipe to load a matrix

```C
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
// Read and convert matrices
read_matrix_vals(filename, matrix, matrix_dims);
```
