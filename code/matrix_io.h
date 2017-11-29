/*
 * Header file for matrix reading, writing, and conversion
 * Read and write operations for matrices of 2, 3, and 4 dimensions
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * Matrices are read in and stored in row major order
 * However, cuSPARSE assumes matrices are stored in column major order
 * Use convert_to_column_major to convert the matrix
 */

 #define index2D(i, j, N) ((i)*(N)) + (j)
 #define index2DCol(i, j, M) ((j)*(M)) + (i)

 #define index3D(ch, i, j, M, N) ((ch)*(M)*(N)) + ((i)*(N)) + (j)
 #define index4D(s, ch, i, j, K, M, N) ((s)*(K)*(M)*(N)) + ((ch)*(M)*(N)) + ((i)*(N)) + (j)
 #define index4DCol(s, ch, i, j, K, M, N) ((s)*(K)*(M)*(N)) + ((ch)*(M)*(N)) + ((j)*(M)) + (i)

 #define MAX(x, y) (((x) > (y)) ? (x) : (y))
 #define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* ======================== Function declarations=========================== */
void read_matrix_vals(const char *, float *, int [],char);
void read_matrix_dims(const char *, int [],int *);
void print_matrix(float *, int [],char);
void write_matrix(char *, float *, int []);
void convert_to_column_major(float *, float *, int []);
void convert_to_row_major(float *, float *, int []);
// void convert_dense_to_coo_4D(float *, float *, int *, int *, int *, int *, int [])
/* ========================================================================= */
