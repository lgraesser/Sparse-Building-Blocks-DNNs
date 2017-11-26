/*
 * Header file for matrix reading, writing, and conversion
 * Read and write operations for matrices of 2, 3, and 4 dimensions
 * Matrices assumed to be generated using generate_sparse_mat.py
 *
 * Matrices are read in and stored in row major order
 * However, cuSPARSE assumes matrices are stored in column major order
 * Use convert_to_column_major to convert the matrix
 */

/* ======================== Function declarations=========================== */
void read_matrix_vals(char *, float *, int [],char);
void read_matrix_dims(char *, int []);
void print_matrix(float *, int [],char);
void write_matrix(char *, float *, int []);
void convert_to_column_major(float *, float *, int []);
void convert_to_row_major(float *, float *, int []);
// void convert_dense_to_coo_4D(float *, float *, int *, int *, int *, int *, int [])
/* ========================================================================= */
