
/*
 * Matrix multiplication experiments.
 */
#include "matrix_io.h"
#include "mm.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <cuda.h>
#include <cudnn.h>
#include <cusparse_v2.h>
#include "conv.h"
#include "safe_call_defs.h"
#include "sparse_conversion.h"
#include "matrix_io.h"

// #define DEBUG

int main(int argc, char * argv[])
{
  struct Matrix row_mat;
  struct Matrix k_mat;
  struct Kernel kernel;
  struct Matrix col_kernel;
  struct Matrix dense_out;
  int k_elems;
  int num_elems;
  const char * filename1;
  const char * filename2;
  double time_taken;
  clock_t start, end;

 int iterate_flag = 0;
 char* alg_type_flag;
 int c,i;

 opterr = 0;

 while ((c = getopt (argc, argv, "an:")) != -1)
   switch (c)
     {
     case 'n':
       iterate_flag = 1;
       break;
     case 'a':
       alg_type_flag = optarg;
       break;
     case '?':
       if (optopt == 'a')
         fprintf (stderr, "Option -%c requires an argument.\n", optopt);
       else if (isprint (optopt))
         fprintf (stderr, "Unknown option `-%c'.\n", optopt);
       else
         fprintf (stderr,
                  "Unknown option character `\\x%x'.\n",
                  optopt);
       return 1;
     default:
       abort ();
     }

 printf ("iterations_flag = %d, alg_type_flag = %c\n",
         iterate_flag, alg_type_flag);

 for (i = optind; i < argc; i++)
   printf ("Non-option argument %s\n", argv[i]);

printf("optind:%d,argc:%d\n",optind,argc);


  if (argc-optind != 2){
    printf("usage ./mm matrix kernel\n");
    printf("Default values are going to be used ./mm data/a.mat data/k.mat\n");
    filename1 = "data/a.mat";
    filename2 = "data/k.mat";
  }
  else{
    filename1 = argv[optind];
    filename2 = argv[optind+1];
  }
  cudaFree(0);
  // Read in matrix and kernel
  read_matrix_dims(filename1, &row_mat, &num_elems);
  row_mat.vals = (float *)calloc(num_elems, sizeof(float));
  read_matrix_vals(filename1, &row_mat, 0);
  #ifdef DEBUG
    print_matrix(&rowmat);
  #endif

  read_matrix_dims(filename2, &k_mat, &k_elems);
  k_mat.vals = (float *)calloc(k_elems, sizeof(float));
  read_matrix_vals(filename2, &k_mat, 0);
  #ifdef DEBUG
    print_matrix(&k_mat);
  #endif

  kernel.vals = (float *)calloc(k_elems, sizeof(float));
  kernel.vals = k_mat.vals;
  for (int i = 0; i < 4; i++)
  {
    kernel.dims[i] = k_mat.dims[i];
  }
  kernel.is_column_first = k_mat.is_column_first;
  kernel.is_on_device = 0;

  /// Convolve dense, no pitch
  int num_its = 1;
  if (iterate_flag == 1)
  {
    num_its = 1000;
  }
  printf("Number of iterations: %d\n", num_its);
  start = clock();
  if (alg_type_flag == 'a')
  {
    printf(" ============= TILED CONVOLUTION ================= \n");
    convolve2DDenseProjectImp(&row_mat,
                              &kernel,
                              &dense_out,
                              0,
                              num_its);
  }
  else if (alg_type_flag == 'b')
  {
    printf(" ============= TILED, PITCHED, CONVOLUTION ================= \n");
    convolve2DDenseProjectImp(&row_mat,
                              &kernel,
                              &dense_out,
                              1,
                              num_its);
  }
  else if (alg_type_flag == 'c')
  {
    printf(" =============  CUDNN CONVOLUTION ================= \n");
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    convolve2DDense(&row_mat,
                    &kernel,
                    &dense_out, // Not initialized
                    cudnn,
                    num_its);
    cudnnDestroy(cudnn);
  }
  else if (alg_type_flag == 'd')
  {
    printf(" ============= SPARSE KERNEL, TILED CONVOLUTION =============== \n");
    clock_t handle_start = clock();
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    clock_t handle_end = clock();
    time_taken = ((double)(handle_end - handle_start))/ CLOCKS_PER_SEC;
    printf("Time taken to convert matrix: %lf \n",time_taken);
    // Convert kernel to sparse matrix
    clock_t convert_start = clock();
    for (int i = 0; i < 4; i++)
    {
      col_kernel.dims[i] = kernel.dims[i];
    }
    num_elems = kernel.dims[2] * kernel.dims[3];
    col_kernel.vals = (float *)calloc(num_elems, sizeof(float));
    convert_to_column_major(&k_mat, &col_kernel);
    struct SparseMat spm;
    spm.is_on_device = 0;
    convert_to_sparse(&spm, &col_kernel, handle);
    printf("Spm on device? %d\n", spm.is_on_device);
    clock_t convert_end = clock();
    clock_t func_start = clock();
    convolve2DSparseProjectImp(&row_mat,
                              &spm,
                              &dense_out,
                              0,
                              num_its);
    clock_t func_end = clock();
    time_taken = ((double)(convert_end - convert_start))/ CLOCKS_PER_SEC;
    printf("Time taken to convert matrix: %lf \n",time_taken);
    time_taken = ((double)(func_end - func_start))/ CLOCKS_PER_SEC;
    printf("Time taken to convolve matrix: %lf \n",time_taken);
    destroySparseMatrix(&spm);
    cusparseDestroy(handle);
  }
  else if (alg_type_flag == 'e')
  {
    printf(" ============= SPARSE KERNEL, TILED CONVOLUTION =============== \n");
    clock_t handle_start = clock();
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    clock_t handle_end = clock();
    time_taken = ((double)(handle_end - handle_start))/ CLOCKS_PER_SEC;
    printf("Time taken to convert matrix: %lf \n",time_taken);
    // Convert kernel to sparse matrix
    clock_t convert_start = clock();
    for (int i = 0; i < 4; i++)
    {
      col_kernel.dims[i] = kernel.dims[i];
    }
    num_elems = kernel.dims[2] * kernel.dims[3];
    col_kernel.vals = (float *)calloc(num_elems, sizeof(float));
    convert_to_column_major(&k_mat, &col_kernel);
    struct SparseMat spm;
    spm.is_on_device = 0;
    convert_to_sparse(&spm, &col_kernel, handle);
    printf("Spm on device? %d\n", spm.is_on_device);
    clock_t convert_end = clock();
    clock_t func_start = clock();
    convolve2DSparseProjectImp(&row_mat,
                              &spm,
                              &dense_out,
                              1,
                              num_its);
    clock_t func_end = clock();
    time_taken = ((double)(convert_end - convert_start))/ CLOCKS_PER_SEC;
    printf("Time taken to convert matrix: %lf \n",time_taken);
    time_taken = ((double)(func_end - func_start))/ CLOCKS_PER_SEC;
    printf("Time taken to convolve matrix: %lf \n",time_taken);
    destroySparseMatrix(&spm);
    cusparseDestroy(handle);
  }
  else
  {
    printf("Unrecognised algorithm. Exiting...\n");
    exit(1);
  }
  end = clock();
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("Time taken is %lf\n", time_taken);
  printf("Time taken per iteration %lf\n", time_taken / (num_its * 1.0));

  #ifdef DEBUG
    printf("Convolution result:\n");
    print_matrix(&dense_out);
  #endif
  destroyMatrix(&dense_out);

  // Free memory
  destroyMatrix(&col_kernel);
  destroyMatrix(&row_mat);
  destroyKernel(&kernel, &k_mat);
}
