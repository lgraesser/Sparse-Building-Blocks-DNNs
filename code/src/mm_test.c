
/*
 * Matrix multiplication experiments.
 */
#include "matrix_io.h"
#include "mm.h"
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <unistd.h>
#include <ctype.h>

// #define DEBUG

int main(int argc, char * argv[])
{
  struct Matrix matrix1, matrix2, matrixResCPU, matrixResGPU;
  int n_elements;
  const char * filename1;
  const char * filename2;
  double time_taken;
  clock_t start, end;
  float diff;

 int time_flag = 0;
 int repetation_flag = 0;
 int correctness_check_flag = 0;
 char *alg_type_flag = NULL;
 int c,i;

 opterr = 0;

 while ((c = getopt (argc, argv, "tca:")) != -1)
   switch (c)
     {
     case 't':
       time_flag = 1;
       break;
     case 'c':
       correctness_check_flag = 1;
       break;
     case 'r':
       repetation_flag = 1;
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

 printf ("time_flag = %d, correctness_check_flag = %d, repetation_flag = %d, alg_type_flag = %s\n",
         time_flag, correctness_check_flag, repetation_flag, alg_type_flag);

 for (i = optind; i < argc; i++)
   printf ("Non-option argument %s\n", argv[i]);

printf("optind:%d,argc:%d\n",optind,argc);
  if (argc-optind != 2){
    printf("usage ./mm matrixA matrixB\n");
    printf("Default values are going to be used ./mm data/a.mat data/b.mat\n");
    filename1 = "data/a.mat";
    filename2 = "data/b.mat";
  }
  else{
    filename1 = argv[optind];
    filename2 = argv[optind+1];
  }
  cudaFree(0);
  read_matrix_dims(filename1, &matrix1, &n_elements);
  matrix1.vals = (float *)calloc(n_elements, sizeof(float));
  read_matrix_vals(filename1, &matrix1,1);
  #ifdef DEBUG
    print_matrix(&matrix1);
  #endif


  read_matrix_dims(filename2, &matrix2, &n_elements);
  matrix2.vals = (float *)calloc(n_elements, sizeof(float));
  read_matrix_vals(filename2, &matrix2,1);
  #ifdef DEBUG
    print_matrix(&matrix2);
  #endif

  int m = matrix1.dims[2];
  int k = matrix1.dims[3];
  int n = matrix2.dims[3];

  if (correctness_check_flag){
    //cpu_mm
    initiliaze2dMatrix(&matrixResCPU,m,n);
    start = clock();
    cpu_mm(&matrix1,&matrix2,&matrixResCPU,m,n,k);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for the cpu_mm is %lf\n", time_taken);

    #ifdef DEBUG
      printf("Cpu result:\n");
      print_matrix(&matrixResCPU);
    #endif
  }

  if (strcmp(alg_type_flag, "denseblas") == 0)
  {
    ///gpu_mm_dense
    // Create a handle for CUBLAS
    initiliaze2dMatrix(&matrixResGPU,m,n);
    cublasHandle_t handleBLAS;
    cublasCreate(&handleBLAS);

    start = clock();
    gpu_mm_dense(&matrix1,&matrix2,&matrixResGPU,m,n,k,handleBLAS,time_flag);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for the gpu_mm_dense is %lf\n", time_taken);
    #ifdef DEBUG
      printf("CuBLAS result:\n");
      print_matrix(&matrixResGPU);
    #endif
    if (correctness_check_flag){
      diff = calculateDistanceMatrix(&matrixResCPU,&matrixResGPU);
      if (diff>1e-7){
          printf("Diff is %.8f\n",diff);
          printf("There might be a problem\n");
      }
      else{
          printf("Diff is less then 1e-7\n",diff);
      }
    }
    destroyMatrix(&matrixResGPU);
    cublasDestroy(handleBLAS);
  }
  else if (strcmp(alg_type_flag, "cusparse") == 0)
  {
    ///gpu_mm_sparse
    // Initialize cusparse library
    initiliaze2dMatrix(&matrixResGPU,m,n);
    cusparseHandle_t handleSparse;
    cusparseCreate(&handleSparse);

    start = clock();
    gpu_mm_sparse(&matrix1,&matrix2,&matrixResGPU,m,n,k,handleSparse,time_flag);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for gpu_mm_sparse is %lf\n", time_taken);
    #ifdef DEBUG
      printf("cuSparse result:\n");
      print_matrix(&matrixResGPU);
    #endif
    if (correctness_check_flag){
      diff = calculateDistanceMatrix(&matrixResCPU,&matrixResGPU);
      if (diff>1e-7){
          printf("Diff is %.8f\n",diff);
          printf("There might be a problem\n");
      }
      else{
          printf("Diff is less then 1e-7\n",diff);
      }
    }
    cusparseDestroy(handleSparse);
    destroyMatrix(&matrixResGPU);
  }
  else if (strcmp(alg_type_flag, "sparseimp") == 0)
  {
    //gpu_mm_sparse_ProjectImp
    initiliaze2dMatrix(&matrixResGPU,m,n);
    cusparseHandle_t handleSparse;
    cusparseCreate(&handleSparse);
    start = clock();
    gpu_mm_sparse_ProjectImp(&matrix1,&matrix2,&matrixResGPU,m,n,k,handleSparse,time_flag);
    end = clock();
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for gpu_mm_sparse_ProjectImp is %lf\n", time_taken);
    #ifdef DEBUG
      printf("cuSparse result:\n");
      print_matrix(&matrixResGPU);
    #endif
    if (correctness_check_flag){
      diff = calculateDistanceMatrix(&matrixResCPU,&matrixResGPU);
      if (diff>1e-7){
          printf("Diff is %.8f\n",diff);
          printf("There might be a problem\n");
      }
      else{
          printf("Diff is less then 1e-7\n",diff);
      }
    }
    cusparseDestroy(handleSparse);
    destroyMatrix(&matrixResGPU);
  }
  else{
    printf("Use denseblas/cusparse/sparseimp with -a flag.\n");
  }

  destroyMatrix(&matrix1);
  destroyMatrix(&matrix2);
  if (correctness_check_flag){
    destroyMatrix(&matrixResCPU);
  }
}
