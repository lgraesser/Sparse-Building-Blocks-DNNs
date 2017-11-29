#define cudaCalloc(A, SIZE) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, SIZE); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, SIZE); \
    } while (0)

void gpu_mm_dense(const float *h_A, const float *h_B, float *h_C, const int m, const int n, const int k);
void cpu_mm(float *a, float *b, float *c, int m, int n, int k);
