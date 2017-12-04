#ifndef _SAFE_CALLS_H__
#define _SAFE_CALLS_H__

#include <stdio.h>
#include <cudnn.h>
#include <cusparse_v2.h>

#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cusparseSafeCall( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )
#define checkCUDNN( err ) __checkCUDNN( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


inline void __checkCUDNN( cudnnStatus_t stat, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUDNN_STATUS_SUCCESS != stat )
    {
        fprintf( stderr, "checkCUDNN() failed at %s:%i : %d\n",
                 file, line, stat);
        exit( -1 );
    }
#endif

    return;
}

inline void __cusparseSafeCall( cusparseStatus_t stat, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUSPARSE_STATUS_SUCCESS != stat )
    {
        fprintf( stderr, "cusparseSafeCall() failed at %s:%i : %d\n",
                 file, line, stat);
        exit( -1 );
    }
#endif

    return;
}

/* Source: https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/ */
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}
#endif
