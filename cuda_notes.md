## Notes on CUDA ###


### cuSPARSE

 Supports float, double, cuComplex, and cuDoubleComplex

 Naming conventions
```C
 cusparse<t>[<matrix data format>]<operation>[<output matrix data format>]

 <t> : S, D, C, Z, X
 <matrix data format>: dense, coo, csr, csc, hyb
```

 All functions have return type `cusparseStatus_t`

 Execution is asynchronous, although cudaMemcpy() (either device to host or vice versa) is blocking with respect to the host

 **Compilation**
 `nvcc myCusparseApp.c  -lcusparse  -o myCusparseApp`

**Usage**
Must call `cusparseCreate()` to initialize the library before calling any cuSPARSE function
Call `cusparseDestroy()` at end


 **Streams**
 To achieve the overlap of computation between the tasks, the developer should create CUDA streams using the function cudaStreamCreate() and set the stream to be used by each individual cuSPARSE library routine by calling cusparseSetStream() just before calling the actual cuSPARSE routine.

**Matrix formats**
- Dense: Assumed to be column major
- COO: Sparse format, matrix assumed to be stored in row major format
- CSR: Sparse format, row major
- CSC: Sparse format, column major
- Ellpack-Itpack: Sparse format,
- Hybrid: Sparse format, mix of a regular part, stored in ELL format, and an irregular part, usually stored in COO format
- BSR: block compressed sparse row format (might be good for convolutions, if block = kernel size)


### cuDNN

Call `cudnnCreate()` and `cudnnDestroy()` at beginning and end

**Useful functions**

```C
cudnnStatus_t cudnnCreateTensorDescriptor(
    cudnnTensorDescriptor_t *tensorDesc)
```

This function creates a generic tensor descriptor object by allocating the memory needed to hold its opaque structure. The data is initialized to be all zero.

```C
cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t     format,
    cudnnDataType_t         dataType,
    int                     n,
    int                     c,
    int                     h,
    int                     w)
```

This function initializes a previously created generic Tensor descriptor object into a 4D tensor.

```C
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc)
```

This function destroys a previously created tensor descriptor object. When the input pointer is NULL, this function performs no destroy operation


```C
cudnnStatus_t cudnnCreateConvolutionDescriptor(
    cudnnConvolutionDescriptor_t *convDesc)
```

This function creates a convolution descriptor object by allocating the memory needed to hold its opaque structure

```C
cudnnStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t    convDesc,
    int                             pad_h,
    int                             pad_w,
    int                             u,
    int                             v,
    int                             dilation_h,
    int                             dilation_w,
    cudnnConvolutionMode_t          mode,
    cudnnDataType_t                 computeType)
```

This function initializes a previously created convolution descriptor object into a 2D correlation. This function assumes that the tensor and filter descriptors corresponds to the formard convolution path and checks if their settings are valid. That same convolution descriptor can be reused in the backward path provided it corresponds to the same layer.


```C
cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t                      handle,
    const cudnnTensorDescriptor_t      xDesc,
    const cudnnFilterDescriptor_t      wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t      yDesc,
    const int                          requestedAlgoCount,
    int                               *returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t     *perfResults)
```
This function attempts all cuDNN algorithms for cudnnConvolutionForward(), using memory allocated via cudaMalloc(), and outputs performance metrics to a user-allocated array of cudnnConvolutionFwdAlgoPerf_t. These metrics are written in sorted fashion where the first element has the lowest compute time.

Note: This function is host blocking.
Note: It is recommend to run this function prior to allocating layer data; doing otherwise may needlessly inhibit some algorithm options due to resource usage.
