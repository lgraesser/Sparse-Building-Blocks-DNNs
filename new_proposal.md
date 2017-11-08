## New Proposal
We decided to to focus in fully connected parts of trained networks.

- We will choose fully connected parts of following networks and write c code to read them into various C formats.
- We need to write code fore doing basic feed forward capability.
- Simulating pruning and comparing it with normal pruning(if you have time). You can compare different simulations and see how pruning statistics effect performance.


### [Highly Parallel Sparse Matrix-Matrix Multiplication, Buluc, Gilbert](https://arxiv.org/pdf/1006.2183.pdf)

- " One of the key linear-algebraic primitives for graph algorithms is computing the product of two sparse matrices (SpGEMM) over a semiring."

### [Fast Sparse Martix Multiplication](https://ac.els-cdn.com/001046559290116G/1-s2.0-001046559290116G-main.pdf?_tid=5317d530-c405-11e7-b03e-00000aab0f26&acdnat=1510091460_0ee8c2533767fe6235271a52ba4a38e1)

### [A work-efficient parallel sparse matrix-sparse vector multiplication algorithm, Azad,  Buluc](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7967159)

### [Sparse Matrix Sparse Vector Multiplication - A Novel Approach](http://ieeexplore.ieee.org/document/7349895/)


### [Sparse Matrix Sparse Matrix multiplication](https://devtalk.nvidia.com/default/topic/744976/problem-of-two-large-sparse-matrices-multiplication-in-cuparse-/)


http://docs.nvidia.com/cuda/cusparse/index.html
```
int baseC, nnzC;
// nnzTotalDevHostPtr points to host memory
int *nnzTotalDevHostPtr = &nnzC;
cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
cusparseXcsrgemmNnz(handle, transA, transB, m, n, k,
        descrA, nnzA, csrRowPtrA, csrColIndA,
        descrB, nnzB, csrRowPtrB, csrColIndB,
        descrC, csrRowPtrC, nnzTotalDevHostPtr );
if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;
}else{
    cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
    nnzC -= baseC;
}
cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
cudaMalloc((void**)&csrValC, sizeof(float)*nnzC);
cusparseScsrgemm(handle, transA, transB, m, n, k,
        descrA, nnzA,
        csrValA, csrRowPtrA, csrColIndA,
        descrB, nnzB,
        csrValB, csrRowPtrB, csrColIndB,
        descrC,
        csrValC, csrRowPtrC, csrColIndC);
```
