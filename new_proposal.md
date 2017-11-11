## New Proposal

### Why we are changing project

Automatic tuning of sparse formats based on the sparsity characteristics of matrices appears well covered (please see a selection of papers below).

Given this, We decided to focus on sparse matrix\*sparse matrix(spMspM) multiplications and particularly in the context of deep learning. We believe in the importance of optimizing spM\*spM multiplications in the context of deep-learning-research and development. Deep learning algorithms are over-parameterized and their parameters can typically be pruned down to 5-10% non zeros per matrix. Training big networks and using them for inference consists of many matrix-matrix multiplications. These matrices may be very sparse.

- [Automatic selection of sparse matrix representation on GPUs,  Sedaghati et al, 2015](http://web.cse.ohio-state.edu/~pouchet.2/doc/ics-article.15b.pdf)
- [Accelerating Sparse Matrix Vector Multiplication in Iterative Methods Using GPU](http://ieeexplore.ieee.org/document/6047229/)
- [SMAT: An input adaptive auto-tuner for sparse matrix-vector multiplication, Li et al, 2013](https://arxiv.org/abs/1210.2536)
- [A lightweight optimization selection method for Sparse Matrix-Vector Multiplication, Jan 2016](https://arxiv.org/pdf/1511.02494.pdf)
- clSpMV framework. This approach is restricted to GPUs.

### Problem setting:
- Neural networks an essential component of the state of the art machine learning models for object detection, object localization, semantic segmentation, image captioning, machine translation, speech recognition, etc.
- At their heart, neural networks are chains of matrix multiplications, interspersed with simple functions applied to the results.
- Optimizing neural networks is not that well understood, but it appears that significant over-parameterization (i.e. really big models) during training helps the model to learn and to find a good optimum.
- Many of these models are so large that the only tractable way to use them for training and inference is to use parallel programs (i.e. GPUs)
- Iterative pruning has become a common way to compress the size of fully trained neural networks. Typically 80-90% of a networks parameters can be pruned away without having any negative effect on performance. This creates significant sparsity in the networks weight matrices.
- Additionally, a considerable group of famous networks uses zero-inducing non-linearities like ReLU or sigmoid and creates sparsity in the resulting matrix.
- Typically the weight matrices of neural networks are represented in a dense format.

### Project
This project explores if, and at what degree of pruning, it is optimal to represent a weight matrix in a sparse format. A secondary question is what amongst the diff
Two interrelated factors are relevant:
1. Space the model occupies
2. Inference speed

It has been observed that if the entire model can fit in the L1 cache of GPU [add reference], significant speedups are possible. Given this, a sparse matrix format might be an optimal solution even if the sparse matrix multiplication is slower.

### Hypotheses:
- Primary: For a sufficiently large neural network, there exists a pruning threshold (measured as an average % of the number of model parameters) above which it is optimal to represent one or more weight matrices in the network as sparse matrices, and to use sparse matrix - sparse matrix routines instead of the more common parallel dense matrix routines
- Secondary: There may exist levels of pruning below this threshold, where is it optimal to represent one or more weight matrices in the network as a hybrid (mix of sparse and dense) format.

### Literature

See below

**Assumptions**:
  - All comparisons are made on GPUs
  - Optimality is a function of the space the model occupies and the speed at which inference using the model takes. A faster, smaller model is better than a larger, slower model.
  - A smaller model is preferable to a larger model, assuming speed is equal.
  - We will analyze both the size of the model and speed of inference

### Notes

We decided to to focus in fully connected parts of trained networks.

- We will choose fully connected parts of following networks and write c code to read them into various C formats.
- We need to write code fore doing basic feed forward capability.
- Simulating pruning and comparing it with normal pruning(if you have time). You can compare different simulations and see how pruning statistics effect performance.

## References

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
