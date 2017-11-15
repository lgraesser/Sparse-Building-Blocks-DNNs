### Project
This project explores if, and at what degree of pruning, it is optimal to represent a weight matrix in a sparse format. A secondary question is if different sparse matrix storage formats make a significant difference.

Two interrelated factors are relevant when evaluating performance:
1. Space the model occupies
2. Inference speed

It has been observed that if the entire model can fit in the L1 cache of GPU [add reference], significant speedups are possible. Given this, a sparse matrix format might be an optimal solution even if the sparse matrix multiplication is slower.

Our question is, given a dense matrix, should we convert it to Sparse Matrix or not? At how many iterations we need to amortize converting. How much sparsity we need?

### Project Phases
1. **Investigating Fully Connected Layer Sparsity Threshold for Efficient Forward Propagation**
  - Generate sample matrices with python(write script) with random sampling. Look at the common datasets to comply with their style/standard. So we can use already written reading code. There should be (matrx1,matrx2)
  - Write down matrix conversion part with cuda. (maybe look at the UF-dataset, if they are written as dense vector, they may have a conversion)
  - Implement Dense-Kernel(call cuda).
  - Implement Sparse-Kernel
    1. using cuSparse
    2. our implementation
  - Write code to check whether the result with sparse kernel is same as its dense-part.

2. **Investigating Convolutional Layer Sparsity Threshold for Efficient Forward Propagation**
  - Generate sample matrices with python(write script) with random sampling. Look at the common datasets to comply with their style/standard. So we can use already written reading code.
  - Implement Dense-Kernel(call cuda).
  - Implement Sparse-Kernel
    1. using cuSparse
    2. our implementation

3. (Optional) 1. **Investigating Sparsity Threshold for Efficient Forward Propagation on Deep Networks**
  - Implement non-linearities with layers (maybe embed them into the matrix multiplication to prevent memory reads)
  - Implement network definition and reading it and recursively calling necesarry kernels (maybe use python for wrapping it, if we figure out how, maybe we can work on pytorch code-base).

4. **Perform experiments, complete report**
  - At this point we might have a real-life sparse matrices(utku aims to prune some deep-networks and you may get different sparse matrices for different pruning strategies) or we will use our sparse generator. One nice thing would be just using the python sampler and compare how much the results vary when we use real pruned networks.
  - Define the sweep range and write script to run the tests. Use AWS or prince to have the full control of the gpu.
  - Plot the results.
