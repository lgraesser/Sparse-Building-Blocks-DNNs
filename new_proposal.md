## New Proposal
We decided to focus on sparse matrix*sparse matrix(spMspM) multiplications and particularly in the context of deep learning. We believe in the importance of optimizing spM*spM multiplications in the context of deep-learning-research&development. Deep learning algorithms are over-parameterized and their parameters can be pruned down to 5-10% non zeros per matrix. Training big networks and using them for inference consists of many matrix-matrix multiplications. Some considerable group of famous networks uses zero-inducing non-linearities like ReLU or sigmoid and creates sparsity in the resulting matrix.

- We will choose fully connected parts of following networks and write c code to read them into various C formats.
- We need to write code fore doing basic feed forward capability.
- Simulating pruning and comparing it with normal pruning(if you have time). You can compare different simulations and see how pruning statistics effect performance.



- We can measure #floating point operations with give sparsity and then calculate empirical bound.
- Dropout sparsity patterns are random, so we can simulate that.
