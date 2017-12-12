# GPU-Project-SpMVM
Evaluating sparse and dense matrix formats for performing the forward pass for sparse weight and input matrices. Currently the evaluation is restricted to fully connected and convolutional neural network layers with non linear activations.


## Running the experiments
```bash
cd code
make clean
make all
```
### Feedforward

### Convolution
```bash
./scripts/run_conv_exp.sh
```

### Generating sparse matrices
```bash
scripts/generate_sparse_mat.py 3,20,200,300 0.9 > data/test.mat
less data/test.mat
```
