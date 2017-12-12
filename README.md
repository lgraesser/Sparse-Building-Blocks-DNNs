# GPU-Project-SpMVM
Evaluating sparse and dense matrix formats for performing the forward pass for sparse weight and input matrices. Currently the evaluation is restricted to fully connected and convolutional neural network layers with non linear activations.


## Running the experiments
```bash
cd code
make clean
make all
```

This would compile the library from the `src/` folder into `obj/` & `bin/` folders. One can run following tests after compiling

### Feedforward
```bash
./scripts/mm_experiment.sh log_name.txt
```
This would run the 180 feedforward experiments mentioned in the report sequencially, each time generating required matrices under `data/` folder and directing the experiment information to the stdout. The results are appended into the file name provided, i.e. `log_name.txt`. If the log file exists the scripts stops executing, so you need to provide a non existing file path.

### Convolution
```bash
cd code/scripts
./generate_conv_data.sh
./run_conv_exp.sh
```

This runs the 200 convolutional experiments mentioned in the report sequentially, each time generating required matrices under `data/` folder and directing the experiment information to the stdout.

### Generating sparse matrices
```bash
scripts/generate_sparse_mat.py 3,20,200,300 0.9 > data/test.mat
less data/test.mat
```
