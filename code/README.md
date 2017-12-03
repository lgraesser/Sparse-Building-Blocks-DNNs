To sample a sparse matrix use following

```bash
scripts/generate_sparse_mat.py 3,20,200,300 0.9 > data/test.mat
less data/test.mat
```

To compile binaries and run the tests

```bash
make
bin/io data/test.mat
bin/sparse_test data/test.mat
```

Check scripts folder to understand how to use binaries.


## utku notes
- why not fill the descrbtiors inside the sparse convertation and include them in the sparse struct
```  
cusparseMatDescr_t descrX;
  cusparseCreateMatDescr(&descrX);
  cusparseSetMatType(descrX, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrX, CUSPARSE_INDEX_BASE_ZERO);
```
- do dense2csr work both on host and device? We need a sparse2dense conversation.

- maybe we don't need to keep have nz_per_row in the struct, it would be nice to have leading mat->dims[2]  inside the sparse struct so that we don't need to give/keep the whole dense matrix when we copy

- I was thinking of unifying the device/host fields within sparse matrix struct, we can have a char flag saying where the sparse matrix is. Usually it is always going to be on device during the calculations. We can create a new struct and provide it to
