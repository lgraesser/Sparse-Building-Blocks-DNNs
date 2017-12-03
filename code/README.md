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
