#!/usr/bin/env bash

# This line to get to the code file, no matter where it is called from
cd $( dirname "${BASH_SOURCE[0]}" )/../
make all
scripts/generate_sparse_mat.py 3,200000 0.1 > data/a.mat
scripts/generate_sparse_mat.py 200000,2 0.1 > data/b.mat
bin/mm data/a.mat data/b.mat
