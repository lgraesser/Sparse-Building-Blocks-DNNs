#!/usr/bin/env bash

# This line to get to the code file, no matter where it is called from
cd $( dirname "${BASH_SOURCE[0]}" )/../
make all
scripts/generate_sparse_mat.py 5,20 0.8 > data/a.mat
scripts/generate_sparse_mat.py 20,2 0.8 > data/b.mat
bin/mm data/a.mat data/b.mat
