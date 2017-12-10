#!/usr/bin/env bash

# This line to get to the code file, no matter where it is called from
cd $( dirname "${BASH_SOURCE[0]}" )/../
make bin/mm
scripts/generate_sparse_mat.py 2000,2000 0.99 > data/a.mat
scripts/generate_sparse_mat.py 2000,1000 0.99 > data/b.mat
bin/mm data/a.mat data/b.mat
