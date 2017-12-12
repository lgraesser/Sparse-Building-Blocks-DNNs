#!/bin/bash

mat=224
sparsity=0
echo "Creating $mat x $mat matrix"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/$mat.mat

mat=512
echo "Creating $mat x $mat matrix"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/$mat.mat

mat=1024
echo "Creating $mat x $mat matrix"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/$mat.mat

mat=2048
echo "Creating $mat x $mat matrix"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/$mat.mat

kernel=3
sparsity=0
sp=0
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=3
sparsity=0.5
sp=05
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=3
sparsity=0.9
sp=09
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=3
sparsity=0.95
sp=095
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=3
sparsity=0.99
sp=099
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=7
sparsity=0
sp=0
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=7
sparsity=0.5
sp=05
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=7
sparsity=0.9
sp=09
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=7
sparsity=0.95
sp=095
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat

kernel=7
sparsity=0.99
sp=099
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k$mat_$sp.mat
