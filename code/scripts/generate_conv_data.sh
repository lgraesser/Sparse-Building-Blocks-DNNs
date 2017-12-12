#!/bin/bash
DATA_FOLDER=data
if [ ! -e ../$DATA_FOLDER ]
then
  echo Folder $DATA_FOLDER "doesnt exists...creating new"
  mkdir ../$DATA_FOLDER/
fi

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
./generate_sparse_mat.py $kernel,$kernel $sparsity > ../data/k${kernel}_${sp}.mat

kernel=3
sparsity=0.5
sp=05
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=3
sparsity=0.9
sp=09
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=3
sparsity=0.95
sp=095
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=3
sparsity=0.99
sp=099
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=7
sparsity=0
sp=0
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=7
sparsity=0.5
sp=05
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=7
sparsity=0.9
sp=09
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=7
sparsity=0.95
sp=095
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

kernel=7
sparsity=0.99
sp=099
echo "Creating $kernel x $kernel kernel, $sparsity sparsity"
./generate_sparse_mat.py $mat,$mat $sparsity > ../data/k${kernel}_${sp}.mat

echo "7 x 7 kernel 0% sparsity"
cat ../data/k7_0.mat
echo "7 x 7 kernel 90% sparsity"
cat ../data/k7_09.mat
echo "7 x 7 kernel 99% sparsity"
cat ../data/k7_099.mat
