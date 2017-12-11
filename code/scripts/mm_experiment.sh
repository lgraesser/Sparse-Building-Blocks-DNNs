#!/usr/bin/env bash

# This line to get to the code file, no matter where it is called from
mkn_s32=(32,256,128 32,512,256 32,1024,512 32,2048,1024 32,4096,2048 32,8192,4096)
# mkn_s32=(1024,256,128 1024,512,256 1024,1024,512 1024,2048,1024 1024,4096,2048 1024,8192,4096)

sparsity=(0 0.5 0.8 0.9 0.95 0.99)


cd $( dirname "${BASH_SOURCE[0]}" )/../
make bin/mm


for i in "${mkn_s32[@]}";
do
  IFS=","
  set -- $i
  for f in "${sparsity[@]}";
  do
    echo m=$1 k=$2 n=$3 sparsity=$f
    FILENAME=data/$1_$2_$f.mat
    if [ ! -e $FILENAME ]
    then
      scripts/generate_sparse_mat.py $1,$2 $f > $FILENAME
    fi
    FILENAME=data/$2_$3_$f.mat
    if [ ! -e $FILENAME ]
    then
      scripts/generate_sparse_mat.py $2,$3 $f > $FILENAME
    fi
  done
done

# bin/mm data/a.mat data/b.mat -t -a cusparse -r
# bin/mm data/a.mat data/b.mat -t -a denseblas -r
# bin/mm data/a.mat data/b.mat -t -a sparseimp -r
