#!/usr/bin/env bash

# This line to get to the code file, no matter where it is called from
mkn_s32=(32,256,128 32,512,256 32,1024,512 32,2048,1024 32,4096,2048 32,8192,4096 1024,256,128 1024,512,256 1024,1024,512 1024,2048,1024 1024,4096,2048 1024,8192,4096)
iterations=(50000 10000 2500 1000 250 100 5000 1000 250 100 25 10)
sparsities=(0.5 0.8 0.9 0.95 0.99)
# mkn_s32=(1024,512,256)
# iterations=(250)
# sparsities=(0.99)

RESULTFILE=$1
if [ -e $RESULTFILE ]
then
  echo Filename $RESULTFILE exists, "so we cant let this happen"
  exit 1
fi


for i in "${!foo[@]}"; do
  printf "%s\t%s\n" "$i" "${foo[$i]}"
done

cd $( dirname "${BASH_SOURCE[0]}" )/../
make bin/mm


for i in "${!mkn_s32[@]}";
do
  IFS=","
  ITER=${iterations[$i]}
  set -- ${mkn_s32[$i]}
  for SPARSITY in "${sparsities[@]}";
  do
    echo m=$1 k=$2 n=$3 sparsity=$SPARSITY ITER=$ITER | tee -a $RESULTFILE
    FILENAME1=../data/$1_$2_$SPARSITY.mat
    FILENAME2=../data/$2_$3_$SPARSITY.mat
    # We want to generate new random matrices. Otherwise comment in the if clauses.
    # if [ ! -e $FILENAME1 ]
    # then
      scripts/generate_sparse_mat.py $1,$2 $SPARSITY > $FILENAME1
    # fi
    # if [ ! -e $FILENAME2 ]
    # then
      scripts/generate_sparse_mat.py $2,$3 $SPARSITY > $FILENAME2


    echo bin/mm $FILENAME1 $FILENAME2 -t -a cusparse -r$ITER | tee -a $RESULTFILE
    bin/mm $FILENAME1 $FILENAME2 -t -a cusparse -r$ITER >>$RESULTFILE
    echo bin/mm $FILENAME1 $FILENAME2 -t -a denseblas -r$ITER | tee -a $RESULTFILE
    bin/mm $FILENAME1 $FILENAME2 -t -a denseblas -r$ITER >>$RESULTFILE
    echo bin/mm $FILENAME1 $FILENAME2 -t -a sparseimp -r$ITER | tee -a $RESULTFILE
    bin/mm $FILENAME1 $FILENAME2 -t -a sparseimp -r$ITER >>$RESULTFILE
  done
done
