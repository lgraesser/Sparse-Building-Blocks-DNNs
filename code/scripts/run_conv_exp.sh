#!/bin/bash

matrices=("224.mat" "512.mat" "1024.mat" "2048.mat")
k3=("k3_0.mat" "k3_05.mat" "k3_09.mat" "k3_095.mat" "k3_099.mat")
k7=("k7_0.mat" "k7_05.mat" "k7_09.mat" "k7_095.mat" "k7_099.mat")
algos=("projdense" "projdensepitch" "densecudnn" "sparse" "sparsepitch")

for i in "${matrices[@]}"
do
  echo "===================== $i MATRIX =================="
  for j in "${k3[@]}"
  do
    echo "===================== $j KERNEL =================="
    for k in "${algos[@]}"
    do
      echo "===================== $k ALGORITHM =================="
      echo "===================== 1 iteration =================="
      ../bin/conv_exp -a $k ../data/$i ../data/$j
      echo "===================== 1000 iterations =================="
      ../bin/conv_exp -n -a $k ../data/$i ../data/$j
    done
  done
  for j in "${k7[@]}"
  do
    echo "===================== $j KERNEL =================="
    for k in "${algos[@]}"
    do
      echo "===================== $k ALGORITHM =================="
      echo "===================== 1 iteration =================="
      ../bin/conv_exp -a $k ../data/$i ../data/$j
      echo "===================== 1000 iterations =================="
      ../bin/conv_exp -n -a $k ../data/$i ../data/$j
    done
  done
done
