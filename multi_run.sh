#!/bin/bash

#for n in 4096 8192 65536 131072 196608
# number of samples, chunk size
for seed in 100 ; do
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 4096 $seed
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 8192 $seed
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 16384 $seed 8192
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 32768 $seed 8192
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 65536 $seed 8192
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 131072 $seed 8192
  sbatch --nodes=1 --ntasks=1 --gres=gpu:8 run.sh 196608 $seed 4096
done

