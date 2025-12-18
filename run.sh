#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=./outputs/ntk.%j.out
#SBATCH --time=24:00:00
#SBATCH --exclude=workergpu156
source ~/rnn_jax/.venv/bin/activate

# Choose a coordinator (rank 0’s node)
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo $HOSTS
MASTER=${HOSTS[0]}

export JAX_COORDINATOR_ADDRESS=${MASTER}:10001
export JAX_PROCESS_COUNT=${SLURM_NTASKS}
ns=$1
seed=$2
chunk_size=$3
echo "Number of tasks $SLURM_NTASKS"
echo "Number nodes $SLURM_JOB_NUM_NODES"
echo "Number of GPUs $SLURM_GPUS_ON_NODE"

for index in "${!HOSTS[@]}"; do
  host="${HOSTS[$index]}"
  echo "Launching rank $index on $host"
  srun --export=ALL -N1 -n1 -w $host --gpus-per-task=$SLURM_GPUS_ON_NODE --output="./outputs/ntk_sub_$index-$ns.out" bash job_sub.sh $index $ns $seed $chunk_size&
done
wait


