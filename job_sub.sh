#!/bin/bash -l

source ~/rnn_jax/.venv/bin/activate

export JAX_PROCESS_ID=$1
ns=$2
seed=$3
chunk_size=$4
# Start GPU monitoring in the background
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,memory.total --format=csv -l 1 > ./outputs/gpu_mem_rank_${JAX_PROCESS_ID}_ns_${ns}.csv &

MONITOR_PID=$!

# Run your program
if [ -n "$chunk_size" ]; then
    python -u tfim_vmc.py --ns="$ns" --seed="$seed" --chunk_size="$chunk_size"
else
    python -u tfim_vmc.py --ns="$ns" --seed="$seed"
fi

# Kill monitor when program ends
kill $MONITOR_PID
wait