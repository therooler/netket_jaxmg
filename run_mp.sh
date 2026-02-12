
num_processes=${1:-2}
shift
# export CUDA_VISIBLE_DEVICES="0"
source ~/netket_jaxmg/.venv/bin/activate
range=$(seq 0 $(($num_processes - 1)))
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo $HOSTS
MASTER=${HOSTS[0]}
for i in $range; do
  export JAX_COORDINATOR_ADDRESS="$MASTER:10001"
  export JAX_PROCESS_ID=$i
  export JAX_PROCESS_COUNT=$num_processes
  if [ $i -eq 0 ]; then
    python -u run_vmc.py "$@" &
  else
    python -u run_vmc.py "$@" > /tmp/run_vmc_$i.out &
  fi
done

wait