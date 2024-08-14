#!/bin/bash

#SBATCH --account=IscrC_LLM-EVAL
#SBATCH --job-name=lm_eval
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00

set -x

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/leonardo_work/IscrC_LLM-EVAL/scolomb1

# Load necessary modules
module purge
module load profile/global
module load cuda cudnn nccl

#Change to your scratch directory
cd $WORK
cd scolomb1


# attivo l'ambiente (come interagisce con i moduli?)
source eval_venv/bin/activate

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


# robe con spazio in IP boh
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block &

# Optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# Number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block &
    sleep 5
done

cd /leonardo_work/IscrC_LLM-EVAL/scolomb1/lm-evaluation-harness

model_path=/leonardo_work/IscrC_LLM-EVAL/scolomb1/hub/models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26
echo $model_path

lm_eval --model vllm --model_args pretrained=$model_path,tensor_parallel_size=16,pipeline_parallel_size=1,max_model_len=4096,distributed_executor_backend='ray' --tasks m_mmlu_it --num_fewshot 5 --batch_size 1

lm_eval --model vllm --model_args pretrained=$model_path,tensor_parallel_size=16,pipeline_parallel_size=1,max_model_len=4096,distributed_executor_backend='ray' --tasks hellaswag_it,arc_it --batch_size 1

echo "Background GPU watcher stopped."
echo "Job completed."
