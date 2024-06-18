#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --account=IscrC_LLM-EVAL
#SBATCH --job-name=accelerate_test_multinode
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
###SBATCH -q boost_qos_dbg

export GPUS_PER_NODE=4

set -x -e

export HF_HOME=/leonardo_work/IscrC_LLM-EVAL/scolomb1
export NCCL_DEBUG=INFO


# Change to your scratch directory
cd $WORK

# Activate the environment
source /leonardo_work/IscrC_LLM-EVAL/scolomb1/eval_venv/bin/activate

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000


export LAUNCHER="accelerate launch \
        --num_processes $NUM_PROCESSES \
        --num_machines $NNODES \
        --machine_rank \$SLURM_PROCID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT"


export SCRIPT="/leonardo_work/IscrC_LLM-EVAL/scolomb1/lm-evaluation-harness/lm_eval/__main__.py"
export SCRIPT_ARGS=" \
        --model hf \
        --model_args pretrained=/leonardo_work/IscrC_LLM-EVAL/scolomb1/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79,dtype="float" \
        --tasks arc_it \
        --num_fewshot 0 \
        --batch_size 1 \
        --limit 10 \
        "
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

srun --jobid $SLURM_JOBID bash -c "$CMD"