#!/bin/bash
#SBATCH --job-name=a2a                   # Name of job
#SBATCH --gres=gpu:8                     # Total number of GPUs requested
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=8              # Total number of tasks per node
#SBATCH --cpus-per-task=12               # CPU-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=640G                       # Memory per node (4G is default)
#SBATCH --time=3:00:00                   # Job time limit
#SBATCH --partition=pli                  # Specify the private node partition (mig, gpu, pli, all)
#SBATCH --account=spectralssmtorch       # Project name

# Check if SLURM environment variables are available and exit if not
if [[ -z $SLURM_NNODES ]] || [[ -z $SLURM_NTASKS_PER_NODE ]]; then
    echo "SLURM environment variables not set. Assuming standalone setup."
    WORLD_SIZE=1
else
    # Calculate WORLD_SIZE safely
    WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    echo "WORLD_SIZE=$WORLD_SIZE"
fi

# Determine the master node address
if [[ -z $SLURM_JOB_NODELIST ]]; then
    echo "SLURM_JOB_NODELIST not set. Using localhost as MASTER_ADDR."
    MASTER_ADDR="localhost"
else
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    if [[ -z $master_addr ]]; then
        echo "Failed to determine MASTER_ADDR. Exiting."
        exit 1
    fi
    MASTER_ADDR=$master_addr
    echo "MASTER_ADDR=$MASTER_ADDR"
fi

# Set an available port for the MASTER_PORT
if [[ -z $SLURM_JOBID ]]; then
    export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
else
    export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
echo "MASTER_PORT=$MASTER_PORT"

# Optional WandB environment settings, uncomment to use
export WANDB_ENTITY=windsornguyen
export WANDB_API_KEY=17dce35b188763800b6e9a443a761a1e713d87ab
export WANDB_PROJECT=compass-finetune
export WANDB_LOG_MODEL=checkpoint

echo "WandB and Slurm Environment Variables:"
printenv | grep -E 'WANDB|SLURM' | sort

# API key check
if [ -z "$WANDB_API_KEY" ]; then
  echo "WANDB_API_KEY: Not found"
else
  echo "WANDB_API_KEY: Found"
fi

# Display the hostname
echo "Running on host $(hostname)"

# Print the GPU information, check for CUDA
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=gpu_name --format=csv,noheader
else
    echo "CUDA not installed or GPUs not available."
fi

# Environment module management
module purge
module load anaconda3/2024.2
module load gcc-toolset/10
module load cudatoolkit/12.2

# Activate the conda environment
conda activate cos429

# Determine the command based on WORLD_SIZE
if [[ "$WORLD_SIZE" -eq "1" ]]; then
    # Standalone mode
    cmd="torchrun --standalone --master_port=$MASTER_PORT"
else
    # Distributed mode with nproc_per_node set from SLURM
    if [ -z "$SLURM_NTASKS_PER_NODE" ]; then
        echo "SLURM_NTASKS_PER_NODE is not set. Exiting."
        exit 1
    fi
    cmd="torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_port=$MASTER_PORT"
fi

python train.py
