#!/bin/bash

# SLURM Multi-Node Training Script for Castor (Flexible Version)
# 
# Usage examples:
# 1. Basic usage with defaults:
#    sbatch submit_multinode_training_flexible.slurm
#
# 2. Specify nodes and partition:
#    sbatch --nodes=4 --partition=gpu submit_multinode_training_flexible.slurm
#
# 3. Override config file:
#    sbatch --export=CONFIG_FILE=apps/Castor/configs/my_config.yaml submit_multinode_training_flexible.slurm
#
# 4. Full customization:
#    sbatch --nodes=4 --partition=gpu --export=CONFIG_FILE=apps/Castor/configs/my_config.yaml,JOB_NAME=my_experiment submit_multinode_training_flexible.slurm

# Default SLURM parameters (can be overridden via command line)
#SBATCH --job-name=castor-multinode
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=learn

#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block

# ============================================================================
# CONFIGURABLE PARAMETERS (can be set via --export)
# ============================================================================

# Default config file (can be overridden)
CONFIG_FILE=${CONFIG_FILE:-"apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml"}

# Default job name (can be overridden)
JOB_NAME=${JOB_NAME:-"castor-multinode"}

# Default project directory (can be overridden)
PROJECT_DIR=${PROJECT_DIR:-"/fsx/ubuntu/workspace/repo/Pollux"}

# Default log directory (can be overridden)
LOG_DIR=${LOG_DIR:-"/fsx/checkpoints/ablations/logs"}

# Default master port (can be overridden)
MASTER_PORT=${MASTER_PORT:-29500}

# Number of GPUs per node (can be overridden)
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Conda environment name (can be overridden)
CONDA_ENV=${CONDA_ENV:-"pollux"}

# Conda installation path (can be overridden)
CONDA_PATH=${CONDA_PATH:-"/fsx/ubuntu/miniconda3"}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Set environment variables
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export ENABLE_INTRA_NODE_COMM=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Create log directory with job-specific subdirectory
mkdir -p ${LOG_DIR}/$SLURM_JOB_ID

echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $JOB_NAME"
echo "Node ID: $SLURM_NODEID"
echo "Node Name: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Config file: $CONFIG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Log directory: ${LOG_DIR}/$SLURM_JOB_ID"
echo "Conda environment: $CONDA_ENV"
echo "Working directory: $(pwd)"
echo "======================="

# ============================================================================
# CONDA ENVIRONMENT SETUP
# ============================================================================

echo "=== Conda Environment Setup ==="
# Set conda path and add to PATH
echo "Using conda installation at: $CONDA_PATH"
export PATH="$CONDA_PATH/bin:$PATH"

# Initialize conda for bash shell
source $CONDA_PATH/etc/profile.d/conda.sh

# Activate the specified conda environment
echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV

# Verify the environment is activated
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Torch available: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "==============================="

# Export environment variables for srun
export PATH
export CONDA_DEFAULT_ENV
export CONDA_PREFIX

# ============================================================================
# DISTRIBUTED TRAINING SETUP
# ============================================================================

# Get the master node hostname
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"

# Set up distributed training environment variables
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

# Change to the project directory
cd $PROJECT_DIR

echo "Changed to project directory: $(pwd)"
echo "Starting training with config: $CONFIG_FILE"

# Get the full path to python from the activated environment
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"

# Run the training with torchrun
srun --export=ALL --output=${LOG_DIR}/$SLURM_JOB_ID/node_%N_%j_%t.log torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$GPUS_PER_NODE \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m apps.Castor.train \
    config=$CONFIG_FILE

echo "Training completed on node $SLURM_NODEID ($(hostname))"
echo "Job finished at: $(date)"