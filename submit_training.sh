#!/bin/bash

# Wrapper script for submitting multi-node training jobs
# This script provides an easy interface to submit SLURM jobs with different configurations

# Default values
NODES=2
PARTITION="learn"
CONFIG_FILE="apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml"
JOB_NAME="castor-multinode"
TIME="72:00:00"
GPUS_PER_NODE=8
CPUS_PER_GPU=24
CONDA_ENV="pollux"
CONDA_PATH="/fsx/ubuntu/miniconda3"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --nodes NODES           Number of nodes (default: $NODES)"
    echo "  -p, --partition PARTITION   SLURM partition (default: $PARTITION)"
    echo "  -c, --config CONFIG_FILE    Path to config file (default: $CONFIG_FILE)"
    echo "  -j, --job-name JOB_NAME     Job name (default: $JOB_NAME)"
    echo "  -t, --time TIME             Time limit (default: $TIME)"
    echo "  -g, --gpus-per-node GPUS    GPUs per node (default: $GPUS_PER_NODE)"
    echo "  --cpus-per-gpu CPUS         CPUs per GPU (default: $CPUS_PER_GPU)"
    echo "  -e, --conda-env ENV         Conda environment name (default: $CONDA_ENV)"
    echo "  --conda-path PATH           Conda installation path (default: $CONDA_PATH)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -n 4 -p gpu                                    # 4 nodes on gpu partition"
    echo "  $0 -n 2 -p learn -c apps/Castor/configs/my.yaml  # Custom config"
    echo "  $0 -n 8 -j my_experiment                          # 8 nodes with custom job name"
    echo "  $0 -n 4 -e my_env                                 # Use different conda environment"
    echo "  $0 --conda-path /opt/conda                        # Use different conda installation"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -g|--gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --cpus-per-gpu)
            CPUS_PER_GPU="$2"
            shift 2
            ;;
        -e|--conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --conda-path)
            CONDA_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! $NODES =~ ^[0-9]+$ ]] || [[ $NODES -lt 1 ]]; then
    echo "Error: Number of nodes must be a positive integer"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist"
    exit 1
fi

# Calculate total GPUs
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

echo "=== Submitting SLURM Job ==="
echo "Nodes: $NODES"
echo "Partition: $PARTITION"
echo "Config file: $CONFIG_FILE"
echo "Job name: $JOB_NAME"
echo "Time limit: $TIME"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "CPUs per GPU: $CPUS_PER_GPU"
echo "Conda environment: $CONDA_ENV"
echo "Conda path: $CONDA_PATH"
echo "=========================="

# Submit the job
sbatch \
    --nodes=$NODES \
    --partition=$PARTITION \
    --job-name=$JOB_NAME \
    --time=$TIME \
    --gres=gpu:$GPUS_PER_NODE \
    --cpus-per-gpu=$CPUS_PER_GPU \
    --export=CONFIG_FILE=$CONFIG_FILE,JOB_NAME=$JOB_NAME,GPUS_PER_NODE=$GPUS_PER_NODE,CONDA_ENV=$CONDA_ENV,CONDA_PATH=$CONDA_PATH \
    submit_multinode_training_flexible.slurm

echo ""
echo "Job submitted! Use 'squeue -u \$USER' to check job status."
echo "Logs will be available in: /fsx/checkpoints/ablations/logs/<JOB_ID>/" 