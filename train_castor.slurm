#!/bin/bash
#SBATCH --job-name=castor_train        # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=8            # Number of tasks per node (1 per GPU)
#SBATCH --gpus-per-node=8              # Request 8 GPUs per node (H100s)
#SBATCH --cpus-per-task=8              # CPU cores per task
#SBATCH --mem=0                        # Request all memory on the node
#SBATCH --time=72:00:00                # Time limit (72 hours)
#SBATCH --output=%x_%j.out             # Standard output log
#SBATCH --error=%x_%j.err              # Standard error log
#SBATCH --partition=debug              # Partition with H100 GPUs (adjust if needed)

# Set up environment variables for CUDA
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export ROOT_DIR=/mnt/pollux

# Configure DNS servers as a fallback in case of DNS issues
export DNS_BACKUP="8.8.8.8 8.8.4.4 1.1.1.1"
echo "Setting up DNS fallback to: $DNS_BACKUP"
cat > /tmp/resolv.conf.new << EOF
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 1.1.1.1
EOF
export HOSTALIASES=/tmp/resolv.conf.new

# Check if the shared environment exists and set it up if not
if [ "$SLURM_PROCID" == "0" ]; then
    echo "Checking if shared environment setup is needed..."
    bash setup_shared_env.sh
    
    # Verify flash_attn_interface.py exists
    if [ ! -f "$ROOT_DIR/environments/pollux_env/lib/python3.12/site-packages/flash_attn_interface.py" ]; then
        echo "Creating flash_attn_interface compatibility module..."
        cat > $ROOT_DIR/environments/pollux_env/lib/python3.12/site-packages/flash_attn_interface.py << EOF
# Compatibility layer for flash_attn_interface imports
from flash_attn.flash_attn_interface import flash_attn_varlen_func
# Export other functions that might be needed
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
# Export any other necessary functions
EOF
    fi
    
    # Patch the Castor component.py file to fix flash_attn import if needed
    COMPONENT_FILE="apps/Castor/modules/component.py"
    if [ -f "$COMPONENT_FILE" ]; then
        echo "Checking if Castor component.py needs patching..."
        if grep -q "from flash_attn_interface import" "$COMPONENT_FILE"; then
            echo "Patching $COMPONENT_FILE to use correct flash_attn import..."
            # Create backup
            cp "$COMPONENT_FILE" "${COMPONENT_FILE}.bak"
            # Replace direct import with proper module path
            sed -i 's/from flash_attn_interface import flash_attn_varlen_func/from flash_attn.flash_attn_interface import flash_attn_varlen_func/' "$COMPONENT_FILE"
            echo "Patched $COMPONENT_FILE"
        else
            echo "$COMPONENT_FILE doesn't need patching"
        fi
    fi
fi

# Wait for rank 0 to finish setting up the environment
sleep 5
srun --ntasks=1 --nodes=1 --ntasks-per-node=1 --wait=0 true

# Load the shared conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ROOT_DIR/environments/pollux_env

# Verify critical packages are installed
echo "Verifying critical packages..."
python -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    import flash_attn
    print(f'Flash Attention version: {flash_attn.__version__}')
    
    # Attempt to import from both paths to test compatibility
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        print('Original flash_attn.flash_attn_interface module is available')
    except ImportError:
        print('Warning: Could not import from flash_attn.flash_attn_interface')
    
    try:
        import flash_attn_interface
        print('Compatibility layer flash_attn_interface is available')
    except ImportError:
        print('Warning: Could not import flash_attn_interface')
        
except ImportError as e:
    print(f'Error: {e}')
    print('Warning: Some required packages are missing but continuing anyway')
"

# Setup distributed training environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Print environment information
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_NODEID: $SLURM_NODEID"

# Set PyTorch options for better performance
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$ROOT_DIR/torch_extensions  # Share compiled extensions

# NCCL settings for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=^lo,docker0

# Set PyTorch to use high precision
export TORCH_FLOAT32_MATMUL_PRECISION=high

# Launch training using srun for optimal resource allocation
srun python -m apps.Castor.train config=apps/Castor/configs/train_bucket_256_Castor_flux_qwen_fixed_siglip2.yaml
