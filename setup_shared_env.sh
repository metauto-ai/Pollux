#!/bin/bash
set -e

ROOT_DIR=/mnt/pollux

# Create shared directories
mkdir -p $ROOT_DIR/environments
mkdir -p $ROOT_DIR/compiled_packages

# Setup conda environment in shared location
if [ ! -d "$ROOT_DIR/environments/pollux_env" ]; then
    echo "Creating conda environment in shared location..."
    conda create -y -p $ROOT_DIR/environments/pollux_env python=3.12.9
    
    # Activate the environment and install PyTorch and other dependencies
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ROOT_DIR/environments/pollux_env
    
    # Install PyTorch with CUDA 12.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    
    # Install xformers and other basic dependencies
    pip install xformers ninja packaging
    pip install --requirement requirements.txt
    
    # Install CLIP
    pip install git+https://github.com/openai/CLIP.git
    
    # Install optional dependencies
    pip install timm torchmetrics
else
    echo "Conda environment already exists at $ROOT_DIR/environments/pollux_env"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ROOT_DIR/environments/pollux_env
fi

# Function to create the compatibility layer
create_compatibility_layer() {
    echo "Creating flash_attn_interface compatibility module..."
    cat > $ROOT_DIR/environments/pollux_env/lib/python3.12/site-packages/flash_attn_interface.py << EOF
# Compatibility layer for flash_attn_interface imports
from flash_attn.flash_attn_interface import flash_attn_varlen_func
# Export other functions that might be needed
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
# Export any other necessary functions
EOF
}

# Build Flash Attention v3 if not already built
if [ ! -f "$ROOT_DIR/compiled_packages/flash-attention/hopper/build/lib.linux-x86_64-cpython-312/flash_attn_3.so" ]; then
    echo "Compiling Flash Attention v3..."
    cd $ROOT_DIR/compiled_packages
    
    # Configure DNS servers as a fallback in case of DNS issues
    export DNS_BACKUP="8.8.8.8 8.8.4.4 1.1.1.1"
    echo "Setting up DNS fallback to: $DNS_BACKUP"
    cat > /tmp/resolv.conf.new << EOF
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 1.1.1.1
EOF
    export HOSTALIASES=/tmp/resolv.conf.new
    
    # First try direct pip install as fallback (might be easier than building from source)
    echo "Trying direct pip install of flash-attention..."
    if pip install flash-attn --no-build-isolation; then
        echo "Successfully installed flash-attention via pip"
        create_compatibility_layer
        echo "Flash Attention installed via pip"
    else
        echo "Pip install failed, attempting to build from source..."
        
        # Retry git clone with exponential backoff
        MAX_RETRIES=5
        retry_count=0
        clone_success=false
        
        while [ $retry_count -lt $MAX_RETRIES ] && [ "$clone_success" = false ]; do
            if [ ! -d "flash-attention" ]; then
                echo "Attempt $(($retry_count + 1))/$MAX_RETRIES: Cloning flash-attention repository..."
                if git clone https://github.com/Dao-AILab/flash-attention.git; then
                    clone_success=true
                    cd flash-attention
                    git checkout v2.7.4.post1
                else
                    retry_count=$((retry_count + 1))
                    if [ $retry_count -lt $MAX_RETRIES ]; then
                        sleep_time=$((2 ** retry_count))
                        echo "Clone failed. Retrying in $sleep_time seconds..."
                        sleep $sleep_time
                    else
                        echo "Failed to clone after $MAX_RETRIES attempts."
                        exit 1
                    fi
                fi
            else
                clone_success=true
                cd flash-attention
                git checkout v2.7.4.post1
            fi
        done
        
        # Try a different approach if the main build method fails
        cd hopper/
        echo "Building Flash Attention with MAX_JOBS=24..."
        if ! MAX_JOBS=24 python setup.py build; then
            echo "Default build method failed, trying alternative approach..."
            # Try alternative build approach
            if ! TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" pip install -e ..; then
                echo "Alternative build method failed, trying pip install again with force..."
                cd ../../
                pip install flash-attn --force-reinstall
            fi
        fi
        
        # Copy built libraries to the environment if available
        if [ -d "build/lib.linux-x86_64-cpython-312" ]; then
            echo "Copying built Flash Attention libraries to Python environment..."
            cp -r build/lib.linux-x86_64-cpython-312/* $ROOT_DIR/environments/pollux_env/lib/python3.12/site-packages/
        fi
        
        create_compatibility_layer
        echo "Flash Attention v3 has been compiled and installed to the shared environment"
    fi
else
    echo "Flash Attention v3 is already compiled"
    
    # Always ensure the compatibility layer exists
    if [ ! -f "$ROOT_DIR/environments/pollux_env/lib/python3.12/site-packages/flash_attn_interface.py" ]; then
        create_compatibility_layer
    fi
fi

# Install COSMOS Tokenizer VAE
if [ ! -d "$ROOT_DIR/environments/pollux_env/lib/python3.12/site-packages/cosmos_tokenizer" ]; then
    echo "Installing COSMOS Tokenizer VAE..."
    cd $(dirname "$0")  # Go to the directory where this script is located
    
    # Check if Cosmos-Tokenizer directory exists
    if [ -d "apps/Cosmos-Tokenizer" ]; then
        cd apps/Cosmos-Tokenizer
        pip install -e .
        cd ../..
        echo "COSMOS Tokenizer VAE installed."
    else
        echo "Warning: apps/Cosmos-Tokenizer directory not found, skipping installation."
        echo "If you need COSMOS Tokenizer, please make sure the repository is properly cloned."
        # Optional: Clone the repository if it doesn't exist
        # git clone https://github.com/your-org/Cosmos-Tokenizer.git apps/Cosmos-Tokenizer
    fi
fi

# Final verification of Flash Attention installation
python -c "
try:
    import flash_attn
    print(f'Flash Attention {flash_attn.__version__} successfully installed')
    import flash_attn_interface
    print('Flash Attention Interface compatibility layer is working')
except ImportError as e:
    print(f'Error importing Flash Attention: {e}')
    exit(1)
"

echo "Environment setup complete. Activate with: conda activate $ROOT_DIR/environments/pollux_env" 
