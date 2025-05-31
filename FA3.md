## Requirements

*   **Cuda:** `12.8`
    *   If CUDA 12.8 is missing, please follow the steps in the "Installing CUDA" section first.
*   **PyTorch:** `2.7`
*   **Python:** `3.12.9`
*   **Ninja:** `1.11.1.4`
*   **CUDA_HOME:** Ensure `CUDA_HOME` is set to `/usr/local/cuda-12.8`.
*   **Disk Space:** At least 40GB free space on the boot disk.

## Flash Attention Setup

1.  Clone the `flash-attention` repository:
    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    ```
2.  Checkout commit `fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1`:
    ```bash
    git checkout fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1
    ```
3.  Install the Hopper-specific version:
    ```bash
    cd hopper/
    MAX_JOBS=24 python setup.py install
    ```

## Installing CUDA 12.8

1.  **Remove existing CUDA installations:**
    ```bash
    # Remove CUDA related packages
    sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
    # Purge all NVIDIA drivers
    sudo apt-get --purge remove "*nvidia*" "libxnvctrl*"
    # Remove orphaned dependencies
    sudo apt-get autoremove
    # Clean up the package cache
    sudo apt-get autoclean
    # (Optional but recommended) Remove residual CUDA directories
    sudo rm -rf /usr/local/cuda*
    # Reboot the system
    sudo reboot
    ```

2.  **Install CUDA 12.8:**
    ```bash
    # Download and set up the CUDA repository pin
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

    # Download and install the local CUDA repository package
    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb

    # Copy the keyring
    sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

    # Update package lists and install CUDA toolkit and drivers
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-8
    sudo apt-get install -y cuda-drivers

    # Reboot the system
    sudo reboot
    ```

## Setup Fabric-Manager

1.  **Install Fabric Manager:**
    ```bash
    sudo apt-get install nvidia-fabricmanager-570
    ```
2.  **Restart and check the service:**
    ```bash
    sudo systemctl restart nvidia-fabricmanager
    sudo systemctl status nvidia-fabricmanager
    ```

## Export CUDA_HOME

1.  Open your `~/.bashrc` file.
2.  Add the following lines to the end of the file if they are not already present or correctly set:

    ```bash
    echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
    echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    ```
3.  Source the `~/.bashrc` file or restart your terminal for the changes to take effect:
    ```bash
    source ~/.bashrc
    ```


## Transformer Engine Installation

```
export CUDNN_PATH=/home/ubuntu/miniconda3/envs/env_name/lib/python3.12/site-packages/nvidia/cudnn/
export CUDNN_INCLUDE_DIR=/home/ubuntu/miniconda3/envs/env_name/lib/python3.12/site-packages/nvidia/cudnn/include/

# Clone repository, checkout stable branch, clone submodules
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git

cd TransformerEngine
pip3 install --no-build-isolation .   # Build and install
```
