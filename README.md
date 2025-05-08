<div align="center">
    <h1 align="center">✨ Pollux: Unified World Model</h1>
</div>


## Install

Follow these steps to set up the environment and install dependencies.

### CUDA Installation
*   Install CUDA 12.8. Follow instructions from `FA3.md`.

### Environment Setup
*   Create and activate the conda environment:
    ```bash
    conda create -n pollux python=3.12.9
    conda activate pollux
    ```

### PyTorch Installation
*   Install PyTorch, torchvision, and torchaudio compatible with CUDA 12.8:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

### Flash Attention v3 Installation
*   Build and install Flash Attention v3. Follow instructions from `FA3.md`.

### Core Package Installation
*   Install xformers, ninja, packaging, and requirements:
    ```bash
    pip install xformers # installs xformers-0.0.30
    pip install ninja packaging
    pip install --requirement requirements.txt
    ```

### COSMOS TVAE Installation
*   Install the COSMOS Tokenizer VAE:
    ```bash
    cd apps/Cosmos-Tokenizer
    pip3 install -e .
    cd ../.. # Return to the root directory
    ```
*   Test the COSMOS VAE installation:
    ```bash
    python apps/main/test_vae.py
    ```

### CLIP Model Installation
*   Install the OpenAI CLIP model:
    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```

### Optional Dependencies (Data Preprocessing)
*   If you need to run data preprocessing, install these packages:
    ```bash
    pip install timm
    pip install torchmetrics
    ```

### Optional Dependencies (Mongo Tools)
*   If you need `mongoexport`:
    1.  Add the MongoDB GPG key:
        ```bash
        wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
        ```
    2.  Create the list file for MongoDB:
        ```bash
        echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
        ```
    3.  Update package lists and install MongoDB tools:
        ```bash
        sudo apt update
        sudo apt install mongodb-database-tools
        ```
    4.  Example `mongoexport` command:
        ```bash
        mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
        --db=world_model \
        --collection=pexel_images \
        --out=/mnt/pollux/mongo_db_cache/pexel_images.json --jsonArray
        ```

## Preliminary Usage

### Training
*   Example command to train the diffusion model on 8 GPUs using `torchrun`:
    ```bash
    torchrun --standalone --nnodes 1 --nproc-per-node 8 -m apps.main.train config=apps/main/configs/train_bucket_256_latent_code.yaml
    ```

### Generating Visualizations
*   Example command to generate visualizations:
    ```bash
    python -m apps.main.generate config=apps/main/configs/eval.yaml
    ```

## Pollux Pipeline
![Pollux Pipeline Diagram](https://github.com/user-attachments/assets/d0ea0b5f-54ed-48fd-92de-b849f07c7548)

## Data Pipeline
*   Haozhe is working on this section.

## TODO
*   See GitHub issues for the current task list.

