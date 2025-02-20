<div align="center">
    <h1 align="center">✨ Pollux: Unified World Model</h1>
</div>


## Install


* Create the environment and install the required packages:

```
conda create -n pollux python=3.11 -y -c anaconda
conda activate pollux
```

* Install packages
```
pip install torch==2.5.0 xformers==0.0.28.post2 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt
```

* Installation of COSMOS TVAE
```bash
cd apps/Cosmos-Tokenizer
pip3 install -e .
```

* Test the installation of COSMOS VAE
```bash
cd apps/main
python test_vae.py
```

* Installation of CLIP MOdel
```bash
pip install git+https://github.com/openai/CLIP.git
```


* If you need to run data preprocessing (Optional dependencies)
```bash
pip install timm
pip install torchmetrics
```

* If you need to run YOLOv10
```bash
pip install ultralytics
```

* If you need mongoexport
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt update
sudo apt install mongodb-database-tools
# A sample:
mongoexport --uri="mongodb+srv://nucleusadmin:eMPF9pgRy2UqJW3@nucleus.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000" \
--db=world_model \
--collection=pexel_images \
--out=/mnt/pollux/mongo_db_cache/pexel_images.json --jsonArray

```
## Preliminary Usages

* We provides a minimal system to train diffusion model on ImageNet with parallelized system. The following example is how we train our pipeline on 8 GPUs.

```
torchrun --standalone --nnodes 1 --nproc-per-node 8 -m apps.main.train config=apps/main/configs/train_bucket_256_latent_code.yaml
```

* Generate visualizations:

```
python -m apps.main.generate config=apps/main/configs/eval.yaml
```

## Pollux Pipeline
![Image](https://github.com/user-attachments/assets/d0ea0b5f-54ed-48fd-92de-b849f07c7548)

## Data Pipeline
haozhe is working on that.

### TODO 

see issues

