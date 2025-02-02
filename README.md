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
cd apps/main/modules/Cosmos-Tokenizer
pip3 install -e .
```

* Test the installation of COSMOS VAE
```bash
cd apps/main
python test_vae.py
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

## Preliminary Usages

* Before we develop a MongoDB dataloader, we could first use this to remove `.lock` files for HFDataLoader.

```bash
find /jfs/data/imagenet-1k/ -type f -name "*.lock" -exec rm -f {} \;
```

* We provides a minimal system to train diffusion model on ImageNet with parallelized system. The following example is how we train our pipeline on 4 GPUs.

```
torchrun --standalone --nnodes 1 --nproc-per-node 8 -m apps.main.train config=apps/main/configs/train.yaml
```

* Generate visualizations:

```
python -m apps.main.generate config=apps/main/configs/eval.yaml
```

* Test each module:

```
python -m apps.main.test
```

## Pollux Pipeline
![pipeline](https://github.com/user-attachments/assets/2289daf0-2639-4bf9-aaaa-e708721fe9e4)


## Data Pipeline
haozhe is working on that.

### TODO 

see issues

