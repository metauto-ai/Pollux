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
pip install torch==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt
```

* Additional Package for Pollux
```
pip install diffusers
pip install datasets
pip install torchvision==0.20.1
pip install wandb
```


## Preliminary Usages

* We provides a minimal system to train diffusion model on ImageNet with parallelized system. The following example is how we train our pipeline on 4 GPUs.

```
torchrun --standalone --nnodes 1 --nproc-per-node 4 -m apps.main.train config=apps/main/configs/pollux_v0.6.yaml
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
![pollux_0 6](https://github.com/user-attachments/assets/f1731dd1-3cf3-435b-9640-7fe7110d8fe7)


### TODO 

see issues

