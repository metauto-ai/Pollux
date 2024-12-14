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
pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt
```

* Additional Package for Pollux
```
pip install diffusers
pip install datasets
pip install torchvision==0.20.0
```


## Preliminary Usages

* We provides a minimal system to train diffusion model on ImageNet with parallelized system. The following example is how we train our pipeline on 4 GPUs.

```
torchrun --standalone --nnodes 1 --nproc-per-node 4 -m apps.Simple_DiT.train config=apps/Simple_DiT/configs/LLAMA_Baseline_1B.yaml
```

* Generate visualizations:

```
python -m apps.Simple_DiT.generate config=apps/Simple_DiT/configs/eval.yaml
```

* Test each module:

```
python -m apps.Simple_DiT.test
```




### TODO 

* P(ure)-DiffusionModel where time embedding/conditional signal is tokenized to the sequential input. 
* Ensemble minimal and efficient Qwen model for image/video understanding.
