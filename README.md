# Pollux
✨ Pollux: Unified Foundation Model


## Environmental Configuration

```
conda create -n lingua python=3.11 -y -c anaconda
conda activate lingua

# Install packages
pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt
# Additional Package for Pollux
pip install diffusers
pip install datasets
pip install torchvision==0.20.0
```


## Quick Start. 

We provide a minimal system to train diffusion model on ImageNet with parallelized system. 


Test Plan: Launch 4 GPU Jobs:

```
torchrun --standalone --nnodes 1 --nproc-per-node 4 -m apps.Simple_DiT.train config=apps/Simple_DiT/configs/LLAMA_Baseline_1B.yaml
```

Generate Visualizations:

```
python -m apps.Simple_DiT.generate config=apps/Simple_DiT/configs/eval.yaml
```

Test each module:

```
python -m apps.Simple_DiT.test
```


### Philosophies for Better Engineering

* The `lingua` folder should contain only commonly used classes and functions, avoiding application-specific elements.
* Each model or pipeline should have its own dedicated folder within the `apps` directory.
* Each Pull Request should focus on a specific new feature (e.g., adding CFG support, implementing a new architecture, or introducing new configurations) and include a detailed test plan to facilitate effective code review.


### TODO 

* Loading Pre-Training Weigts for DM (LLAMA3.2-1B/3B)
* Inference for Latent DM based on Flux RectFlow (`dynamic_shift=True`)
* P(ure)-DiffusionModel where time embedding/conditional signal is tokenized to the sequential input. 
