<div align="center">
  <h1 align="center">Pollux: Unified World Model</h1>
  <p align="center">
    
  </p>
</div>

>  Pollux is a unified world-model training and generation codebase. A collaboration between  <a href="https://www.kaust.edu.sa/en/">KAUST</a>  and <a href="https://www.withnucleus.ai/">Nucleus</a> on world models.

### Install

* **Setup** 

Please see [Install.md](Install.md) for a full installation. 

* **Train**

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 -m apps.main.train \
  config=apps/main/configs/train_bucket_256_latent_code.yaml
```

* **Generate**

```bash
python -m apps.main.generate config=apps/main/configs/eval.yaml
```

### Docs
- [MULTINODE.md](MULTINODE.md) - multinode support
- [DEVELOP.md](DEVELOP.md) – development notes and cluster setup
- [FA3.md](FA3.md) – CUDA 12.8 + FlashAttention v3

### Contributors
<a href="https://github.com/Pollux/metauto-ai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Pollux/metauto-ai" />
</a>

### License
MIT (see [LICENSE](LICENSE)).
