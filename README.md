<div align="center">
  <h1 align="center">Pollux: Unified World Model</h1>
  <p align="center">
    A collaboration between KAUST and <a href="https://www.withnucleus.ai/">Nucleus</a> on world models.
  </p>
</div>

Pollux is a unified world-model training and generation codebase.

## Install
See [Install.md](Install.md).

## Quickstart

### Train (single node)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 -m apps.main.train \
  config=apps/main/configs/train_bucket_256_latent_code.yaml
```

### Generate
```bash
python -m apps.main.generate config=apps/main/configs/eval.yaml
```

### Multi-node / Slurm
See [MULTINODE.md](MULTINODE.md).

## Docs
- [DEVELOP.md](DEVELOP.md) – development notes and cluster setup
- [FA3.md](FA3.md) – CUDA 12.8 + FlashAttention v3

## Pollux Pipeline
![Pollux Pipeline Diagram](https://github.com/user-attachments/assets/d0ea0b5f-54ed-48fd-92de-b849f07c7548)

## Contributors
<a href="https://github.com/metauto-ai/Pollux/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=metauto-ai/Pollux" />
</a>

## License
MIT (see [LICENSE](LICENSE)).
