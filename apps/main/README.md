### Huggingface Credentials

To access models like Llama or FLUX.1-dev, authenticate with Huggingface by following the steps below:

```bash
huggingface-cli login
git config --global credential.helper store
```

### Wandb Integration

If you need access to the Wandb team, please contact Mingchen for an invitation.

1. Obtain your Wandb API key from [this link](https://wandb.me/wandb-server).
2. Log in using the following command:

```bash
wandb login
```

3. Update your configuration file (e.g., `apps/main/configs/LLAMA_Baseline_1B.yaml`) with the relevant Wandb settings:

```yaml
# Update the `name` field for each new run. The `dump_dir` will be auto-generated.
name: "ImageNet_1B_BaseLine_256_Flux_LLAMA_Pre_Train_MC" # Ensure to modify this for each run
```

### Running the Pipeline

Our system provides a streamlined setup for training diffusion models on ImageNet using a parallelized architecture. Below are the steps for training, visualization, and testing:

#### Training

Use the following command to train the model on 4 GPUs:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 -m apps.main.train config=apps/main/configs/pollux_v0.1.yaml
```

#### Visualization

To generate visualizations, run:

```bash
python -m apps.main.generate config=apps/main/configs/eval.yaml
```

#### Module Testing

To test individual modules, execute:

```bash
python -m apps.main.test
```