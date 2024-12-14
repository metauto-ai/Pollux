### Huggingface Credentials

* Put the token here to access Llama or FLUX.1-dev
```
huggingface-cli login
git config --global credential.helper store
```

### Wandb Usage (please ask Mingchen for invitation to Wandb team)

* Set you wandb (get the API key from [here](https://wandb.me/wandb-server))

```
wandb login
```

* Set it into the config (for example, in the `apps/Latent_DIT/configs/LLAMA_Baseline_1B.yaml`)

```
# Only need to change the `name` each time; `dump_dir` will be generated automatically and accordingly
name: "ImageNet_1B_BaseLine_256_Flux_LLAMA_Pre_Train_MC" # Mingchen: Please modify this everytime
```
