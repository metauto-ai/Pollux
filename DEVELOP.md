### PR first

```bash
git checkout -b feature/awesome-feature
git checkout -b fix/awesome-fix
```


### Philosophies for Better Engineering

* The `lingua` folder should contain only commonly used classes and functions, avoiding application-specific elements.
* Each model or pipeline should have its own dedicated folder within the `apps` directory.
* Each Pull Request should focus on a specific new feature (e.g., adding CFG support, implementing a new architecture, or introducing new configurations) and include a detailed test plan to facilitate effective code review.


### Create User

Example:

```
sudo adduser mczhuge
sudo usermod -aG sudo mczhuge
```

### Mount JFS

```bash
sudo curl -L https://juicefs.com/static/juicefs -o /usr/local/bin/juicefs && sudo chmod +x /usr/local/bin/juicefs && sudo /usr/local/bin/juicefs mount world-model /jfs
```

### For Conda Installation

```bash
su - mczhuge
sudo chmod -R u+w $Pollux
sudo chown -R mczhuge $Pollux
```

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source /home/mczhuge/miniconda3/bin/activate
```


### Setup Github SSH

Example:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/mczhuge -C "mczhuge@gmail.com"
cat ~/.ssh/mczhuge.pub
chmod 600 ~/.ssh/mczhuge
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/mczhuge
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts
ssh -T git@github.com
```


### NVIDIA-Fabric Issue

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-535_535.183.01-1_amd64.deb
sudo dpkg -i nvidia-fabricmanager-535_535.183.01-1_amd64.deb
sudo systemctl daemon-reload
sudo systemctl start nvidia-fabricmanager
```

### Login Setup

```bash
sudo mkdir -p /home/mczhuge/.ssh
sudo chmod 700 /home/mczhuge/.ssh
sudo chown mczhuge:mczhuge /home/mczhuge/.ssh
sudo cp /home/ubuntu/.ssh/authorized_keys /home/mczhuge/.ssh/
sudo chmod 600 /home/mczhuge/.ssh/authorized_keys
sudo chown mczhuge:mczhuge /home/mczhuge/.ssh/authorized_keys
```

```bash
sudo vim /etc/ssh/sshd_config
# add
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
AllowUsers ubuntu mczhuge
```

```bash
sudo systemctl restart sshd
```

### Local Run

```
export PYTHONPATH=/jfs/mczhuge/Pollux:$PYTHONPATH
python -m apps.main.train config=apps/main/configs/pollux_v0.5.yaml
```


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


### Training

Use the following command to train the model on 4 GPUs:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 -m apps.main.train config=apps/main/configs/pollux_v0.5.yaml
```
