### PR first

```bash
git checkout -b feature/awesome-feature
git checkout -b fix/awesome-fix
```


### Philosophies for Better Engineering

* The `lingua` folder should contain only commonly used classes and functions, avoiding application-specific elements.
* Each model or pipeline should have its own dedicated folder within the `apps` directory.
* Each Pull Request should focus on a specific new feature (e.g., adding CFG support, implementing a new architecture, or introducing new configurations) and include a detailed test plan to facilitate effective code review.


### For Conda Installation

```bash
su - mczhuge
udo chmod -R u+w $Pollux
sudo chown -R mczhuge $Pollux
```

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source /home/mczhuge/miniconda3/bin/activate
```

### Setup SSH

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

### Conda Install

```
sudo chmod a+w /mnt/data
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
/home/mczhuge/miniconda3/bin/conda init # Change to the specific path
source ~/.bashrc
```

### NVIDIA-Fabric Issue

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-fabricmanager-535_535.183.01-1_amd64.deb
sudo dpkg -i nvidia-fabricmanager-535_535.183.01-1_amd64.deb
sudo systemctl daemon-reload
sudo systemctl start nvidia-fabricmanager
```
