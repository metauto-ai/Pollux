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

or 

```
sudo chmod a+w /mnt/data
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
/home/mczhuge/miniconda3/bin/conda init # Change to the specific path
source ~/.bashrc
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
PermitRootLogin no
PubkeyAuthentication yes
PasswordAuthentication no
AllowUsers ubuntu mczhuge
```

```bash
sudo systemctl restart sshd
```

