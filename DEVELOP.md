### PR first

```
git checkout -b feature/awesome-feature
git checkout -b fix/awesome-fix
```


### Philosophies for Better Engineering

* The `lingua` folder should contain only commonly used classes and functions, avoiding application-specific elements.
* Each model or pipeline should have its own dedicated folder within the `apps` directory.
* Each Pull Request should focus on a specific new feature (e.g., adding CFG support, implementing a new architecture, or introducing new configurations) and include a detailed test plan to facilitate effective code review.


### For Conda Installation

```
su - mczhuge
udo chmod -R u+w $Pollux
sudo chown -R mczhuge $Pollux
```

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source /home/mczhuge/miniconda3/bin/activate
```