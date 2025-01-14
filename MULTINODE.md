## Multi-Node Slurm Training
We use `lingua/stool.py` to submit slurm jobs. By revise the original code, we support unpack the conda env to each compute node when submitting a job.

NOTE: we can continue to use `torchrun --standalone` command in a single node for quick debugging.

 ## Usage

1. Pack the conda environment into a .tar.gz file and put it under the shared file system. For instance,

```bash
pip install conda-pack
conda-pack -n pollux -o /jfs/shuming/code/env/pollux_env.tar.gz 
```
In the above, the name of your conda env is `pollux`, and the packed env will be saved at the jfs.


2. Submit to Slurm through `lingua.stool`. The example config is [train_2x4.yaml](https://github.com/metauto-ai/Pollux/blob/a9491c7457370ab20756c94dbf178458d8474d23/apps/gen_tran/configs/train_2x4.yaml), which will use total 8GPUs for training.

```
python -m lingua.stool script=apps.gen_tran.train config=apps/gen_tran/configs/train_2x4.yaml \
  nodes=2 \
  ngpu=4 \
  ncpu=16 \
  mem=256G \
  partition=debug \
  time=72:00:00 \
  anaconda_zip=/jfs/shuming/code/env/pollux_env.tar.gz \
  anaconda=/tmp/shuming/
```

In the above, `anaconda_zip` is the path of the packed file in step 1, and `anaconda` is the path to unpack it in each compute node, which is recommended to be under `/tmp/` for faster speed.

For other arguments, such as `ngpu`, `ncpu`, you can refer to the detailed explanation [here](https://github.com/metauto-ai/Pollux/blob/a9491c7457370ab20756c94dbf178458d8474d23/lingua/stool.py#L17-L37).

**Note that** the `nodes` and `ngpu` should be consistent with the setting in your yaml file. 

