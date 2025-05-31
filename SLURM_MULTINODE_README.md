# SLURM Multi-Node Training Scripts

This directory contains scripts for running multi-node distributed training on SLURM clusters. The scripts are designed to work with your existing Castor training setup and provide unique logging for each node.

## Files

1. **`submit_multinode_training.slurm`** - Basic SLURM script with fixed parameters
2. **`submit_multinode_training_flexible.slurm`** - Flexible SLURM script with configurable parameters
3. **`submit_training.sh`** - Easy-to-use wrapper script for submitting jobs
4. **`SLURM_MULTINODE_README.md`** - This documentation file

## Quick Start

### Method 1: Using the Wrapper Script (Recommended)

The easiest way to submit jobs is using the wrapper script:

```bash
# Basic usage with defaults (2 nodes, learn partition)
./submit_training.sh

# Specify number of nodes and partition
./submit_training.sh -n 4 -p gpu

# Use a different config file
./submit_training.sh -n 2 -p learn -c apps/Castor/configs/my_config.yaml

# Full customization
./submit_training.sh -n 8 -p gpu -c apps/Castor/configs/my_config.yaml -j my_experiment_name
```

### Method 2: Direct SLURM Submission

You can also submit jobs directly using `sbatch`:

```bash
# Basic submission with defaults
sbatch submit_multinode_training_flexible.slurm

# Override nodes and partition
sbatch --nodes=4 --partition=gpu submit_multinode_training_flexible.slurm

# Override config file and other parameters
sbatch --nodes=4 --partition=gpu \
       --export=CONFIG_FILE=apps/Castor/configs/my_config.yaml,JOB_NAME=my_experiment \
       submit_multinode_training_flexible.slurm
```

## Configuration Options

### Wrapper Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --nodes` | Number of nodes | 2 |
| `-p, --partition` | SLURM partition | learn |
| `-c, --config` | Path to config file | apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml |
| `-j, --job-name` | Job name | castor-multinode |
| `-t, --time` | Time limit | 72:00:00 |
| `-g, --gpus-per-node` | GPUs per node | 8 |
| `--cpus-per-gpu` | CPUs per GPU | 16 |
| `-e, --conda-env` | Conda environment name | pollux |
| `--conda-path` | Conda installation path | /fsx/ubuntu/miniconda3 |

### Environment Variables (for flexible script)

You can override these parameters using the `--export` option:

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_FILE` | Path to config file | apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml |
| `JOB_NAME` | Job name | castor-multinode |
| `PROJECT_DIR` | Project directory | /fsx/ubuntu/workspace/repo/Pollux |
| `LOG_DIR` | Log directory | /fsx/checkpoints/ablations/logs |
| `MASTER_PORT` | Master port for distributed training | 29500 |
| `GPUS_PER_NODE` | Number of GPUs per node | 8 |
| `CONDA_ENV` | Conda environment name | pollux |
| `CONDA_PATH` | Conda installation path | /fsx/ubuntu/miniconda3 |

## Logging

### Log File Naming

Each node generates separate log files with unique names:
- **stdout**: `/fsx/checkpoints/ablations/logs/<JOB_ID>/node_<NODE_NAME>_<JOB_ID>.stdout`
- **stderr**: `/fsx/checkpoints/ablations/logs/<JOB_ID>/node_<NODE_NAME>_<JOB_ID>.stderr`

Where:
- `<JOB_ID>` is the SLURM job ID
- `<NODE_NAME>` is the hostname of each node

### Log Directory Structure

```
/fsx/checkpoints/ablations/logs/
└── <JOB_ID>/
    ├── node_worker1_<JOB_ID>.stdout
    ├── node_worker1_<JOB_ID>.stderr
    ├── node_worker2_<JOB_ID>.stdout
    ├── node_worker2_<JOB_ID>.stderr
    └── ...
```

## Examples

### Example 1: Small Scale Training (2 nodes)

```bash
./submit_training.sh -n 2 -p learn -j small_scale_test
```

### Example 2: Large Scale Training (8 nodes)

```bash
./submit_training.sh -n 8 -p gpu -j large_scale_experiment -t 168:00:00
```

### Example 3: Custom Configuration

```bash
./submit_training.sh \
    -n 4 \
    -p gpu \
    -c apps/Castor/configs/my_custom_config.yaml \
    -j custom_experiment \
    -t 48:00:00
```

### Example 4: Different GPU Configuration

```bash
./submit_training.sh -n 4 -p gpu -g 4 --cpus-per-gpu 8
```

### Example 5: Different Conda Environment

```bash
./submit_training.sh -n 2 -p learn -e my_custom_env
```

## Monitoring Jobs

### Check Job Status

```bash
# Check your jobs
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# Check detailed job info
scontrol show job <JOB_ID>
```

### View Logs

```bash
# View logs in real-time
tail -f /fsx/checkpoints/ablations/logs/<JOB_ID>/node_*_<JOB_ID>.stdout

# View all stdout logs
cat /fsx/checkpoints/ablations/logs/<JOB_ID>/*.stdout

# View error logs
cat /fsx/checkpoints/ablations/logs/<JOB_ID>/*.stderr
```

### Cancel Jobs

```bash
# Cancel a specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

## Troubleshooting

### Common Issues

1. **Config file not found**
   - Ensure the config file path is correct and the file exists
   - Use absolute paths or paths relative to the project directory

2. **Partition not available**
   - Check available partitions: `sinfo`
   - Verify you have access to the specified partition

3. **Insufficient resources**
   - Check available nodes: `sinfo -N`
   - Reduce the number of nodes or wait for resources to become available

4. **Network issues between nodes**
   - Check if the master port (default 29500) is available
   - Try using a different port by setting `MASTER_PORT`

### Debug Mode

To debug issues, you can:

1. Check the job output files for detailed information
2. Use a shorter time limit for testing
3. Start with fewer nodes to isolate issues
4. Check SLURM logs: `scontrol show job <JOB_ID>`

## Comparison with Single Node Training

### Single Node Command (Original)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    -m apps.Castor.train \
    config=apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml
```

### Multi-Node Command (Generated by Scripts)
```bash
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=8 \
    --node-rank=$SLURM_NODEID \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    -m apps.Castor.train \
    config=apps/Castor/configs/aws_256_Castor_flux_qwen_fixed_siglip2.yaml
```

The key differences:
- `--nnodes` is set to the number of SLURM nodes
- `--node-rank` is set automatically for each node
- `--master-addr` and `--master-port` are set for inter-node communication
- `srun` is used to launch the command on all nodes simultaneously

## Notes

- The scripts assume 8 GPUs per node by default (can be changed)
- All nodes share the FSX filesystem, so checkpoints and logs are accessible from all nodes
- The scripts automatically handle distributed training setup
- Environment variables are set to optimize for multi-node training
- The scripts are compatible with your existing config files 

## Conda Environment Setup

The scripts automatically activate the specified conda environment on each node before running the training. By default, they use the `pollux` environment.

### Environment Activation Process

1. **Set conda path**: The scripts add the conda installation directory to PATH (`/fsx/ubuntu/miniconda3/bin` by default)
2. **Initialize conda**: Source the conda setup script (`$CONDA_PATH/etc/profile.d/conda.sh`)
3. **Activate environment**: Run `conda activate <ENV_NAME>` where `<ENV_NAME>` is the specified environment (default: `pollux`)
4. **Verify activation**: Display the active environment name, Python path, version, and PyTorch availability for debugging

### Using Different Conda Environments

You can specify a different conda environment in several ways:

#### Using the wrapper script:
```bash
./submit_training.sh -n 4 -p gpu -e my_environment
```

#### Using direct sbatch:
```bash
sbatch --export=CONDA_ENV=my_environment submit_multinode_training_flexible.slurm
```

### Using Different Conda Installation

If conda is installed in a different location, you can specify the path:

#### Using the wrapper script:
```bash
./submit_training.sh -n 4 -p gpu --conda-path /opt/conda
```

#### Using direct sbatch:
```bash
sbatch --export=CONDA_PATH=/opt/conda submit_multinode_training_flexible.slurm
```

### Prerequisites

- The specified conda environment must exist on all compute nodes
- The environment should have all required dependencies installed (PyTorch, your training code dependencies, etc.)
- Conda must be installed at the specified path on all compute nodes (default: `/fsx/ubuntu/miniconda3`)

### Troubleshooting Conda Issues

If you see errors like "conda: command not found":

1. **Check conda installation path**: Verify that conda is installed at `/fsx/ubuntu/miniconda3` or specify the correct path
2. **Use custom conda path**: If conda is installed elsewhere, use `--conda-path` option
3. **Check environment exists**: Ensure the specified conda environment exists on all compute nodes
4. **Verify shared filesystem**: Make sure the conda installation is accessible from all compute nodes 