import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def main():
    """
    Initializes distributed environment and creates a device mesh.
    """
    # Initialize the process group
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Initialized process group: Rank {rank}/{world_size}")

    # Determine the device type
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        device_type = "cuda"
        # Assign each rank to a unique GPU if available
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank} using CUDA device: {local_rank}")
    else:
        device_type = "cpu"
        print(f"Rank {rank} using CPU (CUDA not available or insufficient GPUs)")

    # Define the mesh shape (e.g., a 2D mesh with 2 replicas and 4 shards)
    # Ensure world_size is 8 (2 * 4) for this specific configuration
    if world_size != 8:
        raise ValueError(f"This example requires world_size=8 for a (2, 4) mesh, but got {world_size}")
    mesh_shape = (2, 4)

    # Create the device mesh
    # Define dimension names for clarity: 'rp' for replication, 'dp' (or 'sp') for data/shard parallelism
    mesh = init_device_mesh(device_type=device_type, mesh_shape=mesh_shape, mesh_dim_names=("rp", "dp"))

    print(f"Rank {rank}: Successfully created 2D device mesh: {mesh}")
    print(f"Rank {rank}: Mesh coordinates: {mesh.get_coordinate()}")
    print(f"Rank {rank}: Mesh groups: {mesh['dp']}")

    # Clean up the process group
    dist.destroy_process_group()
    print(f"Rank {rank}: Destroyed process group.")

if __name__ == "__main__":
    main()
