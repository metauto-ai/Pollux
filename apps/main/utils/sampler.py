import os
import numpy as np

from pprint import pformat
from typing import Iterator, List, Optional
from datetime import datetime
import math

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class StatefulDistributedSampler(Sampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,  # True,
        seed: int = 0,
        drop_last: bool = False,
        # save_dir: str = "/jfs/data/sampler",
    ) -> None:
        """
        Though we don't use torch distributed, we reframe our shrading into distribued like sampler.

        Args:
            dataset: Dataset used for sampling.
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is retrieved from the
                current distributed group.
                * the alias for num_shard
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed
                group.
                * the alias for shard_id
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
                indices.
            seed (int, optional): random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across all
                processes in the distributed group. Default: ``0``.
            drop_last (bool, optional): if ``True``, then the sampler will drop the
                tail of the data to make it evenly divisible across the number of
                replicas. If ``False``, the sampler will add extra indices to make
                the data evenly divisible across the replicas. Default: ``False``.

            * shard_id and num_shard could be changed when loading and saving, we only save start idx
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # * no need to consider sharding
        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples
        self.shuffle = shuffle
        self.seed = seed

        # * most important variable, dataset traverse pointer
        self.start_index: int = 0
        self.shard_id = rank
        self.num_shard = num_replicas
        self.shuffle = shuffle
        # self.save_dir = save_dir

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).numpy()  # type: ignore[arg-type]
        else:
            indices = np.arange(len(self.dataset))
            # list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]))
            else:
                repeat_count = math.ceil(padding_size / len(indices))
                indices = np.concatenate(
                    (indices, np.tile(indices, repeat_count)[:padding_size])
                )
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        # * clip index list from start from state
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self, spefic_index: int = 0) -> None:

        # NOTE: we will have situations that we could mannually recover from a specific index
        #       Beside, I think set `shuffle = False` as default makes more sense according to our usage.
        self.start_index = 0 if spefic_index == 0 else spefic_index

        # NOTE: Previously, we set `shuffle = True`` by default, if so, it it better add the shuffle function.
        #       Otherwise we need to set it to False by default. In our situation, we don't need to shuffle frequently.
        # if shuffle is None:
        #     shuffle = self.shuffle
        # if shuffle:
        #     self.shuffle_indices()

    # def shuffle_indices(self) -> None:

    #     if self.shuffle:
    #         generator = torch.Generator()
    #         generator.manual_seed(self.seed)
    #         indices = torch.randperm(len(self.dataset), generator=generator).tolist()
    #         if hasattr(self.dataset, "indices"):
    #             # Update dataset's internal indices if supported
    #             self.dataset.indices = indices

    def state_dict(self, global_step) -> dict:
        local_step = (global_step * self.batch_size) % self.num_samples
        # * we shouldn't update start index during training
        # as it should only be init once the training start in self.__iter__
        return {"start_index": local_step}

    def load_state_dict(self, state_dict: dict) -> None:
        # self.__dict__.update(state_dict)
        # TODO: Please check here. We may explicit load the controlable attributes.
        self.start_index = state_dict.get("start_index", 0)
        self.seed = state_dict.get("seed", self.seed)
        self.shuffle = state_dict.get("shuffle", self.shuffle)

    # NOTE: we have already save TrainState which includes sampler in https://github.com/metauto-ai/Pollux/blob/main/apps/main/train.py#L543
    # def save_state(self, global_step: int) -> None:

    #     # NOTE: For safety, it is better we save the minimal info,
    #     # e.g., start index, seed, shuffle, global step for recovery.
    #     state_set_info = {
    #         "start_index": self.start_index,
    #         "seed": self.seed,
    #         "shuffle": self.shuffle,
    #         "global_step": global_step,
    #     }
    #     # Create a timestamped folder within the save_dir
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     timestamped_dir = os.path.join(self.save_dir, timestamp)
    #     os.makedirs(timestamped_dir, exist_ok=True)

    #     state_path = os.path.join(
    #         timestamped_dir, f"sampler_state_rank_{self.shard_id}.pth"
    #     )
    #     torch.save(state_set_info, state_path)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class StatefulChunkSampler(Sampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_chunks: int,
        chunk_len: int,
        num_workers: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,  # True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        Though we don't use torch distributed, we reframe our shrading into distribued like sampler.

        Args:
            dataset: Dataset used for sampling.
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is retrieved from the
                current distributed group.
                * the alias for num_shard
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed
                group.
                * the alias for shard_id
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
                indices.
            seed (int, optional): random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across all
                processes in the distributed group. Default: ``0``.
            drop_last (bool, optional): if ``True``, then the sampler will drop the
                tail of the data to make it evenly divisible across the number of
                replicas. If ``False``, the sampler will add extra indices to make
                the data evenly divisible across the replicas. Default: ``False``.

            * shard_id and num_shard could be changed when loading and saving, we only save start idx
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # * no need to consider sharding
        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples
        self.shuffle = shuffle
        self.seed = seed

        # * most important variable, dataset traverse pointer
        self.start_index: int = 0
        self.shard_id = rank
        self.num_shard = num_replicas
        self.shuffle = shuffle
        # self.save_dir = save_dir

        # * for chunk dataloader only
        self.num_chunks = num_chunks
        self.num_workers = num_workers
        self.chunk_len = chunk_len

        logging.info(
            f"Set num_chunks {self.num_chunks} num_workers {self.num_workers} chunk_len {self.chunk_len}"
        )

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).numpy()  # type: ignore[arg-type]
        else:
            indices = np.arange(len(self.dataset))
            # list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]))
            else:
                repeat_count = math.ceil(padding_size / len(indices))
                indices = np.concatenate(
                    (indices, np.tile(indices, repeat_count)[:padding_size])
                )
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        # * == NOTE: chunk aware stateful sampler implementation by jinjie ==
        """
        The way torch datalaoder worker operates is every worker featch a batch by order
        
        # Example input parameters (you should replace these with actual values)
        num_chunks = 8  # Total number of chunks
        chunk_len = 10  # Length of each chunk
        num_workers = 4  # Number of workers
        chunk_index = 1  # Start from this chunk index
        
        worker_chunks:
        worker 0 append [1 2], update current chunk 3
        worker 1 append [3 4], update current chunk 5
        worker 2 append [5 6], update current chunk 7
        worker 3 append [7 7], update current chunk 8
        
        worker_item_indices:
            worker 0
        [[[10 11 12 13 14 15 16 17 18 19] # chunk 1
        [20 21 22 23 24 25 26 27 28 29]] # chunk 2

        [[30 31 32 33 34 35 36 37 38 39]
        [40 41 42 43 44 45 46 47 48 49]]

        [[50 51 52 53 54 55 56 57 58 59]
        [60 61 62 63 64 65 66 67 68 69]]

        [[70 71 72 73 74 75 76 77 78 79]
        [70 71 72 73 74 75 76 77 78 79]]]
        
        
        final global item indices: 
        [10 30 50 70 || 11 31 51 71 || 12 32 52 72 13 33 53 73 14 34 54 74 15 35 55 75
        16 36 56 76 17 37 57 77 18 38 58 78 19 39 59 79 20 40 60 70 21 41 61 71
        22 42 62 72 23 43 63 73 24 44 64 74 25 45 65 75 26 46 66 76 27 47 67 77
        28 48 68 78 29 49 69 79]
                
        """
        # Generate an array of chunk lengths (each chunk has length `chunk_len`)
        chunk_lengths = np.full(self.num_chunks, self.chunk_len)
        # Compute the cumulative sum to determine the boundaries of each chunk
        cumulative_sum = np.cumsum(chunk_lengths)
        # Use np.searchsorted to find which chunk the start index belongs to
        chunk_index = np.searchsorted(cumulative_sum, self.start_index)
        new_start_index = cumulative_sum[chunk_index] - self.chunk_len
        if new_start_index != self.start_index:
            logger.warning(
                f"We should resume from start index {self.start_index}, but for chunk dataset, "
                f"we need to restart from chunk {chunk_index}, this results in moving the new "
                f"start index to the first item of chunk {chunk_index}: item {new_start_index}"
            )
        else:
            logger.warning(f"Successfully start/resume from item {self.start_index}....")
        self.start_index=new_start_index

        # * Step 1: Determine the total remaining chunks to be assigned
        remaining_chunks = self.num_chunks - chunk_index
        # * Step 2: Distribute chunks to workers
        # Calculate how many chunks each worker should handle
        base_chunks_per_worker = remaining_chunks // self.num_workers  # Equal chunks per worker
        extra_chunks = remaining_chunks % self.num_workers  # Any extra chunks to distribute
        # * Step 3: Calculate the chunk indices for each worker
        worker_chunk_ranges = []
        current_chunk = chunk_index
        for i in range(self.num_workers):
            # Calculate the number of chunks this worker will handle
            chunks_for_worker = base_chunks_per_worker + (1 if i < extra_chunks else 0)

            # Assign the chunks to the worker
            chunks = np.arange(current_chunk, current_chunk + chunks_for_worker)

            # If the worker has fewer chunks (i.e., it's one of the last workers), pad the last chunk
            if extra_chunks>0 and i >= extra_chunks:  # These workers got fewer chunks
                # if out of bound, pad last chunk else last chunk of this worker instead
                pad_chunk_index = self.num_chunks - 1 if base_chunks_per_worker == 0 else chunks[-1]
                padding = np.full(1, pad_chunk_index)  # Pad with the last chunk
                chunks = np.concatenate([chunks, padding])
            worker_chunk_ranges.append(chunks)
            current_chunk += chunks_for_worker
            # print(f"worker {i} append {chunks}, update current chunk {current_chunk}")
            # Convert worker_chunk_ranges to a NumPy array for better manipulation

        # Convert worker_chunk_ranges to a NumPy array for better manipulation
        worker_chunks = np.array(
            worker_chunk_ranges
        )  # Shape: (num_workers, max_chunks_per_worker)
        # Step 1: Generate the start indices for each chunk

        logging.info(f"Worker chunk arrangement: {worker_chunks}")

        start_indices = worker_chunks * self.chunk_len  # Shape: (num_workers, max_chunks_per_worker)

        # Step 2: Generate global indices for all chunks by broadcasting np.arange(chunk_len)
        # Repeat np.arange(chunk_len) for each chunk and worker
        repeated_chunk_len = np.tile(
            np.arange(self.chunk_len), (worker_chunks.shape[0], worker_chunks.shape[1], 1)
        )  # Shape: (num_workers, max_chunks_per_worker, chunk_len)

        # Step 3: Add the start indices to the repeated chunk indices
        start_indices = start_indices[
            ..., np.newaxis
        ]  # Reshape to (num_workers, max_chunks_per_worker, 1)

        # Perform broadcasting correctly by adding the chunk indices
        worker_item_indices = (start_indices + repeated_chunk_len).reshape(
            self.num_workers, -1
        )  # Shape: (num_workers, max_chunks_per_worker, chunk_len) -> (num_workers, worker_total_samples)

        logging.info(f"Result in per worker indices: {worker_item_indices}")

        global_indices = []
        worker_total_samples = worker_item_indices.shape[1]
        num_batches_per_worker = (worker_total_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches_per_worker): # local batch idx for each worker
            for worker_idx in range(self.num_workers):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, worker_total_samples)

                # Append indices for this worker's batch if they exist
                if start_idx < worker_total_samples:
                    global_indices.extend(worker_item_indices[worker_idx, start_idx:end_idx])

        # start=0
        # for idx in range(self.num_workers*16):
        #     print(idx, global_item_indices[start:start+self.num_workers])
        #     start+=self.num_workers

        return iter(global_indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self, spefic_index: int = 0) -> None:

        self.start_index = 0 if spefic_index == 0 else spefic_index

    def state_dict(self, global_step) -> dict:
        local_step = (global_step * self.batch_size) % self.num_samples
        # * we shouldn't update start index during training
        # as it should only be init once the training start in self.__iter__
        return {
            "start_index": local_step,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "chunk_len": self.chunk_len,
            "num_chunks": self.num_chunks,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        # self.__dict__.update(state_dict)
        self.start_index = state_dict.get("start_index", 0)

        if "seed" in state_dict and state_dict["seed"] != self.seed:
            raise ValueError(
                f"Mismatch in seed. Expected {self.seed}, got {state_dict['seed']}"
            )

        if "shuffle" in state_dict and state_dict["shuffle"] != self.shuffle:
            raise ValueError(
                f"Mismatch in shuffle. Expected {self.shuffle}, got {state_dict['shuffle']}"
            )

        # * num-workers could be different, but chunk dataset must be the same when resume training
        # if "num_workers" in state_dict and state_dict["num_workers"] != self.num_workers:
        #     raise ValueError(f"Mismatch in num_workers. Expected {self.num_workers}, got {state_dict['num_workers']}")

        if "chunk_len" in state_dict and state_dict["chunk_len"] != self.chunk_len:
            raise ValueError(
                f"Mismatch in chunk_len. Expected {self.chunk_len}, got {state_dict['chunk_len']}"
            )

        if "num_chunks" in state_dict and state_dict["num_chunks"] != self.num_chunks:
            raise ValueError(
                f"Mismatch in num_chunks. Expected {self.num_chunks}, got {state_dict['num_chunks']}"
            )

        # Load the state_dict values
        self.seed = state_dict.get("seed", self.seed)
        self.shuffle = state_dict.get("shuffle", self.shuffle)
        self.num_workers = state_dict.get("num_workers", self.num_workers)
        self.chunk_len = state_dict.get("chunk_len", self.chunk_len)
        self.num_chunks = state_dict.get("num_chunks", self.num_chunks)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
