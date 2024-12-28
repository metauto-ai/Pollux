import os
import numpy as np

from pprint import pformat
from typing import Iterator, List, Optional
from datetime import datetime
import math

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

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
        save_dir: str = "/jfs/data/sampler",
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
        self.save_dir = save_dir


    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
            
        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples
        
        # * clip index list from start from state
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self, spefic_index: int = 0, shuffle: Optional[bool] = False) -> None:

        # NOTE: we will have situations that we could mannually recover from a specific index
        #       Beside, I think set `shuffle = False` as default makes more sense according to our usage.
        self.start_index = 0 if spefic_index == 0 else spefic_index

        # NOTE: Previously, we set `shuffle = True`` by default, if so, it it better add the shuffle function.
        #       Otherwise we need to set it to False by default. In our situation, we don't need to shuffle frequently.
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_indices()

    def shuffle_indices(self) -> None:

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
            if hasattr(self.dataset, "indices"):
                # Update dataset's internal indices if supported
                self.dataset.indices = indices

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

    def save_state(self, global_step: int) -> None:

        # NOTE: For safety, it is better we save the minimal info,
        # e.g., start index, seed, shuffle, global step for recovery.
        state = {
            "start_index": self.start_index,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "global_step": global_step,
        }
        # Create a timestamped folder within the save_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = os.path.join(self.save_dir, timestamp)
        os.makedirs(timestamped_dir, exist_ok=True)

        state_path = os.path.join(
            timestamped_dir, f"sampler_state_rank_{self.shard_id}.pth"
        )
        torch.save(state, state_path)

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
