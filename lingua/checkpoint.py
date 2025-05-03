# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from datetime import datetime

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.format_utils import (
    torch_save_to_dcp,
    dcp_to_torch_save,
)
import torch.optim.optimizer

from lingua.distributed import get_is_master

logger = logging.getLogger("CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")


@dataclass
class SaveEvery:
    every: int = 1000
    keep: int = 0


@dataclass
class CheckpointArgs:
    dump: SaveEvery = field(default_factory=SaveEvery)
    eval: SaveEvery = field(default_factory=SaveEvery)
    path: Optional[str] = None
    init_ckpt_path: Optional[str] = None
    continue_training_from_init: bool = False


def _get_key_step(name: str):
    return int(re.findall(RE_DIGITS, name)[-1])


def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path.mkdir(exist_ok=True)
        logger.info(f"Consolidating to: {str(consolidate_path)}")
        dcp_to_torch_save(ckpt_dir, str(consolidate_path / CONSOLIDATE_NAME))
        (consolidate_path / CONFIG_NAME).write_text(
            (Path(ckpt_dir) / CONFIG_NAME).read_text()
        )
        logger.info("Consolidated !")
    return consolidate_path


def load_from_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    if not (Path(ckpt_dir) / ".metadata").exists():
        raise ValueError(
            f"Please convert the checkpoint distcp format using `torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    state_dict = {}
    if optimizer is not None:
        state_dict[model_key], state_dict[optim_key] = get_state_dict(model, optimizer)
    else:
        state_dict[model_key] = get_model_state_dict(model)
        if model_key == "":  # If only loading a model directly, the key should be empty
            state_dict = state_dict.pop(model_key)

    dcp.load(state_dict, checkpoint_id=ckpt_dir)


class CheckpointManager:
    def __init__(self, args: CheckpointArgs):
        self.path = args.path
        self.dump_every = args.dump
        self.eval_every = args.eval
        self.init_ckpt_path = args.init_ckpt_path
        self.continue_training_from_init = args.continue_training_from_init

        assert os.path.exists(
            self.path
        ), f"Path {self.path} does not exist and needs to be created before using CheckpointManager (use instantiate_and_make_dir)"

        self.existing_saves = self.get_existing_saves()
        self.checkpoint_future = None
        self.checkpoint_process_group = None

    def get_existing_saves(self) -> List[Path]:
        folders = [
            p
            for p in Path(self.path).iterdir()
            if p.is_dir() and re.match(RE_FOLDER, p.name)
        ]
        folders.sort(key=lambda p: _get_key_step(p.name))
        return folders

    def clean_up(self):
        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_folders = []
        other_folders = []
        for p in self.existing_saves:
            is_dump = _get_key_step(p.name) % self.dump_every.every == 0
            is_eval = _get_key_step(p.name) % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval:
                eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval folders: {eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders = eval_folders[-self.eval_every.keep :]

        folder_to_keep = set(other_folders + dump_folders + eval_folders)
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if dist.get_rank() == 0:
            for folder in folder_to_remove:
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        assert file.name in [CONSOLIDATE_FOLDER]
                        for f in file.iterdir():
                            f.unlink()
                        file.rmdir()
                folder.rmdir()

        dist.barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(p.name))

    def get_last_step_path(self, dp_rank: int = 0) -> Optional[Path]:
        path = None
        for p in reversed(self.existing_saves):
            if (p / TRAIN_STATE_NAME.format(dp_rank)).is_file():
                path = p
                break
        return path

    def _create_folder(self, base_path: Path, folder_name: str) -> Path:
        folder = base_path / folder_name
        if get_is_master():
            folder.mkdir(parents=False, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return folder

    def _get_dp_tp_mesh(
        self, device_mesh: Optional[DeviceMesh] = None
    ) -> Tuple[int, int]:
        dp_rank = 0
        tp_rank = 0
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                dp_rank = device_mesh.get_local_rank("dp_replicate")
                if "dp_shard" in device_mesh.mesh_dim_names:
                    dp_rank = dp_rank * device_mesh[
                        "dp_replicate"
                    ].size() + device_mesh.get_local_rank("dp_shard")
            if "tp" in device_mesh.mesh_dim_names:
                tp_rank = device_mesh.get_local_rank("tp")
        return dp_rank, tp_rank

    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
    ):
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def _get_optimal_thread_count(self):
        """Calculates the optimal thread count for I/O operations based on system resources.
        Aims to balance between using sufficient threads for I/O without overcommitting.
        """
        try:
            import multiprocessing
            # Use half of available CPU cores, but at least 2 and at most 8
            cpu_count = multiprocessing.cpu_count()
            return max(2, min(cpu_count // 2, 8))
        except:
            # Default to 4 threads if we can't determine CPU count
            return 4

    def _create_optimized_writer(self, checkpoint_dir):
        """Creates an optimized FileSystemWriter with pinned memory cache for faster GPU-to-CPU transfers."""
        thread_count = self._get_optimal_thread_count()
        return FileSystemWriter(
            path=checkpoint_dir,
            single_file_per_rank=True,  # One file per rank is more efficient
            sync_files=True,
            thread_count=thread_count,  # Use multiple threads for I/O operations
            per_thread_copy_ahead=50000000,  # 50MB buffer
            cache_staged_state_dict=True,  # Enable pinned memory caching
            overwrite=True
        )

    def save(
        self,
        model,
        optimizer,
        train_state,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> bool:

        # When creating directory check if only rank0 or is there other solution
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving asynchronously to: {str(curr_save_dir)}")

        if self.checkpoint_future is not None and not self.checkpoint_future.done():
            logger.info("Waiting for previous checkpoint save to finish...")
            try:
                # Set a timeout for waiting for the previous save operation
                self.checkpoint_future.result(timeout=1800)  # 30 minutes timeout
                logger.info("Previous checkpoint save finished.")
            except TimeoutError:
                logger.error("Previous checkpoint save timed out after 30 minutes! Proceeding with new save.")
                # The checkpoint_future is now considered invalid, so we'll null it out
                self.checkpoint_future = None
            except Exception as e:
                logger.error(f"Error during previous checkpoint save: {str(e)}")
                # The checkpoint_future is now considered invalid, so we'll null it out
                self.checkpoint_future = None

        if dist.is_initialized():
            dist.barrier()

        # Ensure we have a checkpoint process group
        if self.checkpoint_process_group is None:
            self.initialize_checkpoint_process_group()
            
        logger.info("Getting state dict for async save...")
        state_dict = self.get_state_dict(model, optimizer)

        # Create an optimized storage writer with pinned memory for faster GPU-to-CPU transfers
        storage_writer = self._create_optimized_writer(curr_save_dir)
        
        logger.info("Initiating asynchronous save...")
        self.checkpoint_future = dcp.async_save(
            state_dict,
            storage_writer=storage_writer,
            process_group=self.checkpoint_process_group,  # Use dedicated process group
        )
        logger.info("Asynchronous save initiated.")

        if get_is_master():
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            logger.info(
                f"Saving train state to: {str(curr_save_dir / train_state_name)}"
            )
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        self.existing_saves.append(curr_save_dir)

        self.clean_up()

        return True

    def wait_for_final_save(self):
         if self.checkpoint_future is not None and not self.checkpoint_future.done():
            logger.info("Waiting for final checkpoint save to finish...")
            try:
                # Set a timeout for the final save to avoid hanging indefinitely
                self.checkpoint_future.result(timeout=1800)  # 30 minutes timeout
                logger.info("Final checkpoint save finished.")
            except TimeoutError:
                logger.error("Final checkpoint save timed out after 30 minutes. This may indicate network issues or resource contention.")
                if dist.is_initialized():
                    dist.barrier()  # Ensure all processes are synchronized
            except Exception as e:
                logger.error(f"Error during final checkpoint save: {str(e)}")
                if dist.is_initialized():
                    dist.barrier()  # Ensure all processes are synchronized

    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: Optional[Path] = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            logger.info("No checkpoints found ! Init train state from sratch...")
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        logger.info("Reloading train state")
        with open(path / train_state_name, "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        # Ensure we have a checkpoint process group for loading
        if self.checkpoint_process_group is None:
            self.initialize_checkpoint_process_group()

        logger.info(f"Loading from: {str(path)}")
        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        dcp.load(
            state_dict, 
            checkpoint_id=path,
            process_group=self.checkpoint_process_group,  # Use dedicated process group for loading too
        )
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")

    @classmethod
    def instantiate_and_make_dir(cls, args: CheckpointArgs):
        if get_is_master():
            os.makedirs(args.path, exist_ok=True)
        dist.barrier()

        return cls(args)

    def initialize_checkpoint_process_group(self):
        """
        Initialize a separate process group dedicated for checkpoint operations.
        This helps avoid NCCL conflicts between training and checkpoint operations.
        
        Call this after the main process group is initialized.
        """
        if dist.is_initialized() and self.checkpoint_process_group is None:
            # Use gloo backend as it's CPU-based and works well for checkpoint operations
            try:
                # Create a unique name for the checkpoint process group
                group_name = f"checkpoint_group_{int(datetime.now().timestamp())}"
                # Initialize the process group with gloo backend
                # We need to create a new process group with gloo backend to avoid NCCL conflicts
                gloo_pg = dist.new_group(
                    ranks=list(range(dist.get_world_size())),
                    backend="gloo",
                    group_desc=group_name
                )
                self.checkpoint_process_group = gloo_pg
                logger.info(f"Initialized separate process group for checkpointing: {group_name}")
                return self.checkpoint_process_group
            except Exception as e:
                logger.error(f"Failed to initialize checkpoint process group: {str(e)}")
                return None
        return self.checkpoint_process_group

    def get_checkpoint_status(self):
        """
        Returns the status of the current checkpoint operation.
        
        Returns:
            dict: Status information about current checkpoint
                - has_pending: Whether there's a pending checkpoint
                - is_complete: Whether the checkpoint is complete (if any)
                - error: Error message (if any)
        """
        status = {
            "has_pending": self.checkpoint_future is not None,
            "is_complete": False,
            "error": None
        }
        
        if self.checkpoint_future is not None:
            status["is_complete"] = self.checkpoint_future.done()
            if self.checkpoint_future.done():
                try:
                    # Check if there was an exception
                    self.checkpoint_future.result(timeout=0)
                except Exception as e:
                    status["error"] = str(e)
                    
        return status
