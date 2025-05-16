# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import logging
import os
import sys
import time

from omegaconf import OmegaConf
from tqdm import tqdm

cli_args = OmegaConf.from_cli()
file_cfg = OmegaConf.load(cli_args.config)
os.environ["CUDA_VISIBLE_DEVICES"] = file_cfg.distributed.gpus
os.environ["NCCL_DEBUG"] = "WARN"

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed
import wandb
import xformers.profiler
from apps.Castor.model import (Castor, ModelArgs, build_fsdp_grouping_plan,
                               get_no_recompute_ops, tp_parallelize)
from apps.main.data import AutoDataLoader, DataArgs
from apps.main.modules.schedulers import SchedulerArgs
from apps.main.utils.cal_flops import get_num_flop_per_token
from apps.main.utils.dict_tensor_data_load import DictTensorBatchIterator
from apps.main.utils.sampler import StatefulDistributedSampler
from apps.Castor.utils.flop_meter import FlopsMeter

from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import (CheckpointArgs, CheckpointManager,
                               load_from_checkpoint)
from lingua.distributed import (DistributedArgs, EnvironmentArgs,
                                check_model_value_range, dist_mean_dict,
                                get_device_mesh, get_is_master, get_local_rank,
                                get_world_size, init_signal_handler,
                                parallelize_model, requeue_slurm_job,
                                setup_env, setup_torch_distributed)
from lingua.logger import init_logger
from lingua.metrics import (GPUMemoryMonitor, LoggingArgs, MetricLogger,
                            get_num_params)
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import lr_scheduler

from transformer_engine.common.recipe import Format, DelayedScaling
import transformer_engine.pytorch as te

logger = logging.getLogger()


@dataclass
class TrainArgs:

    name: str = "Castor"
    version: str = "v1.0"
    train_stage: str = "preliminary"  # Align with `data` configuration
    output_dir: str = "/mnt/data/dump"
    dump_dir: str = ""
    seed: int = 42
    # shuffle: bool = False  # NOTE: detect the step = 0 to shuffle otherwise not shuffle

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None

    # Nb optimizer steps to take
    steps: int = 1000

    data: List[DataArgs] = field(default_factory=list)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: ModelArgs = field(default_factory=ModelArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    sampler: StatefulDistributedSampler

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "sampler": self.sampler.state_dict(self.step),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.sampler.load_state_dict(state_dict["sampler"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        logger.info(
            f"Resume training with distributed sampler state to {self.sampler.start_index} local step."
        )
        logger.info(
            "TrainState is loading state_dict: step, acc-step, sampler, scheduler are loaded."
        )


def validate_train_args(args: TrainArgs):
    # assert args.dump_dir, "Dump dir not set" # Mingchen: no need any more

    # Minchen: generate the dump dir according to the config
    if not args.dump_dir:
        # args.dump_dir = f"/mnt/data/dump/{args.name}"
        args.dump_dir = str(Path(args.output_dir) / f"{args.name}")

    logger.info(f"Dump dir set to {args.dump_dir}")

    if args.logging.wandb is not None:
        if not args.logging.wandb.name:
            args.logging.wandb.name = args.name
        logger.info(f"Wandb name set to {args.logging.wandb.name}")
        if not args.logging.wandb.dir:
            args.logging.wandb.dir = str(Path(args.dump_dir) / "wandb")

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {args.checkpoint.path}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    # TODO: Mingchen: here need to support multiple source later as in the original lingua codebase
    for data_args in args.data:
        if data_args.use:
            if data_args.source == "local" and not os.path.exists(data_args.root_dir):
                raise ValueError(
                    f"Local dataset root_dir '{data_args.root_dir}' does not exist."
                )
            if data_args.source == "huggingface" and not os.path.exists(
                data_args.cache_dir
            ):
                raise ValueError(
                    f"HuggingFace cache_dir '{data_args.cache_dir}' does not exist."
                )

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        assert get_world_size() % args.distributed.dp_shard == 0
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Preemption ! checkpointing asap and exiting.")
    preemption_flag["flag"] = True


def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        validate_train_args(
            args,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        # For handling preemption signals.
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        # build dataloader
        # need dp world size and rank
        print("**** world_mesh['dp_shard']", world_mesh["dp_shard"], "world_mesh['dp_replicate']", world_mesh["dp_replicate"])
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()
        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * dp_degree + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        logger.info(f"Running on dp rank : {dp_rank}")
        logger.info(f"Running on dp size : {dp_degree}")

        torch.manual_seed(args.seed)
        logger.info("Building model")

        model = Castor(args.model)
        logger.info("Model is built !")

        model_param_count = get_num_params(model)
        flops_meter = FlopsMeter(args.model, model)

        torch.manual_seed(args.seed)
        model.init_weights(args.model)
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )
        model = model.to(device="cuda")

        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        active_data = [d for d in args.data if d.stage == args.train_stage and d.use]
        data_loader_factory = AutoDataLoader(
            shard_id=dp_rank,
            num_shards=dp_degree,
            train_stage=args.train_stage,
            data_config=active_data,  # Pass the filtered data configuration
        )
        data_loader, sampler = data_loader_factory.create_dataloader()
        logger.info("Data loader is built !")
        logger.info(f"Data loader size: {len(data_loader)}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)

        train_state = TrainState(
            step=0,
            acc_step=0,
            sampler=sampler,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        # Either load from latest checkpoint or start from scratch

        fp8_recipe = None
        if args.model.diffusion_model.use_fp8_ffn: 
            logger.info("FP8 is enabled. Defining FP8 recipe.")
            # Example recipe, adjust as needed
            fp8_format = Format.HYBRID # Or Format.E4M3
            fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
            # You might want to make recipe parameters configurable via TrainArgs

        gc.disable()

        # train loop
        model.set_train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        torch_profiler = context_stack.enter_context(
            maybe_run_profiler(args.dump_dir, model, args.profiling)
        )

        dataloader_iterator = iter(data_loader)
        nwords_since_last_log = 0
        failure_rate = 0
        time_last_log = timer()
        max_data_load_time = 0.0
        gc.collect()

        pb = tqdm(total=args.steps, initial=train_state.step, desc="Training Steps")

        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            try:
                batch = next(parquet_iterator)
            except:
                try:
                    batch = next(dataloader_iterator)
                except Exception as e:
                    logger.error(f"Error getting next batch: {e}")
                    logger.error("Resetting dataloader")
                    sampler.reset()
                    dataloader_iterator = iter(data_loader)
                    batch = next(dataloader_iterator)
                parquet_iterator = DictTensorBatchIterator(
                    batch, active_data[0].dataloader.batch_size
                )
                batch = next(parquet_iterator)
            if "_id" in batch:
                failure_rate = batch["_id"].count("-1") / len(batch["_id"])
            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()
            if "latent_code" in batch:
                if isinstance(batch["latent_code"], list):
                    batch["latent_code"] = [
                        latent_code.cuda() for latent_code in batch["latent_code"]
                    ]
                    nwords_since_last_log += batch["latent_code"][0].numel() * len(
                        batch["latent_code"]
                    )
                else:
                    batch["latent_code"] = batch["latent_code"].cuda()
                    nwords_since_last_log += batch["latent_code"].numel()
            elif "image" in batch:
                if isinstance(batch["image"], list):
                    batch["image"] = [
                        image.to(device="cuda") for image in batch["image"]
                    ]
                    batch["image_cond"] = [
                        image_cond.to(device="cuda") for image_cond in batch["image_cond"]
                    ]
                    nwords_since_last_log += batch["image"][0].numel() * len(
                        batch["image"]
                    )
                else:
                    batch["image"] = batch["image"].to(device="cuda")
                    batch["image_cond"] = batch["image_cond"].to(device="cuda")
                    nwords_since_last_log += batch["image"].numel()
            else:
                raise ValueError("No image or latent code in batch")
            max_data_load_time = max(max_data_load_time, round(timer() - data_load_start, 4))

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            with te.fp8_autocast(enabled=args.model.diffusion_model.use_fp8_ffn, fp8_recipe=fp8_recipe, fp8_group=world_mesh["dp_shard"].get_group()):
                outputs = model(batch, flops_meter)
            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = outputs.loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.optim.clip, foreach=True
            )

            grad_norm = (
                grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
            ).item()

            # optimizer step
            if train_state.acc_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # if profiler is active
            if torch_profiler:
                xformers.profiler.step()

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                wps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = total_acc_steps * active_data[0].dataloader.batch_size
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                # Use xformer's analyze profile trace to get actual measurement
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "data_failure_rate": failure_rate,
                        "speed": {
                            "wps": wps,
                            "FLOPS": flops_meter.get_total_flops(time_delta),
                            "MFU": flops_meter.get_mfu(time_delta),
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": max_data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_samples": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                to_sync["loss/target"] = outputs.target_loss.item()
                if outputs.align_loss is not None:
                    to_sync["loss/align"] = outputs.align_loss.item()
                metrics.update(dist_mean_dict(to_sync))


                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {flops_meter.get_total_flops(time_delta):.2e}"
                    f"  mfu: {flops_meter.get_mfu(time_delta):.2e}"
                    f"  wps: {wps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {max_data_load_time:>5}"
                    f"  data_failure_rate: {round(failure_rate,4):>7}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W",
                )
                # Reset accumulator and counter for the next logging interval
                max_data_load_time = 0.0
                flops_meter.reset()


            saved = False
            
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ) or every_n_steps(train_state, args.checkpoint.eval.every, acc_step=0):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

            # if args.eval is not None and every_n_steps(
            #     train_state, args.checkpoint.eval.every, acc_step=0
            # ):
            #     logger.info("Evaluation Start")
            #     start_time = time.time()
            #     eval_args = dataclass_from_dict(EvalArgs, args.eval)
            #     eval_args.global_step = train_state.step
            #     eval_args.ckpt_dir = str(checkpoint.existing_saves[-1])
            #     eval_args.dump_dir = str(
            #         os.path.join(
            #             args.dump_dir,
            #             "evals",
            #             EVAL_FOLDER_NAME.format(train_state.step),
            #         )
            #     )
            #     # launch_eval(eval_args)# TODO: update eval.py later
            #     end_time = time.time()
            #     logger.info(
            #         f"Evaluation End! Take total time (sec): {end_time-start_time}"
            #     )
            # TODO: add some images to wandb for visualization
            if preemption_flag["flag"]:
                if not saved:
                    checkpoint.save(
                        model,
                        optimizer,
                        train_state,
                        args,
                        device_mesh=world_mesh,
                    )
                # Wait for the potentially just-started async save to finish
                logger.info("Waiting for preemption checkpoint save to complete...")
                checkpoint.wait_for_final_save()
                logger.info("Preemption checkpoint save complete.")
                requeue_slurm_job()
                sys.exit(0)
            
            pb.update(1)

        pb.close()

        if not saved:
            logger.info("Performing final save after training loop...")
            checkpoint.save(
                model,
                optimizer,
                train_state,
                args,
                device_mesh=world_mesh,
            )

        # Wait for the last save operation (either from last step or the final one above)
        logger.info("Waiting for final checkpoint save to complete before exiting...")
        checkpoint.wait_for_final_save()
        logger.info("Final checkpoint save complete.")

        gc.collect()
        logger.info("Training finished successfully.")


def main():

    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
