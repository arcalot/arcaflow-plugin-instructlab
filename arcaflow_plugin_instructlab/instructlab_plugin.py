#!/usr/bin/env python3

import sys
import typing
from arcaflow_plugin_sdk import plugin
from instructlab_plugin_schema import (
    InputParams,
    SuccessOutput,
    ErrorOutput,
)
from instructlab.training.main_ds import run_training
import instructlab.training.config
import os


@plugin.step(
    id="run-train",
    name="Run instructlab training",
    description="Starts training with InstructLab",
    outputs={"success": SuccessOutput, "error": ErrorOutput},
)
def run_train(
        params: InputParams,
) -> typing.Tuple[str, typing.Union[SuccessOutput, ErrorOutput]]:
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch_args = instructlab.training.TorchrunArgs(
        nproc_per_node=params.pytorch_args.nproc_per_node,
        nnodes=params.pytorch_args.nnodes,
        node_rank=0,
        rdzv_id=123,
        rdzv_endpoint='0.0.0.0:8888'
    )
    if params.training_args.cpu_offload_enabled:
        deepspeed_options = instructlab.training.config.DeepSpeedOptions(
            cpu_offload_optimizer=params.training_args.cpu_offload_enabled,
            cpu_offload_optimizer_pin_memory=params.training_args.cpu_offload_settings.pin_memory,
            cpu_offload_optimizer_ratio=params.training_args.cpu_offload_settings.optimizer_ratio,
        )
    else:
        deepspeed_options = instructlab.training.config.DeepSpeedOptions(
            cpu_offload_optimizer=False,
            cpu_offload_optimizer_ratio=1,
            cpu_offload_optimizer_pin_memory=False,
        )
    train_args = instructlab.training.TrainingArgs(
        model_path=params.training_args.model_path,
        data_path=params.training_args.data_path,
        ckpt_output_dir=params.training_args.ckpt_output_dir,
        data_output_dir=params.training_args.data_output_dir,
        max_seq_len=4096,
        max_batch_len=params.training_args.max_batch_len,
        num_epochs=params.training_args.num_epochs,
        effective_batch_size=params.training_args.effective_batch_size,
        save_samples=params.training_args.samples_before_checkpoint,
        learning_rate=2e-5,
        warmup_steps=385,
        is_padding_free=params.training_args.is_padding_free,
        random_seed=42,
        deepspeed_options=deepspeed_options,
    )
    run_training(torch_args, train_args)
    return "success", SuccessOutput("ran; no output")


if __name__ == "__main__":
    sys.exit(
        plugin.run(
            plugin.build_schema(
                # List your step functions here:
                run_train,
            )
        )
    )
