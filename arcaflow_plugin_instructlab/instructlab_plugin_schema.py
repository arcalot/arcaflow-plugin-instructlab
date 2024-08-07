#!/usr/bin/env python3

import typing
from dataclasses import dataclass
from arcaflow_plugin_sdk import validation, schema


@dataclass
class TorchrunArgs:
    nproc_per_node: typing.Annotated[
        typing.Optional[int],
        schema.name("pmlogger metrics"),
        schema.description("pmlogger metrics to report"),
    ] = 1
    nnodes: typing.Annotated[
        schema.Optional[int],
        schema.name("number of nodes"),
        schema.description("Number of nodes used for training"),
    ] = 1
    # node_rank
    # rdzv_id
    # rdzv_endpoint


@dataclass
class CpuOffloadSettings:
    pin_memory: typing.Annotated[
        schema.Optional[bool],
        schema.name("pin memory"),
        schema.description("whether to pin memory")
    ] = False
    optimizer_ratio: typing.Annotated[
        schema.Optional[float],
        schema.name("optimizer ratio"),
        schema.description("Adjust the ratio of parameters updating (i.e. optimizer step) on CPU side")
    ] = 1.0


@dataclass
class TrainingArgs:
    model_path: typing.Annotated[
        str,
        schema.name("model path"),
        schema.description("path to the model used for training")
    ]
    data_path: typing.Annotated[
        str,
        schema.name("data path"),
        schema.description("the path to the training data (jsonl files)")
    ]
    ckpt_output_dir: typing.Annotated[
        schema.Optional[str],
        schema.name("checkpoint output directory"),
        schema.description("Path to the output model checkpoints")
    ] = "/tmp"
    data_output_dir: typing.Annotated[
        schema.Optional[str],
        schema.name("data output directory"),
        schema.description("Directory to output preprocessed data")
    ] = "/dev/shm"
    num_epochs: typing.Annotated[
        schema.Optional[int],
        schema.name("epoch count"),
        schema.description("the number of epochs to train")
    ] = 1
    is_padding_free: typing.Annotated[
        schema.Optional[bool],
        schema.name("is padding free"),
        schema.description("whether the model is padding free; necessary for models with dolomite format")
    ] = False
    effective_batch_size: typing.Annotated[
        schema.Optional[int],
        schema.name("effective batch size"),
        schema.description("Batch size scaling factor")
    ] = 3840
    max_batch_len: typing.Annotated[
        schema.Optional[int],
        schema.name("maximum batch length"),
        schema.description("Maximum length of a batch")
    ] = 60000
    cpu_offload_enabled: typing.Annotated[
        schema.Optional[bool],
        schema.name("cpu offload enabled"),
        schema.description("whether cpu offload is enabled; if enabled, provide cpu_offload_settings")
    ] = False
    cpu_offload_settings: typing.Annotated[
        schema.Optional[CpuOffloadSettings],
        schema.name("cpu offload settings"),
        schema.description("cpu offload settings, for use when CPU offload is enabled")
    ] = None
    samples_before_checkpoint: typing.Annotated[
        schema.Optional[int],
        schema.name("samples before checkpoint"),
        schema.description("Number of samples to process before saving a checkpoint, default is intentionally high"
                           " to prevent checkpoint saving")
    ] = 47999


@dataclass
class InputParams:
    """
    This is the data structure for the input parameters of the step defined
    below.
    """
    pytorch_args: TorchrunArgs
    training_args: TrainingArgs


@dataclass
class SuccessOutput:
    """
    This is the output data structure for the success case.
    """

    message: str


@dataclass
class ErrorOutput:
    """
    This is the output data structure in the error  case.
    """

    error: str
