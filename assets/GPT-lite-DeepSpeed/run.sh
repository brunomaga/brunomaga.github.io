#!/bin/bash

# When DeepSpeed deadlocks with "~/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...", run:
rm -rf ~/.cache/torch_extensions/py*_cu* #remove all cache of cuda python extensions

# No activation checkpoint until issue is fixed: https://github.com/microsoft/DeepSpeed/issues/4274
export ACTIVATION_CHECKPOINT_ARG="--activation_checkpoint_interval 0"
export PIPELINING_ARG="--pipeline_parallel_size 4 --pipeline_spec_layers"
export NUM_GPUS=8

# single node, single GPU:
python train.py --deepspeed_config ds_config_serial.json $ACTIVATION_CHECKPOINT_ARG

# distributed data parallelism, no ZeRO
deepspeed --num_gpus=$NUM_GPUS train.py --deepspeed_config ds_config_ddp.json $ACTIVATION_CHECKPOINT_ARG

# deepspeed, 1 node, NUM_GPUS GPUs, ZeRO-3:
deepspeed --num_gpus=$NUM_GPUS train.py --deepspeed_config ds_config_ZeRO3.json $ACTIVATION_CHECKPOINT_ARG

# deepspeed, 1 node, NUM_GPUS GPUs, ZeRO-3, with CPU offloading:
deepspeed --num_gpus=$NUM_GPUS train.py --deepspeed_config ds_config_offload.json $ACTIVATION_CHECKPOINT_ARG

# deepspeed, 1 node, NUM_GPUS GPUs, pipeline parallelism, ZeRO-1:
deepspeed --num_gpus=$NUM_GPUS train.py --deepspeed_config ds_config_pipe.json $ACTIVATION_CHECKPOINT_ARG  $PIPELINING_ARG


