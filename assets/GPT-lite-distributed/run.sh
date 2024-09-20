#!/bin/bash

# When DeepSpeed deadlocks with "~/.cache/torch_extensions/py310_cu117 as PyTorch extensions root...", run:
rm -rf ~/.cache/torch_extensions/py*_cu* #remove all cache of cuda python extensions

# No activation checkpoint until issue is fixed: https://github.com/microsoft/DeepSpeed/issues/4274
export ACTIVATION_CHECKPOINT_ARG="--activation_checkpoint_interval 0"
export PIPELINING_ARG="--pipeline_num_stages 4 --pipeline_spec_layers"
export DEEPSPEED_ARGS="--num_gpus=8"

# single node, single GPU:
python train.py --deepspeed --deepspeed_config ds_config.json $ACTIVATION_CHECKPOINT_ARG

# distributed data parallelism (stage: 0), ZeRO-1, ZeRO-2 and ZeRO-3 (stage: 1, 2 or 3).  Replace "stage" : X, in config file
deepspeed $DEEPSPEED_ARGS train.py --deepspeed --deepspeed_config ds_config.json $ACTIVATION_CHECKPOINT_ARG

# deepspeed, 1 node, NUM_GPUS GPUs, ZeRO-3, with CPU offloading:
deepspeed $DEEPSPEED_ARGS train.py --deepspeed --deepspeed_config ds_config_offload.json $ACTIVATION_CHECKPOINT_ARG

# deepspeed, 1 node, NUM_GPUS GPUs, pipeline parallelism. Replace "stage" : 1
deepspeed $DEEPSPEED_ARGS train.py --deepspeed --deepspeed_config ds_config.json $ACTIVATION_CHECKPOINT_ARG  $PIPELINING_ARG

