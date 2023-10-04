#!/bin/bash

# single node, single GPU:
python train.py --deepspeed --deepspeed_config ds_config.json

# distributed data parallelism, no ZeRO
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config_ddp.json

# deepspeed, 1 node, 8 GPUs, ZeRO-3:
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json

# deepspeed, 1 node, 8 GPUs, ZeRO-3, with CPU offloading:
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config_offload.json

# deepspeed, 1 node, 8 GPUs, pipeline parallelism, ZeRO-1:
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config_pipe.json --pipeline --pipeline_spec_layers 


