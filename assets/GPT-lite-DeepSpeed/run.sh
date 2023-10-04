#!/bin/bash

# single node, single GPU:
# python train.py --deepspeed --deepspeed_config ds_config.json

# deepspeed, 1 node, 8 GPUs:
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json #--run_memory_estimation_only

# deepspeed, 1 node, 8 GPUs, pipeline parallelism:
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json --pipeline --pipeline_spec_layers  #--run_memory_estimation_only


