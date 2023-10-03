#!/bin/bash

# serial, single GPU:
# python train.py --deepspeed --deepspeed_config ds_config.json

# deepspeed, 8 GPUs:
deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json

