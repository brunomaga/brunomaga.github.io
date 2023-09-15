#!/bin/bash

# serial, single GPU:
# python gptlite_deepspeed.py --deepspeed --deepspeed_config gptlite_config_ds.json

# deepspeed, 8 GPUs:
deepspeed --num_gpus=8 gptlite_deepspeed.py --deepspeed --deepspeed_config gptlite_config_ds.json
