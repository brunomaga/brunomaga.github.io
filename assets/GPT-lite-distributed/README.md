Support material for the post [Distributed training of a large GPT model with DeepSpeed](https://brunomaga.github.io/GPT-lite-DeepSpeed). 
- `train.py` is the main loop. Launch with `python train.py` for a single-node single-GPU run or launch with `deepspeed train.py` for distributed runs;
- `gptlite.py` and `benchmark.py` are the GPTlite and benchmark model implementations, adapted for DeepSped;
- `run.sh` contains several command line executions that replicate the post results;
- `ds_config.json` and `ds_config_offload.json` are the DeepSpeed config files;
