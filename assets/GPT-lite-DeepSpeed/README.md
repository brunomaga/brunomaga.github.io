Code for the post [Distributed training of a large GPT model with DeepSpeed](https://brunomaga.github.io/GPT-lite-DeepSpeed). 
- `train.py` is the main loop;
- `gptlite.py` and `benchmark.py` are the GPTlite and benchmark model implementations, adapted for DeepSped;
- `run.sh` contains several command line executions that replicate the post results;
- `ds_config.json` and `ds_config_offload.json` are the DeepSpeed config files;
