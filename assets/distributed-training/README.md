Support material for the distributed training posts, starting with [Distributed training: data parallelism, sharding and CPU offloading](https://brunomaga.github.io/data-parallelism).
- `train.py` is the main loop. Launch with `python train.py` for a single-node single-GPU run or launch with `deepspeed train.py` for distributed runs;
- `gptlite_ds.py` and `benchmark.py` are the GPTlite and benchmark model implementations, adapted for DeepSpeed;
- `megatron_model_parallelism.py`, `ulysses_sequence_parallelism.py` and `ring_attention_sequence_parallelism.py` implement tensor parallelism (Megatron-LM) and sequence/context parallelism (Ulysses and Ring attention);
- `moe.py` implements the Mixture-of-Experts layers, and `moe_dist.py` their distributed version with one expert per GPU (expert parallelism);
- `run.sh` contains several command line executions that replicate the post results;
- `ds_config.json` and `ds_config_offload.json` are the DeepSpeed config files;
- `imgs/` holds the figures of the posts and their sources;
