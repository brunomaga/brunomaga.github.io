import torch
import torch.distributed as dist
import os
import deepspeed
from datetime import datetime


# DeepSpeed requires a distributed environment even when only one process is used.
if os.environ.get("WORLD_SIZE") is None:
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "9991"  # modify if RuntimeError: errno: 98 - Address already in use
  os.environ["RANK"] = "0"
  os.environ["LOCAL_RANK"] = "0"
  os.environ["WORLD_SIZE"] = "1"


def get_cmd_line_args(description='GPT lite on DeepSpeed'):
  import argparse
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
  parser.add_argument('--activation_checkpoint_interval', type=int, default=0,
                      help='activation checkpoint interval (0 means disabled)')
  parser.add_argument('--pipeline_num_stages', type=int, default=0,
                      help='enable pipeline parallelism with N stages (0 means disabled)')
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable SpecLayers in pipeline parallelism")
  # Include DeepSpeed configuration arguments (--deepspeed, --deepspeed_config, ...)
  parser = deepspeed.add_config_arguments(parser)
  return parser.parse_args()


  
def measure_parameters_memory(model, args):

  ranks = list(range(dist.get_world_size())) if args.pipeline_num_stages else [0]
  for r in ranks:
    if args.local_rank == r:
      print(f"\n\nEstimating memory requirements at local rank {r}")
      param_size_GB = sum([p.nelement() * p.element_size() for p in model.parameters()])/1024**3
      print(f"Native model parameters size: {round(param_size_GB, 3)}GB.")

      from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
      estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
      
      from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
      estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

    torch.distributed.barrier()
    
class CrossEntropyLoss_FlatView(torch.nn.Module):
  def forward(self, logits, labels):
    # from shape (B,T,C) to (B*T,C)
    B, T, C = logits.shape
    logits = logits.view(B*T,C)
    labels = labels.view(-1)
    # fix RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    # logits = logits.requires_grad_(True) # https://github.com/microsoft/DeepSpeed/issues/4274
    return torch.nn.functional.cross_entropy(logits, labels)

def main_deepspeed(n_epochs=100, random_seed=42, model='gptlite'):
  torch.manual_seed(random_seed)  #set random seed
  deepspeed.runtime.utils.set_random_seed(random_seed)
  deepspeed.init_distributed() #initialize distributed network
  args = get_cmd_line_args() # get command line arguments
  criterion = CrossEntropyLoss_FlatView() #initialize loss function

  get_model_kwargs = {
    'criterion': criterion,
    'pipeline_num_stages': args.pipeline_num_stages,
    'pipeline_spec_layers': args.pipeline_spec_layers,
    'activation_checkpoint_interval': args.activation_checkpoint_interval,
    }

  if model.lower()=='gptlite':
    import gptlite_ds as gptlite
    train_dataset, _, vocab_size = gptlite.get_dataset()
    model = gptlite.get_model(vocab_size, **get_model_kwargs)
  elif model.lower()=='benchmark':
    import benchmark
    W, L = 8192, 3 # wide model
    W, L = 256, 2048 # deep model
    train_dataset, _ = benchmark.get_dataset(in_size=W, out_size=W)
    model = benchmark.get_model(W, L, in_size=W, out_size=W, **get_model_kwargs)

  #estimate parameters memory requirements
  measure_parameters_memory(model, args)

  engine, _, train_dataloader , _ = deepspeed.initialize(
    args=args, model=model, training_data=train_dataset,
    #model_parameters=[p for p in model.parameters() if p.requires_grad] #optional
    #config=args.deepspeed_config, #only needed when args.deepspeed_config does not exist
    )

  assert engine.local_rank == dist.get_rank()
  print(f"{str(datetime.now())[:22]} :: Starting training. Rank {engine.local_rank} on {engine.device}")

  for epoch in range(n_epochs):
    if args.pipeline_num_stages:
      step_count = len(train_dataset)//engine.gradient_accumulation_steps()
      for step in range(step_count):
        loss = engine.train_batch()
    else:
      for step, data in enumerate(train_dataloader):
        inputs = data[0].to(engine.device)
        labels = data[1].to(engine.device)
              
        outputs = engine(inputs) #fwd pass
        loss = criterion(outputs, labels)
        engine.backward(loss) #backprop
        engine.step() #update weights, no zero-ing

        # from deepspeed.runtime.utils import memory_status
        # memory_status("Memory stats after training step")

    if engine.local_rank == 0: # time, epoch and loss
      print(f"{str(datetime.now())[:22]} :: Epoch: {epoch}, Loss: {loss}")
  

if __name__ == "__main__":
  main_deepspeed()


