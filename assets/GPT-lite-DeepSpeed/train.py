import torch
import torch.distributed as dist
import os
import deepspeed
import gptlite

# DeepSpeed requires a distributed environment even when only one process is used.
if os.environ.get("WORLD_SIZE") is None:
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
  os.environ["RANK"] = "0"
  os.environ["LOCAL_RANK"] = "0"
  os.environ["WORLD_SIZE"] = "1"


def load_tiny_shakespeare_data(filename="tinyshakespeare.txt"):
  rank = dist.get_rank()

  #load input data
  with open(filename) as f:
      text = f.read()
  if rank==0: print("input data loaded. Length of text: ", len(text))

  #collect all ordered and unique characters in the text
  chars = sorted(list(set(text)))
  if rank==0: print("unique chars: ", "".join(chars))
  if rank==0: print("length of chars: ", len(chars))

  #map characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long) #encode text to integers
  decode = lambda x: ''.join([itos[i] for i in x]) #decode integers to text
  vocab_size = len(stoi)
  if rank==0: print("vocab size: ", vocab_size)
  if rank==0: print(encode("Hello world"))
  if rank==0: print(decode(encode("Hello world").tolist()))
  if rank==0: print("character zero is:", decode([0]), "<end>")

  # collect input data, break dataset in train/validation
  data = encode(text)
  n = int(0.9*len(data))
  train_data, valid_data = data[:n], data[n:]
  if rank==0: print("Train data encoded", data.shape, train_data.shape, valid_data.shape)
  return train_data, valid_data, vocab_size


def get_deepspeed_args(description='GPT lite on DeepSpeed'):
  import argparse
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank passed from distributed launcher')
  parser.add_argument('--activation_checkpoint_interval', type=int, default=0,
                      help='activation checkpoint interval (0 means disabled)')
  parser.add_argument('--pipeline_parallel_size', type=int, default=0,
                      help='enable pipeline parallelism with N stages')
  parser.add_argument("--pipeline_spec_layers", action="store_true",
                      help="enable SpecLayers in pipeline parallelism")
  parser.add_argument("--run_memory_estimation_only", action="store_true",
                      help="run parameter memory estimation and quit")
  # Include DeepSpeed configuration arguments (--deepspeed, --deepspeed_config, ...)
  parser = deepspeed.add_config_arguments(parser)
  return parser.parse_args()

  
def get_dataset():
  
  class GPTliteDataset(torch.utils.data.Dataset):

      def __init__(self, train_data, block_size):
        self.train_data = train_data
        self.block_size = block_size

      def __len__(self):
        return len(self.train_data)

      def __getitem__(self, idx):
        # generate 1 random offset on the data
        ix = torch.randint(len(self.train_data)-self.block_size , size=())
        # input is a random subset of tokens
        x = self.train_data[ix   : ix+self.block_size]
        # target is just x shifted right (ie the next predicted word)
        y = self.train_data[ix+1 : ix+1+self.block_size]
        return x, y

  train_data, _, vocab_size = load_tiny_shakespeare_data() #load data encodings from file
  dataset = GPTliteDataset(train_data, gptlite.block_size)
  return dataset, vocab_size

  
def measure_parameters_memory(model, args):

  ranks = list(range(dist.get_world_size())) if args.pipeline_parallel_size else [0]
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
    
  
def main_deepspeed(n_epochs=100, random_seed=42):
  deepspeed.init_distributed()
  args = get_deepspeed_args() 
  train_dataset, vocab_size = get_dataset()
  torch.manual_seed(random_seed)

  if args.pipeline_parallel_size:
    #inspired by DeepSpeedExamples/training/pipeline_parallelism/train.py
    deepspeed.runtime.utils.set_random_seed(random_seed)
    pipe_kwargs={
      'num_stages': args.pipeline_parallel_size,
      'activation_checkpoint_interval': args.activation_checkpoint_interval, 
      'loss_fn': torch.nn.CrossEntropyLoss(),
      }

    if args.pipeline_spec_layers:
      model = gptlite.GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)
    else:
      device_str = f'cuda:{dist.get_rank()}'
      model = gptlite.GPTlite(vocab_size).to(device_str)
      model = deepspeed.pipe.PipelineModule(layers=model.to_layers(), **pipe_kwargs)
  else:
    model = gptlite.GPTlite(vocab_size,
              activation_checkpoint_interval=args.activation_checkpoint_interval)
    criterion = torch.nn.CrossEntropyLoss()
    
  #estimate parameters memory requirements
  if args.run_memory_estimation_only:
    measure_parameters_memory(model, args)
    return

  engine, _, train_dataloader , _ = deepspeed.initialize(
    args=args, model=model, training_data=train_dataset,
    model_parameters=[p for p in model.parameters() if p.requires_grad] #optional
    #config=args.deepspeed_config, #only needed when args.deepspeed_config does not exist
    )

  assert engine.local_rank == dist.get_rank()
  print(f"Starting training. Rank {engine.local_rank} on {engine.device}")

  for epoch in range(n_epochs):
    if args.pipeline_parallel_size:
      loss = engine.train_batch()
    else:
      for step, data in enumerate(train_dataloader):
        inputs = data[0].to(engine.device)
        labels = data[1].to(engine.device)
              
        outputs = engine(inputs) #fwd pass
        loss = criterion(outputs, labels)
        engine.backward(loss) #backprop
        engine.step() #update weights, no zero-ing

    if engine.local_rank == 0: print(f"Epoch: {epoch}, Loss: {loss}")
  

if __name__ == "__main__":
  main_deepspeed()



