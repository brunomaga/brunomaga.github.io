import torch
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
  #load input data
  with open(filename) as f:
      text = f.read()
  print("input data loaded. Length of text: ", len(text))

  #collect all ordered and unique characters in the text
  chars = sorted(list(set(text)))
  print("unique chars: ", "".join(chars))
  print("length of chars: ", len(chars))

  #map characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long) #encode text to integers
  decode = lambda x: ''.join([itos[i] for i in x]) #decode integers to text
  vocab_size = len(stoi)
  print("vocab size: ", vocab_size)
  print(encode("Hello world"))
  print(decode(encode("Hello world").tolist()))
  print("character zero is:", decode([0]), "<end>")

  # collect input data, break dataset in train/validation
  data = encode(text)
  n = int(0.9*len(data))
  train_data, valid_data = data[:n], data[n:]
  print("Train data encoded", data.shape, train_data.shape, valid_data.shape)
  return train_data, valid_data, vocab_size


def get_deepspeed_args(description='GPT lite on DeepSpeed'):
  import argparse
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
  parser.add_argument("--pipeline", action="store_true",
                      help="enable pipeline parallelism")
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

  ranks = [0] if not args.pipeline else list(range(int(os.environ["WORLD_SIZE"])))
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
    
  
def main_deepspeed():

  import deepspeed
  deepspeed.init_distributed()
  args = get_deepspeed_args() 

  print("ARGS:", args)
  train_dataset, vocab_size = get_dataset()

  if not args.pipeline:
    model = gptlite.GPTlite(vocab_size)
  else:
    num_stages=int(os.environ["WORLD_SIZE"])
    if not args.pipeline_spec_layers:
      model = gptlite.GPTlitePipe(vocab_size)
      layers = model.to_layers()
      model = deepspeed.pipe.PipelineModule(layers=layers, num_stages=num_stages)
    else:
      model = gptlite.GPTlitePipeLayers(vocab_size, pipe_kwargs={'num_stages':num_stages})
    
  #estimate memory requirements for the stage 3 implementation
  if args.run_memory_estimation_only:
    measure_parameters_memory(model, args)
    return

  # parameters = filter(lambda p: p.requires_grad, model.parameters())
  model_engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    args=args, model=model, training_data=train_dataset,
    #model_parameters=parameters, #optional, to specify which params to optimize 
    #config=args.deepspeed_config, #only needed when args.deepspeed_config does not exist
    )

  local_device = deepspeed.accelerator.get_accelerator().device_name(model_engine.local_rank)
  local_rank = model_engine.local_rank
  print(f"Starting training, I'm rank {local_rank} on {local_device}")

  target_dtype = None # For float32, target_dtype is None so no datatype conversion needed
  if   model_engine.bfloat16_enabled(): target_dtype=torch.bfloat16
  elif model_engine.fp16_enabled():     target_dtype=torch.half
      
  criterion = torch.nn.CrossEntropyLoss()
  
  for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for step, data in enumerate(train_dataloader):
      print(f"Epoch {epoch}, Step {step} on {local_device}")

      inputs = data[0].to(local_device)
      labels = data[1].to(local_device)

      if target_dtype != None:
        inputs = inputs.to(target_dtype)
            
      outputs = model_engine(inputs) #fwd pass
      loss = criterion(outputs, labels)
      running_loss += loss.item()

      model_engine.backward(loss) #backprop
      model_engine.step() #update weights, no zero-ing

    # print statistics
    if local_rank == 0: print("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, running_loss / step))
  
  if local_rank == 0: print('Finished Training')
  

if __name__ == "__main__":
  main_deepspeed()


