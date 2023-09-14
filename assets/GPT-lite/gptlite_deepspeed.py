import torch
import os

# DeepSpeed requires a distributed environment even when only one process is used.
if os.environ.get("WORLD_SIZE") is None:
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
  os.environ["RANK"] = "0"
  os.environ["LOCAL_RANK"] = "0"
  os.environ["WORLD_SIZE"] = "1"


def get_deepspeed_args(description='CIFAR'):
  import argparse
  parser = argparse.ArgumentParser(description=description)
  #deepspeed_config used by deepspeed.initialize(args=args, ...)
  parser.add_argument('--deepspeed_config', type=str, default='./ds_config.json',
                        help='DeepSpeed config file.')
  # Mandatory deepspeed arg  set by `deepspeed` launcher
  parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
  return parser.parse_args()

  
def get_model_and_dataset():

  import gptlite_model as gptlite # use out GPT-lite model

  class GPTliteDataset(torch.utils.data.Dataset):

      def __init__(self, train_data, block_size):
        self.train_data = train_data
        self.block_size = block_size

      def __len__(self):
        return len(self.train_data)

      def __getitem__(self, idx):
        # generate 1 random offsets on the data
        ix = torch.randint(len(self.train_data)-self.block_size)
        # input is a random subset of tokens
        x = torch.Tensor(self.train_data[ix : ix+self.block_size])
        # target is just x shifted right (ie the next predicted word)
        y = torch.Tensor(self.train_data[ix+1 : ix+1+self.block_size])
        
        return gptlite.get_batch(self.train_data, batch_size=1) #get 1 item

  model = gptlite.GPTlite()
  dataset = GPTliteDataset(gptlite.train_data, gptlite.block_size)
  return model, dataset

  
def main_deepspeed():

  import deepspeed
  deepspeed.init_distributed()
  args = get_deepspeed_args('CIFAR') 
  model, train_dataset = get_model_and_dataset()
  parameters = filter(lambda p: p.requires_grad, model.parameters())

  model_engine, optimizer, train_dataloader , _ = deepspeed.initialize(
    args=args, #needs args.local_rank and args.deepspeed_config (or argument config)
    model=model,
    model_parameters=parameters, #maybe just mode.parameters() ?
    training_data=train_dataset,
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


