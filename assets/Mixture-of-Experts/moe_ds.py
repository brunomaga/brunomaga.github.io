import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils
from torch.utils.data import DistributedSampler, DataLoader

# use current diectory as import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir))
from moe import Router

#use base GPTlite model from the GPT-lite post
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import n_embd, dropout, FeedForward, GPTlite

#user helper functions from the GPT-lite deepspeed post
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite-DeepSpeed'))
from gptlite_ds import get_dataset

assert 'LOCAL_RANK' in os.environ and 'RANK' in os.environ, "env vars not set. Launch with torchrun."
local_rank = int(os.environ['LOCAL_RANK']) #set by torchrun
global_rank = int(os.environ['RANK']) #set by torchrun
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
dist.init_process_group(backend='nccl', init_method='env://')

class MoE_ds(nn.Module):
  def __init__(self, k=2, capacity_factor=1.25, padding_val=0, local_rank=local_rank):
    super().__init__()
    self.capacity_factor = capacity_factor
    self.padding_val = padding_val

    # number of devices is the same as number of experts, as per paper: " Switch Transformers
    # will allocate all of their cores to the data partitioning dimension n, which will also
    # correspond to the number of experts in the model."
    self.num_experts = dist.get_world_size()
    self.k = k
    self.router = DDP(Router(n_embd, self.num_experts, dropout=dropout).to(device), device_ids=[local_rank])
    self.expert = FeedForward(n_embd).to(device) # 1 expert per GPU

  def forward(self, x):
    B,T,C = x.shape
    return x.view(B,T,C)


if __name__ == "__main__":
  torch.manual_seed(local_rank)  #set random seed
  vocab_size, batch_size = 65, 1 
  n_epochs = 100
  criterion = torch.nn.CrossEntropyLoss() #initialize loss function
  dataset, _, vocab_size = get_dataset()
  sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), drop_last=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=dist.get_world_size(), drop_last=True)

  # instantiate model and apply DDP to all layers except our MoE FeedForward
  model = GPTlite(vocab_size).to(device)
  model.token_embedding_table = DDP(model.token_embedding_table.to(device), device_ids=[local_rank])
  model.position_embedding_table = DDP(model.position_embedding_table.to(device), device_ids=[local_rank])
  model.ln = DDP(model.ln.to(device), device_ids=[local_rank])
  model.lm_head = DDP(model.lm_head.to(device), device_ids=[local_rank])
  for block in model.blocks: 
    block.sa = DDP(block.sa.to(device), device_ids=[local_rank])
    block.ln1 = DDP(block.ln1.to(device), device_ids=[local_rank])
    block.ln2 = DDP(block.ln2.to(device), device_ids=[local_rank])
    block.ffwd = MoE_dist().to(device) #replace FeedForward with Mixture of Experts

  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  model.train()
  for epoch in range(n_epochs):
      for step, data in enumerate(dataloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)
 
        outputs = model(inputs) #fwd pass
        loss = criterion(outputs, labels)
        loss.backward() #backprop
        print(f"Iteration: {step}, Loss: {loss}")
        optimizer.step() #update weights, no zero-ing
        optimizer.zero_grad()

      print(f"Epoch: {epoch}, Loss: {loss}")
  