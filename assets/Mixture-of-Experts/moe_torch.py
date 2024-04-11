import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils
from torch.utils.data import DistributedSampler, DataLoader

#use base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import n_embd, dropout, FeedForward, GPTlite

#user helper functions from the GPT-lite deepspeed post
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite-DeepSpeed'))
from gptlite_ds import get_dataset

local_rank = int(os.environ['LOCAL_RANK']) #set by torchrun
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
dist.init_process_group(backend='nccl', init_method='env://')

class FeedForward_MoE(nn.Module):
  def __init__(self, k=2, capacity=10, padding_token=0, local_rank=local_rank):
    super().__init__()
    self.capacity = capacity
    self.padding_token = padding_token

    # number of devices is the same as number of experts, as per paper: " Switch Transformers
    # will allocate all of their cores to the data partitioning dimension n, which will also
    # correspond to the number of experts in the model."
    self.n_experts = dist.get_world_size()
    self.k = k
    self.router = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(n_embd, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, self.n_experts), nn.Softmax(dim=-1)
    )
    self.router = DDP(self.router.to(device), device_ids=[local_rank])
    self.expert = FeedForward(n_embd).to(device) # each GPU has 1 expert, so no DDP here

  def forward(self, x):
    B,T,C = x.shape
    assignments = self.router(x) #get assignements from router, shape B * T * n_experts
    topk_probs, topk_ids = torch.topk(assignments, k=self.k) # top-k experts per sentence

    # 1. PERMUTATION: collect the tokens to send to each expert, sort and send them
    tokens_per_expert = {i: [] for i in range(self.n_experts)}
    for b in range(B):
      for t in range(T):
        for e in topk_ids[b, t]:
          tokens_per_expert[e.item()].append( (b, t) )
    tokens_per_expert = {expert: sorted(tokens) for expert, tokens in tokens_per_expert.items()}

    # each data-parallel unit has the assignments for each of its inputs, so
    # we'll do an all-to-all to exchange the count of inputs to send/receive to each processor
    send_count = [torch.tensor([len(tokens)], dtype=torch.int64, device=device) for tokens in tokens_per_expert.values()]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in tokens_per_expert]
    dist.all_to_all(recv_count, send_count)
    fn_count = lambda tensor, scale=1: [x.item()*scale for x in tensor] 

    # send the metadata sentence_id+token_id to the appropriate processors
    M = 2 # metadata columns
    send_meta = [ torch.tensor((b,t)) for expert_id in range(self.n_experts) for b,t in tokens_per_expert[expert_id] ]
    send_meta = torch.cat(send_meta, dim=0).to(device) #flatten
    recv_meta = torch.zeros(sum(recv_count)*M, dtype=send_meta.dtype).to(device)
    dist.all_to_all_single(recv_meta, send_meta, fn_count(recv_count,M), fn_count(send_count,M))
    recv_meta = recv_meta.view(-1, M) # reshape to M columns 

    # now we send the elements themselves to the appropriate processors, as flattened tensors
    send_embd = [ x[b, t] for expert_id in range(self.n_experts) for b,t in tokens_per_expert[expert_id] ]
    send_embd = torch.cat(send_embd, dim=0).to(device) #flatten
    recv_embd = torch.zeros(sum(recv_count)*C, dtype=send_embd.dtype).to(device)
    dist.all_to_all_single(recv_embd, send_embd, fn_count(recv_count,C), fn_count(send_count,C))
    recv_embd = recv_embd.view(-1, C) # reshape to C columns 

    #crop or pad received items to max capacity
    if recv_embd.shape[0]>self.capacity: #crop
      recv_embd = recv_embd[:self.capacity]
      recv_meta = recv_meta[:self.capacity]
    else: # pad
      recv_embd = F.pad(recv_embd, (0,0,0,self.capacity-recv_embd.shape[0]), value=self.padding_token)
      recv_meta = F.pad(recv_meta, (0,0,0,self.capacity-recv_meta.shape[0]), value=self.padding_token)

    # 2. COMPUTATION: pass received tokens through this device's expert
    x = self.expert(recv_embd)

    # 3. UN-PERMUTATION: revert to B * T * C shape 
    send_count = [torch.tensor([len(tokens)], dtype=torch.int64, device=device) for tokens in tokens_per_expert.values()]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in tokens_per_expert]


    # 4. SCALE: multiply by the probabilities assigned to each token
    recv_topk_probs = recv_meta[:,2] 
    x = x*recv_topk_probs.unsqueeze(1)

    # all reduce sum of expert outputs
    x = torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


if __name__ == "__main__":
  torch.manual_seed(local_rank)  #set random seed
  vocab_size, batch_size = 65, 8 
  n_epochs = 100
  criterion = torch.nn.CrossEntropyLoss() #initialize loss function
  dataset, _, vocab_size = get_dataset()
  sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), drop_last=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=dist.get_world_size(), drop_last=True)

  # instantiate model and apply DDP to all layers except our MoE FeedForward
  model = DDP( GPTlite(vocab_size).to(device), device_ids=[local_rank])
  for block in model.module.blocks:
    block.ffwd = FeedForward_MoE().to(device) #replace DDP of FFN with MoE

  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  model.train()
  for epoch in range(n_epochs):
      for step, data in enumerate(dataloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)
 
        outputs = model(inputs) #fwd pass
        loss = criterion(outputs, labels)
        loss.backward() #backprop
        optimizer.step() #update weights, no zero-ing
        optimizer.zero_grad()

      print(f"Epoch: {epoch}, Loss: {loss}")
  