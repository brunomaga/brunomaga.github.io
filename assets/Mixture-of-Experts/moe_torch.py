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
global_rank = int(os.environ['RANK']) #set by torchrun
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
dist.init_process_group(backend='nccl', init_method='env://')

class MoE(nn.Module):
  def __init__(self, k=2, capacity_factor=1.25, padding_val=0, local_rank=local_rank):
    super().__init__()
    self.capacity_factor = capacity_factor
    self.padding_val = padding_val

    # number of devices is the same as number of experts, as per paper: " Switch Transformers
    # will allocate all of their cores to the data partitioning dimension n, which will also
    # correspond to the number of experts in the model."
    self.num_experts = dist.get_world_size()
    self.k = k
    self.router = nn.Sequential( #a DNN to route tokens to experts
      nn.Dropout(dropout),
      nn.Linear(n_embd, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, self.num_experts), nn.Softmax(dim=-1)
    )
    self.router = DDP(self.router.to(device), device_ids=[local_rank])
    self.expert = FeedForward(n_embd).to(device) # 1 expert per GPU

  def forward(self, x):
    B,T,C = x.shape

    # 0. SETUP: collect batch size and first row index of each processor and expert
    # if drop_last=False, then the batch size will be always B
    batch_size = torch.tensor([B], dtype=torch.int64, device=device)
    batch_sizes = [torch.tensor([0], dtype=torch.int64, device=device) for _ in range(self.num_experts)]
    dist.all_gather(batch_sizes, batch_size)
    batch_inits = torch.cumsum(torch.tensor([0]+[b.item() for b in batch_sizes]), dim=0)

    # 1. ROUTING 
    assignments = self.router(x) #get assignements from router, shape B * T * n_experts
    topk_probs, topk_experts = torch.topk(assignments, k=self.k) # top-k experts per sentence

    # 2. PERMUTATION: collect and sort the coordinates of the tokens to send to each expert
    ids_per_expert = [ (topk_experts==expert).nonzero()[:,:2] for expert in range(self.num_experts) ]
    ids_per_expert = [ sorted(ids.tolist()) for ids in ids_per_expert ]

    # all-to-all to exchange the count of inputs to send/receive to/from each processor
    send_count = [torch.tensor([len(ids)], dtype=torch.int64, device=device) for ids in ids_per_expert]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in ids_per_expert]
    dist.all_to_all(recv_count, send_count)
    fn_count = lambda tensor, scale=1: [x.item()*scale for x in tensor] 

    # send/receive the metadata row_id+token_id to/from the appropriate processors
    M = 2 # metadata columns
    send_meta = [ torch.tensor((batch_inits[global_rank]+b,t)) for e in range(self.num_experts) for b,t in ids_per_expert[e] ]
    send_meta = torch.cat(send_meta, dim=0).to(device) #flatten
    recv_meta = torch.zeros(sum(recv_count)*M, dtype=send_meta.dtype).to(device)
    dist.all_to_all_single(recv_meta, send_meta, fn_count(recv_count,M), fn_count(send_count,M))
    recv_meta = recv_meta.view(-1, M) # reshape to M columns 

    # group received metadata by row id 
    uniq_rows, recv_row_lens = recv_meta[:,0].unique(sorted=True, return_counts=True)
    recv_row_offsets = [0] + torch.cumsum(recv_row_lens, dim=0).tolist()

    # send/receive input tokens to/from the appropriate processors
    send_toks = [ x[b, t] for e in range(self.num_experts) for b,t in ids_per_expert[e] ]
    send_toks = torch.cat(send_toks, dim=0).to(device) #flatten
    recv_toks = torch.zeros(sum(recv_count)*C, dtype=send_toks.dtype).to(device)
    dist.all_to_all_single(recv_toks, send_toks, fn_count(recv_count,C), fn_count(send_count,C))
    recv_toks = recv_toks.view(-1, C) # reshape to C columns 

    # crop or pad received items PER SENTENCE to max capacity. Batch shape: Rows * Capacity * C
    capacity = int( T / self.num_experts *self.capacity_factor)
    batch_toks = torch.full( (len(uniq_rows), capacity, C), self.padding_val, dtype=recv_toks.dtype, device=device) # Rows * Capacity * C
    used_token_ids_per_row = []
    for row_id in range(len(uniq_rows)):
      row_toks = recv_toks[recv_row_offsets[row_id]:recv_row_offsets[row_id+1]] # split by row id
      token_count = row_toks.shape[0]
      if token_count>capacity: # crop
        ids = torch.linspace(0, token_count-1, capacity).int() # pick intervealed
        batch_toks[row_id] = row_toks[ids]
        used_token_ids_per_row.append(ids)
      else: # fill with padding
        batch_toks[row_id, :token_count] = row_toks
        used_token_ids_per_row.append(torch.tensor(range(token_count)))

    # 3. COMPUTATION: pass received tokens through this device's expert
    batch_toks = self.expert(batch_toks) # Rows * Capacity * C

    # 4. UN-PERMUTATION: send metadata and results back to the appropriate data-loader processors 
    # re-use send_count, recv_count, send_meta, recv_meta, send_toks, recv_toks
    recv_toks = recv_toks.fill_(self.padding_val) # reset recv_toks, will be used to SEND results back
    send_toks = send_toks.fill_(self.padding_val) # reset send_toks, will be used to RECEIVE results back
    for row_id in range(len(uniq_rows)):
      row_offset = recv_row_offsets[row_id]
      used_token_ids = used_token_ids_per_row[row_id]
      recv_toks[row_offset+used_token_ids] = batch_toks[row_id, :len(used_token_ids)]
    dist.all_to_all_single(send_toks, recv_toks.flatten(), fn_count(send_count,C), fn_count(recv_count,C))
    x = send_toks.view(B,T,C) # reshape received buffer to B*T*C columns

    # 5. SCALE: multiply by the probabilities assigned to each token
    x = x*topk_probs.unsqueeze(1)
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
    block.ffwd = MoE().to(device) #replace DDP of FFN with MoE

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
  