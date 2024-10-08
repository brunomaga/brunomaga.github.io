import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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

# how to do data-parallelism of non-MoEs: pick DDP or FSDP
# DataParallel = lambda model: torch.distributed.fsdp.FullyShardedDataParallel(model.to(device), device_id=local_rank)
DataParallel = lambda model: torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])

class MoE_dist(nn.Module):
  def __init__(self, k=2, capacity_factor=1.25, padding_val=0, local_rank=local_rank):
    super().__init__()
    self.capacity_factor = capacity_factor
    self.padding_val = padding_val

    # number of devices is the same as number of experts, as per paper: " Switch Transformers
    # will allocate all of their cores to the data partitioning dimension n, which will also
    # correspond to the number of experts in the model."
    self.num_experts = dist.get_world_size()
    self.k = k
    self.router = DataParallel(Router(n_embd, self.num_experts, dropout=dropout))
    self.expert = FeedForward(n_embd).to(device) # 1 expert per GPU

  def forward(self, x):
    B,T,C = x.shape

    # 1. ROUTING 
    probs = self.router(x) #get assignements from router, shape B * T * n_experts
    topk_probs, topk_experts = torch.topk(probs, k=self.k) # top-k experts per sentence
    ids_per_expert = [ (topk_experts==expert).nonzero() for expert in range(self.num_experts) ]
    probs_per_expert = [ topk_probs[topk_experts==expert] for expert in range(self.num_experts) ]

    # 2. PERMUTATION: collect and sort the coordinates of the tokens to send to each expert

    # all-to-all to exchange the count of inputs to send/receive to/from each processor
    send_count = [torch.tensor([len(ids)], dtype=torch.int64, device=device) for ids in ids_per_expert]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in ids_per_expert]
    dist.all_to_all(recv_count, send_count)
    fn_count = lambda tensor, scale=1: [x.item()*scale for x in tensor] 

    # send/receive the metadata row_id+token_id to/from the appropriate processors
    M = ids_per_expert[0].shape[-1] # number of columns in metadata
    send_ids = torch.cat(ids_per_expert, dim=0).to(device)
    send_ids[:,0] += global_rank*B # add processor's batch offset
    recv_ids = torch.zeros(sum(recv_count)*M, dtype=send_ids.dtype).to(device)
    dist.all_to_all_single(recv_ids, send_ids.flatten(), fn_count(recv_count,M), fn_count(send_count,M))
    recv_ids = recv_ids.view(-1, M) # reshape to M columns 

    # send/receive input tokens to/from the appropriate processors
    send_toks = torch.cat([ x[ids[:,:2].T.tolist()] for ids in ids_per_expert], dim=0).to(device)
    recv_toks = torch.zeros(sum(recv_count)*C, dtype=x.dtype).to(device)
    dist.all_to_all_single(recv_toks, send_toks.flatten(), fn_count(recv_count,C), fn_count(send_count,C))
    recv_toks = recv_toks.view(-1, C) # reshape to C columns 

    if len(recv_toks)>0:
      # 3. COMPUTATION: pass received tokens through this device's expert
      # group received metadata by row id 
      uniq_rows, recv_row_lens = recv_ids[:,0].unique(sorted=True, return_counts=True)
      recv_row_offsets = [0] + torch.cumsum(recv_row_lens, dim=0).tolist()
      recv_row_slice = lambda row: slice(recv_row_offsets[row], recv_row_offsets[row+1])

      # crop or pad received items PER SENTENCE to max capacity. Batch shape: Rows * Capacity * C
      capacity = int( T / self.num_experts *self.capacity_factor)
      pad_fn = lambda toks, value = self.padding_val: F.pad(toks, (0,0,0,capacity-toks.shape[0]), value=value) #pad or crop
      batch_toks = torch.stack([ pad_fn(recv_toks[recv_row_slice(i)]) for i in range(len(uniq_rows))], dim=0).to(device)

      batch_row_len = torch.tensor([ min(recv_row_lens[r], capacity) for r in range(len(uniq_rows))])
      batch_toks = self.expert(batch_toks) # Rows * Capacity * C

      # 4. UN-PERMUTATION: send metadata and results back to the appropriate data-loader processors 
      recv_toks = recv_toks.fill_(self.padding_val) # reset recv_toks, will be used to SEND results
      recv_tok_offsets  = np.concatenate([ range(recv_row_offsets[i], recv_row_offsets[i]+batch_row_len[i]) for i in range(len(uniq_rows)) ])
      batch_tok_offsets = np.concatenate([ [ [i]*batch_row_len[i], range(batch_row_len[i]) ] for i in range(len(uniq_rows)) ], axis=1)
      recv_toks[recv_tok_offsets] = batch_toks[batch_tok_offsets[0], batch_tok_offsets[1]] # fill recv_toks with results

    send_toks = send_toks.fill_(self.padding_val).flatten() # reset send_toks, will be used to RECEIVE results
    dist.all_to_all_single(send_toks, recv_toks.flatten(), fn_count(send_count,C), fn_count(recv_count,C))
    x = send_toks.view(-1,C)

    # 5. SCALE: multiply by the probabilities assigned to each token
    x *= torch.concatenate(probs_per_expert).view(-1,1) # multiply by router probabilities
    if self.k>1: # sum over k probabilities for each input
      x = torch.stack( [ x[send_ids[:,-1]==k] for k in range(self.k)]).sum(dim=0)
    return x.view(B,T,C)


if __name__ == "__main__":
  torch.manual_seed(local_rank)  #set random seed
  vocab_size, batch_size = 65, 1 
  n_epochs = 100
  criterion = torch.nn.CrossEntropyLoss() #initialize loss function
  dataset, _, vocab_size = get_dataset()
  sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), drop_last=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=dist.get_world_size(), drop_last=True)

  # instantiate model and apply DataParallel to all layers except our MoE FeedForward
  model = GPTlite(vocab_size).to(device)
  model.token_embedding_table = DataParallel(model.token_embedding_table)
  model.position_embedding_table = DataParallel(model.position_embedding_table)
  model.ln = DataParallel(model.ln)
  model.lm_head = DataParallel(model.lm_head)
  for block in model.blocks: 
    block.sa = DataParallel(block)
    block.ln1 = DataParallel(block.ln1)
    block.ln2 = DataParallel(block.ln2)
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
  
