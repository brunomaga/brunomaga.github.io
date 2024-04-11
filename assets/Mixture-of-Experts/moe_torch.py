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
from gptlite import n_layer, n_embd, n_head, block_size, MultiHeadAttention, FeedForward, GPTlite

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
    self.n_experts = torch.cuda.device_count
    self.k = k
    self.router = DDP(FeedForward(n_embd).to(device), device_ids=[local_rank])
    self.expert = FeedForward(n_embd).to(device) # each GPU has 1 expert, so no DDP here

  def forward(self, x):
    B,T,C = x.shape
    assignments = self.router(x) #get assignements from router, shape B * T * n_experts
    expert_values, expert_ids = torch.topk(assignments, k=self.k) # top-k experts per sentence
    expert_probs = F.softmax(expert_values, dim=-1) #convert to probabilities

    # collect the tokens to send to each expert
    tokens_per_expert = {i: [] for i in range(self.n_experts)}
    for sentence_id, topk_tokens_in_sentence in enumerate(expert_ids):
      for token_id, topk_experts_for_sentence in enumerate(topk_tokens_in_sentence):
        for expert_id in topk_experts_for_sentence:
          tokens_per_expert[expert_id].append( (sentence_id, token_id) )

    # each data-parallel unit has the assignments for each of its inputs, so
    # we'll do an all-to-all to exchange the count of inputs to send/receive to each processor
    send_count = [torch.tensor([len(tokens)*C], dtype=torch.int64, device=device) for tokens in tokens_per_expert]
    recv_count = [torch.tensor([0], dtype=torch.int64, device=device) for _ in tokens_per_expert]
    dist.all_to_all(recv_count, send_count)

    # now we send the elements themselves to the appropriate processors, as flattened tensors
    send = [ x[sentence_id, token_id] for expert_id in range(self.n_experts) for sentence_id, token_id in tokens_per_expert[expert_id] ]
    send = torch.tensor(send, dim=0).flatten().to(device)
    recv = torch.zeros(sum(recv_count), dtype=send.dtype).to(device)
    send_count = [s.item() for s in send_count]  # convert to list of ints
    recv_count = [r.item() for r in recv_count]
    dist.all_to_all_single(recv, send, recv_count, send_count)

    # now we reshape the received tensor back to the original shape
    recv = recv.view(-1, C) # B*T x C
    #crop received items by padding to max capacity
    recv = F.pad(recv, (0,0, 0, self.capacity-recv.shape[0], 0, 0), value=self.padding_token)

    # pass received tokens through this device's expert
    x = self.expert(recv)

    # multiply by the probabilities of this expert
    x = x * expert_probs

    # all reduce sum of expert outputs
    x = torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x


if __name__ == "__main__":
  torch.manual_seed(local_rank)  #set random seed
  vocab_size, batch_size = 65, 4 
  n_epochs = 100
  criterion = torch.nn.CrossEntropyLoss() #initialize loss function
  dataset, _, vocab_size = get_dataset()
  sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), drop_last=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=dist.get_world_size(), drop_last=True)

  # instantiate model and apply DDP to all layers except our MoE FeedForward
  model = GPTlite(vocab_size).to(device)
  model.token_embedding_table = DDP(model.token_embedding_table, device_ids=[local_rank])
  model.position_embedding_table = DDP(model.position_embedding_table, device_ids=[local_rank])
  model.ln = DDP(model.ln, device_ids=[local_rank])
  model.lm_head = DDP(model.lm_head, device_ids=[local_rank])
  for block in model.blocks:
    block.sa = DDP(block.sa, device_ids=[local_rank])
    block.ln1 = DDP(block.ln1, device_ids=[local_rank])
    block.ln2 = DDP(block.ln2, device_ids=[local_rank])
    del block.ffwd
    block.ffwd = FeedForward_MoE().to(device)

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
  