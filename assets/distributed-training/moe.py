import torch.nn as nn
import torch

class Router(nn.Module):
  """ a DNN to route tokens to experts """

  def __init__(self, n_embd, num_experts, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(n_embd, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, n_embd*4), nn.ReLU(),
      nn.Linear(n_embd*4, num_experts), nn.Softmax(dim=-1)
    )

  def forward(self, x):
    return self.net(x)


class FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper"""

  def __init__(self, input_size, output_size, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(input_size, input_size*4), nn.ReLU(),
        nn.Linear(input_size*4, input_size*4), nn.ReLU(),
        nn.Linear(input_size*4, output_size),
        nn.Dropout(dropout),
      )
  def forward(self, x):
    return self.net(x)
  

class MoE_1991(nn.Module):

  def __init__(self, input_size, output_size, num_experts, dropout=0.1):
    super().__init__()
    self.router = Router(input_size, num_experts, dropout=dropout)
    self.experts = nn.ModuleList([
      FeedForward(input_size, output_size, dropout=dropout) for _ in range(num_experts)
      ])

  def forward(self, x):
    probs = self.router(x)
    outputs = torch.stack([ expert(x) for expert in self.experts ], dim=-2) # B*T*Experts*C
    weighted_outputs = outputs * probs.unsqueeze(-1) # B*T*E*C x B*T*1*C -> B*T*E*C
    weighted_sum = weighted_outputs.sum(dim=-2) # sum over experts: B*T*E*C -> B*T*C
    return weighted_sum, probs, outputs

  def loss_per_token(self, probs, outputs, labels):
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[-1])
    mse_expert_i = lambda i: (one_hot_labels-outputs[:,:,i,:]).square().mean()  # B*T*1 * B*T*classes
    loss_per_expert = (torch.stack([mse_expert_i(i) for i in range(8)])*probs) # B*T*experts
    return loss_per_expert.sum(-1) # B*T, 1 value per token


if __name__ == "__main__":
  B, T, C = 2, 256, 512 # batch size, sequence length, input size
  n_experts = 8
  n_classes = 1024
  model = MoE_1991(C, n_classes, n_experts)
  x= torch.randn(B, T, C)
  labels = torch.randint(0, n_classes, (B, T))
  y, probs, outputs = model(x)
  loss = model.loss_per_token(probs, outputs, labels).sum()
  print(y.shape, loss)