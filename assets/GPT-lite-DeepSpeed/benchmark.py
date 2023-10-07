import torch
import torch.nn as nn
import deepspeed

class BenchmarkModel(nn.Module):
  """" simple DNN with W input features, W neurons per layer, and L layers """

  def __init__(self, W=1024, L=16, activation_checkpoint_interval=0):
    super(BenchmarkModel, self).__init__()
    self.activation_checkpoint_interval = activation_checkpoint_interval
    self.layers = [ [nn.Linear(W, W), nn.ReLU()] ] * L
    self.layers = [l for sublist in self.layers for l in sublist][:-1]
    self.layers = nn.Sequential(*self.layers)

  def to_layers(self):  
      return [*self.layers,]
    
  def forward(self, x):
    if self.activation_checkpoint_interval > 0:
      for l, layer in enumerate(self.to_layers()):
        is_checkpoint = l % self.activation_checkpoint_interval == 0 
        x = deepspeed.checkpointing.checkpoint(layer, x) if is_checkpoint else layer(x)
      return x
    
    return self.layers(x)
  
  

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, input_size, num_labels, len=2**16):
      self.input_size = input_size
      self.num_labels = num_labels

    def __len__(self):
      return self.len

    def __getitem__(self, _):
      x = torch.Tensor(self.input_size).uniform_(-1, 1)
      y = int( x @ x % self.num_labels)
      return x, torch.tensor(y, dtype=torch.long)

def get_model_and_dataset(W=1024, L=15):
  return BenchmarkModel(W=W, L=L), BenchmarkDataset(input_size=W, num_labels=W)


