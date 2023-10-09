import torch
import torch.nn as nn
import deepspeed
import torch.distributed as dist


class BenchmarkModel(nn.Module):
  """" DNN with W input features, W neurons per layer, W output classes and L layers """

  def __init__(self, W, L, activation_checkpoint_interval=0):
    super(BenchmarkModel, self).__init__()
    self.activation_checkpoint_interval = activation_checkpoint_interval
    self.layers = []
    for _ in range(L):
      self.layers += [nn.Linear(W, W), nn.ReLU()]
    self.layers = nn.Sequential(*self.layers)

  def to_layers(self):  
      return [*self.layers]
    
  def forward(self, x):
    if self.activation_checkpoint_interval > 0:
      for l, layer in enumerate(self.to_layers()):
        is_checkpoint = l % self.activation_checkpoint_interval == 1 
        x = deepspeed.checkpointing.checkpoint(layer, x) if is_checkpoint else layer(x)
      return x
    
    return self.layers(x)
  
  
from deepspeed.pipe import PipelineModule, LayerSpec
class BenchmarkModelPipeSpec(PipelineModule):

  def __init__(self, W, L, pipe_kwargs={}):
    self.specs = []
    for _ in range(L):
      self.specs += [ LayerSpec(nn.Linear, W, W), LayerSpec(nn.ReLU) ]
    super().__init__(layers=self.specs, **pipe_kwargs)
    

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, W, len=2**16):
      self.W = W
      self.len = len

    def __len__(self):
      return self.len

    def __getitem__(self, _):
      x = torch.Tensor(self.W).uniform_(-1, 1)
      y = int( x @ x % self.W)
      return x, torch.tensor(y, dtype=torch.long)


def get_model(W, L, criterion, pipeline_num_stages=0, pipeline_spec_layers=False, activation_checkpoint_interval=0):

  if pipeline_num_stages:
    pipe_kwargs={
      'num_stages': pipeline_num_stages,
      'activation_checkpoint_interval': activation_checkpoint_interval, 
      'loss_fn': criterion,
      }

    if pipeline_spec_layers:
      model = BenchmarkModelPipeSpec(W, L, pipe_kwargs=pipe_kwargs)
    else:
      device_str = f'cuda:{dist.get_rank()}'
      model = BenchmarkModel(W, L).to(device_str)
      model = deepspeed.pipe.PipelineModule(layers=model.to_layers(), **pipe_kwargs)
  else:
    model = BenchmarkModel(W, L, activation_checkpoint_interval=activation_checkpoint_interval)  
  return model


def get_dataset(W):
  return BenchmarkDataset(W)
