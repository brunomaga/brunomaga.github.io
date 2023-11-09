import torch
import torch.nn as nn
import deepspeed
import torch.distributed as dist


class BenchmarkModel(nn.Module):
  """" DNN with L layers and W neurons per layer """

  def __init__(self, W, L, in_size, out_size, activation_checkpoint_interval=0):
    super(BenchmarkModel, self).__init__()
    self.activation_checkpoint_interval = activation_checkpoint_interval
    self.layers = [nn.Linear(in_size, W), nn.ReLU()]
    for _ in range(L-2):
      self.layers += [nn.Linear(W, W), nn.ReLU()]
    self.layers += [nn.Linear(W, out_size), nn.ReLU()]
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

  def __init__(self, W, L, in_size, out_size, pipe_kwargs={}):
    self.specs = [ LayerSpec(nn.Linear, in_size, W), LayerSpec(nn.ReLU) ]
    for l in range(L-2):
      self.specs += [ LayerSpec(nn.Linear, W, W), LayerSpec(nn.ReLU) ]
    self.specs += [ LayerSpec(nn.Linear, W, out_size), LayerSpec(nn.ReLU) ]
    super().__init__(layers=self.specs, **pipe_kwargs)
    

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, in_size, out_size, len=2**16):
      self.in_size = in_size
      self.len = len
      self.out_size = out_size

    def __len__(self):
      return self.len

    def __getitem__(self, _):
      x = torch.Tensor(self.in_size).uniform_(-10, 10)
      y = int( x @ x % self.out_size)
      return x, torch.tensor(y, dtype=torch.long)


def get_model(W, L, in_size, out_size, 
              criterion=None, pipeline_num_stages=0,
              pipeline_spec_layers=False, activation_checkpoint_interval=0):

  if pipeline_num_stages:
    assert criterion is not None, "for pipeline runs, need to specify criterion"
    pipe_kwargs={
      'num_stages': pipeline_num_stages,
      'activation_checkpoint_interval': activation_checkpoint_interval, 
      'loss_fn': criterion,
      }

    if pipeline_spec_layers:
      model = BenchmarkModelPipeSpec(W, L, in_size, out_size, pipe_kwargs=pipe_kwargs)
    else:
      device_str = f'cuda:{dist.get_rank()}'
      model = BenchmarkModel(W, L, in_size, out_size).to(device_str)
      model = deepspeed.pipe.PipelineModule(layers=model.to_layers(), **pipe_kwargs)
  else:
    model = BenchmarkModel(W, L, in_size, out_size, activation_checkpoint_interval=activation_checkpoint_interval)  
  return model


def get_dataset(in_size, out_size, len=2**16):
  return BenchmarkDataset(in_size, out_size, int(len*0.8) ), \
         BenchmarkDataset(in_size, out_size, int(len*0.2) )
