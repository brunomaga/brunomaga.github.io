from torch import nn

class BenchmarkModel(nn.Module):
  """" DNN with L layers and W neurons per layer """

  def __init__(self, W, L, in_size, out_size):
    super(BenchmarkModel, self).__init__()
    self.layers = [nn.Linear(in_size, W), nn.ReLU()]
    for _ in range(L-2):
      self.layers += [nn.Linear(W, W), nn.ReLU()]
    self.layers += [nn.Linear(W, out_size), nn.ReLU()]
    self.layers = nn.Sequential(*self.layers)
 
  def forward(self, x):
    return self.layers(x)

