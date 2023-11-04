from torch import nn

class BenchmarkModel(nn.Module):
  """" DNN with W input features, W neurons per layer, W output classes and L layers """

  def __init__(self, W, L):
    super(BenchmarkModel, self).__init__()
    self.layers = []
    for _ in range(L):
      self.layers += [nn.Linear(W, W), nn.ReLU()]
    self.layers = nn.Sequential(*self.layers)
 
  def forward(self, x):
    return self.layers(x)


