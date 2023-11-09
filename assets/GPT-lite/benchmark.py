from torch import nn

class BenchmarkModel(nn.Module):
  """" DNN with L layers and W neurons per layer """

  def __init__(self, W, L, in_size, out_size):
    super(BenchmarkModel, self).__init__()
    self.layers = []
    for l in range(L):
      self.layers.append( nn.Linear(
        in_size  if l==0   else W,
        out_size if l==L-1 else W) )
      self.layers.append( nn.ReLU() )
    self.layers = nn.Sequential(*self.layers)
 
  def forward(self, x):
    return self.layers(x)

