import torch 
from torch import nn 

import numpy as np
from sys import exit as e 



class ConvNet(nn.Module):
  def __init__(self, in_chan, out_chan, kernel=3):
    super().__init__()
    self.conv1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=kernel, padding=1, stride=1),
                               nn.ReLU(),
                               nn.MaxPool2d(2)
    )

  def forward(self, x):
    out = self.conv1(x)
    return out

class ConvBlock(nn.Module):
  def __init__(self, in_chan, n_block, kernel=3, expansion=32, max_features=512):
    super().__init__()

    self.blocks = nn.ModuleList()
    for i in range(n_block):
      # in_channel = (in_chan if i == 0 else min(max_features, expansion * (2**i)))
      in_channel = in_chan if i == 0 else min(max_features, expansion * (2**(i-1)))
      out_channel = min(max_features, expansion * (2**(i)))
      # out_channel = min(max_features, expansion* (2**(i+1)))
      self.blocks.append(ConvNet(in_channel, out_channel, kernel=kernel))

    
  def forward(self, x):
    out = x
    for block in self.blocks:
      out = block(out)
    return out

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.lin_blocks = nn.Sequential(
            # nn.Linear(in_features, in_features//2),
            # nn.ReLU(),
            nn.Linear(in_features, out_features),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.lin_blocks(x)
    


class ClassNet(nn.Module):
  def __init__(self, in_chan, n_block, img_size, n_class, kernel=3, max_features=512, expansion=32):
    super().__init__()
    # Max number of ConvBlocks allowed
    curr_size = img_size//(2**n_block)
    max_allowed = np.floor(np.log2(img_size))
    assert curr_size > 0, e(f"Too many num_blocks. Max allowed num_blocks is {int(max_allowed)}")
    
    self.convnet = ConvBlock(in_chan, n_block, kernel)
    flattened_feats = (expansion * (2 ** (n_block - 1))) * (((img_size//(2**n_block))**2))
    self.linear = LinearBlock(flattened_feats, n_class)

  def forward(self, x):
    out = self.convnet(x)
    out = out.flatten(1)
    out = self.linear(out)
    return out


    
