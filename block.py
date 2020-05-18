import numpy as np
from torch import nn


class SNDCNNBlock(nn.Module):
    def __init__(self, num_blocks, n_channels, kernel_size):
        super().__init__()

        padding = int(kernel_size / 2)
        self.layers = nn.ModuleList()

        for i in range(num_blocks):
            conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding)
            conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding)

            # LeCun initialization
            conv1.weight.data.normal_(0.0, np.sqrt(1. / np.prod(conv1.weight.shape[1:])))
            conv2.weight.data.normal_(0.0, np.sqrt(1. / np.prod(conv2.weight.shape[1:])))

            combined = nn.Sequential(conv1, nn.SELU(), conv2)
            self.layers.append(combined)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x
