from block import SNDCNNBlock
import torch

model = SNDCNNBlock(3, 3, 3)

minibatch_images = torch.randn(size=(5, 55, 55, 3), dtype=torch.float32).reshape((5, 3, 55, 55))

x = model(minibatch_images)