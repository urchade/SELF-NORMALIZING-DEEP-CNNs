from block import SNDCNNBlock
import torch

model = SNDCNNBlock(3, 3, 3)

minibatch_images = torch.randn(size=(5, 3, 55, 55), dtype=torch.float32)

x = model(minibatch_images)
