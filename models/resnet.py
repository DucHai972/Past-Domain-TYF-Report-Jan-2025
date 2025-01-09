import torch.nn as nn

class Block(nn.Module):
    def __init__(self, inp_channels, out_channels, is_down):
        # Your Block implementation
        pass

    def forward(self, x):
        pass

class Resnet(nn.Module):
    def __init__(self, inp_channels, num_residual_block, num_class):
        # Your ResNet implementation
        pass

    def forward(self, x):
        pass
