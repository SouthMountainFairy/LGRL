import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal


class part_att_fc(nn.Module):
    def __init__(self):
        super(part_att_fc, self).__init__()
        self.head_layer = nn.Linear(1024, 64)

    def forward(self, x):
        fatt1 = self.head_layer(x)
        return fatt1


class sketch_fc(nn.Module):
    def __init__(self):
        super(sketch_fc, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2048, 64)
        )

    def forward(self, x):
        x = self.linear(x)
        return x
