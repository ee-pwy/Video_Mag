from magnet import *
from data_load import *
from torch.optim import lr_scheduler
import torch.optim as optim

class radar_Net(nn.Module):
    def __init__(self):
        super(radar_Net, self).__init__()
        self.fc = nn.linear()
        self.res_blk = nn.ModuleList([Residual(in_channels=8) for i in range(10)])