from magnet import *
from data_load import *
from torch.optim import lr_scheduler
import torch.optim as optim


class radar_Net(nn.Module):
    def __init__(self, size=0):
        super(radar_Net, self).__init__()
        self.size = size
        self.pad2 = nn.ReplicationPad2d(2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.res_blk = nn.ModuleList([Residual(in_channels=16) for i in range(3)])

    def forward(self, radar_image):
        if self.size != 0:
            x = F.interpolate(radar_image, size=self.size)
        else:
            x = radar_image
        x = self.pad2(x)
        x = F.relu(self.conv1(x))
        x = self.res_blk(x)
        return x


class fusion(nn.Module):
    def __init__(self, path, size=0, batchsize=0):
        super(fusion, self).__init__()
        self.batchsize = batchsize
        self.size = size
        self.radar = radar_Net()
        self.video = torch.load(path)
        self.pad1 = nn.ReplicationPad2d(1)
        self.vid_conv = nn.Conv2d(in_channels=228, out_channels=32, kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        expansion = 3
        self.pool = nn.AdaptiveAvgPool2d(expansion)
        self.fc = nn.Linear(expansion*expansion*128, 1)

    def forward(self, radar_image, video_frames):
        video_feature = torch.randn(self.batchsize, 0, self.size[0], self.size[1])
        for i in range(len(video_frames)):
            _, shape = self.video.encoder(video_frames[i])
            video_feature = torch.cat((video_feature, shape), 1)
        video_feature = F.relu(self.vid_conv(self.pad1(video_feature)))
        radar_feature = self.radar(radar_image)

        x = torch.cat((video_feature, radar_feature), 1)
        x = F.relu(self.conv1(self.pad1(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
