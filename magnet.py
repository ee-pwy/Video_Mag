import torch.nn as nn
import torch
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        return self.relu(x + self.bn2(self.conv2(self.pad(F.relu(self.bn1(self.conv1(self.pad(x))))))))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # network structure of encoder
        self.pad_1 = nn.ReplicationPad2d(1)
        self.pad_2 = nn.ReplicationPad2d(2)
        self.pad_3 = nn.ReplicationPad2d(3)
        self.conv_input_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)
        self.bn_input_1 = nn.BatchNorm2d(16)
        self.conv_input_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2)
        self.bn_input_2 = nn.BatchNorm2d(8)
        self.conv_text_1 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2)
        self.bn_text_1 = nn.BatchNorm2d(4)
        self.conv_text_2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1)
        self.bn_text_2 = nn.BatchNorm2d(2)
        self.conv_shape_1 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2)
        self.bn_shape_1 = nn.BatchNorm2d(4)
        self.conv_shape_2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1)
        self.bn_shape_2 = nn.BatchNorm2d(2)
        self.res_blk_encoder = nn.ModuleList([Residual(in_channels=8) for i in range(3)])
        self.res_blk_text = nn.ModuleList([Residual(in_channels=2) for i in range(2)])
        self.res_blk_shape = nn.ModuleList([Residual(in_channels=2) for i in range(2)])

        # network structure of manipulator
        self.conv_man_1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=1)
        self.bn_man_1 = nn.BatchNorm2d(2)
        self.conv_man_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=1)
        self.bn_man_2 = nn.BatchNorm2d(2)
        self.res_blk_man = nn.ModuleList([Residual(in_channels=2) for i in range(1)])

        # network structure of decoder
        self.conv_cat_1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1)
        self.bn_cat_1 = nn.BatchNorm2d(8)
        self.conv_cat_2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1)
        self.bn_cat_2 = nn.BatchNorm2d(4)
        self.res_blk_decoder = nn.ModuleList([Residual(in_channels=4) for i in range(9)])
        self.conv_output_1 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1)
        self.bn_output_1 = nn.BatchNorm2d(3)
        self.conv_output_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1)
        self.bn_output_2 = nn.BatchNorm2d(3)

    def encoder(self, x):
        x = self.pad_3(x)
        x = F.relu(self.bn_input_1(self.conv_input_1(x)))
        x = self.pad_1(x)
        x = F.relu(self.bn_input_2(self.conv_input_2(x)))
        for block in self.res_blk_encoder:
            x = block(x)

        text = self.pad_1(x)
        text = F.relu(self.bn_text_1(self.conv_text_1(text)))
        text = self.pad_1(text)
        text = F.relu(self.bn_text_2(self.conv_text_2(text)))
        for block in self.res_blk_text:
            text = block(text)

        shape = self.pad_1(x)
        shape = F.relu(self.bn_shape_1(self.conv_shape_1(shape)))
        shape = self.pad_1(shape)
        shape = F.relu(self.bn_shape_2(self.conv_shape_2(shape)))
        for block in self.res_blk_shape:
            shape = block(shape)

        return text, shape

    def manipulator(self, shape_a, shape_b, amplification_factor):
        diff = shape_b - shape_a
        diff = self.pad_2(diff)
        diff = F.relu(self.bn_man_1(self.conv_man_1(diff)))
        diff = (diff.transpose(0, 3)*amplification_factor).transpose(0, 3)
        diff = self.pad_2(diff)
        diff = F.relu(self.bn_man_2(self.conv_man_2(diff)))
        for block in self.res_blk_man:
            diff = block(diff)
        return shape_b + diff

    def decoder(self, text, shape):  # tensor: N(batchsize)*C(channel)*H*W
        x = torch.cat((text, shape), 1)
        x = self.pad_1(x)
        x = F.relu(self.bn_cat_1(self.conv_cat_1(x)))
        x = F.interpolate(x, scale_factor=2)
        x = self.pad_1(x)
        x = F.relu(self.bn_cat_2(self.conv_cat_2(x)))
        for block in self.res_blk_decoder:
            x = block(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.pad_1(x)
        x = F.relu(self.bn_output_1(self.conv_output_1(x)))
        x = self.pad_3(x)
        x = F.relu(self.conv_output_2(x))
        return x

    def forward(self, image_a, image_b, amplification_factor):
        self.text_a, self.shape_a = self.encoder(image_a)
        self.text_b, self.shape_b = self.encoder(image_b)
        encode_shape = self.manipulator(self.shape_a, self.shape_b, amplification_factor)
        output = self.decoder(self.text_b, encode_shape)
        return output


class origin_Net(nn.Module):
    def __init__(self):
        super(origin_Net, self).__init__()

        # network structure of encoder
        self.pad_1 = nn.ReplicationPad2d(1)
        self.pad_2 = nn.ReplicationPad2d(2)
        self.pad_3 = nn.ReplicationPad2d(3)
        self.conv_input_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1)
        self.conv_input_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv_text_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv_text_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv_shape_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv_shape_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.res_blk_encoder = nn.ModuleList([Residual(in_channels=32) for i in range(3)])
        self.res_blk_text = nn.ModuleList([Residual(in_channels=32) for i in range(2)])
        self.res_blk_shape = nn.ModuleList([Residual(in_channels=32) for i in range(2)])

        # network structure of manipulator
        self.conv_man_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.conv_man_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.res_blk_man = nn.ModuleList([Residual(in_channels=32) for i in range(1)])

        # network structure of decoder
        self.conv_cat_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv_cat_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.res_blk_decoder = nn.ModuleList([Residual(in_channels=64) for i in range(9)])
        self.conv_output_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.conv_output_2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, stride=1)

    def encoder(self, x):
        x = self.pad_3(x)
        x = F.relu(self.conv_input_1(x))
        x = self.pad_1(x)
        x = F.relu(self.conv_input_2(x))
        for block in self.res_blk_encoder:
            x = block(x)

        text = self.pad_1(x)
        text = F.relu(self.conv_text_1(text))
        text = self.pad_1(text)
        text = F.relu(self.conv_text_2(text))
        for block in self.res_blk_text:
            text = block(text)

        shape = self.pad_1(x)
        shape = F.relu(self.conv_shape_1(shape))
        shape = self.pad_1(shape)
        shape = F.relu(self.conv_shape_2(shape))
        for block in self.res_blk_shape:
            shape = block(shape)

        return text, shape

    def manipulator(self, shape_a, shape_b, amplification_factor):
        diff = shape_b - shape_a
        diff = self.pad_1(diff)
        diff = F.relu(self.conv_man_1(diff))
        diff = (diff.transpose(0, 3)*amplification_factor).transpose(0, 3)
        diff = self.pad_1(diff)
        diff = F.relu(self.conv_man_2(diff))
        for block in self.res_blk_man:
            diff = block(diff)
        return shape_b + diff

    def decoder(self, text, shape):  # tensor: N(batchsize)*C(channel)*H*W
        x = torch.cat((text, shape), 1)
        x = self.pad_1(x)
        x = F.relu(self.conv_cat_1(x))
        x = F.interpolate(x, scale_factor=2)
        x = self.pad_1(x)
        x = F.relu(self.conv_cat_2(x))
        for block in self.res_blk_decoder:
            x = block(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.pad_1(x)
        x = F.relu(self.conv_output_1(x))
        x = self.pad_3(x)
        x = F.relu(self.conv_output_2(x))
        return x

    def forward(self, image_a, image_b, amplification_factor):
        self.text_a, self.shape_a = self.encoder(image_a)
        self.text_b, self.shape_b = self.encoder(image_b)
        encode_shape = self.manipulator(self.shape_a, self.shape_b, amplification_factor)
        output = self.decoder(self.text_b, encode_shape)
        return output
