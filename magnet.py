import torch.nn as nn
import torch.nn.functional as F


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class Residual(nn.Module):
    def __init__(self, in_channels, activation='relu'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels, kernel_size=3, stride=1, padding=1,
                               padding_mode='circular')

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # network structure of encoder
        self.conv_input_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3,
                                      padding_mode='circular')
        self.conv_input_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1,
                                      padding_mode='circular')
        self.conv_text_1 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1,
                                     padding_mode='circular')
        self.conv_text_2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1,
                                     padding_mode='circular')
        self.conv_shape_1 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1,
                                      padding_mode='circular')
        self.conv_shape_2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1,
                                      padding_mode='circular')
        self.res_blk_encoder = nn.ModuleList([Residual() for i in range(3)])
        self.res_blk_text = nn.ModuleList([Residual() for i in range(2)])
        self.res_blk_shape = nn.ModuleList([Residual() for i in range(2)])

        # network structure of manipulator
        self.conv_man_1 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular')
        self.conv_man_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular')
        self.res_blk_man = nn.ModuleList([Residual() for i in range(1)])

        # network structure of decoder
        self.conv_cat_1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular')
        self.conv_cat_2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular')
        self.res_blk_decoder = nn.ModuleList([Residual() for i in range(9)])
        self.conv_output_1 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1,
                                       padding_mode='circular')
        self.conv_output_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3,
                                       padding_mode='circular')

        self.fc3 = nn.Linear(84, 10)

    def encoder(self, x):
        x = F.relu(self.conv_input_1(x))
        x = F.relu(self.conv_input_2(x))
        for block in self.res_blk_encoder:
            x = block(x)

        text = F.relu(self.conv_text_1(x))
        text = F.relu(self.conv_text_2(text))
        for block in self.res_blk_text:
            text = block(text)

        shape = F.relu(self.conv_text_1(x))
        shape = F.relu(self.conv_text_2(shape))
        for block in self.res_blk_shape:
            shape = block(shape)

        return text, shape

    def manipulator(self, shape_a, shape_b, amplification_factor):
        diff = shape_b - shape_a
        diff = amplification_factor * F.relu(self.conv_man_1(diff))
        diff = F.relu(self.conv_man_2(diff))
        for block in self.res_blk_man:
            diff = block(diff)
        return shape_b + diff

    def decoder(self, text, shape):  # tensor: N(batchsize)*C(channel)*H*W
        x = F.cat((text, shape), 1)
        x = F.relu(self.conv_cat_1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv_cat_2(x))
        for block in self.res_blk_decoder:
            text = block(text)

        x = F.interpolate(x, scale_factor=4)
        x = F.relu(self.conv_output_1(x))
        x = F.relu(self.conv_output_2(x))
        return x

    def forward(self, image_a, image_b, amplification_factor):
        text_a, shape_a = self.encoder(image_a)
        text_b, shape_b = self.encoder(image_b)
        encode_shape = self.manipulator(shape_a, shape_b, amplification_factor)
        output = self.decoder(text_b, encode_shape)
        return output
