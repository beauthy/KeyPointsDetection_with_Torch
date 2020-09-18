# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 16:54
# @Author  : Haiyan Tan 
# @File    : keypoints_Net.py

from collections import OrderedDict
import torch
import torch.nn as nn
import dsntnn


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class KeyPointsModel(nn.Module):
    def __init__(self):
        super(KeyPointsModel, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(2, stride=2)

        # stage 1
        block1_0_0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
        ])
        block1_0_1 = OrderedDict([
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
        ])
        block1_0_2 = OrderedDict([
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
        ])
        block1_0_3 = OrderedDict([
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3', [512, 512, 3, 1, 1]),
            ('conv4_4', [512, 512, 3, 1, 1]),
            ('conv5_1', [512, 512, 3, 1, 1]),
            ('conv5_2', [512, 512, 3, 1, 1]),
            ('conv5_3_CPM', [512, 128, 3, 1, 1])
        ])



        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 24, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0_0'] = block1_0_0
        blocks['block1_0_1'] = block1_0_1
        blocks['block1_0_2'] = block1_0_2
        blocks['block1_0_3'] = block1_0_3

        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                ('Mconv1_stage%d' % i, [152, 128, 7, 1, 3]),
                ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d' % i, [128, 24, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0_0 = blocks['block1_0_0']
        self.model1_0_1 = blocks['block1_0_1']
        self.model1_0_2 = blocks['block1_0_2']
        self.model1_0_3 = blocks['block1_0_3']

        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        # block0
        out1_0_0 = self.model1_0_0(x)
        output, indices = self.maxpool(out1_0_0)
        output_un_pool = self.maxunpool(output, indices)

        out1_0_1 = self.model1_0_1(output_un_pool)
        output, indices = self.maxpool(out1_0_1)
        output_un_pool = self.maxunpool(output, indices)

        out1_0_2 = self.model1_0_2(output_un_pool)
        output, indices = self.maxpool(out1_0_2)
        output_un_pool = self.maxunpool(output, indices)

        out1_0 = self.model1_0_3(output_un_pool)

        # block1
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


class CoordRegression(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = KeyPointsModel()
        self.hm_conv = nn.Conv2d(24, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # run images trough FCN
        fcn_out = self.fcn(images)
        # use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)

        # calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = KeyPointsModel()
    model = CoordRegression(24)
    summary(model.cuda(), (3, 128, 128))
