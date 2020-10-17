import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)

class I2VNet(torch.nn.Module):
    def __init__(self, mode='tag', tag_len=1539):
        super(I2VNet, self).__init__()
        self.mode = mode
        self.tag_len = tag_len
        self.block = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('conv4_1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('conv5_1', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('conv6_1', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu6_1', nn.ReLU(inplace=True)),
            ('conv6_2', nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu6_2', nn.ReLU(inplace=True)),
            ('conv6_3', nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('relu6_3', nn.ReLU(inplace=True)),
            ('dropout6_3', nn.Dropout(p=0.5, inplace=True))
        ]))

        self.tag_classifier = nn.Sequential(OrderedDict([
            ('conv6_4', nn.Conv2d(in_channels=1024, out_channels=self.tag_len, kernel_size=(3, 3), stride=(1, 1), padding=1)),
            ('pool6', nn.AvgPool2d(kernel_size=(7,7), stride=(1,1)))
        ]))

        self.prob = nn.Sigmoid()

        self.flatten = Flatten()

        self.encode1 = nn.Linear(in_features=7*7*1024, out_features=4096, bias=True)
        self.encode1neuron = nn.Sigmoid()

        # for train or fine-tuning
        self.encode2 = nn.Linear(in_features=4096, out_features=self.tag_len, bias=True)

    def _change_mode(self, new_mode='tag'):
        self.mode = new_mode
        if new_mode == 'tag':
            self.module_list =  nn.ModuleList([self.block, self.tag_classifier, self.prob])
        elif new_mode == 'feature':
            self.module_list =  nn.ModuleList([self.block, self.flatten, self.encode1])

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x


# net = I2VNet(mode='feature')
#
# print(net.state_dict()['encode1.bias'])
# print(net.state_dict()['encode2.bias'])
# load_weights = torch.load('../models/i2v_pretrained_ver200_pytorch.bin',
#                                   map_location={'cuda:0': 'cpu'})
# net.load_state_dict(load_weights, strict=False)
# print(net.state_dict()['encode1.bias'][0])
# print(net.state_dict()['encode2.bias'][0])
#
# feature_premodel = np.load('../models/i2v_ver200.npz')
# print(feature_premodel['encode1/b'])
# print(feature_premodel['encode1/W'])
#
# for param in net.parameters():
#     print(param)
