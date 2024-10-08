import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import copy


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find(
            'ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass




class ResNet_FE(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet18(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        self.bottle = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)

    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        out = self.bn(self.bottle(out))
        return out

class ResNet_FE_ADKD(nn.Module):
    def __init__(self, model_resnet):
        super().__init__()
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        self.bottle = nn.Linear(512, 512)
        self.bn = nn.BatchNorm1d(512)

    def forward(self, x):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        # out = self.bn(self.bottle(out))
        return out

