from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is a list of features, each feature in the list with various shape
        feat0 = x[0]
        feat1 = x[1]
        feat2 = x[2]
        feat3 = x[3]
        _, _, h, w = feat3.size()

        # Average pooling, the features feat0, feat1, feat2 are resized to the same size as feat3
        Avg_pool = nn.AdaptiveAvgPool2d((h, w))
        feat0 = Avg_pool(feat0)
        feat1 = Avg_pool(feat1)
        feat2 = Avg_pool(feat2)

        feat = torch.cat((feat0, feat1, feat2, feat3), 1)
        b, c, _, _ = feat.size()

        # Calculate the weights
        y = self.avg_pool(feat).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return feat * y.expand_as(feat)

class SELayer_Single(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_Single, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_S, x_T):
        # x_S is a list of features from the student, each feature in the list with various shape
        # x_T is a list of features from the teacher, each feature in the list with various shape
        feat0_T = x_T[0]
        feat1_T = x_T[1]
        feat2_T = x_T[2]
        feat3_T = x_T[3]
        _, _, h, w = feat3_T.size()

        # Average pooling, the features feat0_T, feat1_T, feat2_T are resized to the same size as feat3_T
        Avg_pool = nn.AdaptiveAvgPool2d((h, w))
        feat0_T = Avg_pool(feat0_T)
        feat1_T = Avg_pool(feat1_T)
        feat2_T = Avg_pool(feat2_T)

        feat0_S = x_S [0]
        feat1_S  = x_S[1]
        feat2_S  = x_S[2]
        feat3_S  = x_S[3]
        _, _, h, w = feat3_S .size()

        Avg_pool = nn.AdaptiveAvgPool2d((h, w))
        feat0_S  = Avg_pool(feat0_S)
        feat1_S  = Avg_pool(feat1_S)
        feat2_S  = Avg_pool(feat2_S)

        feat_T = torch.cat((feat0_T, feat1_T, feat2_T, feat3_T), 1)
        feat_S = torch.cat((feat0_S, feat1_S, feat2_S, feat3_S), 1)

        # Calculate the weights from the teacher
        b, c, _, _ = feat_T.size()
        y = self.avg_pool(feat_T).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return feat_S * y.expand_as(feat_S), feat_T * y.expand_as(feat_T)


class BlockHintLoss(nn.Module):
    def __init__(self):
        super(BlockHintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t, weights=None):

        loss = 0
        # Calculate the MSE between the intermediate layers of the student and teacher
        for i in range(len(f_s)):
            f_s[i] = f_s[i] * weights[i].expand_as(f_s[i])
            f_t[i] = f_t[i] * weights[i].expand_as(f_t[i])
            loss += self.crit(f_s[i], f_t[i])
        return loss * 0.25
