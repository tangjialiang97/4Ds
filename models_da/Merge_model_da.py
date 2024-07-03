import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math
cfg_block = {
        'vgg16_bn': [7, 14, 24, 34, 44],
        'vgg19_bn': [7, 14, 27, 40, 53],
        'vgg13_bn': [7, 14, 21, 28, 35],
        'vgg11_bn': [4, 8, 15, 22, 29],
}
def replace_denormals(x, threshold=1e-5):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y

def decompose(x, mode='all'):
    fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
    if mode == 'all' or mode == 'amp':
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(replace_denormals(fft_amp))
    else:
        fft_amp = None

    if mode == 'all' or mode == 'phase':
        fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
    else:
        fft_pha = None
    return fft_pha, fft_amp


def compose(phase, amp):
    x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
    x = x / math.sqrt(x.shape[2] * x.shape[3])
    x = torch.view_as_complex(x)
    return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Adapter_conv(nn.Module):
    def __init__(self, c_in, reduction=4):
        norm_layer = nn.BatchNorm2d
        super(Adapter_conv, self).__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.T = 0.1
        self.fc = nn.Sequential(
            conv1x1(c_in, c_in // reduction),
            norm_layer(c_in // reduction),
            nn.ReLU(inplace=True),
            conv1x1(c_in // reduction, c_in),
            norm_layer(c_in),
        )

    def forward(self, x):
        weight = F.softmax(self.weight / self.T, dim=0)
        x_adapt = self.fc(x)
        phase_ori, amp_ori = decompose(x, 'all')
        phase_adapt, amp_adapt = decompose(x_adapt, 'all')
        amp = amp_adapt * weight[0] + amp_ori * weight[1]
        # print('Weights:')
        # print(weight[0], weight[1])

        x = compose(phase_ori, amp)

        return x, phase_ori

class Adapter_conv_temp(nn.Module):
    def __init__(self, c_in, reduction=4):
        norm_layer = nn.BatchNorm2d
        super(Adapter_conv_temp, self).__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.T = 0.1
        self.fc = nn.Sequential(
            conv1x1(c_in, c_in // reduction),
            norm_layer(c_in // reduction),
            nn.ReLU(inplace=True),
            conv1x1(c_in // reduction, c_in),
            norm_layer(c_in),
        )

    def forward(self, x):
        weight = F.softmax(self.weight / self.T, dim=0)
        phase_ori, amp_ori = decompose(x, 'all')
        amp_adapt = self.fc(amp_ori)
        amp = amp_adapt * weight[0] + amp_ori * weight[1]

        x = compose(phase_ori, amp)

        return x, phase_ori

class Adapter_linear(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_linear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.T = 0.1
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )

    def forward(self, x):
        weight = F.softmax(self.weight / self.T, dim=0)
        x_adapt = self.fc(x)
        phase_ori, amp_ori = decompose(x, 'all')
        phase_adapt, amp_adapt = decompose(x_adapt, 'all')
        amp = amp_adapt * weight[0] + amp_ori * weight[1]

        x = compose(phase_ori, amp)

        return x


class Merge_model_da_vgg(nn.Module):
    def __init__(self, model, feat_dim, model_name='vgg16_bn'):
        super(Merge_model_da_vgg, self).__init__()
        self.model = model

        blocks = cfg_block[model_name]

        self.layer0 = nn.Sequential(*list(self.model.features.children())[0:blocks[0]])
        self.layer1 = nn.Sequential(*list(self.model.features.children())[blocks[0]:blocks[1]])
        self.layer2 = nn.Sequential(*list(self.model.features.children())[blocks[1]:blocks[2]])
        self.layer3 = nn.Sequential(*list(self.model.features.children())[blocks[2]:blocks[3]])
        self.layer4 = nn.Sequential(*list(self.model.features.children())[blocks[3]:blocks[4]])
        self.classifier = self.model.classifier

        self.adapter1 = Adapter_conv(feat_dim[1])
        self.adapter2 = Adapter_conv(feat_dim[2])
        self.adapter3 = Adapter_conv(feat_dim[3])
        self.adapter4 = Adapter_conv(feat_dim[4])


    def forward(self, x, is_feat=False):
        x = self.layer0(x)
        f0 = x
        x = self.layer1(x)
        x, fo1 = self.adapter1(x)
        x = self.layer2(x)
        x, fo2 = self.adapter2(x)
        x = self.layer3(x)
        x, fo3 = self.adapter3(x)
        x = self.layer4(x)
        x, fo4 = self.adapter4(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)


        if is_feat == True:
            # return [f0, f1, f2, f3, f4, f5], x
            return [fo1, fo2, fo3, fo4], x
        else:
            return x

class Merge_model_da(nn.Module):
    def __init__(self, model, feat_dim):
        super(Merge_model_da, self).__init__()
        self.model = model

        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool

        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

        self.adapter0 = Adapter_conv(feat_dim[0])
        self.adapter1 = Adapter_conv(feat_dim[1])
        self.adapter2 = Adapter_conv(feat_dim[2])
        self.adapter3 = Adapter_conv(feat_dim[3])
        self.adapter4 = Adapter_conv(feat_dim[4])
        self.adapter5 = Adapter_linear(feat_dim[5])

        self.avgpool = self.model.avgpool
        self.fc = self.model.fc

    def forward(self, x, is_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.adapter0(x)
        f0 = x

        x = self.layer1(x)
        x, fo1 = self.adapter1(x)
        f1 = x

        x = self.layer2(x)
        x, fo2 = self.adapter2(x)
        f2 = x

        x = self.layer3(x)
        x, fo3 = self.adapter3(x)
        f3 = x

        x = self.layer4(x)
        x, fo4 = self.adapter4(x)
        f4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.adapter5(x)
        f5 = x
        x = self.fc(x)

        if is_feat == True:
            # return [f0, f1, f2, f3, f4, f5], x
            return [fo1, fo2, fo3, fo4], x
        else:
            return x


class Adapter_conv_var(nn.Module):
    def __init__(self, c_in, reduction=4, num_conv=1):
        norm_layer = nn.BatchNorm2d
        super(Adapter_conv_var, self).__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.T = 0.1
        if num_conv == 1:
            self.fc = nn.Sequential(
                conv1x1(c_in, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 2:
            self.fc = nn.Sequential(
                conv1x1(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 3:
            self.fc = nn.Sequential(
                conv1x1(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 4:
            self.fc = nn.Sequential(
                conv1x1(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 5:
            self.fc = nn.Sequential(
                conv1x1(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv1x1(c_in // reduction, c_in),
                norm_layer(c_in),
            )

    def forward(self, x):
        weight = F.softmax(self.weight / self.T, dim=0)
        x_adapt = self.fc(x)
        phase_ori, amp_ori = decompose(x, 'all')
        phase_adapt, amp_adapt = decompose(x_adapt, 'all')
        amp = amp_adapt * weight[0] + amp_ori * weight[1]

        x = compose(phase_ori, amp)

        return x, phase_ori


class Adapter_conv_var3x3(nn.Module):
    def __init__(self, c_in, reduction=4, num_conv=1):
        norm_layer = nn.BatchNorm2d
        super(Adapter_conv_var3x3, self).__init__()
        self.weight = nn.Parameter(torch.zeros(2))
        self.T = 0.1
        if num_conv == 1:
            self.fc = nn.Sequential(
                conv3x3(c_in, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 2:
            self.fc = nn.Sequential(
                conv3x3(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 3:
            self.fc = nn.Sequential(
                conv3x3(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 4:
            self.fc = nn.Sequential(
                conv3x3(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in),
                norm_layer(c_in),
            )
        elif num_conv == 5:
            self.fc = nn.Sequential(
                conv3x3(c_in, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in // reduction),
                norm_layer(c_in // reduction),
                nn.ReLU(inplace=True),
                conv3x3(c_in // reduction, c_in),
                norm_layer(c_in),
            )

    def forward(self, x):
        weight = F.softmax(self.weight / self.T, dim=0)
        x_adapt = self.fc(x)
        phase_ori, amp_ori = decompose(x, 'all')
        phase_adapt, amp_adapt = decompose(x_adapt, 'all')
        amp = amp_adapt * weight[0] + amp_ori * weight[1]

        x = compose(phase_ori, amp)

        return x, phase_ori