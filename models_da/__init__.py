from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x48d, resnext101_32x8d, resnext50_32x4d, wide_resnet50_2, wide_resnet101_2
from .vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
    'resnext101_32x48d': resnext101_32x48d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnext50_32x4d': resnext50_32x4d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
}