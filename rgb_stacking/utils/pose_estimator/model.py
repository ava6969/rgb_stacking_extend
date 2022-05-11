from typing import Optional, Callable

import torch.nn as nn
from torch import Tensor
import numpy as np
import torch
from torchvision.models.resnet import BasicBlock, conv1x1, Bottleneck

from rgb_stacking.utils.pose_estimator.models.backbone import Backbone, Joiner
from rgb_stacking.utils.pose_estimator.models.detr import DETR
from rgb_stacking.utils.pose_estimator.models.position_encoding import PositionEmbeddingSine
from rgb_stacking.utils.pose_estimator.models.transformer import Transformer


def conv2x2(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)


def _make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class DETRWrapper(nn.Module):

    def __init__(self):
        super().__init__()

        hidden_dim = 512
        backbone = Backbone("resnet50", train_backbone=True, return_interm_layers=False, dilation=False)
        pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = Joiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels
        transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=False)
        self.detr = DETR(backbone_with_pos_enc, transformer, num_classes=7, num_queries=3)

    def forward(self, inputs):
        inputs = torch.stack([inputs[k] for k in ['fl', 'fr', 'bl']], 1)
        # propagate inputs through ResNet-50 up to avg-pool layer
        B = inputs.size(0)
        x = inputs.flatten(0, 1)

        x = self.detr(x)[0]

        return x.view(B, -1)

class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=0, bias=False)

        self.bn0 = nn.BatchNorm2d(3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.dropout = torch.nn.Dropout(0.9)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.layer1_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.layer1 = self._make_layer(64, 3)

        self.layer2_conv = nn.Conv2d(self.inplanes, 128, kernel_size=2, stride=2)
        self.inplanes = 128
        self.layer2 = self._make_layer(128, 4, stride=2)

        self.layer3_conv = nn.Conv2d(self.inplanes, 256, kernel_size=2, stride=2)
        self.inplanes = 256
        self.layer3 = self._make_layer(256, 6, stride=2)

        self.layer4_conv = nn.Conv2d(self.inplanes, 512, kernel_size=2, stride=2)
        self.inplanes = 512
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(inplanes=self.inplanes,
                                 planes=planes,
                                 stride=stride,
                                 downsample=downsample,
                                 base_width=self.base_width,
                                 norm_layer=nn.BatchNorm2d))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes=self.inplanes,
                                     planes=planes,
                                     base_width=self.base_width,
                                     norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        # See note [TorchScript super()]

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.bn1(x)

        x = self.layer1_conv(x)
        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2_conv(x)
        x = self.layer2(x)

        x = self.layer3_conv(x)
        x = self.layer3(x)

        x = self.layer4_conv(x)
        x = self.layer4(x)

        x = self.maxpool2(x)
        x = torch.flatten(x, 1)

        return x


class LargeVisionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = ResNet()
        self.relu = torch.nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6144, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 21)
        self.bn_f1 = torch.nn.BatchNorm1d(512)
        self.bn_f2 = torch.nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(0.9)

    def parse_image(self, inputs):
        B = inputs.size(0)
        x = inputs.flatten(0, 1)
        return B, self.backbone(x)

    def forward(self, inputs) -> torch.Tensor:
        inputs = torch.stack([inputs[k] for k in ['fl', 'fr', 'bl']], 1)
        B, x = self.parse_image(inputs)
        x = x.view(B, 3, -1).flatten(1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(self.bn_f1(x))
        x = self.relu(self.bn_f2(self.fc2(x)))
        return self.fc3(x)


class VisionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 5, 1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1)
        self.max_pool = torch.nn.MaxPool2d(3, 3)
        self.relu = torch.nn.ReLU()
        self.resnet_block_1 = _make_layer(32, BasicBlock, 16, 1, 3)
        self.resnet_block_2 = _make_layer(16, BasicBlock, 32, 2, 3)
        self.resnet_block_3 = _make_layer(32, BasicBlock, 64, 2, 3)
        self.resnet_block_4 = _make_layer(64, BasicBlock, 64, 2, 3)
        self.soft_max = torch.nn.Softmax2d()

        self.fc1 = torch.nn.Linear(192, 128)
        self.outputs =  nn.Linear(128, 21)

    def forward(self, inputs):
        inputs = torch.stack([inputs[k] for k in ['fl', 'fr', 'bl']], 1)
        B = inputs.size(0)
        x = inputs.flatten(0, 1)

        x = self.max_pool(self.relu(self.conv2(self.relu(self.conv1(x)))))

        x = self.resnet_block_4(self.resnet_block_3(self.resnet_block_2(self.resnet_block_1(x))))
        x = self.soft_max(x).flatten(1)

        x = x.view(B, 3, -1).flatten(1)
        x = self.relu(x)
        x = self.relu(self.fc1(x))
        return self.outputs(x)


if __name__ == '__main__':
    img = dict(fl=torch.randn((3, 3, 200, 200)),
               fr=torch.randn((3, 3, 200, 200)),
               bl=torch.randn((3, 3, 200, 200)) )

    model = DETRWrapper()
    print(model)

    print(model(img).shape)

    model = LargeVisionModule()
    print(model)

    print(model(img).shape)

    model = VisionModule()
    print(model)

    print(model(img).shape)
