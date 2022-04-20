from typing import Optional, Callable

import torch.nn as nn
from torchvision.models import resnet50
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
import numpy as np

from torch import Tensor

from rgb_stacking.utils.pose_estimator.models.backbone import Backbone, Joiner
from rgb_stacking.utils.pose_estimator.models.detr import DETR
from rgb_stacking.utils.pose_estimator.models.position_encoding import PositionEmbeddingSine
from rgb_stacking.utils.pose_estimator.models.transformer import Transformer


def conv2x2(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)


class CustomBottleneck(nn.Module):

    expansion: int = 4
    conv_fn = conv1x1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(CustomBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = CustomBottleneck.conv_fn(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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

    return nn.Sequential(*layers), inplanes

class DETRWrapper(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = resnet50(False)

        with torch.no_grad():
            _, z = self.parse_image( torch.randn(3, 3, 3, 200, 200, requires_grad=False))
            sz = np.product( z.shape[1:] )

        self.fc1 = nn.Linear(sz, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 21)

        # pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        # backbone_with_pos_enc = Joiner(backbone, pos_enc)
        # backbone_with_pos_enc.num_channels = backbone.num_channels
        # transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=False)
        # self.detr = DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=3)

    def parse_image(self, inputs):
        B = inputs.size(0)
        x = inputs.flatten(0, 1)

        x = self.backbone(x)

        x = x.flatten(1)
        x = x.view(B, 3, x.shape[-1]).flatten(1)
        return B, x

    def forward(self, inputs):

        B, x = self.parse_image(inputs)
        x = self.dropout( x )
        x = self.relu( self.bn_f1( self.fc1(x) ) )
        x = self.relu( self.bn_f2( self.fc2(x) ) )
        return  self.fc3(x)

class LargeVisionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn1 = torch.nn.BatchNorm2d(3)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=0)
        self.drop_out = torch.nn.Dropout(p=0.9)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.relu = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(64)

        inplanes = 64
        self.resnet_block_1, inplanes  = _make_layer(inplanes=inplanes, block=CustomBottleneck, planes=64, blocks=3, stride=1)

        CustomBottleneck.conv_fn = conv2x2
        self.resnet_block_2, inplanes = _make_layer(inplanes=inplanes, block=CustomBottleneck, planes=128, blocks=4, stride=2)
        self.resnet_block_3, inplanes = _make_layer(inplanes=inplanes, block=CustomBottleneck, planes=256, blocks=6, stride=2)
        self.resnet_block_4, inplanes = _make_layer(inplanes=inplanes, block=CustomBottleneck, planes=512, blocks=3, stride=2)

        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        with torch.no_grad():
            _, z = self.parse_image( torch.randn(3, 3, 3, 200, 200, requires_grad=False))
            sz = np.product( z.shape[1:] )

        self.fc1 = nn.Linear(sz, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 21)

    def parse_image(self, inputs):
        B = inputs.size(0)
        x = inputs.flatten(0, 1)

        x = self.bn1(x)
        x =  self.relu( self.drop_out( self.conv1(x) ) )

        x = self.relu( self.bn2( self.max_pool(x) ) )

        x = self.resnet_block_1(x)
        x = self.drop_out(x)

        x = self.resnet_block_2(x)
        x = self.resnet_block_3(x)
        x = self.resnet_block_4(x)

        x = self.max_pool_2(x)
        x = x.flatten(1)
        x = x.view(B, 3, x.shape[-1]).flatten(1)
        return B, x

    def forward(self, inputs):

        B, x = self.parse_image(inputs)

        x = self.dropout( x )
        x = self.relu( self.bn_f1( self.fc1(x) ) )
        x = self.relu( self.bn_f2( self.fc2(x) ) )
        return  self.fc3(x)

class VisionModule(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self):
        super().__init__()

        # 3, 200, 200
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
        self.fc2 = torch.nn.Linear(128, 21)

    def forward(self, inputs):
        B = inputs.size(0)
        x = inputs.flatten(0, 1)
        
        x = self.max_pool( self.relu( self.conv2( self.relu( self.conv1(x) ) ) ))
        
        x = self.resnet_block_4( self.resnet_block_3( self.resnet_block_2( self.resnet_block_1(x) ) ) )
        x = self.soft_max(x).flatten(1)

        x = x.view(B, 3, -1).flatten(1)
        x = self.relu(x)
        return  self.fc2( self.relu( self.fc1(x) ) )