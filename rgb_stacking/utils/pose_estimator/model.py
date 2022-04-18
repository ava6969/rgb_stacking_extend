import torch.nn as nn
from torchvision.models import resnet50
import torch
from torchvision.models.resnet import BasicBlock
import numpy as np
import detr.hubconf as hc


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

        hidden_dim = 512
        backbone = hc.Backbone("resnet50", train_backbone=True, return_interm_layers=False, dilation=False)
        pos_enc = hc.PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = hc.Joiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels
        transformer = hc.Transformer(d_model=hidden_dim, return_intermediate_dec=True)
        self.detr = hc.DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=5)

    def forward(self, inputs):

        # propagate inputs through ResNet-50 up to avg-pool layer
        B = inputs.size(0)
        x = inputs.flatten(0, 1)

        x = self.detr(x)

        # finally project transformer outputs to class labels and bounding boxes
        return x

class LargeVisionModule(nn.Module):
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

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc
        del self.backbone.avgpool

        self.backbone.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)
        self.backbone.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = torch.nn.Dropout(0.9)
        self.backbone.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn_f1 = torch.nn.BatchNorm1d(512)
        self.bn_f2 = torch.nn.BatchNorm1d(256)

        self.relu = torch.nn.ReLU()

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
        x =  self.relu( self.dropout( self.backbone.conv1(x) ) )

        x = self.relu( self.bn2( self.backbone.maxpool(x) ) )

        x = self.backbone.layer1(x)
        x = self.dropout(x)

        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.maxpool2(x)
        x = x.flatten(1).view(B, -1)
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