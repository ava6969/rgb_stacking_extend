import torch.nn as nn
from torchvision.models import resnet50
import torch
from torchvision.models.resnet import BasicBlock
import numpy as np


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

class DETR(nn.Module):
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
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.skip_trans =  False
        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(3, hidden_dim))
        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):

        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        if self.fc:
            return self.fc( h.flatten(1) )

        # construct positional encodings
        H, W = h.shape[-2:]
        print(H, W)
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return self.linear_class(h)

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
        self.dropout = torch.nn.Dropout(0.9, True)
        self.backbone.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn_f1 = torch.nn.BatchNorm1d(512)
        self.bn_f2 = torch.nn.BatchNorm1d(256)

        self.relu = torch.nn.ReLU(True)

        with torch.no_grad():
            _, z = self.parse_image( torch.randn(3, 3, 3, 200, 200))
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