import torch.nn as nn
from torchvision.models import resnet50
import torch


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
    def __init__(self, num_classes):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        self.backbone.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)
        self.do = torch.nn.Dropout(0.9)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn_f1 = torch.nn.BatchNorm1d(512)
        self.bn_f2 = torch.nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(12 * 12 * 2048 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 21)

        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        B = inputs.size(0)
        x = inputs.flatten(0, 1)
        x = self.bn1(x)

        x = self.do( self.backbone.conv1(x) )
        x = self.mp1( x )
        x = self.bn2(x)
        x = self.backbone.relu( x )

        x = self.backbone.layer1(x)
        x = self.do( x )

        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.mp2(x)
        x = x.view(B, 3, -1).flatten(1)

        x = self.do( x )
        x = self.relu( self.bn_f1( self.fc1(x) ) )
        x = self.relu( self.bn_f2( self.fc2(x) ) )
        return  self.fc3(x)

