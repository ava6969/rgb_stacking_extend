from typing import List, Dict

import numpy as np
import torch.nn
import torch.nn as nn
from rgb_stacking.contrib.common import init, Flatten


class NatureNet(nn.Module):
    def __init__(self, image_shape, out_size):
        super(NatureNet, self).__init__()
        n, w, h = image_shape
        img_init = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.base = nn.Sequential(
            img_init(nn.Conv2d(n, 32, kernel_size=(8, 8), stride=(4, 4))), nn.ReLU(),
            img_init(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))), nn.ReLU(),
            img_init(nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))), nn.ReLU(), Flatten(),
            img_init(nn.Linear(32 * 7 * 7, out_size)), nn.ReLU())

    def forward(self, inputs):
        return self.base(inputs / 255.0)


class ResidualBlock(nn.Module):
    def __init__(self, image_shape, filter_sz, kernel_sz=(3, 3), flatten_out=True, relu_out=True):
        super(ResidualBlock, self).__init__()

        c, h, w = image_shape
        img_init = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = img_init(torch.nn.Conv2d(c, filter_sz, kernel_sz, (1, 1), 'same'))
        self.conv2 = img_init(torch.nn.Conv2d(c, filter_sz, kernel_sz, (1, 1), 'same'))
        self.flatten_out = flatten_out
        self.relu_out = relu_out
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(x)
        x = self.relu( self.conv1(x) )
        x = self.conv2(x)
        x += identity

        x = torch.flatten(x) if self.flatten_out else x
        return self.relu(x) if self.relu else x


class ImpalaResidualBlock(nn.Module):
    def __init__(self, image_shape: List, filter_sz, flatten_out, relu_out, apply_batch_norm):
        super(ImpalaResidualBlock, self).__init__()

        c, h, w = image_shape
        img_init = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = img_init(torch.nn.Conv2d(c, filter_sz, (3, 3), (1, 1), 'same'))
        self.max_pool = torch.nn.MaxPool2d(3, stride=2, padding=1)

        with torch.no_grad():
            x = torch.zeros([1] + image_shape, requires_grad=False)
            y = self.max_pool(self.conv1(x))
            _image_shape = list(y.shape[1:])

        self.res_block1 = ResidualBlock(_image_shape, filter_sz, (3, 3), False, False)
        self.res_block2 = ResidualBlock(_image_shape, filter_sz, (3, 3), flatten_out, relu_out)
        if apply_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(filter_sz)
        else:
            self.batch_norm = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm(out) if self.batch_norm is not None else out
        return self.res_block2(self.res_block1(self.max_pool(out)))


class ImpalaResnet(nn.Module):
    def __init__(self, image_shape: List, filters, flatten_out):
        super(ImpalaResnet, self).__init__()

        self.residual_blocks = torch.nn.Sequential()
        self.output_shape = image_shape
        for i, filter in enumerate(filters, 1):
            with torch.no_grad():
                m = ImpalaResidualBlock(self.output_shape, filter, i == len(filters) and flatten_out, False, True)
                v = torch.zeros([1] + self.output_shape, requires_grad=False)
                out = m(v)
                self.output_shape = list(out.shape[1:])
                self.residual_blocks.add_module('residual_block_{}'.format(i), m)

        if flatten_out:
            self.output_shape = np.prod(image_shape)

    def forward(self, x):
        return self.residual_blocks(x)


class RelationalImpalaResnet(nn.Module):
    def __init__(self, _shape, filters ):
        super(RelationalImpalaResnet, self).__init__()

        image_shape = list(_shape)
        self.resnet = ImpalaResnet(image_shape, filters, False)
        with torch.no_grad():
            var = torch.zeros(4, *image_shape, requires_grad=False)
            var = self.resnet(var)
            var = var.flatten(start_dim=-2).transpose(-1, -2)
        c = torch.zeros(*var.shape[:-1], 1)
        var = torch.cat([var, c, c], dim=-1)
        self.output_size = var.shape[-2:]

    def forward(self, x):
        x = self.resnet(x)
        ncols = x.shape[-1]
        x = x.flatten(start_dim=-2).transpose(-1, -2)
        c = torch.arange(x.shape[-2]).expand(*x.shape[:-1]).to(x.dtype)
        x_coord = (c % ncols).view(*x.shape[:-2], -1, 1)
        y_coord = (c // ncols).view(*x.shape[:-2], -1, 1)
        return torch.cat([x, x_coord, y_coord], dim=-1)


class RobotImageResnetModule(nn.Module):
    def __init__(self, image_shape, filters):
        super().__init__()

        self.image_bl = ImpalaResnet(image_shape, filters, False)
        self.image_fl = ImpalaResnet(image_shape, filters, False)
        self.image_fr = ImpalaResnet(image_shape, filters, False)
        x = [3] + self.image_bl.output_shape
        self._flat_size = np.prod(x)

    def forward(self, x_dict: Dict):
        bl = self.image_bl( x_dict.pop('image_bl'))  # N, 4, 130
        fl = self.image_fl( x_dict.pop('image_fl'))
        fr = self.image_fr( x_dict.pop('image_fr'))
        x_cat = torch.cat([bl, fl, fr], 1)

        x_flat = torch.flatten(x_cat, 1)
        for k, v in x_dict.items():
            if 'past' not in k:
                x_dict[k] = torch.cat([v, x_flat], 1)

        return x_dict

    @property
    def flat_size(self):
        return self._flat_size

def main():

    x = torch.zeros((4, 3, 128, 128))
    # m = ImpalaResnet(list(x.shape[1:]), [32, 32, 64, 64, 128, 128], False)
    # print(m)
    #
    # y = m(x)
    # print(y.shape)

    # m = RelationalImpalaResnet(x.shape[1:], [32, 32, 64, 64, 128, 128])
    # print(m)
    #
    # y = m(x)
    # print(y.shape)
    # print(y)
    # assert y.shape[1:] == m.output_size

    m = RobotImageResnetModule( list(x.shape[1:]), [32, 32, 64, 64, 64, 128, 128])
    print(m)

    x = dict(box=torch.zeros((4, 12)), agent=torch.zeros((4, 87)), image_bl=x, image_fl=x, image_fr=x  )
    y = m(x)
    print(y)



if __name__ == '__main__':
    main()