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
