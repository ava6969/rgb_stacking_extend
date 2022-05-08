#!/usr/bin/env python -W ignore::DeprecationWarning
import argparse
import asyncio
from _weakref import ref

from absl import app
from stable_baselines3.common.vec_env import CloudpickleWrapper

from rgb_stacking.run import init_env
import tensorflow as tf
import socket
from numpy import uint8
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Lambda
from rgb_stacking.utils.pose_estimator.model import VisionModule, LargeVisionModule
from rgb_stacking.utils.pose_estimator.dataset import CustomDataset, Buffer, VecBuffer
from rgb_stacking.utils.pose_estimator.lars import LARS
import os
import multiprocessing as mp
import torch, tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rgb_stacking.utils.pose_estimator.util.misc import setup_for_distributed
import logging

logging.disable(logging.CRITICAL)

def optimize(model,
             N_training_samples,
             batch_size,
             optimizer,
             criterion,
             step_lr,
             file,
             train_loss_min,
             name,
             img_transform, target_transform,
             data):
    epoch, (fr, fl, bl, poses) = data

    train_batch = CustomDataset(dict(fl=fl, fr=fr, bl=bl, poses=poses), img_transform, target_transform)

    train_dataloader = DataLoader(train_batch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=mp.cpu_count(),
                                  pin_memory=True)

    train_loss = train_per_batch(train_dataloader,
                                 model,
                                 N_training_samples,
                                 optimizer,
                                 criterion, batch_size)
    step_lr.step()

    # calculate average losses
    train_loss = train_loss / N_training_samples
    print('\nEpoch: {} \tTraining Loss: {:.6f} LR: {}'.format(epoch, train_loss, step_lr.get_last_lr()))
    file.add_scalar("Train Loss", train_loss, epoch)

    # save model if validation loss has decreasedN_total_batches
    if train_loss <= train_loss_min:
        print('\nTraining loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min,
            train_loss))
        torch.save(model, '{}_model_{}.pt'.format(name, batch_size))
        train_loss_min = train_loss
    return train_loss_min


def train(N_workers,
          N_total_batches,
          N_training_samples,
          batch_size,
          no_dr,
          debug=False):

    if no_dr:
        img_transform = Lambda(lambd=lambda x: x / 255)
    else:
        img_transform = None

    target_transform = ToTensor()

    train_loss_min = np.inf
    file = SummaryWriter()
    criterion = torch.nn.MSELoss()
    print(torch.cuda.device_count())
    model = VisionModule().cuda()
    print(model)
    name = "large" if isinstance(model, LargeVisionModule) else "small"
    name += "no_dr" if no_dr else "dr"

    if batch_size > 64:
        optimizer = LARS(model.parameters(),
                         lr=0.5,
                         max_epoch=N_total_batches)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=5e-4,
                                     weight_decay=1e-3)

    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 40000, gamma=0.5)

    data_gen = VecBuffer(N_training_samples, N_workers, no_dr, debug, "forkserver")
    epoch = 0
    for data in data_gen.gather(N_total_batches):
        epoch += 1
        train_loss_min = optimize(model,
                 N_training_samples,
                 batch_size,
                 optimizer,
                 criterion,
                 step_lr,
                 file,
                 train_loss_min,
                 name,
                 img_transform, target_transform,
                 (epoch, data))



def train_per_batch(train_loader, model, total, optimizer, criterion, batch_size):
    ###################
    # train the model #
    ###################
    model.train()

    # keep track of training and validation loss
    train_loss = 0.0

    for ii, (data, target) in tqdm.tqdm(enumerate(train_loader), total=total // batch_size):
        data, target = {k: d.cuda() for k, d in data.items()}, target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        l = loss.item() * batch_size

        loss.backward()

        optimizer.step()

        train_loss += l

    return train_loss


def main(argv):
    parser = argparse.ArgumentParser('Runner')
    parser.add_argument('-l', '--debug_specs', type=bool, default=False)
    # parser.add_argument('-r', '--root', type=int)
    args = parser.parse_args()

    init_env()
    # HOME = os.environ["HOME"]
    # print(HOME)

    N = 32
    no_dr = True
    debug = False
    batch_size = 256
    N_training_samples = int(1e4)
    N_total_batches = 300000

    train(N, N_total_batches, N_training_samples, batch_size, no_dr, debug)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
