#!/usr/bin/env python -W ignore::DeprecationWarning
import argparse
import asyncio
from _weakref import ref

import torchvision.transforms
from absl import app
from stable_baselines3.common.vec_env import CloudpickleWrapper

from rgb_stacking.run import init_env
import tensorflow as tf
import socket
from numpy import uint8
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Lambda
from rgb_stacking.utils.pose_estimator.model import VisionModule, LargeVisionModule, DETRWrapper
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
             img_transform, target_transform, total_training_batches,
             data):
    epoch, (fr, fl, bl, poses) = data

    train_batch = CustomDataset(dict(fl=fl, fr=fr, bl=bl, poses=poses), img_transform, target_transform)

    train_dataloader = DataLoader(train_batch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=mp.cpu_count(),
                                  pin_memory=True)

    train_loss, total_training_batches = train_per_batch(train_dataloader,
                                 model,
                                 N_training_samples,
                                 optimizer,
                                 criterion, batch_size, step_lr, total_training_batches)

    # calculate average losses
    train_loss = train_loss / N_training_samples
    print('\nEpoch: {}, Total_training_batches {} \tTraining Loss: {:.6f} LR: {}'.format(epoch, total_training_batches, train_loss, step_lr.get_last_lr()))
    file.add_scalar("Train Loss", train_loss, epoch)

    # save model if validation loss has decreasedN_total_batches
    if train_loss <= train_loss_min:
        print('\nTraining loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min,
            train_loss))
        torch.save(model, '{}_model_{}.pt'.format(name, batch_size))
        train_loss_min = train_loss
    return train_loss_min, total_training_batches


def train(N_workers,
          model_name,
          N_training_samples,
          no_dr,
          debug=False,
          focal_loss=False):

    batch_size = 64
    if model_name == "core":
        model = VisionModule().cuda()
    elif model_name == "detr":
        model = DETRWrapper().cuda()
    else:
        model = LargeVisionModule().cuda()
        batch_size = 256

    N_total_batches = 300000
    name = model_name
    name += "no_dr" if no_dr else "dr"
    name += socket.gethostname()

    criterion = torch.nn.MSELoss() if not focal_loss else None

    if no_dr:
        img_transform = Lambda(lambd=lambda x: x / 255)
    else:
        img_transform = None

    target_transform = ToTensor()

    train_loss_min = np.inf
    file = SummaryWriter()

    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel( model )
        

    print(model)

    if model_name == "core":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=5e-4,
                                     weight_decay=1e-3)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 20000, gamma=0.5)
    elif model_name == 'detr':
        optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=1e-4,
                                          weight_decay=1e-4)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 200000)
    else:
        optimizer = LARS(model.parameters(),
                         lr=0.5,
                         max_epoch=N_total_batches)
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 40000, gamma=0.5)


    data_gen = VecBuffer(N_training_samples, N_workers, no_dr, debug, "forkserver")
    epoch = 0
    total_training_batches = 0
    for data in data_gen.gather(N_total_batches):
        epoch += 1
        train_loss_min, total_training_batches = optimize(model,
                                                          N_training_samples,
                                                          batch_size,
                                                          optimizer,
                                                          criterion,
                                                          step_lr,
                                                          file,
                                                          train_loss_min,
                                                          name,
                                                          img_transform, target_transform, total_training_batches,
                                                          (epoch, data))



def train_per_batch(train_loader, model, total, optimizer, criterion, batch_size, step_lr, total_training_batches):
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
        total_training_batches += 1
        step_lr.step()

    return train_loss, total_training_batches


def main(argv):
    init_env()
    model = argv[1]

    N = 100
    no_dr = False
    debug = True
    N_training_samples = int(1.5e5)

    train(N, model, N_training_samples, no_dr, debug)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
