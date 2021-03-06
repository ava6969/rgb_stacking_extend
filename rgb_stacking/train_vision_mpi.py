#!/usr/bin/env python -W ignore::DeprecationWarning
import argparse

import mpi4py
from absl import app

mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
from rgb_stacking.run import init_env
import tensorflow as tf
import socket
from numpy import uint8
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Lambda
from rgb_stacking.utils.pose_estimator.model import VisionModule, LargeVisionModule
from rgb_stacking.utils.pose_estimator.dataset import CustomDataset, Buffer
from rgb_stacking.utils.pose_estimator.lars import LARS
import os
import multiprocessing as mp
import torch, tqdm
import rgb_stacking.utils.mpi_tools as mt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rgb_stacking.utils.pose_estimator.util.misc import setup_for_distributed
import logging

logging.disable(logging.CRITICAL)

def train(N_total_batches,
          N_training_samples,
          img_transform,
          target_transform,
          batch_size,
          no_dr,
          debug=False,
          root=0):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices([], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU, using:", gpus[0])
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    print('saved buffer')
    data_gen = Buffer( mt.proc_id(), N_training_samples, mt.num_procs() - 1, no_dr, debug)

    fl, fr, bl, poses = None, None, None, None

    if mt.proc_id() == root:
        train_loss_min = np.inf
        file = SummaryWriter()
        criterion = torch.nn.MSELoss()
        model = VisionModule().cuda()
        print(model)

        name = "large" if isinstance(model, LargeVisionModule) else "small"
        name += "no_dr" if no_dr else "dr"

        if batch_size > 64:
            optimizer = LARS(model.parameters(), lr=5e-4, max_epoch=N_total_batches * N_training_samples // batch_size)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 20000, gamma=0.5)

        groups = 4
        N = mt.num_procs()
        img_sz = [groups, N] + data_gen.img_size
        pose_size = [groups, N] + data_gen.pose_size

        fl, fr, bl, poses = np.zeros(img_sz, dtype=uint8),\
                            np.zeros(img_sz, dtype=uint8),\
                            np.zeros(img_sz, dtype=uint8), \
                            np.zeros(pose_size, dtype=float)

    flattened_img_size = [-1] + data_gen.img_size[1:]
    print('started training')
    for epoch in tqdm.tqdm( range(N_total_batches) ):

        if mt.proc_id() != root:
            data_gen.gather()

        mpi4py.MPI.COMM_WORLD.Gather(data_gen.fl, fl, root)
        mpi4py.MPI.COMM_WORLD.Gather(data_gen.fr, fr, root)
        mpi4py.MPI.COMM_WORLD.Gather(data_gen.bl, bl, root)
        mpi4py.MPI.COMM_WORLD.Gather(data_gen.poses, poses, root)

        if mt.proc_id() == root:
            train_batch = CustomDataset(dict(fl=fl.reshape(flattened_img_size),
                                             fr=fr.reshape(flattened_img_size),
                                             bl=bl.reshape(flattened_img_size),
                                             poses=poses.reshape(-1, 21) ), img_transform, target_transform)
            print(mp.cpu_count())
            train_dataloader = DataLoader(train_batch,
                                          batch_size=batch_size,
                                          shuffle=True,
                                        #   num_workers=0
                                          )

            train_loss = train_per_batch(train_dataloader,
                                         model,
                                         N_training_samples,
                                         optimizer,
                                         criterion, batch_size)
            step_lr.step()

            # calculate average losses
            train_loss = train_loss / N_training_samples
            print('Epoch: {} \tTraining Loss: {:.6f} LR: {}'.format(epoch, train_loss, step_lr.get_last_lr()))
            file.add_scalar("Train Loss", train_loss, epoch)

            # save model if validation loss has decreasedN_total_batches
            if train_loss <= train_loss_min:
                print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    train_loss_min,
                    train_loss))
                torch.save(model, '{}_model_{}.pt'.format(name, batch_size))
                train_loss_min = train_loss


def train_per_batch(train_loader, model, total, optimizer, criterion, batch_size):

    ###################
    # train the model #
    ###################
    model.train()

    # keep track of training and validation loss
    train_loss = 0.0

    for ii, (data, target) in tqdm.tqdm( enumerate(train_loader), total=total // batch_size):

        data, target = {k : d.cuda()for k, d in data.items()}, target.cuda()

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        l = loss.item()*batch_size

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

    N = mp.cpu_count()
    no_dr = True
    debug = False
    batch_size = 64
    N_training_samples = int(1e6) 
    N_total_batches = 400000

    if no_dr:
        img_transform = Lambda(lambd= lambda x: x/255 )
    else:
        img_transform = None

    target_transform = ToTensor()

    root = 0
    mt.msg( f"root {root}, host_name {socket.gethostname()} , N {mt.num_procs()} ")

    setup_for_distributed(mt.proc_id() == root)
    train(N_total_batches, N_training_samples, img_transform, target_transform, batch_size, no_dr, debug, root)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
