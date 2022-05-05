import mpi4py.MPI
import tensorflow as tf
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


def train(N_total_batches,
          N_training_samples,
          img_transform,
          target_transform,
          batch_size,
          no_dr,
          debug=False):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU, using:", gpus[0])
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    data_gen = Buffer(N_training_samples, mt.num_procs(), no_dr, debug)
    fl, fr, bl, poses = None, None, None, None

    if mt.proc_id() == 0:
        train_loss_min = np.inf
        file = SummaryWriter()
        criterion = torch.nn.MSELoss()
        model = VisionModule().to('cuda:1')
        print(model)

        name = "large" if isinstance(model, LargeVisionModule) else "small"
        name += "no_dr" if no_dr else "dr"

        if batch_size > 64:
            optimizer = LARS(model.parameters(), lr=5e-4, max_epoch=N_total_batches * N_training_samples // batch_size)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 40000, gamma=0.5)

        N = mt.num_procs()
        img_sz = [N] + data_gen.img_size
        pose_size = [N] + data_gen.pose_size

        fl, fr, bl, poses = np.empty(img_sz, dtype=uint8),\
                            np.empty(img_sz, dtype=uint8),\
                            np.empty(img_sz, dtype=uint8), \
                            np.empty(pose_size, dtype=float)

    flattened_img_size = [-1] + data_gen.img_size[1:]

    for epoch in tqdm.tqdm( range(N_total_batches) ):

        data_gen.gather()

        mpi4py.MPI.COMM_WORLD.Gather(data_gen.fl, fl, 0)
        mpi4py.MPI.COMM_WORLD.Gather(data_gen.fr, fr, 0)
        mpi4py.MPI.COMM_WORLD.Gather(data_gen.bl, bl, 0)
        mpi4py.MPI.COMM_WORLD.Gather(data_gen.poses, poses, 0)

        if mt.proc_id() == 0:
            train_batch = CustomDataset(dict(fl=fl.reshape(flattened_img_size),
                                             fr=fr.reshape(flattened_img_size),
                                             bl=bl.reshape(flattened_img_size),
                                             poses=poses.reshape(-1, 21) ), img_transform, target_transform)

            train_dataloader = DataLoader(train_batch,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=mp.cpu_count())

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

    for ii, (data, target) in tqdm.tqdm( enumerate(train_loader), total=total):

        data, target = {k : d.to('cuda:1') for k, d in data.items()}, target.to('cuda:1')

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        l = loss.item()*batch_size

        loss.backward()

        optimizer.step()

        train_loss += l

    return train_loss


if __name__ == '__main__':

    setup_for_distributed(mt.proc_id() == 0)

    HOME = os.environ["HOME"]
    print(HOME)

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

    train(N_total_batches, N_training_samples, img_transform, target_transform, batch_size, no_dr, debug)