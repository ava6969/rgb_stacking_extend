import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms.transforms import Lambda
from model import VisionModule, LargeVisionModule, DETRWrapper
from dataset import CustomDataset, load_data
from lars import LARS
import os
import multiprocessing as mp
import torch, tqdm
import rgb_stacking.utils.mpi_tools as mt
import numpy as np
import math
import sys
from torch.utils.tensorboard import SummaryWriter


def train(train_loader, valid_loader, model,
          device, batch_size, total, valid_total,
          optimizer,
          n_epochs=10):

    file = SummaryWriter()
    criterion = torch.nn.MSELoss()
    valid_loss_min = np.inf
    valid_loss = 0
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 20000, gamma=0.5)
    total_batches = 0
    name = "large" if isinstance(model, LargeVisionModule) else "small"

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()

        for ii, (data, target) in tqdm.tqdm( enumerate(train_loader), total=total):
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            
            # update training loss
            l = loss.item()*data.size(0)
            
             # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
   
            optimizer.step()
            # step_lr.step()

            train_loss += l

            total_batches += data.shape[0]
            
            if ii == total:
                break
            
        model.eval()
        for ii, (data, target) in tqdm.tqdm( enumerate(valid_loader), total=valid_total):
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            with torch.no_grad():
                output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

        
        # calculate average losses
        train_loss = train_loss/total
        valid_loss = valid_loss/valid_total
  
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} Validation Loss: {:.6f} LR: {}'.format(epoch, train_loss, valid_loss, step_lr.get_last_lr()) )
        file.add_scalar("Train Loss", train_loss, epoch)
        file.add_scalar("Validation Loss", valid_loss, epoch)
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model, '{}_model_{}.pt'.format(name, batch_size))
            valid_loss_min = valid_loss


if __name__ == '__main__':

    HOME = os.environ["HOME"]
    print(HOME)
    N = mp.cpu_count()
    batch_size = 256
    epochs = 300
    
    examples = load_data( HOME + '/rgb_stacking_extend/rgb_stacking', jobs=N)
    
    sz = len(examples)
    
    print(f'Total Examples: {sz}')
    
    img_transform = Lambda(lambd= lambda x: x/255 )
    target_transform = ToTensor()

    N = int(1e6)
    train_ds = CustomDataset(examples[:N], img_transform, target_transform)
    
    valid_ds = CustomDataset(examples[N:], img_transform, target_transform)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=mp.cpu_count())
    
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=mp.cpu_count())

    model = VisionModule(True)
    model.to( 'cuda' )
    
    if torch.cuda.device_count() > 1:
                
        model = torch.nn.DataParallel(model)
        
        print('device: {} GPUS'.format(torch.cuda.device_count()))


    print(model)
    
    if batch_size >= 64:
        optimizer = LARS(model.parameters(), lr=5e-4,max_epoch=epochs*1e6//batch_size)
    else:
        optimizer = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=1e-3)
    
    train(train_dataloader, 
          valid_dataloader, 
          model, "cuda",
          batch_size,
          n_epochs=epochs,
          total= len(train_ds)//batch_size,
          valid_total=len(valid_ds) // batch_size,
          optimizer=optimizer)
