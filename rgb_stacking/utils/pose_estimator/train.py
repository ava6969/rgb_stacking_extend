import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from model import VisionModule
from dataset import CustomDataset, load_data
from lars import LARS
import os
import multiprocessing as mp
import torch, tqdm
import numpy as np


def train(train_loader, valid_loader, model, device, batch_size, total, valid_total, n_epochs=10, lr=0.5):
    optimizer = LARS(model.parameters(), lr, 0.9, 0.0001)
    criterion = torch.nn.MSELoss()
    valid_loss_min = -np.inf
    stepT = 40000 // batch_size
    step_lr = torch.optim.lr_scheduler.ConstantLR(optimizer, 0.5, stepT)

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        
        for ii, (data, target) in tqdm.tqdm( enumerate(train_loader), total= total):
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            step_lr.step()
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for i, (data, target) in tqdm.tqdm( enumerate(valid_loader), total=valid_total):
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/total
        valid_loss = valid_loss/valid_total
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t LR: {}'.format(
            epoch, train_loss, valid_loss, step_lr.get_last_lr()))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss


if __name__ == '__main__':
    HOME = os.environ["HOME"]
    print(HOME)
    N = mp.cpu_count()
    batch_size = 64
    examples = load_data( HOME + '/rgb_stacking_extend/rgb_stacking', jobs=N)
    sz = len(examples)
    print(f'Total Examples: {sz}')
    train_sz, valid_sz = int(0.6 * sz), int(0.2 * sz)
    train_dt, valid_dt, test_dt = examples[:train_sz], examples[train_sz: train_sz + valid_sz], examples[train_sz + valid_sz:]

    img_transform = Normalize(0.1307, 0.3081)
    target_transform = ToTensor()

    train_ds, valid_ds, test_ds = CustomDataset(train_dt, img_transform, target_transform), \
                                  CustomDataset(valid_dt, img_transform, target_transform), \
                                  CustomDataset(test_dt, img_transform, target_transform)
                                  
    import utils.mpi_tools as mt
    i = mt.proc_id()
    s = N // mt.num_procs() 
    N = mt.num_procs()

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=min(s, batch_size))
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=min(s, batch_size))
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)


    model = VisionModule()
    model.to( 'cuda:0' )
    
    if torch.cuda.device_count() > 1:
                
        model = torch.nn.DataParallel(model)
        
        print('device: {} GPUS'.format(torch.cuda.device_count()))


    print(model)

    train(train_dataloader, valid_dataloader, model, "cuda:0", batch_size,
          total=len(train_ds) // batch_size,
          valid_total=len(valid_dt) // batch_size)
