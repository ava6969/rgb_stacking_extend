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


def train(train_loader, model,
          device, batch_size, total,
          optimizer,
          n_epochs=10):

    file = SummaryWriter()
    criterion = torch.nn.MSELoss()
    train_loss_min = np.inf
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 200)
    total_batches = 0

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
        
            
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))
                sys.exit()
            
             # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
            optimizer.step()
            
            train_loss += l

            total_batches += data.shape[0]

        step_lr.step()
        # calculate average losses
        train_loss = train_loss/total
  
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}  \t LR: {}'.format(
            epoch, train_loss, step_lr.get_last_lr()))
        file.add_scalar("Loss", train_loss, epoch)
        # save model if validation loss has decreased
        if train_loss <= train_loss_min:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min,
            train_loss))
            torch.save(model.state_dict(), 'model.pt')
            train_loss_min = train_loss


if __name__ == '__main__':

    print(DETRWrapper()(torch.rand(64, 3, 3, 200, 200)).shape)
    # utils.init_distributed_mode(args)
    HOME = os.environ["HOME"]
    print(HOME)
    N = mp.cpu_count()
    batch_size = 64
    examples = load_data( HOME + '/rgb_stacking_extend/rgb_stacking', jobs=N)
    sz = len(examples)
    print(f'Total Examples: {sz}')

    # img_transform = Normalize((0.485, 0.486, 0.406), (0.229, 0.224, 0.225))
    img_transform = Lambda(lambd= lambda x: x/255 )
    target_transform = ToTensor()

    train_ds = CustomDataset(examples, img_transform, target_transform)

    i = mt.proc_id()
    s = N // mt.num_procs() 
    N = mt.num_procs()

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=min(s, batch_size))

    #model = DETRWrapper(7)
    model = LargeVisionModule()
    model.to( 'cuda:0' )
    
    if torch.cuda.device_count() > 1:
                
        model = torch.nn.DataParallel(model)
        
        print('device: {} GPUS'.format(torch.cuda.device_count()))


    print(model)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=1e-3)
    train(train_dataloader, model, "cuda:0", batch_size,
          n_epochs=1000,
          total=len(train_ds) // batch_size,
          optimizer=optimizer)
