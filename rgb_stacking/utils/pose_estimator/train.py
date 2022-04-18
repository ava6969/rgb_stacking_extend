import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms.transforms import Lambda
from model import VisionModule, LargeVisionModule, DETR
from dataset import CustomDataset, load_data
from lars import LARS
import os
import multiprocessing as mp
import torch, tqdm
import numpy as np


def train(train_loader, model,
          device, batch_size, total,
          optimizer,
          n_epochs=10):

    criterion = torch.nn.MSELoss()
    train_loss_min = np.inf
    stepT = 40000 // batch_size
    step_lr = torch.optim.lr_scheduler.ConstantLR(optimizer, 0.5, stepT)
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
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            l = loss.item()*data.size(0)
            
            assert( l != np.nan )
            train_loss += l

            total_batches += data.shape[0]

            step_lr.step()

        # calculate average losses
        train_loss = train_loss/total
  
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}  \t LR: {}'.format(
            epoch, train_loss, step_lr.get_last_lr()))
        
        # save model if validation loss has decreased
        if train_loss <= train_loss_min:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min,
            train_loss))
            torch.save(model.state_dict(), 'model.pt')
            train_loss_min = train_loss


if __name__ == '__main__':

    HOME = os.environ["HOME"]
    print(HOME)
    N = mp.cpu_count()
    batch_size = 64
    examples = load_data( HOME + '/rgb_stacking_extend/rgb_stacking', jobs=N, sz=60)
    sz = len(examples)
    print(f'Total Examples: {sz}')


    img_transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    target_transform = ToTensor()

    train_ds = CustomDataset(examples, img_transform, target_transform)
                                  
    import utils.mpi_tools as mt
    i = mt.proc_id()
    s = N // mt.num_procs() 
    N = mt.num_procs()

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=min(s, batch_size))

    model = DETR(21)
    model.to( 'cuda:0' )
    
    if torch.cuda.device_count() > 1:
                
        model = torch.nn.DataParallel(model)
        
        print('device: {} GPUS'.format(torch.cuda.device_count()))


    print(model)
    optimizer = LARS(model.parameters(), 1e-4, weight_decay=0.0001)
    train(train_dataloader, model, "cuda:0", batch_size,
          n_epochs=100,
          total=len(train_ds) // batch_size,
          optimizer=optimizer)
