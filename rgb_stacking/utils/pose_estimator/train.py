import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from model import VisionModule, DETR
from dataset import CustomDataset, load_data
from lars import lars
import tqdm


def train(train_loader, valid_loader, model, device, batch_size, n_epochs=10, lr=0.5):
    optimizer = lars.LARS(model.parameters(), lr, 0.9, 0.0001)
    criterion = torch.nn.MSELoss()
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
        for data, target in train_loader:
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
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
            
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
    examples = load_data('/home/dewe/rgb_stacking_extend/rgb_stacking', 1)
    sz = len(examples)
    print(f'Total Examples: {sz}')
    batch_size = 4 
    train_sz, valid_sz = int(0.7 * sz), int(0.9 * sz)
    train_dt, valid_dt, test_dt = examples[:train_sz], examples[train_sz:valid_sz], examples[valid_sz:]

    img_transform = Normalize(0.1307, 0.3081)
    target_transform = ToTensor()

    train_ds, valid_ds, test_ds = CustomDataset(train_dt, img_transform, target_transform), \
                                  CustomDataset(valid_dt, img_transform, target_transform), \
                                  CustomDataset(test_dt, img_transform, target_transform)

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=32)
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model = VisionModule(7)
    model.to( 'cuda' )

    print(model)

    train(train_dataloader, valid_dataloader, model, "cuda", batch_size)