import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import scipy
from scipy.ndimage import grey_dilation, grey_erosion

from models.modnet import MODNet
from dataloader import * 

class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input 
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()
            
        return self.op(x)
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blurer = GaussianBlurLayer(1, 3).to(device)

class MODNetLoss(nn.Module):
    def __init__(self, semantic_scale = 1.0,detail_scale=10.0, matte_scale=1.0):
        super(MODNetLoss, self).__init__()
        self.semantic_scale = semantic_scale
        self.detail_scale = detail_scale
        self.matte_scale = matte_scale
    def forward(self, pred_semantic, pred_detail, pred_matte, image, gt_matte, trimap):
        global blurer
        gt_semantic = F.interpolate(gt_matte, scale_factor=1/16, mode='bilinear')
        gt_semantic = blurer(gt_semantic.float())
        semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
        semantic_loss = self.semantic_scale * semantic_loss
        
        boundaries = (trimap < 0.5) + (trimap > 0.5)

        pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
        gt_detail = torch.where(boundaries, trimap, gt_matte)
        
        
        detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
        
        
        detail_loss = self.detail_scale * detail_loss
        pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
        matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
        matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
            + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
        matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
        matte_loss = self.matte_scale * matte_loss
        return semantic_loss, detail_loss, matte_loss


def train_model(model, dataloaders, criterion, optimizer,epochs=40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to GPU or CPU
    
    best_train_model = None
    best_valid_model = None
    best_train_loss = float('inf')  
    best_valid_loss = float('inf') 
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)
    train_loss = []
    val_loss = []
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+ 1}/{epochs}')
        print('-' * 50)

        # Training and validation phases
        for phase in ['train', 'val500p']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            semantic_running_loss = 0.0
            detail_running_loss = 0.0
            matte_running_loss = 0.0

            for idx, (images, alphas, trimaps) in enumerate(dataloaders[phase]):
                images = images.to(device)  # Move data to GPU or CPU
                alphas = alphas.to(device)  # Move data to GPU or CPU
                trimaps = trimaps.to(device)  # Move data to GPU or CPU

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    pred_semantic, pred_detail, pred_matte = model(images, False)
                    semantic_loss, detail_loss, matte_loss = criterion(pred_semantic, pred_detail, pred_matte, images, alphas, trimaps)
                    loss = semantic_loss + detail_loss + matte_loss

                    # Backward pass and optimization in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                semantic_running_loss += semantic_loss.item()
                detail_running_loss += detail_loss.item()
                matte_running_loss += matte_loss.item()
                        
                if (idx + 1) % 100 == 0:  # Print every 100 batches
                    print(f"Epoch [{epoch + 1}/{epochs}], Phase : {phase}, Batch [{idx + 1}/{len(dataloaders[phase])}], Loss : {loss.item()}")

            epoch_semantic_loss = semantic_running_loss / len(dataloaders[phase].dataset)
            epoch_detail_loss = detail_running_loss / len(dataloaders[phase].dataset)
            epoch_matte_loss = matte_running_loss / len(dataloaders[phase].dataset)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f'{phase} | semantic_loss: {epoch_semantic_loss} | detail_loss: {epoch_detail_loss} | matte_loss: {epoch_matte_loss} | Loss: {epoch_loss}')
            
            # Save the best model for the training phase
            if phase == 'train' and epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                best_train_model = model.state_dict()

            # Save the best model for the validation phase
            if phase == 'val500p' and epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss
                best_valid_model = model.state_dict()
                
            if phase == 'train':
                train_loss.append(epoch_matte_loss)
            if phase == 'val500p':
                val_loss.append(epoch_matte_loss)
                
        # Step the learning rate scheduler after each epoch
        lr_scheduler.step()
                
    return best_train_model, best_valid_model, model, train_loss, val_loss, lr_scheduler


if __name__ == '__main__' :
     
    # load image path and mask path

    train_paths =  generate_paths_for_dataset('TRAIN') # => [image_path, mask_path]
    val500p_paths = generate_paths_for_dataset('VAL500P') # => [image_path, mask_path]
    val500np_paths = generate_paths_for_dataset('VAL500NP') # => [image_path, mask_path]

   

    train_dataset = MattingDataset(datasets = train_paths, phase= 'train', transform= MattingTransform())
    val500p_dataset = MattingDataset(datasets = val500p_paths, phase= 'validation', transform= MattingTransform())
    val500np_dataset = MattingDataset(datasets = val500np_paths, phase= 'validation', transform= MattingTransform())
        

    batch_size = 1
    train_dataloader = data.DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    val500p_dataloader = data.DataLoader(val500p_dataset, batch_size= batch_size, shuffle= True)
    val500np_dataloader = data.DataLoader(val500np_dataset, batch_size= batch_size, shuffle= True)
    
    dataload_dir = {
        "train" : train_dataloader,
        "val500p" : val500p_dataloader,
        "val500np" : val500np_dataloader,
    }

    # start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

    lr = 0.01
    epochs = 20
    pretrained_ckpt = 'modnet_webcam_portrait_matting.ckpt'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    modnet = modnet.to(device)
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location= device))


    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    best_train_model, best_valid_model, final_model , train_loss, val_loss, lr_scheduler = train_model(modnet, dataload_dir,MODNetLoss(), optimizer, epochs=epochs)