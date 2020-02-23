# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import numpy as np
import torch

import utils
from q0_hello_mnist import SimpleCNN
from Caffe_Net import CaffeNet
from voc_dataset import VOCDataset
import ipdb

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

from tensorboardX import SummaryWriter
import time
from torchvision.utils import make_grid

def visualize_data_sample(tensor_img):
    tensor_img = tensor_img.int()
    plt.imshow(tensor_img.permute(1, 2, 0))
    plt.show()

def visualize_filter(kernels,epoch,path):
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    img = make_grid(kernels)
    plt.imshow(img.permute(1, 2, 0))
    plt.savefig(path+'_'+str(epoch))

def main():
    # TODO:  Initialize your visualizer here!
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+timestr+'/'
    writer = SummaryWriter(save_path)

    # TODO: complete your dataloader in voc_dataset.py
    train_loader = utils.get_data_loader('voc', train=True, batch_size=args.batch_size, split='trainval')
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')

    # 2. define the model, and optimizer.
    # TODO: modify your model here!
    # bad idea of use simple CNN, but let's give it a shot!
    # In task 2, 3, 4, you might want to modify this line to be configurable to other models.
    # Remember: always reuse your code wisely.
    if(int(args.model_to_use)==1):
        model = SimpleCNN(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3).to(device)
        MODEL_SAVE_PATH = "../Saved_Models/trained_simpleCNNmodel"
        print("Using SimpleCNN")
    elif(int(args.model_to_use)==2):
        model = CaffeNet(num_classes=len(VOCDataset.CLASS_NAMES), inp_size=64, c_dim=3,dropout_prob=0.5).to(device)
        MODEL_SAVE_PATH = "../Saved_Models/trained_CaffeNet"
        print("Using CaffeNet")
    else:
        print("Select Correct model_to_use")
        raise NotImplementedError

    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    cnt = 0

    #Model Save Code
    model_save_epochs = np.arange(args.epochs/5,args.epochs,step=args.epochs/5,dtype=int)
    if(model_save_epochs[-1]!=args.epochs):
        model_save_epochs = np.append(model_save_epochs,args.epochs)
    index_model_save_epochs = 0

    #Visualize Filter Code
    filter_visualization_epochs = np.arange(args.epochs/3,args.epochs,step=args.epochs/3,dtype=int)
    if(filter_visualization_epochs[-1]!=args.epochs):
        filter_visualization_epochs = np.append(filter_visualization_epochs,args.epochs)
    index_filter_visualization_epochs = 0    


    for epoch in range(args.epochs):
        print("---------------EPOCH: ",epoch+1," ------------------------")
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            # Get a batch of data
            data, target, wgt = data.to(device), target.to(device), wgt.to(device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            # TODO: your loss for multi-label clf?
            loss = F.multilabel_soft_margin_loss(output,target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                # todo: add your visualization code
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                # ap, map = utils.eval_dataset_map(model, device, test_loader)
                model.train()
            cnt += 1
        scheduler.step()
        writer.add_scalar('Train/Loss', loss, epoch)

        if(model_save_epochs[index_model_save_epochs] == epoch+1):
            print("Saving Model ",epoch+1)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, MODEL_SAVE_PATH)
            index_model_save_epochs+=1

        if(filter_visualization_epochs[index_filter_visualization_epochs] == epoch+1):
            print("Extracting Filter",epoch+1)
            index_filter_visualization_epochs+=1
            kernels = model.conv1.weight.detach().clone()
            visualize_filter(kernels,epoch,MODEL_SAVE_PATH)

    # Validation iteration
    # test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test')
    # ap, map = utils.eval_dataset_map(model, device, test_loader)

    print('----test-----')
    # print(ap)
    # print('mAP: ', map)
    # writer.close()



if __name__ == '__main__':
    args, device = utils.parse_args()
    main()